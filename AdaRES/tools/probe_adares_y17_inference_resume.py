"""Student inference identity and native checkpoint restore for Y17."""

import argparse
import copy
import gc
import importlib
import json
import time
from pathlib import Path

import gorilla
import torch

from diagnose_adares_y14_reliability import scene_indices
from ipdn.dataset import build_dataset
from ipdn.model_paper5.adares_y17_target_equivalent_pairing import IPDN, build_train_dataset
from ipdn.utils import get_root_logger
from probe_adares_y15_resume import cpu_state, differences, rng, restore
from train import load_shape_compatible_pretrain


def step(model, optimizer, batch, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, values = model(batch, mode="loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    optimizer.step()
    result = {"loss": float(loss), "pair_js": float(values["pair_js_loss"]),
              "finite": bool(torch.isfinite(loss)), "pair_count": model.last_pair_count}
    return result


def predict(model, batch):
    captured = []
    hook = model.dec.register_forward_hook(
        lambda module, inputs, out: captured.append(out))
    ret = model(batch, mode="predict")
    hook.remove()
    return {"probabilities": [p.detach().cpu().clone() for p in ret["pred_pmask"]],
            "indices": captured[0]["indis"].softmax(-1)[..., 1].argmax(-1).cpu(),
            "seeds": captured[0]["sample_inds"].cpu(),
            "queries": captured[0]["masks"].shape[1]}


def comparison(a, b):
    return {"probability_max": max(float((x-y).abs().max())
                                  for x, y in zip(a["probabilities"], b["probabilities"])),
            "binary_flips": sum(int((x.gt(.5) != y.gt(.5)).sum())
                                for x, y in zip(a["probabilities"], b["probabilities"])),
            "selection_flips": int((a["indices"] != b["indices"]).sum()),
            "seed_flips": int((a["seeds"] != b["seeds"]).sum())}


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "output", "mode"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    if args.mode == "inference":
        dc = cfg.data.train.copy()
        dc.training = dc.aug = dc.with_elastic = False
        dataset = build_dataset(dc, get_root_logger())
        model = IPDN(**cfg.model).cuda()
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
        state = cpu_state(model)
        reference_cls = importlib.import_module(
            "ipdn.model_paper5.adares_w2_pointmass_competition").IPDN
        parent = reference_cls(**cfg.model).cuda()
        incompatible = parent.load_state_dict(state, strict=True)
        assert not incompatible.missing_keys and not incompatible.unexpected_keys
        model.eval()
        parent.eval()
        result = {"scope": "train8 unaugmented identical state, label counterfactual",
                  "rows": [], "parameters": sum(p.numel() for p in model.parameters()),
                  "same_state_keys": state.keys() == parent.state_dict().keys(),
                  "complete": False}
        with torch.no_grad():
            for item in scene_indices(dataset.scene_inputs, 8):
                batch = dataset.collate_fn([dataset[item]])
                changed = dict(batch)
                changed["gt_pmasks"] = [torch.zeros_like(v) for v in batch["gt_pmasks"]]
                changed["gt_spmasks"] = [torch.zeros_like(v) for v in batch["gt_spmasks"]]
                first, repeat = predict(model, batch), predict(model, batch)
                modified = predict(model, changed)
                baseline = predict(parent, batch)
                times = {}
                for name, arm in (("y17", model), ("w2", parent)):
                    for _ in range(2):
                        arm(batch, mode="predict")
                    torch.cuda.reset_peak_memory_stats()
                    elapsed = []
                    for _ in range(5):
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        arm(batch, mode="predict")
                        torch.cuda.synchronize()
                        elapsed.append(time.monotonic()-start)
                    times[name] = {"seconds": elapsed,
                                   "allocated_gib_process": torch.cuda.max_memory_allocated()/2**30,
                                   "note": "both models resident; not standalone peak"}
                row = {"scan_id": batch["scan_ids"][0],
                       "repeat": comparison(first, repeat),
                       "label_change": comparison(first, modified),
                       "parent": comparison(first, baseline),
                       "queries": first["queries"], "timing": times}
                result["rows"].append(row)
                Path(args.output).write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
        result["complete"] = True
    else:
        dataset = build_train_dataset(cfg.data.train, get_root_logger())
        batch = dataset.collate_fn([dataset[i] for i in scene_indices(dataset.scene_inputs, 2)])
        model = IPDN(**cfg.model).cuda()
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
        opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
        sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
        first = step(model, opt, batch, cfg)
        sched.step()
        checkpoint = str(Path(args.output).resolve().with_suffix(".pth"))
        gorilla.save_checkpoint(model, checkpoint, opt, sched, dict(epoch=1, best_miou=0.0))
        saved = torch.load(checkpoint, map_location="cpu")
        random_state = rng()
        uninterrupted = step(model, opt, batch, cfg)
        final_model, final_opt = cpu_state(model), copy.deepcopy(opt.state_dict())
        del model, opt, sched
        gc.collect()
        torch.cuda.empty_cache()
        resumed = IPDN(**cfg.model).cuda()
        opt = gorilla.build_optimizer(resumed, copy.deepcopy(cfg.optimizer))
        sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
        meta = gorilla.resume(model=resumed, filename=checkpoint, optimizer=opt,
                              scheduler=sched, strict=False, map_location="cpu")
        allowed = {"text_encoder.embeddings.position_ids",
                   "text_encoder.embeddings.token_type_ids"}
        before = {
            "model": differences({k: v for k, v in saved["model"].items()
                                  if k not in allowed}, cpu_state(resumed)),
            "optimizer": differences(saved["optimizer"], opt.state_dict()),
            "scheduler": differences(saved["scheduler"], sched.state_dict()),
            "epoch": meta["epoch"]}
        assert all(before[k]["equal"] for k in ("model", "optimizer", "scheduler"))
        restore(random_state)
        second = step(resumed, opt, batch, cfg)
        result = {"scope": "paired native gorilla save/resume, RNG diagnostic replay",
                  "before_continuation": before, "first": first,
                  "uninterrupted": uninterrupted, "resumed": second,
                  "continuation_model": differences(final_model, cpu_state(resumed)),
                  "continuation_optimizer": differences(final_opt, opt.state_dict()),
                  "complete": True, "whole_dataloader_rng_resume_claim": False}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
