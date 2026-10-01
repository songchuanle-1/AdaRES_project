"""Isolated native train, inference identity and checkpoint-resume checks."""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

import gorilla
import torch

from probe_adares_y18_point_measure import acquire, digest
from diagnose_adares_y14_reliability import scene_indices
from probe_adares_y15_resume import rng, restore, differences, cpu_state
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain, train


class Batches:
    sampler = None

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def step(model, opt, batch, cfg):
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, values = model(batch, mode="loss")
    loss.backward()
    if not torch.isfinite(loss).all():
        raise RuntimeError("Nonfinite loss")
    grads = {n: {"norm": float(p.grad.norm()),
                 "finite": bool(torch.isfinite(p.grad).all())}
             for n, p in model.named_parameters()
             if n.startswith("dec.") and p.grad is not None}
    if not all(v["finite"] for v in grads.values()):
        raise RuntimeError("Nonfinite decoder gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    opt.step()
    return {"loss": float(loss), "decoder_gradients": grads}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("integration", "profile", "inference"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--parent-config")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise RuntimeError("Refusing overwrite")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    lease, occupied = acquire(1)
    torch.set_num_threads(4)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    cls = resolve_model_class(args.config)
    logger = get_root_logger()
    result = {"complete": False, "mode": args.mode, "gpu_precheck": occupied,
              "config_sha256": digest(args.config), "tool_sha256": digest(__file__),
              "class_module": cls.__module__}
    dc = cfg.data.train.copy()
    if args.mode == "inference":
        dc.training, dc.aug, dc.with_elastic, dc.lang_num_max = False, False, False, 4
    data = build_dataset(dc, logger)
    selected = scene_indices(data.scene_inputs, 4 if args.mode == "integration" else 8)
    if args.mode == "profile":
        manifest = json.loads(Path(args.manifest).read_text())
        selected = [r["item"] for r in manifest["selected"]]
    batches = []
    width = 1 if args.mode == "inference" else 2
    for offset in range(0, len(selected), width):
        gorilla.set_random_seed(1999 + offset)
        batches.append(data.collate_fn([data[i] for i in selected[offset:offset + width]]))
    result["scans"] = [s for b in batches for s in b["scan_ids"]]
    model = cls(**cfg.model).cuda()
    allowed = {"text_encoder.embeddings.position_ids", "text_encoder.embeddings.token_type_ids"}
    if args.mode == "inference":
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        load = model.load_state_dict(state, strict=False)
        if load.missing_keys or set(load.unexpected_keys) - allowed:
            raise RuntimeError(str(load))
        del checkpoint
        model.eval()
        # Native prediction identity is checked serially, not with co-resident models.
        captures = []
        repeats = []
        counterfactuals = []
        for b in batches:
            gorilla.set_random_seed(1999)
            with torch.no_grad():
                captures.append(model(b, mode="predict"))
                gorilla.set_random_seed(1999)
                repeats.append(model(b, mode="predict"))
                changed = dict(b)
                changed["gt_pmasks"] = [1 - t for t in b["gt_pmasks"]]
                changed["gt_spmasks"] = [1 - t for t in b["gt_spmasks"]]
                gorilla.set_random_seed(1999)
                counterfactuals.append(model(changed, mode="predict"))
        parent_cls = resolve_model_class(args.parent_config)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        parent = parent_cls(**cfg.model).cuda()
        load = parent.load_state_dict(state, strict=False)
        if load.missing_keys or set(load.unexpected_keys) - allowed:
            raise RuntimeError(str(load))
        del state
        parent.eval()
        rows = []
        for b, candidate, repeated, counter in zip(
                batches, captures, repeats, counterfactuals):
            gorilla.set_random_seed(1999)
            with torch.no_grad():
                baseline = parent(b, mode="predict")
            probs = [(a - c).abs().max().item()
                     for a, c in zip(candidate["pred_pmask"], baseline["pred_pmask"])]
            flips = [int(((a >= .5) != (c >= .5)).sum())
                     for a, c in zip(candidate["pred_pmask"], baseline["pred_pmask"])]
            score_deltas = [float((a - c).abs().max())
                            for a, c in zip(candidate["indi"], baseline["indi"])]
            entry = {"scans": b["scan_ids"], "expressions": len(b["ann_ids"]),
                         "max_mask_probability_change": max(probs),
                         "binary_flips": sum(flips), "max_indicator_change": max(score_deltas),
                         "parent_selection_flips": sum(int(a.argmax() != c.argmax())
                             for a, c in zip(candidate["indi"], baseline["indi"]))}
            for name, other in (("repeat", repeated), ("counterfactual", counter)):
                entry[name] = {
                    "max_mask_probability_change": max(float((a-c).abs().max())
                        for a, c in zip(candidate["pred_pmask"], other["pred_pmask"])),
                    "binary_flips": sum(int(((a >= .5) != (c >= .5)).sum())
                        for a, c in zip(candidate["pred_pmask"], other["pred_pmask"])),
                    "selection_flips": sum(int(a.argmax() != c.argmax())
                        for a, c in zip(candidate["indi"], other["indi"]))}
            rows.append(entry)
        result["rows"] = rows
        result["checkpoint_sha256"] = digest(args.checkpoint)
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
        opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
        sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
        result["parameters"] = sum(p.numel() for p in model.parameters())
        if args.mode == "profile":
            result["rows"] = []
            for b in batches:
                for _ in range(2):
                    step(model, opt, b, cfg)
                readings = []
                for repeat in range(3):
                    opt.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.monotonic()
                    row = step(model, opt, b, cfg)
                    torch.cuda.synchronize()
                    readings.append({"seconds": time.monotonic() - start,
                                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                                     "loss": row["loss"]})
                result["rows"].append({"scans": b["scan_ids"],
                                       "points": len(b["superpoints"]),
                                       "superpoints": int(b["batch_offsets"][-1]),
                                       "readings": readings})
        else:
            cfg.dist = False
            before = cpu_state(model)
            native_calls = []
            hook = model.register_forward_hook(
                lambda module, inputs, out: native_calls.append(float(out[0])))
            train(1, model, Batches(batches), opt, sched, cfg, logger, Writer())
            hook.remove()
            if len(native_calls) != len(batches) or not all(
                    torch.isfinite(torch.tensor(v)) for v in native_calls):
                raise RuntimeError("Native training entry did not execute every batch")
            result["native_train"] = {"losses": native_calls,
                                      "model_change": differences(before, cpu_state(model))}
            del before
            checkpoint = str(path.with_suffix(".pth"))
            gorilla.save_checkpoint(model, checkpoint, opt, sched,
                                   dict(epoch=1, best_miou=0.0))
            saved = torch.load(checkpoint, map_location="cpu")
            random_state = rng()
            continuous = step(model, opt, batches[0], cfg)
            final_model = cpu_state(model)
            final_opt = copy.deepcopy(opt.state_dict())
            del model, opt, sched
            gc.collect()
            torch.cuda.empty_cache()
            model = cls(**cfg.model).cuda()
            opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
            sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
            meta = gorilla.resume(model=model, filename=checkpoint, optimizer=opt,
                                  scheduler=sched, strict=False, map_location="cpu")
            result["before_continuation"] = {
                "model": differences({k: v for k, v in saved["model"].items()
                                      if k not in allowed}, cpu_state(model)),
                "optimizer": differences(saved["optimizer"], opt.state_dict()),
                "scheduler": differences(saved["scheduler"], sched.state_dict()),
                "meta_epoch": int(meta["epoch"])}
            if not all(result["before_continuation"][k]["equal"]
                       for k in ("model", "optimizer", "scheduler")):
                raise RuntimeError("Checkpoint resume state mismatch")
            restore(random_state)
            resumed = step(model, opt, batches[0], cfg)
            result["continuation"] = {"loss_difference": resumed["loss"] - continuous["loss"],
                                       "model": differences(final_model, cpu_state(model)),
                                       "optimizer": differences(final_opt, opt.state_dict()),
                                       "decoder_gradient_tensors": len(resumed["decoder_gradients"])}
            result["limitation"] = "Explicit RNG same-batch replay; not whole-dataloader resume"
    result["complete"] = True
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    lease.close()


if __name__ == "__main__":
    main()
