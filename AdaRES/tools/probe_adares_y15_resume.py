"""Native checkpoint round trip, next-step replay and lazy-teacher RNG audit."""

import argparse
import copy
import gc
import importlib
import json
import random
from pathlib import Path

import gorilla
import numpy as np
import torch

from diagnose_adares_y14_reliability import scene_indices
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain


def rng():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all())


def restore(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def cpu_state(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()}


def differences(a, b):
    if isinstance(a, torch.Tensor):
        return {"equal": torch.equal(a.cpu(), b.cpu()),
                "max": float((a.cpu().double()-b.cpu().double()).abs().max())
                if a.numel() else 0.0}
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return {"equal": False, "max": None}
        parts = [differences(a[k], b[k]) for k in a]
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return {"equal": False, "max": None}
        parts = [differences(x, y) for x, y in zip(a, b)]
    else:
        return {"equal": a == b, "max": 0 if a == b else None}
    return {"equal": all(p["equal"] for p in parts),
            "max": max((p["max"] for p in parts if p["max"] is not None),
                       default=0)}


def step(model, optimizer, batch, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, values = model(batch, mode="loss")
    loss.backward()
    gradients = {name: parameter.grad.detach().cpu().clone()
                 for name, parameter in model.named_parameters()
                 if name.startswith("dec.completion_query.")
                 and parameter.grad is not None}
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    optimizer.step()
    result = {"loss": float(loss), "finite": bool(torch.isfinite(loss).all()),
              "gradient_tensors": len(gradients),
              "gradients": gradients}
    del loss, values
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", required=True, type=int)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    indices = scene_indices(dataset.scene_inputs, 2)
    batch = dataset.collate_fn([dataset[i] for i in indices])
    module = importlib.import_module(args.module)
    model = module.IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
    scheduler = gorilla.build_lr_scheduler(optimizer, copy.deepcopy(cfg.lr_scheduler))
    first = step(model, optimizer, batch, cfg)
    scheduler.step()
    checkpoint = str(Path(args.output).resolve().with_suffix(".pth"))
    gorilla.save_checkpoint(model, checkpoint, optimizer, scheduler,
                           dict(epoch=1, best_miou=0.0))
    saved = torch.load(checkpoint, map_location="cpu")
    state = rng()
    uninterrupted = step(model, optimizer, batch, cfg)
    final_model = cpu_state(model)
    final_optimizer = copy.deepcopy(optimizer.state_dict())
    final_gradients = uninterrupted.pop("gradients")
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    restored = module.IPDN(**cfg.model).cuda()
    opt = gorilla.build_optimizer(restored, copy.deepcopy(cfg.optimizer))
    sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
    incompatible = restored.load_state_dict(saved["model"], strict=False)
    allowed = {"text_encoder.embeddings.position_ids",
               "text_encoder.embeddings.token_type_ids"}
    if incompatible.missing_keys or set(incompatible.unexpected_keys)-allowed:
        raise ValueError(str(incompatible))
    meta = gorilla.resume(model=restored, filename=checkpoint, optimizer=opt,
                          scheduler=sched, strict=False, map_location="cpu")
    before = {
        "model": differences({k: v for k, v in saved["model"].items()
                              if k not in allowed}, cpu_state(restored)),
        "optimizer": differences(saved["optimizer"], opt.state_dict()),
        "scheduler": differences(saved["scheduler"], sched.state_dict()),
        "meta_epoch": meta.get("epoch"),
        "student_contains_teacher": any("teacher" in k for k in saved["model"]),
        "fresh_teacher_absent": getattr(restored, "_training_teacher", None)
        is None,
        "allowed_nonpersistent_keys": incompatible.unexpected_keys,
    }
    restore(state)
    resumed = step(restored, opt, batch, cfg)
    lazy = {"model": differences(final_model, cpu_state(restored)),
            "optimizer": differences(final_optimizer, opt.state_dict()),
            "gradients": differences(final_gradients, resumed.pop("gradients")),
            "loss_difference": resumed["loss"]-uninterrupted["loss"]}
    # Repeat from the same saved state, now with its frozen teacher constructed.
    gorilla.resume(model=restored, filename=checkpoint, optimizer=opt,
                   scheduler=sched, strict=False, map_location="cpu")
    restore(state)
    warmed = step(restored, opt, batch, cfg)
    repeat = {"model": differences(final_model, cpu_state(restored)),
              "optimizer": differences(final_optimizer, opt.state_dict()),
              "gradients": differences(final_gradients, warmed.pop("gradients")),
              "loss_difference": warmed["loss"]-uninterrupted["loss"]}
    first.pop("gradients")
    output = {"module": args.module, "scans": batch["scan_ids"],
              "scope": "same augmented batch, native gorilla save/resume",
              "first_step": first, "uninterrupted": uninterrupted,
              "before_continuation": before, "lazy_continuation": lazy,
              "warmed_repeat": repeat, "complete": True,
              "limitation": "RNG explicitly replayed by probe, not native "
              "whole-dataloader resume guarantee"}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
