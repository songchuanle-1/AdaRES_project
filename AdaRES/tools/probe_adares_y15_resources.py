"""Independent clean-process arm resource measurement on matched real batches."""

import argparse
import gc
import importlib
import json
import time
from pathlib import Path

import gorilla
import torch

from diagnose_adares_y14_reliability import scene_indices
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    indices = scene_indices(dataset.scene_inputs, 4)
    batches = [dataset.collate_fn([dataset[i] for i in indices[j:j+2]])
               for j in (0, 2)]
    model = importlib.import_module(args.module).IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    rows = []
    for batch in batches:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, values = model(batch, mode="loss")
            loss.backward()
            optimizer.step()
            del loss, values
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        loss, values = model(batch, mode="loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        rows.append({
            "scans": batch["scan_ids"], "finite_loss": bool(torch.isfinite(loss).all()),
            "seconds": time.monotonic()-start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
            "student_parameters": sum(p.numel() for p in model.parameters()),
            "student_trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad),
        })
        del loss, values
    Path(args.output).write_text(json.dumps({
        "scope": "two augmented batches, warmed optimizer, isolated process per arm",
        "config": args.config, "module": args.module, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
