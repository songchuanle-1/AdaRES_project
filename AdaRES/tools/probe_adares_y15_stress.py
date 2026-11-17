"""Prediction-blind train metadata and isolated native stress measurements."""

import argparse
import gc
import importlib
import json
import time
from pathlib import Path

import gorilla
import numpy as np
import torch

from diagnose_adares_y14_reliability import scene_indices
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import load_shape_compatible_pretrain


def dump(path, result):
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--module")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    gorilla.set_random_seed(1999)
    torch.set_num_threads(4)
    cfg = gorilla.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train, get_root_logger())
    if args.metadata:
        scan_count = len({str(rows[0]["scene_id"])
                          for rows in dataset.scene_inputs})
        distinct = scene_indices(dataset.scene_inputs, scan_count)
        rows = []
        for item in distinct:
            scan = str(dataset.scene_inputs[item][0]["scene_id"])
            files = [p for p in dataset.sp_filenames if scan in Path(p).name]
            if len(files) != 1:
                raise ValueError("Ambiguous point-cloud path: " + scan)
            data = dataset.load(files[0])
            rows.append({"scan_id": scan, "item": item,
                         "points": int(data[0].shape[0]),
                         "superpoints": int(np.unique(data[2]).size)})
        selected = []
        for key in ("points", "superpoints"):
            for row in sorted(rows, key=lambda r: (-r[key], r["scan_id"]))[:2]:
                if row not in selected:
                    selected.append(row)
        result = {"scope": "all distinct training scans, no model outcomes",
                  "scans": len(rows), "selected": selected, "rows": rows}
        dump(args.manifest, result)
        dump(args.output, {"scans": len(rows), "selected": selected})
        print(json.dumps({"metadata_complete": True, "selected": selected}),
              flush=True)
        return
    gorilla.set_cuda_visible_devices([args.gpu_id], 1)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    selected = manifest["selected"]
    model = importlib.import_module(args.module).IPDN(**cfg.model).cuda()
    load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    optimizer = gorilla.build_optimizer(model, cfg.optimizer)
    result = {"scope": "native augmented stress, isolated process per arm",
              "module": args.module, "manifest": args.manifest, "rows": []}
    dump(args.output, result)
    for offset in range(0, len(selected), 2):
        batch_rows = selected[offset:offset+2]
        gorilla.set_random_seed(1999 + offset)
        batch = dataset.collate_fn([dataset[r["item"]] for r in batch_rows])
        for warm in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, values = model(batch, mode="loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            del loss, values
        times, resources = [], []
        for repeat in range(3):
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
            if not torch.isfinite(loss).all():
                raise ValueError("Nonfinite stress loss")
            times.append(time.monotonic()-start)
            resources.append({
                "allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "reserved_gib": torch.cuda.max_memory_reserved()/2**30})
            del loss, values
        row = {"selection": batch_rows, "actual_points": len(batch["superpoints"]),
               "actual_superpoints": int(batch["batch_offsets"][-1]),
               "scans": batch["scan_ids"], "expressions": len(batch["ann_ids"]),
               "step_seconds": times, "peaks": resources, "finite_loss": True,
               "parameters": sum(p.numel() for p in model.parameters()),
               "trainable": sum(p.numel() for p in model.parameters()
                                if p.requires_grad)}
        result["rows"].append(row)
        dump(args.output, result)
        print(json.dumps(row), flush=True)
        del batch
    result["complete"] = True
    dump(args.output, result)


if __name__ == "__main__":
    main()
