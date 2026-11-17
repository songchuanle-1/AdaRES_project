"""Read-only native best-checkpoint evaluation with expression-level repeats."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def summarize(rows):
    return {
        "expressions": len(rows),
        "scenes": len({row["scan_id"] for row in rows}),
        "miou": statistics.mean(row["piou"] for row in rows),
        "acc25": statistics.mean(row["piou"] > 0.25 for row in rows),
        "acc50": statistics.mean(row["piou"] > 0.5 for row in rows),
        "spmiou": statistics.mean(row["spiou"] for row in rows),
    }


def identity(row):
    return row["scan_id"], tuple(row["object_ids"]), row["ann_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, default=0, choices=(0,))
    args = parser.parse_args()
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite evaluation evidence")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import fcntl
    import gorilla
    import torch
    from ipdn.dataset import build_dataset, build_dataloader
    from ipdn.utils import get_root_logger
    from train import resolve_model_class
    import subprocess

    resource_hostname = subprocess.check_output(["hostname"], universal_newlines=True).strip()
    if resource_hostname != "dl02-MS-7B73":
        raise RuntimeError("WRONG_ORIGINAL_SERVER")
    lease = open("/data1/songcl/Project/AdaRES_project/.foreground_gpu0_diagnostic.lock", "a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resource_power = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], universal_newlines=True).strip()
    if float(resource_power) != 350.0:
        raise RuntimeError("WRONG_ORIGINAL_POWER")
    resource_compute = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], universal_newlines=True).strip()
    if resource_compute:
        raise RuntimeError("GPU_COMPUTE_OCCUPIED: " + resource_compute)
    occupation = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu_id),
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], universal_newlines=True)
    used, util = (int(value.strip()) for value in occupation.strip().split(","))
    if used >= 1024 or util >= 10:
        raise RuntimeError("GPU_NOT_IDLE: " + occupation.strip())
    torch.set_num_threads(4)
    gorilla.set_random_seed(1999)
    cfg = gorilla.Config.fromfile(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    raw_meta = checkpoint.get("meta", {})
    meta = {"epoch": int(raw_meta["epoch"]),
            "best_miou": float(raw_meta["best_miou"])}
    if Path(args.config).stem == "adares_b1_standard_res":
        registered = gorilla.LOSSES.obj_map.get("Criterion")
        if registered is not None:
            if registered.__module__ != "ipdn.model_paper5.adares_b3_mask_evidence.loss":
                raise RuntimeError("Unexpected inherited Criterion registry entry")
            # Root ipdn imports B3; B1 directly constructs its own Criterion.
            gorilla.LOSSES.obj_map.pop("Criterion")
    model = resolve_model_class(args.config)(**cfg.model).cuda()
    loaded = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    allowed = {"text_encoder.embeddings.position_ids",
               "text_encoder.embeddings.token_type_ids"}
    if loaded.missing_keys or set(loaded.unexpected_keys) - allowed:
        raise RuntimeError("Checkpoint mismatch: " + str(loaded))
    del checkpoint
    model.eval()
    dataset = build_dataset(cfg.data.val.copy(), get_root_logger())
    loader = build_dataloader(dataset, batch_size=1, num_workers=2,
                              training=False, persistent_workers=True)
    result = {
        "config": args.config, "config_sha256": digest(args.config),
        "checkpoint": args.checkpoint, "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_meta": meta, "seed": 1999,
        "ordering": "all val chunks, shuffle=False, drop_last=False",
        "parameters": sum(p.numel() for p in model.parameters()),
        "completed_passes": 0, "complete": False, "passes": [],
        "official_unique_multiple": "unknown: official metadata not supplied",
        "size_cuts_from_training": [1487, 3325],
        "gpu_precheck": {"memory_mib": used, "utilization": util},
    }
    timings = []
    query_counts = set()
    start = time.monotonic()
    with torch.no_grad():
        for repeat in range(2):
            gorilla.set_random_seed(1999)
            rows = []
            pass_start = time.monotonic()
            for batch_index, batch in enumerate(loader):
                if repeat == 0 and batch_index == 0:
                    for _ in range(2):
                        model(batch, mode="predict")
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    for _ in range(5):
                        torch.cuda.synchronize()
                        clock = time.monotonic()
                        model(batch, mode="predict")
                        torch.cuda.synchronize()
                        timings.append(time.monotonic() - clock)
                    gorilla.set_random_seed(1999)
                torch.cuda.synchronize()
                output = model(batch, mode="predict")
                torch.cuda.synchronize()
                for i, scan_id in enumerate(output["scan_id"]):
                    truth = output["gt_pmask"][i]
                    point_count = int(truth.gt(0.5).sum())
                    query_counts.add(int(output["indi"][i].numel()))
                    rows.append({
                        "scan_id": str(scan_id),
                        "object_ids": [int(v) for v in output["object_ids"][i]],
                        "ann_id": str(output["ann_id"][i]),
                        "piou": float(output["piou"][i]),
                        "spiou": float(output["spiou"][i]),
                        "target_points": point_count,
                        "size": "small" if point_count <= 1487 else
                                "medium" if point_count <= 3325 else "large",
                    })
                if (batch_index + 1) % 60 == 0:
                    print(json.dumps({"pass": repeat, "chunks": batch_index + 1,
                                      "total_chunks": len(loader)}), flush=True)
            if len({identity(row) for row in rows}) != len(rows):
                raise RuntimeError("Duplicate expression identities")
            result["passes"].append({
                "summary": summarize(rows), "rows": rows,
                "seconds": time.monotonic() - pass_start})
            result["completed_passes"] += 1
            Path(args.output).write_text(json.dumps(result), encoding="utf-8")
    first, second = (item["rows"] for item in result["passes"])
    if [identity(row) for row in first] != [identity(row) for row in second]:
        raise RuntimeError("Repeated expression ordering differs")
    result["repeat_max_iou_difference"] = max(
        abs(a["piou"] - b["piou"]) for a, b in zip(first, second))
    result["repeat_miou_difference"] = abs(
        summarize(first)["miou"] - summarize(second)["miou"])
    result["active_queries"] = sorted(query_counts)
    result["selected_queries"] = 1
    result["warm_first_chunk_seconds"] = timings
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    result["elapsed_seconds"] = time.monotonic() - start
    result["complete"] = True
    Path(args.output).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "passes"}), flush=True)
    print(json.dumps(result["passes"][0]["summary"]), flush=True)


if __name__ == "__main__":
    main()
