"""Scene-clustered paired analysis of frozen native full-validation captures."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed(rows):
    values = {}
    for row in rows:
        key = (row["scan_id"], tuple(row["object_ids"]), row["ann_id"])
        if key in values:
            raise ValueError("Duplicate expression identity")
        values[key] = row
    return values


def comparison(candidate, reference, seed=1999, draws=10000):
    left = keyed(candidate["passes"][0]["rows"])
    right = keyed(reference["passes"][0]["rows"])
    if left.keys() != right.keys():
        raise ValueError("Expression population mismatch")
    if not all(left[k]["target_points"] == right[k]["target_points"] for k in left):
        raise ValueError("Matched target geometry differs")
    candidate_repeat = keyed(candidate["passes"][1]["rows"])
    reference_repeat = keyed(reference["passes"][1]["rows"])
    results = {}
    for group in ("all", "small", "medium", "large"):
        scenes = defaultdict(list)
        for key, row in left.items():
            if group == "all" or row["size"] == group:
                scenes[row["scan_id"]].append(row["piou"] - right[key]["piou"])
        ordered = [scenes[k] for k in sorted(scenes)]
        if len(ordered) < 2:
            results[group] = {"status": "insufficient independent scenes"}
            continue
        totals = np.array([sum(values) for values in ordered])
        counts = np.array([len(values) for values in ordered])
        selected = [k for k in left if group == "all" or left[k]["size"] == group]
        candidate_noise = abs(np.mean([
            left[k]["piou"] - candidate_repeat[k]["piou"] for k in selected]))
        reference_noise = abs(np.mean([
            right[k]["piou"] - reference_repeat[k]["piou"] for k in selected]))
        noise = float(max(candidate_noise, reference_noise))
        rng = np.random.default_rng(seed)
        choices = rng.integers(0, len(ordered), size=(draws, len(ordered)))
        sampled = totals[choices].sum(1) / counts[choices].sum(1)
        lower, upper = np.quantile(sampled, [0.025, 0.975])
        results[group] = {
            "expressions": int(counts.sum()), "scenes": len(ordered),
            "paired_miou_difference": float(totals.sum() / counts.sum()),
            "scene_clustered_ci95": [float(lower), float(upper)],
            "observed_group_mean_repeat_noise": noise,
            "max_expression_repeat_iou_difference": float(max(
                max(abs(left[k]["piou"] - candidate_repeat[k]["piou"])
                    for k in selected),
                max(abs(right[k]["piou"] - reference_repeat[k]["piou"])
                    for k in selected))),
            "material_harm_within_observed_noise": bool(upper < -noise),
            "positive_difference_beyond_observed_noise": bool(lower > noise),
            "status": "interval includes zero" if lower <= 0 <= upper else
                      "negative interval" if upper < 0 else "positive interval",
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--w2", required=True)
    parser.add_argument("--b1", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise RuntimeError("Refusing to overwrite evidence")
    paths = {"y17": args.candidate, "w2": args.w2, "b1": args.b1}
    captures = {name: json.loads(Path(path).read_text(encoding="utf-8"))
                for name, path in paths.items()}
    if not all(value["complete"] and value["completed_passes"] == 2
               for value in captures.values()):
        raise ValueError("Incomplete capture")
    result = {
        "method": "expression-weighted paired delta, resample whole scan clusters",
        "seed": 1999, "bootstrap_draws": 10000,
        "sources": {name: {"path": path, "sha256": file_hash(path)}
                    for name, path in paths.items()},
        "summary": {name: value["passes"][0]["summary"]
                    for name, value in captures.items()},
        "versus_w2": comparison(captures["y17"], captures["w2"]),
        "versus_b1": comparison(captures["y17"], captures["b1"]),
        "target": 0.515,
        "target_reached": captures["y17"]["passes"][0]["summary"]["miou"] >= 0.515,
        "cost": {name: {
            "parameters": value["parameters"],
            "warm_first_chunk_median_seconds":
                float(np.median(value["warm_first_chunk_seconds"])),
            "peak_allocated_gib": value["peak_allocated_gib"],
            "peak_reserved_gib": value["peak_reserved_gib"],
            "available_queries": value["active_queries"],
            "selected_queries": value["selected_queries"],
        } for name, value in captures.items()},
        "limitations": [
            "Official unique/multiple metadata unavailable; not substituted with object names.",
            "Group mIoU repeat noise and maximum per-expression change are distinct; both reported.",
            "Two repeat passes do not establish a universal numerical-noise upper bound.",
            "One seed/checkpoint per method; cluster intervals are not training-seed uncertainty.",
            "Latency is five warm forwards on the first identical chunk, not full latency distribution.",
            "No exact regrouped pair-off full-training control; no causal or novelty claim.",
            "Training-size cuts1487/3325 frozen before validation; no tuning from subgroup outcomes.",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
