import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1999)
    args = parser.parse_args()
    rows = json.loads(args.probe.read_text(encoding="utf-8"))["rows"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_index"]].append(
            row["actual_iou"] - row["uniform_iou"]
        )
    scene_ids = sorted(groups)
    if len(scene_ids) < 2:
        raise ValueError("Need at least two independent scenes")
    rng = random.Random(args.seed)
    estimates = []
    for _ in range(args.replicates):
        selected = rng.choices(scene_ids, k=len(scene_ids))
        values = [value for scene in selected for value in groups[scene]]
        estimates.append(statistics.fmean(values))
    estimates.sort()
    changed = [
        row for row in rows if row["actual_query"] != row["uniform_query"]
    ]
    print(json.dumps({
        "scenes": len(scene_ids),
        "expressions": len(rows),
        "mean_delta_iou": statistics.fmean(
            row["actual_iou"] - row["uniform_iou"] for row in rows
        ),
        "scene_bootstrap_95pct": [
            estimates[int(0.025 * args.replicates)],
            estimates[int(0.975 * args.replicates)],
        ],
        "scene_bootstrap_fraction_nonpositive": (
            sum(value <= 0 for value in estimates) / args.replicates
        ),
        "scene_positive": sum(statistics.fmean(groups[scene]) > 0 for scene in scene_ids),
        "scene_negative": sum(statistics.fmean(groups[scene]) < 0 for scene in scene_ids),
        "scene_equal": sum(statistics.fmean(groups[scene]) == 0 for scene in scene_ids),
        "changed_top1": len(changed),
        "changed_top1_equal_iou": sum(
            abs(row["actual_iou"] - row["uniform_iou"]) < 1e-5
            for row in changed
        ),
        "mean_oracle_rank_actual": statistics.fmean(
            row["actual_oracle_rank"] for row in rows
        ),
        "mean_oracle_rank_uniform": statistics.fmean(
            row["uniform_oracle_rank"] for row in rows
        ),
        "oracle_rank_improved": sum(
            row["actual_oracle_rank"] < row["uniform_oracle_rank"]
            for row in rows
        ),
        "oracle_rank_worsened": sum(
            row["actual_oracle_rank"] > row["uniform_oracle_rank"]
            for row in rows
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
