import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


KEYS = ("scene_index", "expression_index", "scan_id", "ann_id")


def load_index(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {tuple(row[key] for key in KEYS): row for row in data["rows"]}
    if len(rows) != len(data["rows"]) or len(rows) != data["expressions"]:
        raise ValueError(f"Duplicate or missing expression in {path}")
    return rows, data


def quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def group_summary(records, size_field, low, high):
    result = {}
    for name, predicate in (
        ("small", lambda x: x <= low),
        ("middle", lambda x: low < x < high),
        ("large", lambda x: x >= high),
    ):
        subset = [row for row in records if predicate(row[size_field])]
        scenes = defaultdict(list)
        for row in subset:
            scenes[row["scene_index"]].append(row)
        result[name] = {
            "expressions": len(subset),
            "scenes": len(scenes),
            "selected_delta": statistics.fmean(row["selected_delta"] for row in subset)
            if subset else None,
            "oracle_delta": statistics.fmean(row["oracle_delta"] for row in subset)
            if subset else None,
            "scene_mean_selected_delta": statistics.fmean(
                statistics.fmean(row["selected_delta"] for row in rows)
                for rows in scenes.values()) if scenes else None,
            "scene_mean_oracle_delta": statistics.fmean(
                statistics.fmean(row["oracle_delta"] for row in rows)
                for rows in scenes.values()) if scenes else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analyses_dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = Path(args.analyses_dir)
    cohorts = {}
    for cohort in ("train64", "val32"):
        w2, w2_data = load_index(base / f"AdaRES-y1_w2_{cohort}_20260928.json")
        b4, b4_data = load_index(base / f"AdaRES-y1_b4_{cohort}_20260928.json")
        sizes, size_data = load_index(base / f"AdaRES-y2_size_{cohort}_20260928.json")
        if not (w2.keys() == b4.keys() == sizes.keys()):
            raise ValueError(f"Unmatched expression keys in {cohort}")
        if any(data["scenes"] != size_data["scenes"]
               for data in (w2_data, b4_data)):
            raise ValueError(f"Unmatched scene count in {cohort}")
        rows = []
        for key in sorted(sizes):
            size = sizes[key]
            if size["scene_points"] <= 0 or not 0 <= size["target_fraction"] <= 1:
                raise ValueError(f"Invalid target fraction: {key}")
            rows.append({
                "scene_index": key[0],
                "expression_index": key[1],
                "target_fraction": size["target_fraction"],
                "selected_delta": w2[key]["selected_iou"] - b4[key]["selected_iou"],
                "oracle_delta": w2[key]["oracle_iou"] - b4[key]["oracle_iou"],
            })
        cohorts[cohort] = rows
    low = quantile([row["target_fraction"] for row in cohorts["train64"]], 0.25)
    high = quantile([row["target_fraction"] for row in cohorts["train64"]], 0.75)
    result = {"train_q1": low, "train_q3": high, "cohorts": {}}
    for cohort, rows in cohorts.items():
        by_scene = defaultdict(list)
        for row in rows:
            by_scene[row["scene_index"]].append(row)
        for scene in by_scene:
            group = sorted(by_scene[scene], key=lambda row: row["expression_index"])
            for index, row in enumerate(group):
                row["rotated_fraction"] = group[(index + 1) % len(group)]["target_fraction"]
        result["cohorts"][cohort] = {
            "expressions": len(rows),
            "scenes": len(by_scene),
            "original": group_summary(rows, "target_fraction", low, high),
            "within_scene_rotated": group_summary(rows, "rotated_fraction", low, high),
        }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
