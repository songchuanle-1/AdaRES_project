"""Summarize frozen layer-aware controls without treating layers as replicates."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise RuntimeError("Refusing to overwrite analysis evidence")
    runs = [json.loads(Path(p).read_text(encoding="utf-8"))
            for p in (args.first, args.repeat)]
    if not all(r["complete"] for r in runs) or runs[0]["selection"] != runs[1]["selection"]:
        raise RuntimeError("Incomplete or different scene captures")
    summaries = []
    for run in runs:
        groups = {}
        for batch in run["rows"]:
            count = batch["expressions"]
            if count != len(batch["scans"]) * 4 or len(batch["captures"]) != count * 7:
                raise RuntimeError("Unexpected expression/layer capture alignment")
            for expression, row in enumerate(batch["captures"][:count]):
                scan = batch["scans"][expression // 4]
                groups.setdefault(scan, []).append(row)
        rows = [r for group in groups.values() for r in group]
        def stats(read):
            values = [statistics.mean(read(r) for r in g) for g in groups.values()]
            return {"scene_mean": statistics.mean(values), "minimum": min(values),
                    "maximum": max(values), "positive_scenes": sum(v > 0 for v in values),
                    "negative_scenes": sum(v < 0 for v in values)}
        current = {"scans": len(groups), "expressions": len(rows),
                   "mixed_expressions": sum(r["mixed_superpoints"] > 0 for r in rows),
                   "gradient_separated": sum(r["gradient_change_norm"] > 0 for r in rows),
                   "old_alignment": stats(lambda r: r["old_vs_negative_point_iou_cos"]),
                   "point_alignment": stats(lambda r: r["point_vs_negative_point_iou_cos"]),
                   "point_minus_old_alignment": stats(
                       lambda r: r["point_vs_negative_point_iou_cos"]
                       - r["old_vs_negative_point_iou_cos"]),
                   "adverse_expressions": sum(
                       r["point_vs_negative_point_iou_cos"]
                       < r["old_vs_negative_point_iou_cos"] for r in rows)}
        for key in ("mass_only", "occupancy_only", "equal_size"):
            current[key + "_alignment"] = stats(
                lambda r: r[key]["vs_negative_point_iou_cos"])
            current["point_minus_" + key] = stats(
                lambda r: r["point_vs_negative_point_iou_cos"]
                - r[key]["vs_negative_point_iou_cos"])
        summaries.append(current)
    noise = {}
    for key in ("old_bce", "point_bce", "old_dice", "point_dice",
                "gradient_change_norm", "old_vs_negative_point_iou_cos",
                "point_vs_negative_point_iou_cos"):
        changes = [abs(a[key] - b[key])
                   for x, y in zip(runs[0]["rows"], runs[1]["rows"])
                   for a, b in zip(x["captures"][:x["expressions"]],
                                   y["captures"][:y["expressions"]])]
        noise[key] = {"mean_absolute": statistics.mean(changes),
                      "maximum_absolute": max(changes)}
    result = {"complete": True, "summary": summaries, "repeat_noise": noise,
              "sources": [{"path": p, "sha256": sha(p)}
                          for p in (args.first, args.repeat)],
              "unit": "Final-layer scene means; auxiliary layers not independent repeats",
              "scope": "Train-only local derivative diagnosis, not trained efficacy",
              "limitation": "Mass/occupancy/smoothing confounds reported; two repeats "
              "are not a universal numerical bound or multi-seed evidence"}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
