"""Exact train-only superpoint IoU ceilings and frozen target-size tertiles."""

import argparse
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def ceiling(mass, target):
    total = target.sum()
    if total <= 0:
        raise ValueError("Empty target")
    active = np.flatnonzero(target > 0)
    order = active[np.argsort(-(target[active] / mass[active]), kind="stable")]
    intersections = np.cumsum(target[order])
    unions = total + np.cumsum(mass[order] - target[order])
    return float(np.max(intersections / unions))


def test_ceiling():
    cases = 0
    for mass in itertools.product((1, 2, 3), repeat=3):
        mass = np.asarray(mass)
        for target in itertools.product(*(range(int(n) + 1) for n in mass)):
            target = np.asarray(target)
            if not target.sum():
                continue
            brute = max(
                float(target[mask].sum() /
                      (target.sum() + (mass[mask] - target[mask]).sum()))
                for bits in itertools.product((False, True), repeat=3)
                for mask in [np.asarray(bits)])
            if not np.isclose(ceiling(mass, target), brute, rtol=0, atol=1e-12):
                raise AssertionError((mass, target, brute))
            cases += 1
    return cases


def summarize(rows, weighted):
    weights = np.asarray([r["expressions"] if weighted else 1 for r in rows])
    keys = ("ceiling_iou", "majority_iou", "mixed_target_fraction")
    return {key: float(np.average([r[key] for r in rows], weights=weights))
            for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    tests = test_ceiling()
    if args.self_test:
        print(json.dumps({"exhaustive_cases": tests, "passed": True}))
        return
    root = Path(args.data_root)
    annotation = root / "ScanRefer" / "ScanRefer_train_new.json"
    annotations = json.loads(annotation.read_text())
    grouped = defaultdict(Counter)
    for row in annotations:
        ids = tuple(sorted(int(v) for v in row.get(
            "object_ids", [row.get("object_id")])))
        grouped[str(row["scene_id"])][ids] += 1
    start = time.monotonic()
    rows, sources, empty = [], [], []
    result = {"scope": "all raw training scans; no model or validation",
              "annotation": {"path": str(annotation),
                             "sha256": sha256(annotation)},
              "unit_tests": tests, "rows": rows, "point_sources": sources,
              "empty_targets": empty, "complete": False}
    output = Path(args.output)
    for scan in sorted(grouped):
        files = sorted((root / "scannetv2" / "train").glob(scan + "*_refer.pth"))
        if len(files) != 1:
            raise ValueError("Ambiguous/missing scan: " + scan)
        data = torch.load(files[0], map_location="cpu")
        sp = np.asarray(data[2])
        labels = np.asarray(data[4])
        if len(sp) != len(labels):
            raise ValueError("Point label alignment: " + scan)
        _, inverse = np.unique(sp, return_inverse=True)
        mass = np.bincount(inverse).astype(np.float64)
        for ids, expressions in sorted(grouped[scan].items()):
            truth = np.isin(labels, ids)
            target = np.bincount(inverse, weights=truth,
                                 minlength=len(mass)).astype(np.float64)
            total = int(target.sum())
            if not total:
                empty.append({"scan_id": scan, "object_ids": ids,
                              "expressions": expressions})
                continue
            majority = target > mass / 2
            mixed = (target > 0) & (target < mass)
            rows.append({
                "scan_id": scan, "object_ids": ids, "expressions": expressions,
                "target_points": total, "points": len(sp),
                "superpoints": len(mass), "ceiling_iou": ceiling(mass, target),
                "majority_iou": float(target[majority].sum() /
                    (total + (mass[majority] - target[majority]).sum())),
                "mixed_target_fraction": float(target[mixed].sum() / total)})
        sources.append({"scan_id": scan, "path": str(files[0]),
                        "sha256": sha256(files[0])})
        if len(sources) % 50 == 0:
            output.write_text(json.dumps(result, indent=2))
            print(json.dumps({"scans": len(sources), "targets": len(rows),
                              "elapsed_s": time.monotonic() - start}), flush=True)
    result["elapsed_s"] = time.monotonic() - start
    result["actual_scans"] = len(sources)
    result["unique_targets"] = len(rows)
    result["expressions"] = sum(r["expressions"] for r in rows)
    if empty:
        output.write_text(json.dumps(result, indent=2))
        raise ValueError("Empty targets retained; no complete size freeze")
    sizes = np.repeat([r["target_points"] for r in rows],
                      [r["expressions"] for r in rows])
    cuts = np.quantile(sizes, [1 / 3, 2 / 3], interpolation="linear").tolist()
    result["size_tertiles"] = {
        "distribution": "raw train target points, expression weighted",
        "quantile_method": "linear", "numpy_version": np.__version__,
        "cutpoints": cuts, "rule": "small<=cut1; medium<=cut2; else large",
        "counts": [int((sizes <= cuts[0]).sum()),
                   int(((sizes > cuts[0]) & (sizes <= cuts[1])).sum()),
                   int((sizes > cuts[1]).sum())]}
    result["summary_unique_targets"] = summarize(rows, False)
    result["summary_expression_weighted"] = summarize(rows, True)
    result["scene_mean_ceiling"] = float(np.mean([
        np.average([r["ceiling_iou"] for r in rows if r["scan_id"] == scan],
                   weights=[r["expressions"] for r in rows
                            if r["scan_id"] == scan]) for scan in grouped]))
    result["complete"] = True
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in (
        "complete", "actual_scans", "unique_targets", "expressions",
        "size_tertiles", "summary_expression_weighted", "elapsed_s")}), flush=True)


if __name__ == "__main__":
    main()
