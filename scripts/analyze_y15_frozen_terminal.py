"""Frozen Y15 scene-clustered efficacy analysis, without modifying old reports."""

import json
from pathlib import Path
import statistics

from analyze_frozen_full import comparison, file_hash


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "pipeline/analyses"
    paths = {
        "y15": folder / "y15_y15_v2_963720b2.json",
        "w2": folder / "y15_w2_v2_963720b2.json",
        "b1": folder / "y15_b1_v2_963720b2.json",
    }
    output = folder / "y15_clustered_comparison_963720b2.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite evidence")
    captures = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if not all(value["complete"] and value["completed_passes"] == 2
               for value in captures.values()):
        raise ValueError("Incomplete full-validation capture")
    result = {
        "candidate": "AdaRES-Y15",
        "method": "expression-weighted paired delta, resample whole scan clusters",
        "seed": 1999,
        "bootstrap_draws": 10000,
        "sources": {name: {"path": str(path), "sha256": file_hash(path)}
                    for name, path in paths.items()},
        "summary": {name: value["passes"][0]["summary"]
                    for name, value in captures.items()},
        "versus_w2": comparison(captures["y15"], captures["w2"]),
        "versus_b1": comparison(captures["y15"], captures["b1"]),
        "target": 0.515,
        "target_reached": captures["y15"]["passes"][0]["summary"]["miou"] >= 0.515,
        "cost": {name: {
            "parameters": value["parameters"],
            "warm_first_chunk_median_seconds":
                statistics.median(value["warm_first_chunk_seconds"]),
            "peak_allocated_gib": value["peak_allocated_gib"],
            "peak_reserved_gib": value["peak_reserved_gib"],
            "available_queries": value["active_queries"],
            "selected_queries": value["selected_queries"],
        } for name, value in captures.items()},
        "limitations": [
            "Official unique/multiple metadata unavailable; no guessed substitutes.",
            "Two repeats do not establish a universal numerical-noise upper bound.",
            "One seed/checkpoint per method; intervals are not training-seed uncertainty.",
            "Latency is five warm forwards on the first identical chunk, not a full distribution.",
            "No trained matched GT-only control; teacher necessity remains unknown.",
            "No thresholds, epochs or weights tuned on validation or subgroup readouts.",
            "Training-size cuts1487/3325 were frozen before validation.",
        ],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
