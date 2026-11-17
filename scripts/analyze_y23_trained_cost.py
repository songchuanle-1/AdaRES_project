"""Describe the frozen paired cost readings without claiming tail precision."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
source = ROOT / ("pipeline/analyses/y23_trained_cost_serial_" + args.token + ".json")
capture = json.loads(source.read_text())
assert capture["complete"] and capture["batches"] == 10 and capture["scans"] == 10


def summarize(readings):
    seconds = [r["seconds"] for r in readings]
    return dict(readings=len(seconds), median_seconds=statistics.median(seconds),
                descriptive_p95_seconds=float(np.percentile(seconds, 95)),
                peak_allocated_gib=max(r["peak_allocated_gib"] for r in readings),
                peak_reserved_gib=max(r["peak_reserved_gib"] for r in readings))


result = dict(source=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
              observed_local=datetime.now().astimezone().isoformat(),
              manifest_indices=capture["manifest_indices"], rows={}, aggregate={},
              checkpoints=capture["checkpoint_sources"], parameters=capture["parameters"],
              limitations=[
                  "Three readings per batch give a descriptive percentile, not reliable tail-latency precision.",
                  "Train8/largest-pair pooled readings mix scenes and sizes; they are not independent scenes for a significance claim. Largest-pair scenes run serially at native predict batch=1.",
                  "Serial fixed candidate-then-parent order; transient hardware effects are not isolated.",
                  "Cost readings do not establish causal efficacy, equivalence, or Pareto dominance."])
for name, rows in capture["rows"].items():
    result["rows"][name] = [dict(scan_ids=row["scan_ids"], points=row["points"],
                               expressions=row["expressions"], **summarize(row["readings"])) for row in rows]
    result["aggregate"][name] = dict(
        train8=summarize([reading for row in rows[:8] for reading in row["readings"]]),
        largest_pair=summarize([reading for row in rows[-2:] for reading in row["readings"]]))
assert [r["scan_ids"] for r in result["rows"]["candidate"]] == [r["scan_ids"] for r in result["rows"]["parent"]]
result["candidate_vs_parent"] = {
    group: {metric: result["aggregate"]["candidate"][group][metric] /
            result["aggregate"]["parent"][group][metric] - 1.
            for metric in ("median_seconds", "descriptive_p95_seconds",
                           "peak_allocated_gib", "peak_reserved_gib")}
    for group in ("train8", "largest_pair")}
output = ROOT / ("pipeline/analyses/y23_trained_cost_summary_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
print("sha256=" + hashlib.sha256(output.read_bytes()).hexdigest())
