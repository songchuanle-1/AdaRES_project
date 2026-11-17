"""Use the existing paired scene bootstrap on the frozen Y18 captures."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

from analyze_frozen_full import comparison

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
paths = {name: ROOT / ("pipeline/analyses/y18_" + name + "_full_eval_" + args.token + ".json")
         for name in ("y18", "w2", "b1")}
captures = {name: json.loads(path.read_text()) for name, path in paths.items()}
assert all(c["complete"] and c["completed_passes"] == 2 for c in captures.values())
assert all(c["size_cuts_from_training"] == [1487, 3325] for c in captures.values())
output = ROOT / ("pipeline/analyses/y18_clustered_comparison_" + args.token + ".json")
assert not output.exists()
result = dict(id="AdaRES-Y18", method="Expression-weighted paired delta; resample whole scan clusters",
    seed=1999, bootstrap_draws=10000, size_cuts_from_training=[1487, 3325],
    sources={name: dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
             for name,path in paths.items()},
    summary={name:c["passes"][0]["summary"] for name,c in captures.items()},
    versus_w2=comparison(captures["y18"],captures["w2"]),
    versus_b1=comparison(captures["y18"],captures["b1"]),
    target=0.515,target_reached=captures["y18"]["passes"][0]["summary"]["miou"]>=0.515,
    cost={name:dict(parameters=c["parameters"],
        warm_first_chunk_median_seconds=statistics.median(c["warm_first_chunk_seconds"]),
        peak_allocated_gib=c["peak_allocated_gib"],peak_reserved_gib=c["peak_reserved_gib"],
        available_queries=c["active_queries"],selected_queries=c["selected_queries"])
        for name,c in captures.items()},
    limitations=["One training seed; best-checkpoint comparison is not causal mechanism proof.",
        "Two native passes quantify observed repeat noise, not a universal bound.",
        "Warm first-chunk timing is not a latency distribution or p95/Pareto proof.",
        "Official unique/multiple metadata unavailable; no subgroup tuning."],
    observed_local=datetime.now().astimezone().isoformat())
output.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
print("sha256="+hashlib.sha256(output.read_bytes()).hexdigest())
