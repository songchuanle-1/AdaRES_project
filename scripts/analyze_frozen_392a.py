"""FIRST frozen results and predeclared scene/subgroup comparisons."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from analyze_y27_frozen_terminal import comparison

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("name", choices=("y43", "y44"))
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
path = ROOT / ("pipeline/analyses/" + args.name + "_frozen_392a_" + args.token + ".json")
value = json.loads(path.read_text(encoding="utf-8"))
captures = value["rows"]
assert value["complete"] and set(captures) == {args.name, "w2", "b1"}
assert all(c["complete"] and c["completed_passes"] == 2 for c in captures.values())
result = dict(
    complete=True, id="AdaRES-" + args.name.upper(),
    source=dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    observed_at_local=datetime.now().astimezone().isoformat(),
    primary="FIRST full pass, not favorable repeat",
    bootstrap=dict(draws=10000, seed=1999, unit="whole scene"),
    summary={k: c["passes"][0]["summary"] for k, c in captures.items()},
    repeated_miou={k: c["passes"][1]["summary"]["miou"] for k, c in captures.items()},
    versus_w2=comparison(captures[args.name], captures["w2"]),
    versus_b1=comparison(captures[args.name], captures["b1"]),
    target=.515, target_reached=captures[args.name]["passes"][0]["summary"]["miou"] >= .515,
    cost_first_chunk_only={k: dict(parameters=c["parameters"],
                                 median_seconds=statistics.median(c["warm_first_chunk_seconds"]),
                                 peak_allocated_gib=c["peak_allocated_gib"],
                                 peak_reserved_gib=c["peak_reserved_gib"],
                                 repeat_max_iou_difference=c["repeat_max_iou_difference"],
                                 repeat_miou_difference=c["repeat_miou_difference"])
                           for k, c in captures.items()},
    object_count_groups={
        group: dict(expressions=sum(len(r["object_ids"]) == count for r in captures[args.name]["passes"][0]["rows"]),
                    scenes=len({r["scan_id"] for r in captures[args.name]["passes"][0]["rows"]
                                if len(r["object_ids"]) == count}))
        for group, count in (("single_object", 1), ("zero_object", 0))},
    limitations=["Official unique/multiple metadata unknown; object-count strata are not substitutes.",
                 "Two repeats do not quantify training-seed uncertainty; no no-harm/equivalence claim.",
                 "First-chunk timings are not paired trained Pareto evidence."])
multi = [r for r in captures[args.name]["passes"][0]["rows"] if len(r["object_ids"]) > 1]
result["object_count_groups"]["multiple_objects"] = dict(
    expressions=len(multi), scenes=len({r["scan_id"] for r in multi}))
result["frozen_harm_triggered_groups"] = {
    ref: [group for group, row in result["versus_" + ref].items()
          if row.get("frozen_material_harm_ci_upper_below_zero")]
    for ref in ("w2", "b1")}
output = ROOT / ("pipeline/analyses/" + args.name + "_clustered_392a_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: result[k] for k in ("id", "summary", "repeated_miou", "versus_w2",
                                       "versus_b1", "target_reached", "frozen_harm_triggered_groups")}, indent=2))
