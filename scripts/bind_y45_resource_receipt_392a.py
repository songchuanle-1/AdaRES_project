"""Additive identity clarification; never overwrite the raw measurement."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
args = parser.parse_args()
owner = json.loads((ROOT / "pipeline/headless/.foreground.lockdir/owner.json").read_text())
assert owner["token"] == args.token and owner["thread_id"] == os.environ["CODEX_THREAD_ID"]
assert datetime.now(timezone.utc) < datetime.fromisoformat(owner["deadline_at"])
raw = ROOT / ("pipeline/analyses/y45_resources_392a_" + args.token + ".json")
value = json.loads(raw.read_text(encoding="utf-8"))
source = ROOT / "scripts/probe_y45_resources_392a.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
configs = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
           and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "config"]
assert configs == ["configs_song/model_paper5/adares_y45_metric_decomposition_res.yaml"]
assert value["complete"] and value["id"] == "AdaRES-Y43"
physical = json.loads((ROOT / ("pipeline/analyses/y45_physical_native_392a_" + args.token + ".json")).read_text(encoding="utf-8"))
assert value["summary"]["candidate"]["parameters"] == physical["parameters"] == 164827272
assert value["parameters_added"] == physical["added_parameters"] == 197635
assert value["summary"]["candidate"]["stress_train_reserved"] < 24
assert value["parameters_added"] < .01 * value["summary"]["parent"]["parameters"]
output = ROOT / ("pipeline/analyses/y45_resources_binding_392a_" + args.token + ".json")
assert not output.exists()
result = dict(complete=True, id="AdaRES-Y45", raw_result_id=value["id"],
              correction="Export identity string inherited Y43; tested config/source and independent parameter count bind the actual Y45. Raw receipt unchanged; no repeated measurement.",
              raw=dict(path=raw.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
              executed_source=dict(path=source.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
              actual_config=configs[0], summary=value["summary"],
              added_parameters=197635, scientific_status="resource_feasibility_not_trained_Pareto")
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(dict(path=str(output), raw_unchanged=True, canonical_id="AdaRES-Y45")))
