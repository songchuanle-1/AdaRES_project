"""Preserve the failed adapter and bind the actual protected GPU1 command."""

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
source = ROOT / "scripts/probe_y41_trained_cost_0422.py"
tree = ast.parse(source.read_text())
changed = []
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and node.value == b"configs_song/model_paper5/adares_y41_boundary_probability_res.yaml":
        node.value = b"configs_song/model_paper5/adares_y42_point_covariance_res.yaml"
        changed.append("protected_config_bytes")
    elif isinstance(node, ast.Constant) and node.value == b"0":
        node.value = b"1"
        changed.append("protected_gpu_bytes")
assert sorted(changed) == ["protected_config_bytes", "protected_gpu_bytes"]
target = ROOT / "scripts/probe_y41_trained_cost_0422_v2.py"
assert not target.exists()
ast.fix_missing_locations(tree)
text = ast.unparse(tree) + "\n"
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
log = ROOT / ("pipeline/analyses/y41_trained_cost_0422_" + args.token + ".log")
record = {
    "complete": True, "scope": "Adapter repair only, no scientific protocol/source change.",
    "failure": {"path": log.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                "reason": "Protected PID91215 is Y42/GPU1; first adapter expected Y41/GPU0 bytes."},
    "old_program_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "retry_program": {"path": target.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
}
output = ROOT / ("pipeline/analyses/y41_cost_binding_repair_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record))
