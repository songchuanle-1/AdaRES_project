"""Repair result keyword names only; retain the original failed native adapter."""

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
source = ROOT / "scripts/probe_y43_native_controls_0422.py"
tree = ast.parse(source.read_text())
changes = []
for node in ast.walk(tree):
    if isinstance(node, ast.keyword) and node.arg:
        replacement = node.arg.replace("covariance", "third").replace("diagonal_only", "zero_third")
        replacement = replacement.replace("y41_", "y42_")
        if replacement != node.arg:
            changes.append([node.arg, replacement])
            node.arg = replacement
assert changes
target = ROOT / "scripts/probe_y43_native_controls_0422_v2.py"
assert not target.exists()
text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
log = ROOT / ("pipeline/analyses/y43_native_controls_0422_" + args.token + ".log")
record = {
    "complete": True, "scope": "Result keyword fields only; no model/control/gate/tolerance changes.",
    "failure": {"path": log.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                "reason": "Dict keyword labels retained covariance names while indexing expected third-moment names."},
    "old_program_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "retry_program": {"path": target.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
    "changes": changes,
}
output = ROOT / ("pipeline/analyses/y43_native_field_repair_" + args.token + ".json")
assert not output.exists()
output.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record))
