"""Bind AST adapter to the actual decoder method; retain both failed probes."""

import argparse
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
source = ROOT / "scripts/probe_y42_parent_semantics_v2.py"
text = source.read_text(encoding="utf-8")
anchor = 'node.func.attr in ("res", "dec")'
assert text.count(anchor) == 1
text = text.replace(anchor, 'node.func.attr in ("forward_iter_pred", "dec")')
target = ROOT / "scripts/probe_y42_parent_semantics_v3.py"
assert not target.exists()
compile(text, str(target), "exec")
target.write_text(text, encoding="utf-8", newline="\n")
log = ROOT / ("pipeline/analyses/y42_parent_semantics_v2_" + args.token + ".log")
output = ROOT / ("pipeline/analyses/y42_parent_ast_binding_repair_" + args.token + ".json")
assert not output.exists() and log.exists()
output.write_text(json.dumps({
    "complete": True, "original_program": {
        "path": source.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
    "failed_log": {"path": log.relative_to(ROOT).as_posix(),
                   "sha256": hashlib.sha256(log.read_bytes()).hexdigest()},
    "new_program": {"path": target.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
    "reason": "Actual DEC method is forward_iter_pred, not the guessed res.",
    "scope": "Exact call-name binding only; whole-module equality and removal "
             "counts unchanged; no model edit or weakened assertion.",
}, indent=2), encoding="utf-8")
print(json.dumps({"path": str(target), "repair": str(output)}))
