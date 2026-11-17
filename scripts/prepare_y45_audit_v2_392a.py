"""Correct only Python3.8 Index handling; preserve failed audit and model."""

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
parent = ROOT / "AdaRES/ipdn/model_paper5/adares_w2_pointmass_competition"
capture = ROOT / ("pipeline/analyses/w2_native_source_392a_" + args.token)
identity = {}
for path in parent.glob("*.py"):
    actual = capture / path.name
    assert ast.dump(ast.parse(path.read_text(encoding="utf-8"))) == ast.dump(ast.parse(actual.read_text(encoding="utf-8")))
    identity[path.name] = dict(local_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               native_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
                               AST_equal=True)
source = ROOT / "scripts/probe_y45_package_cpu_392a.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
count = 0
for node in ast.walk(tree):
    if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
        if any(isinstance(n, ast.Constant) and n.value == "point_positive" for n in ast.walk(node.test)) and any(
                isinstance(n, ast.Attribute) and n.attr == "slice" for n in ast.walk(node.test)):
            node.test = ast.parse(
                "isinstance(target, ast.Subscript) and subscript_key(target) == 'point_positive'",
                mode="eval").body
            count += 1
assert count == 1
helper = ast.parse('''
def subscript_key(node):
    index=node.slice
    if isinstance(index,ast.Index):
        index=index.value
    return index.value if isinstance(index,ast.Constant) else None
''').body
tree.body[0:0] = helper
output = ROOT / "scripts/probe_y45_package_cpu_392a_v2.py"
assert not output.exists()
output.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n", encoding="utf-8")
proof = ROOT / ("pipeline/analyses/y45_audit_repair_392a_" + args.token + ".json")
assert not proof.exists()
result = dict(complete=True, id="AdaRES-Y45", original_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              repaired_audit_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
              change="Only normalize Python3.8 ast.Index in the audit; all model/config code unchanged.",
              parent_sources=identity)
proof.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(dict(path=str(output), model_changes=0, verified_parent_modules=6)))
