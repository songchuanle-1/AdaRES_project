"""Mechanically adapt only experiment bindings in the existing cost probe."""

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
source = ROOT / "scripts/probe_y44_trained_cost_392a.py"
mapping = {
    "/tmp/adares_y44_terminal_392a_": "/tmp/adares_y45_terminal_601b_",
    "/tmp/adares_y44_frozen_392a_": "/tmp/adares_y45_frozen_601b_",
    "y44": "y45",
    "AdaRES-Y44": "AdaRES-Y45",
}
counts = dict.fromkeys(mapping, 0)


class Bindings(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in mapping:
            counts[node.value] += 1
            return ast.copy_location(ast.Constant(value=mapping[node.value]), node)
        return node


tree = ast.fix_missing_locations(Bindings().visit(ast.parse(source.read_text())))
assert list(counts.values()) == [2, 1, 1, 1]
text = ast.unparse(tree) + "\n"
destination = ROOT / "scripts/probe_y45_trained_cost_601b.py"
compile(text, str(destination), "exec")
with destination.open("x", encoding="utf-8") as stream:
    stream.write(text)
result = dict(
    complete=True, id="AdaRES-Y45",
    preregistered_at=datetime.now().astimezone().isoformat(),
    source=dict(path=source.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    generated=dict(path=destination.relative_to(ROOT).as_posix(),
                   sha256=hashlib.sha256(destination.read_bytes()).hexdigest()),
    only_changed_constant_bindings=counts,
    protocol="Unchanged TRAIN8/largest-pair, native predict2warmups3samples, separate trained native optimizer stress2warmups3samples; ephemeral updates only.")
manifest = ROOT / ("pipeline/analyses/y45_cost_binding_prereg_601b_" + args.token + ".json")
with manifest.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
