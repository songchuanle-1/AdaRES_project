"""Preserve the failed probe; correct only its serialized gradient field."""

import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "scripts/probe_y37_native_controls_v2.py"
target = source.with_name("probe_y37_native_controls_v3.py")
assert not target.exists()
before = source.read_bytes()


class Field(ast.NodeTransformer):
    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg == "edge_gradients":
            node.arg = "pair_gradients"
        return node


tree = ast.fix_missing_locations(Field().visit(ast.parse(before)))
after = (ast.unparse(tree) + "\n").encode()
target.write_bytes(after)
print(json.dumps(dict(
    original_sha256=hashlib.sha256(before).hexdigest(),
    corrected_sha256=hashlib.sha256(after).hexdigest(),
    scope="One output keyword only; original probe and failed raw log preserved.",
)))
