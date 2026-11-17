"""Repair only generated control-result keyword names, preserving the failure."""

import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
original = root / "scripts/probe_y36_native_controls.py"
target = root / "scripts/probe_y36_native_controls_v2.py"
assert not target.exists()


class Fields(ast.NodeTransformer):
    def visit_keyword(self, node):
        if node.arg is not None:
            node.arg = node.arg.replace("wrong_tokens", "wrong_edges").replace(
                "token_gradients", "edge_gradients")
        return self.generic_visit(node)


tree = ast.fix_missing_locations(Fields().visit(ast.parse(original.read_text(encoding="utf-8"))))
source = ast.unparse(tree) + "\n"
compile(source, str(target), "exec")
target.write_text(source, encoding="utf-8")
print(json.dumps(dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                     target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                     changes="dict keyword field spelling only; same model, dataset, seed, modes and gate thresholds.")))
