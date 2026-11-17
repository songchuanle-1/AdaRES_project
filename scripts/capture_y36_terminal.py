"""Capture original Y36 completion evidence without changing its lifecycle."""

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "scripts/capture_y18_terminal.py"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "afd6a67d5a56f894137bb8e89d22f1d4115cf12572a2e3f6954283ca01b27a36"


class Identity(ast.NodeTransformer):
    def visit_Assert(self, node):
        if ast.unparse(node.test) == "server['allowed_gpus'] == [1]":
            return ast.copy_location(
                ast.parse("assert server['allowed_gpus'] == [0, 1]").body[0], node
            )
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value == 7449:
            return ast.copy_location(ast.Constant(value=100018), node)
        if isinstance(node.value, str):
            value = node.value.replace("Y18", "Y36").replace("y18", "y36")
            value = value.replace("7449", "100018")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(
    compile(tree, str(source_path), "exec"),
    {"__name__": "__main__", "__file__": str(source_path)},
)
