"""Reuse the hash-bound terminal reader for the resumed Y23 identity."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "afd6a67d5a56f894137bb8e89d22f1d4115cf12572a2e3f6954283ca01b27a36"
tree = ast.parse(source, filename=str(source_path))


class Y23Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y18", "Y23").replace("y18", "y23").replace("7449", "129584")
            return ast.copy_location(ast.Constant(value=value), node)
        if node.value == 7449:
            return ast.copy_location(ast.Constant(value=129584), node)
        return node


adapted = ast.fix_missing_locations(Y23Identity().visit(tree))
exec(compile(adapted, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
