"""Reuse the frozen GPU1 evaluator with the completed Y23 identity."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_frozen_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8adbfad10fa4c6e77ee4e89e926a04ac25e6d6f17403b60c235a900c14ce409a"


class Y23Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("adares_y18_point_measure_res.yaml", "adares_y23_valid_token_support_res.yaml")
            value = value.replace("Y18", "Y23").replace("y18", "y23").replace("7449", "129584")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.parse(source, filename=str(source_path))
adapted = ast.fix_missing_locations(Y23Identity().visit(tree))
exec(compile(adapted, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
