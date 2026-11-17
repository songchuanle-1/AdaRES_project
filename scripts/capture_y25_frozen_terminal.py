"""Reuse the frozen two-pass GPU1 evaluator for the completed Y25 run."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_frozen_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8adbfad10fa4c6e77ee4e89e926a04ac25e6d6f17403b60c235a900c14ce409a"


class Y25Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("adares_y18_point_measure_res.yaml", "adares_y25_mask_feedback_res.yaml")
            value = value.replace("Y18", "Y25").replace("y18", "y25").replace("7449", "98611")
            if "s=p['server'];root=pathlib.Path" in value:
                value = value.replace(
                    "assert not pathlib.Path('/proc/98611').exists()",
                    "assert not pathlib.Path('/proc/98611').exists()\n"
                    "assert (root/'.codex_source_commit').read_text().strip()"
                    " == '94918c1a008e21c72b9478c28f934e8b329e0c97'",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Y25Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
