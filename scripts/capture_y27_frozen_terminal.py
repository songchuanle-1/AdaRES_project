"""Reuse the frozen two-pass GPU1 evaluator for the completed Y27 run."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_frozen_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "8adbfad10fa4c6e77ee4e89e926a04ac25e6d6f17403b60c235a900c14ce409a"


class Y27Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("adares_y18_point_measure_res.yaml", "adares_y27_relative_peer_res.yaml")
            value = value.replace("Y18", "Y27").replace("y18", "y27").replace("7449", "9901")
            if "s=p['server'];root=pathlib.Path" in value:
                value = value.replace(
                    "assert not pathlib.Path('/proc/9901').exists()",
                    "assert not pathlib.Path('/proc/9901').exists()\n"
                    "assert (root/'.codex_source_commit').read_text().strip()"
                    " == '359a89d59dcf3990f4e45b1202d11d3257233e76'",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Y27Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
