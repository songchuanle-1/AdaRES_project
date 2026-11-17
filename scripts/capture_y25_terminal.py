"""Reuse the bounded terminal reader for the original Y25 run."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "afd6a67d5a56f894137bb8e89d22f1d4115cf12572a2e3f6954283ca01b27a36"


class Y25Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y18", "Y25").replace("y18", "y25").replace("7449", "98611")
            if value.startswith("\nimport hashlib,json,pathlib,re,subprocess"):
                value = value.replace(
                    "task=__TASK__",
                    "task=__TASK__\n"
                    "assert pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES/.codex_source_commit').read_text().strip()"
                    " == '94918c1a008e21c72b9478c28f934e8b329e0c97'",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        if node.value == 7449:
            return ast.copy_location(ast.Constant(value=98611), node)
        return node


tree = ast.fix_missing_locations(Y25Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
