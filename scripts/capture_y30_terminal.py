"""Reuse the bounded terminal reader with the original Y30 identity."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "afd6a67d5a56f894137bb8e89d22f1d4115cf12572a2e3f6954283ca01b27a36"


class Y30Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y18", "Y30").replace("y18", "y30").replace("7449", "39709")
            if value.startswith("\nimport hashlib,json,pathlib,re,subprocess"):
                value = value.replace(
                    "task=__TASK__",
                    "task=__TASK__\n"
                    "assert task['config'] == 'configs_song/model_paper5/adares_y30_query_trajectory_res.yaml'\n"
                    "assert task['log'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/"
                    "train_adares_y30_query_trajectory_res_20261006_193015.log'\n"
                    "assert pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES/.codex_source_commit')"
                    ".read_text().strip() == '4f5c31f57994b8f9fb640d652c913863dc310a95'",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        if node.value == 7449:
            return ast.copy_location(ast.Constant(value=39709), node)
        return node


tree = ast.fix_missing_locations(Y30Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
