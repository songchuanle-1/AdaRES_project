"""Reuse the bounded terminal reader for the original Y28 run."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "afd6a67d5a56f894137bb8e89d22f1d4115cf12572a2e3f6954283ca01b27a36"


class Y28Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y18", "Y28").replace("y18", "y28").replace("7449", "19745")
            if value.startswith("\nimport hashlib,json,pathlib,re,subprocess"):
                value = value.replace(
                    "task=__TASK__",
                    "task=__TASK__\n"
                    "assert task['config'] == 'configs_song/model_paper5/adares_y28_mask_readback_res.yaml'\n"
                    "assert task['log'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/"
                    "train_adares_y28_mask_readback_res_20261005_202229.log'\n"
                    "assert pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES/.codex_source_commit')"
                    ".read_text().strip() == 'ce2dc75ee9e706da68c5768d9b3c855f33bda7fb'",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        if node.value == 7449:
            return ast.copy_location(ast.Constant(value=19745), node)
        return node


tree = ast.fix_missing_locations(Y28Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
