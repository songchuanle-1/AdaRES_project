"""Reuse the bounded terminal reader with the original Y31 identity."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("capture_y18_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "afd6a67d5a56f894137bb8e89d22f1d4115cf12572a2e3f6954283ca01b27a36"


class Y31Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y18", "Y31").replace("y18", "y31").replace("7449", "66677")
            if value.startswith("\nimport hashlib,json,pathlib,re,subprocess"):
                value = value.replace(
                    "task=__TASK__",
                    "task=__TASK__\n"
                    "assert task['config'] == 'configs_song/model_paper5/adares_y31_mask_shape_res.yaml'\n"
                    "assert task['log'] == '/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/"
                    "train_adares_y31_mask_shape_res_20261007_191327.log'\n"
                    "assert pathlib.Path('/bak_home/songcl/Project/AdaRES_project/AdaRES/.codex_source_commit')"
                    ".read_text().strip() == '5f2489de78196e8d89c004e452a665c0946f2a94'",
                )
            return ast.copy_location(ast.Constant(value=value), node)
        if node.value == 7449:
            return ast.copy_location(ast.Constant(value=66677), node)
        return node


tree = ast.fix_missing_locations(Y31Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
