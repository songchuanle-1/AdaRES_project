"""Adapt the bounded terminal reader without modifying historical evidence."""

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "scripts/capture_y18_terminal.py"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "afd6a67d5a56f894137bb8e89d22f1d4115cf12572a2e3f6954283ca01b27a36"
task = dict(
    id="AdaRES-Y33", pid=64797, gpu=1,
    config="configs_song/model_paper5/adares_y33_token_attention_res.yaml",
    log="/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y33_token_attention_res_20261008_182513.log",
)


class Identity(ast.NodeTransformer):
    def visit_Assert(self, node):
        if ast.unparse(node.test) == "server['allowed_gpus'] == [1]":
            return ast.copy_location(ast.parse("assert server['allowed_gpus'] == [0, 1]").body[0], node)
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "task" for t in node.targets):
            return ast.copy_location(ast.parse("task=" + repr(task)).body[0], node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value == 7449:
            return ast.copy_location(ast.Constant(value=64797), node)
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=node.value.replace(
                "Y18", "Y33").replace("y18", "y33").replace("7449", "64797")), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
