"""Adapt the preserved bounded terminal reader to the original Y35 job."""

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source_path = ROOT / "scripts/capture_y18_terminal.py"
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "afd6a67d5a56f894137bb8e89d22f1d4115cf12572a2e3f6954283ca01b27a36"
task = dict(
    id="AdaRES-Y35", pid=108331, gpu=0,
    config="configs_song/model_paper5/adares_y35_language_evidence_res.yaml",
    log="/bak_home/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y35_language_evidence_res_20261009_091225.log",
)


class Identity(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "task" for target in node.targets):
            return ast.copy_location(ast.parse("task=" + repr(task)).body[0], node)
        return self.generic_visit(node)

    def visit_Assert(self, node):
        test = ast.unparse(node.test)
        if test == "server['allowed_gpus'] == [1]":
            return ast.copy_location(ast.parse("assert server['allowed_gpus'] == [0, 1]").body[0], node)
        if test == "task['pid'] == 7449 and str(task['gpu']) == '1'":
            return ast.copy_location(ast.parse(
                "assert task['pid'] == 108331 and str(task['gpu']) == '0'").body[0], node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y18", "Y35").replace("y18", "y35")
            value = value.replace("7449", "108331").replace("GPU1", "GPU0")
            value = value.replace("gpu1_", "gpu0_").replace("'nvidia-smi','-i','1'", "'nvidia-smi','-i','0'")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
