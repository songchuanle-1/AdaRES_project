"""Resolve the exact-fixture adaptation locally for Python3.8 execution."""

import ast
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
original = root / "scripts/probe_y36_structure_cpu.py"
target = root / "scripts/probe_y36_structure_cpu_v2.py"


class Fixture(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (any(isinstance(t, ast.Name) and t.id == "visual" for t in node.targets)
                and ast.unparse(node.value) == "torch.randn(1, 6, 256)"):
            node.value = ast.parse("torch.randint(-8, 9, (1, 6, 256)).float() / 8").body[0].value
        return self.generic_visit(node)

    def visit_Assert(self, node):
        if ast.unparse(node.test).startswith("torch.allclose(visual.mean"):
            node.test = ast.parse("torch.equal(visual.mean(1), changed.mean(1))").body[0].value
        return self.generic_visit(node)


tree = ast.fix_missing_locations(Fixture().visit(ast.parse(original.read_text(encoding="utf-8"))))
source = ast.unparse(tree) + "\n"
compile(source, str(target), "exec")
target.write_text(source, encoding="utf-8")
print("original_sha256=" + hashlib.sha256(original.read_bytes()).hexdigest())
print("fixed_fixture_sha256=" + hashlib.sha256(target.read_bytes()).hexdigest())
