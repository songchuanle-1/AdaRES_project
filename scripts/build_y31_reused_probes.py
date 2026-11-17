"""Mechanical AST adaptation of tested native/resume/resource harnesses."""

import ast
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent


class Adapt(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("Y30", "Y31").replace(
                "y30_query_trajectory", "y31_mask_shape").replace(
                "trajectory_readout", "shape_readout").replace(
                "temporal confidence", "mask-shape confidence")
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "trajectory_readout":
            node.attr = "shape_readout"
        return node

    def visit_Assert(self, node):
        self.generic_visit(node)
        for part in ast.walk(node.test):
            if (isinstance(part, ast.Compare) and isinstance(part.left, ast.Call)
                    and isinstance(part.left.func, ast.Name) and part.left.func.id == "len"):
                for comparator in part.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value == 8:
                        comparator.value = 6
        return node


for stem, original in (
        ("physical_native", "probe_y30_physical_native_v2.py"),
        ("resources", "probe_y30_resources.py")):
    source = root / original
    target = root / ("probe_y31_" + stem + ".py")
    assert not target.exists()
    tree = Adapt().visit(ast.parse(source.read_text(encoding="utf-8")))
    text = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    assert "trajectory_readout" not in text and "AdaRES-Y30" not in text
    compile(text, str(target), "exec")
    target.write_text(text, encoding="utf-8")
    print(target.name, hashlib.sha256(target.read_bytes()).hexdigest())
