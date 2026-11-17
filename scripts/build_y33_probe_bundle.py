"""Mechanically specialize guarded Y31 test bundle and native harnesses."""

import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
replacements = {
    "Y31": "Y33", "y31": "y33",
    "adares_y31_mask_shape": "adares_y33_token_attention",
    "adares_y31_mask_shape_res": "adares_y33_token_attention_res",
    "MaskShapeReadout": "MaskTokenAttentionReadout",
    "shape_readout": "token_readout",
    "4f5c31f57994b8f9fb640d652c913863dc310a95": "5f2489de78196e8d89c004e452a665c0946f2a94",
}


def specialized(value):
    for old in sorted(replacements, key=len, reverse=True):
        value = value.replace(old, replacements[old])
    return value


class Adapt(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = specialized(node.value)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = specialized(node.attr)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if (isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name)
                and node.left.func.id == "len"
                and not any(isinstance(arg, ast.Name) and arg.id == "files"
                            for arg in node.left.args)):
            for value in node.comparators:
                if isinstance(value, ast.Constant) and value.value == 6:
                    value.value = 7
        return node


for original, target in (
    ("build_y31_test_bundle.py", "build_y33_test_bundle.py"),
    ("probe_y31_physical_native.py", "probe_y33_physical_native.py"),
    ("probe_y31_resources.py", "probe_y33_resources.py"),
):
    source = root / "scripts" / original
    output = root / "scripts" / target
    assert not output.exists()
    tree = ast.fix_missing_locations(Adapt().visit(ast.parse(source.read_text())))
    compile(tree, str(output), "exec")
    output.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
    print(str(output))
