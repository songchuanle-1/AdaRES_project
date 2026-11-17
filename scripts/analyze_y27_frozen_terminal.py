"""Reuse the corrected repeat-noise and paired whole-scene analysis for Y27."""

import ast
import hashlib
from pathlib import Path

from analyze_y23_frozen_terminal import comparison as corrected_comparison

source_path = Path(__file__).with_name("analyze_y18_frozen_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "966d40b34402f0ea779e357a4edad5aac5054fade627a3c975584b97e1d5e719"


def comparison(candidate, reference):
    for capture in (candidate, reference):
        assert all(p["summary"]["expressions"] == 9508 and p["summary"]["scenes"] == 141
                   for p in capture["passes"])
    results = corrected_comparison(candidate, reference)
    for result in results.values():
        if "scene_clustered_ci95" in result:
            result["frozen_material_harm_ci_upper_below_zero"] = result["scene_clustered_ci95"][1] < 0
    return results


class Y27Identity(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "analyze_frozen_full":
            return ast.copy_location(ast.Pass(), node)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(
                ast.Constant(value=node.value.replace("Y18", "Y27").replace("y18", "y27")), node)
        return node


tree = ast.fix_missing_locations(Y27Identity().visit(ast.parse(source)))
if __name__ == "__main__":
    exec(compile(tree, str(source_path), "exec"),
         {"__name__": "__main__", "__file__": str(source_path), "comparison": comparison})
