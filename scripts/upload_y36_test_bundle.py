"""Reuse additive upload checks for the independent Y36 test package."""

import ast
import hashlib
from pathlib import Path

source_path = Path(__file__).with_name("build_y35_test_bundle.py")
source = source_path.read_bytes()


class Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y35", "Y36").replace(
                "y35_language_evidence", "y36_local_adjacency").replace(
                "y35_", "y36_").replace("MaskLanguageEvidenceReadout", "MaskLocalAdjacencyReadout")
            value = value.replace("eaf330166aed361ffc2532a52e352b70cba4e2fd",
                                  "b49b744dc2943a38ab4c3aeb788acc6a2cb90011")
            value = value.replace("Y33 untouched", "Y35/Y33 source untouched")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
