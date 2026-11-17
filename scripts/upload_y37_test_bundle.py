"""Reuse scoped additive upload; never overwrite Y35/Y36 source files."""

import ast
from pathlib import Path

source_path = Path(__file__).with_name("build_y35_test_bundle.py")
source = source_path.read_bytes()


class Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y35", "Y37").replace(
                "y35_language_evidence", "y37_pair_distribution").replace(
                "y35_", "y37_").replace("MaskLanguageEvidenceReadout", "MaskPairDistributionReadout")
            value = value.replace("eaf330166aed361ffc2532a52e352b70cba4e2fd",
                                  "6725f9bd6a8f9b47addae6a4b8336d004f73f1ae")
            value = value.replace("Y33 untouched", "Y35/Y36 source untouched")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
