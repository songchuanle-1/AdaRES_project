"""Reuse strict candidate-only committed-byte alignment on idle GPU1."""

import ast
from pathlib import Path

source_path = Path(__file__).with_name("normalize_y35_committed_bytes.py")
source = source_path.read_bytes()


class Identity(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y35", "Y36").replace(
                "y35_language_evidence", "y36_local_adjacency").replace("y35_", "y36_")
            value = value.replace("eaf330166aed361ffc2532a52e352b70cba4e2fd",
                                  "b49b744dc2943a38ab4c3aeb788acc6a2cb90011")
            value = value.replace(".foreground_gpu0_diagnostic.lock", ".y15_gpu1_diagnostic.lock")
            value = value.replace("'nvidia-smi','-i','0'", "'nvidia-smi','-i','1'")
            return ast.copy_location(ast.Constant(value=value), node)
        return node

    def visit_keyword(self, node):
        if node.arg == "physical_gpu":
            node.value = ast.Constant(value=1)
        return self.generic_visit(node)


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
