"""Candidate-only LF alignment, specialization of tested scoped helper."""
import ast
from pathlib import Path

source_path = Path(__file__).with_name("normalize_y35_committed_bytes.py")
source = source_path.read_bytes()


class Identity(ast.NodeTransformer):
    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg == "physical_gpu":
            assert node.value.value == 0
            node.value = ast.copy_location(ast.Constant(value=1), node.value)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value.replace("Y35", "Y39").replace(
                "y35_language_evidence", "y39_point_distribution").replace("y35_", "y39_")
            value = value.replace("eaf330166aed361ffc2532a52e352b70cba4e2fd",
                                  "67f4a1b2a4e7cac0eaa1a6074104cfc4193c2773")
            value = value.replace("Y33 and original evidence", "Y37 and original evidence")
            value = value.replace(".foreground_gpu0_diagnostic.lock", ".y15_gpu1_diagnostic.lock")
            value = value.replace("'nvidia-smi','-i','0'", "'nvidia-smi','-i','1'")
            return ast.copy_location(ast.Constant(value=value), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path)})
