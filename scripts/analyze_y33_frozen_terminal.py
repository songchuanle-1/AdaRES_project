"""Preserve the corrected paired whole-scene comparison protocol for Y33."""

import ast
import hashlib
from pathlib import Path
from analyze_y27_frozen_terminal import comparison

source_path = Path(__file__).with_name("analyze_y18_frozen_terminal.py")
source = source_path.read_bytes()
assert hashlib.sha256(source).hexdigest() == "966d40b34402f0ea779e357a4edad5aac5054fade627a3c975584b97e1d5e719"


class Identity(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "analyze_frozen_full":
            return ast.copy_location(ast.Pass(), node)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=node.value.replace("Y18", "Y33").replace("y18", "y33")), node)
        return node


tree = ast.fix_missing_locations(Identity().visit(ast.parse(source)))
exec(compile(tree, str(source_path), "exec"),
     {"__name__": "__main__", "__file__": str(source_path), "comparison": comparison})
