"""Mechanical specialization of hash-restricted new-candidate LF alignment."""

import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "scripts/normalize_y31_committed_bytes.py"
output = root / "scripts/normalize_y33_committed_bytes.py"
assert not output.exists()
replacements = {
    "adares_y31_mask_shape": "adares_y33_token_attention",
    "Y31": "Y33", "y31": "y33",
    "4f5c31f57994b8f9fb640d652c913863dc310a95": "5f2489de78196e8d89c004e452a665c0946f2a94",
    "Strict git/blob/local/deployed raw-byte mismatch before any admission/queue write; only CRLF/LF.":
        "Prospective strict git/blob/local/deployed LF alignment before admission, not a failed admission.",
}


class Adapt(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for a in sorted(replacements, key=len, reverse=True):
                node.value = node.value.replace(a, replacements[a])
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if isinstance(node.left, ast.Name) and node.left.id == "commit":
            for i, v in enumerate(node.comparators):
                if isinstance(v, ast.Constant) and v.value == "5f2489de78196e8d89c004e452a665c0946f2a94":
                    node.comparators[i] = ast.Attribute(value=ast.Name(id="args", ctx=ast.Load()),
                                                       attr="commit", ctx=ast.Load())
        return node


tree = Adapt().visit(ast.parse(source.read_text()))
for index, node in enumerate(tree.body):
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
        tree.body.insert(index, ast.parse("parser.add_argument('--commit', required=True)").body[0])
        break
tree = ast.fix_missing_locations(tree)
compile(tree, str(output), "exec")
output.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
print(str(output))
