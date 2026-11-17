"""Mechanical specialization with explicit wrong-token intervention."""

import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "scripts/probe_y31_initial_loss_investigation.py"
output = root / "scripts/probe_y33_initial_loss_investigation.py"
assert not output.exists()
substitutions = {
    "adares_y31_mask_shape_res": "adares_y33_token_attention_res",
    "shape_readout": "token_readout",
    "Y31": "Y33",
    "uniform": "wrong_tokens",
    "only new6": "only new7",
}


class Adapt(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for a, b in substitutions.items():
                node.value = node.value.replace(a, b)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "shape_readout":
            node.attr = "token_readout"
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Name) and node.left.func.id == "len":
            for v in node.comparators:
                if isinstance(v, ast.Constant) and v.value == 6:
                    v.value = 7
        return node

    def visit_FunctionDef(self, node):
        if node.name != "shape_input":
            return self.generic_visit(node)
        return ast.parse("""
def shape_input(module, args):
    q, masks, pad, mass, tokens = args
    if mode == 'rolled':
        return q, masks.roll(1, 1), pad, mass, tokens
    if mode == 'wrong_tokens':
        changed = tokens.clone()
        for b in range(tokens.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = tokens[b, :count].roll(1, 0)
        return q, masks, pad, mass, changed
""").body[0]


tree = ast.fix_missing_locations(Adapt().visit(ast.parse(source.read_text())))
compile(tree, str(output), "exec")
output.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
print(str(output))
