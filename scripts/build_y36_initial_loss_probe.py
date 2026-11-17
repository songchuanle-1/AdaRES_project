"""Reuse the fixed temporary-head investigation with edge interventions."""

import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
original = root / "scripts/probe_y33_initial_loss_investigation.py"
target = root / "scripts/probe_y36_initial_loss_investigation.py"
assert not target.exists()


class Edges(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "token_readout":
            node.attr = "edge_readout"
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("Y33", "Y36").replace(
                "y33_token_attention", "y36_local_adjacency").replace(
                "token_readout", "edge_readout").replace("wrong_tokens", "wrong_edges").replace(
                "new7 tensors", "new8 tensors")
        elif node.value == 7:
            node.value = 8
        return node

    def visit_FunctionDef(self, node):
        if node.name == "shape_input":
            return ast.parse("""
def shape_input(module, args):
    q, masks, pad, mass, tokens, coords = args
    if mode == "rolled":
        return q, masks.roll(1, 1), pad, mass, tokens, coords
    if mode == "wrong_edges":
        changed = coords.clone()
        for b in range(coords.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = coords[b, :count].roll(1, 0)
        return q, masks, pad, mass, tokens, changed
""").body[0]
        return self.generic_visit(node)


tree = ast.fix_missing_locations(Edges().visit(ast.parse(original.read_text(encoding="utf-8"))))
source = ast.unparse(tree) + "\n"
compile(source, str(target), "exec")
target.write_text(source, encoding="utf-8")
print(json.dumps(dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                     target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                     changes="Candidate8tensors/6-input path/wrong-coordinate edges only; fixed native2 temporary steps/no diagnostic weight deployment.")))
