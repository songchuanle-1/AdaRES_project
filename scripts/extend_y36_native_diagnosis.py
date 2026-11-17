"""Add explicit native edge controls without changing the frozen cohort."""

import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
original = root / "scripts/probe_y36_edge_diagnosis.py"
target = root / "scripts/probe_y36_edge_diagnosis_v2.py"
assert not target.exists()
extension = """
edge_contrasts = []
changed_coords = coords.clone()
for b in range(coords.shape[0]):
    keep = ~padding[b]
    positions = coords[b, keep]
    probabilities = masks[b, :, keep].sigmoid()
    distance = torch.cdist(positions, positions)
    distance.fill_diagonal_(float("inf"))
    adjacency = (distance == distance.min(-1, keepdim=True).values).to(probabilities.dtype)
    adjacency = adjacency / adjacency.sum(-1, keepdim=True)
    edge_contrasts.append(float((probabilities - probabilities @ adjacency.transpose(0, 1)).abs().mean()))
    count = int(keep.sum())
    changed_coords[b, :count] = coords[b, :count].roll(1, 0)
wrong_edges = readout(query, masks, padding, mass, tokens, changed_coords)
row["local_mask_edge_contrast"] = statistics.mean(edge_contrasts)
row["wrong_edge_delta"] = float((direct - wrong_edges).abs().max())
assert row["local_mask_edge_contrast"] > 0 and row["wrong_edge_delta"] > 0
"""


class NativeEdges(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "row" for t in node.targets):
            return [node] + ast.parse(extension).body
        if any(isinstance(t, ast.Name) and t.id == "result" for t in node.targets):
            for key, value in (
                ("positive_local_edge_contrast_scans", "sum(r['local_mask_edge_contrast'] > 0 for r in rows)"),
                ("wrong_edges_nonzero_scans", "sum(r['wrong_edge_delta'] > 0 for r in rows)"),
                ("minimum_wrong_edge_delta", "min(r['wrong_edge_delta'] for r in rows)"),
            ):
                node.value.keywords.append(ast.keyword(arg=key, value=ast.parse(value).body[0].value))
        return self.generic_visit(node)


tree = ast.fix_missing_locations(NativeEdges().visit(ast.parse(original.read_text(encoding="utf-8"))))
source = ast.unparse(tree) + "\n"
compile(source, str(target), "exec")
target.write_text(source, encoding="utf-8")
print(json.dumps(dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                     target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                     changes="Explicit local mask edge contrast and wrong-coordinate adjacency; native TRAIN64/seed/checkpoint unchanged.")))
