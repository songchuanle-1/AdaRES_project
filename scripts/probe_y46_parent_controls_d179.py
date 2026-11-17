"""Actual Y31/W2/Y36 equal normalized-shape/local-summary graph controls."""

import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch import nn
import torch.nn.functional as F

assert P["physical_gpu"] is None
torch.set_num_threads(4)
classes, hashes = {}, {}
for name, path in (
        ("MaskShapeReadout", "ipdn/model_paper5/adares_y31_mask_shape/dec.py"),
        ("QueryCompetitionIndicator", "ipdn/model_paper5/adares_w2_pointmass_competition/dec.py"),
        ("MaskLocalAdjacencyReadout", "ipdn/model_paper5/adares_y36_local_adjacency/dec.py")):
    source = Path(path).read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), path, "exec"), namespace)
    classes[name] = namespace[name]
    hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()

ring = [[i, 0, 0] for i in range(4)] + [[4, i, 0] for i in range(4)]
ring += [[4-i, 4, 0] for i in range(4)] + [[0, 4-i, 0] for i in range(4)]
squares = [[x + i, y + j, 0] for x in (-4, 4) for y in (-4, 4)
           for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))]
positions = [torch.tensor(points, dtype=torch.double).unsqueeze(0) for points in (ring, squares)]
components = []
for points in positions:
    distance = torch.cdist(points[0], points[0])
    distance.fill_diagonal_(float("inf"))
    graph = (distance == distance.min(-1, keepdim=True).values).numpy()
    assert (graph.sum(-1) == 2).all()
    count, labels = connected_components(csr_matrix(graph), directed=False)
    components.append(sorted(np.bincount(labels).tolist()))
assert components == [[16], [4, 4, 4, 4]]
cases = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    q = torch.randn(1, 128, 256, dtype=torch.double)
    features = torch.randn(1, 1, 256, dtype=torch.double).expand(1, 16, 256).clone()
    masks = torch.full((1, 128, 16), 8., dtype=torch.double)
    padding = torch.zeros(1, 16, dtype=torch.bool)
    mass = torch.ones(1, 16, dtype=torch.double)
    shape = classes["MaskShapeReadout"](256).double()
    local = classes["MaskLocalAdjacencyReadout"](256).double()
    w2 = classes["QueryCompetitionIndicator"](256).double()
    with torch.no_grad():
        descriptor = [shape.descriptor(masks, padding, mass, p) for p in positions]
        shape_score = [shape(q, masks, padding, mass, p) for p in positions]
        local_score = [local(q, masks, padding, mass, features, p) for p in positions]
        parent_score = [w2(q, masks, padding, mass) for _ in positions]
    gaps = {}
    for name, scores in (("Y31_descriptor", descriptor), ("Y31_actual", shape_score),
                         ("Y36_actual", local_score), ("W2_actual", parent_score)):
        gap = float((scores[0] - scores[1]).abs().max())
        tolerance = 8 * torch.finfo(torch.double).eps * 16 * max(
            1., float(scores[0].abs().max()), float(scores[1].abs().max()))
        assert gap <= tolerance, name
        gaps[name] = dict(max_difference=gap, prospective8eps_nodes_bound=tolerance)
    cases.append(dict(seed=seed, comparisons=gaps))
result = dict(complete=True, id="AdaRES-Y46", cases=cases, CPU_cases=32,
              component_masses=components, exact_nearest_tie_degree=2, actual_source_hashes=hashes,
              maximum_actual_parent_difference=max(v["max_difference"] for c in cases for v in c["comparisons"].values()),
              scientific_status="actual_W2_Y31_Y36_structural_information_witness",
              limitations=["Equal normalized shape, local and competition outputs for these controlled inputs only.",
                           "No universal representational-blindness, trained efficacy or novelty claim.",
                           "Four-component true objects remain legal; no single-component prior.",
                           "No Y43 rejection reopened, tuning, native backward or admission."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
