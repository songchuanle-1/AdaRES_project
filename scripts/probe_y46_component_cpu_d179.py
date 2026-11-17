"""All-component CPU prototype on exact prediction-blind native captures."""

import base64
import hashlib
import json
from pathlib import Path
import time
import zlib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch
from torch import nn

assert P["physical_gpu"] is None
torch.set_num_threads(4)
torch.manual_seed(1999)
capture_path = Path("/tmp/adares_y46_native_capture_d179_" + P["token"] + ".json")
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == "5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89"
captured = json.loads(capture_path.read_text())
assert captured["complete"] and captured["native_scenes"] == 8


def unpack(item):
    raw = zlib.decompress(base64.b64decode(item["zlib_base64"]))
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    return np.frombuffer(raw, dtype=np.dtype(item["dtype"])).reshape(item["shape"]).copy()


def component_inputs(graph, masks, masses, tokens, collapse=False):
    """Retain every component, including background; no component filtering."""
    rows, weights, groups = [], [], []
    for query, mask in enumerate(masks):
        for side, selected in enumerate((mask, ~mask)):
            indices = np.flatnonzero(selected)
            if not len(indices):
                continue
            if collapse:
                count, labels = 1, np.zeros(len(indices), dtype=np.int32)
            else:
                count, labels = connected_components(graph[indices][:, indices], directed=False)
            component_mass = np.bincount(labels, weights=masses[indices], minlength=count)
            feature_sum = np.zeros((count, tokens.shape[1]), dtype=np.float64)
            np.add.at(feature_sum, labels, tokens[indices] * masses[indices, None])
            total_mass = masses[indices].sum()
            assert component_mass.sum() == total_mass
            assert (component_mass > 0).all()
            fraction = component_mass / total_mass
            rows.append(np.concatenate((feature_sum / component_mass[:, None], fraction[:, None]), 1))
            weights.append(fraction)
            groups.append(np.full(count, 2 * query + side, dtype=np.int64))
    return (torch.from_numpy(np.concatenate(rows)), torch.from_numpy(np.concatenate(weights)),
            torch.from_numpy(np.concatenate(groups)))


class AllComponentReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.component_encoder = nn.Sequential(
            nn.Linear(width + 1, width), nn.GELU(), nn.Linear(width, width))
        self.head = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, query, inputs):
        rows, weights, groups = inputs
        encoded = self.component_encoder(rows)
        pooled = encoded.new_zeros((2 * len(query), query.shape[-1]))
        pooled.index_add_(0, groups, weights[:, None] * encoded)
        pooled = pooled.reshape(len(query), -1)
        normalized_query = nn.functional.layer_norm(query, (query.shape[-1],))
        return self.head(torch.cat((normalized_query, pooled), -1)).flatten()


def bound(a, b, nodes):
    # Prospective reduction-length floating-point envelope, not fitted to outputs.
    return 8 * torch.finfo(torch.double).eps * max(1, nodes) * max(
        1., float(a.abs().max()), float(b.abs().max()))


prototype = AllComponentReadout(256).double()
cases = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    module = AllComponentReadout(256).double()
    edges_a = [(i, (i + 1) % 8) for i in range(8)]
    edges_b = [(i, (i // 4) * 4 + (i + 1) % 4) for i in range(8)]
    graphs = []
    for edges in (edges_a, edges_b):
        undirected = edges + [(j, i) for i, j in edges]
        i, j = zip(*undirected)
        graphs.append(csr_matrix((np.ones(len(i), bool), (i, j)), shape=(8, 8)))
    query = torch.randn(128, 256, dtype=torch.double)
    tokens = torch.randn(1, 256, dtype=torch.double).expand(8, 256).numpy().copy()
    masks = np.ones((128, 8), dtype=bool)
    inputs = [component_inputs(g, masks, np.ones(8), tokens) for g in graphs]
    collapsed = [component_inputs(g, masks, np.ones(8), tokens, collapse=True) for g in graphs]
    with torch.no_grad():
        a, b = [module(query, item) for item in inputs]
        ca, cb = [module(query, item) for item in collapsed]
    assert torch.equal(ca, cb)
    difference = float((a - b).abs().max())
    assert difference > bound(a, b, 8), "Unexplained mechanism no-op"
    cases.append(dict(seed=seed, component_score_difference=difference, pooled_control_difference=0,
                      prospective8eps_nodes_bound=bound(a, b, 8)))

rows = []
start = time.monotonic()
for scene in captured["rows"]:
    for expression in scene["expressions"]:
        arrays = {key: unpack(value) for key, value in expression["packed_native_inputs"].items()}
        query = torch.from_numpy(arrays["query"].astype(np.float64))
        tokens = arrays["tokens"].astype(np.float64)
        masses = arrays["masses"].astype(np.float64)
        masks = arrays["masks"] > 0
        n = len(masses)
        edges = arrays["graph_edges"]
        graph = csr_matrix((np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])), shape=(n, n))
        assert masks.shape == (128, n) and tokens.shape == (n, 256)
        inputs = component_inputs(graph, masks, masses, tokens)
        collapsed = component_inputs(graph, masks, masses, tokens, collapse=True)
        reversed_order = np.arange(n - 1, -1, -1)
        permuted = component_inputs(graph[reversed_order][:, reversed_order], masks[:, reversed_order],
                                    masses[reversed_order], tokens[reversed_order])
        wrong = component_inputs(graph, np.roll(masks, 1, axis=0), masses, tokens)
        prototype.zero_grad(set_to_none=True)
        score = prototype(query, inputs)
        assert torch.isfinite(score).all()
        # This is a prototype graph test, explicitly NOT a native training loss.
        score.square().mean().backward()
        gradients = {name: float(value.grad.norm()) for name, value in prototype.named_parameters()}
        assert all(np.isfinite(value) and value > 0 for value in gradients.values())
        with torch.no_grad():
            repeated = prototype(query, inputs)
            permutation_score = prototype(query, permuted)
            collapsed_score = prototype(query, collapsed)
            wrong_score = prototype(query, wrong)
        assert torch.equal(repeated, score.detach())
        permutation_error = float((score - permutation_score).abs().max())
        prospective = bound(score, permutation_score, n)
        assert permutation_error <= prospective
        separate = float((score - collapsed_score).abs().max())
        wrong_change = float((score - wrong_score).abs().max())
        assert separate > bound(score, collapsed_score, n)
        assert wrong_change > bound(score, wrong_score, n)
        rows.append(dict(dataset_index=scene["dataset_index"], scan_id=scene["scan_id"],
                         ann_id=expression["ann_id"], nodes=n,
                         all_FG_BG_component_rows=len(inputs[0]), finite_nonzero_gradients=gradients,
                         exact_cached_repeat=True, joint_permutation_error=permutation_error,
                         prospective8eps_nodes_bound=prospective,
                         component_vs_pooled_max_difference=separate,
                         cyclic_wrong_mask_max_difference=wrong_change))
result = dict(
    complete=True, id="AdaRES-Y46", scientific_status="CPU_prototype_controls_not_native_backward",
    cases=cases, rows=rows, CPU_cases=32, native_scenes=8, native_expressions=len(rows), queries=128,
    new_parameter_tensors=len(list(prototype.parameters())),
    prototype_parameters=sum(value.numel() for value in prototype.parameters()),
    minimum_new_tensor_gradient=min(v for r in rows for v in r["finite_nonzero_gradients"].values()),
    minimum_component_vs_pooled_difference=min(r["component_vs_pooled_max_difference"] for r in rows),
    minimum_wrong_mask_difference=min(r["cyclic_wrong_mask_max_difference"] for r in rows),
    maximum_joint_permutation_error=max(r["joint_permutation_error"] for r in rows),
    native_cpu_seconds=time.monotonic() - start,
    input_capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
    readout_source=Path(__file__).read_text() if "__file__" in globals() else "Source retained by lease runner.",
    limitations=["All components retained; true multi-component targets are not treated as errors.",
                 "Prototype receives exact cached native tensors, not full native training-loss/backward.",
                 "Graph/labels are detached; no gradients through SciPy or hard mask topology.",
                 "Eight synthetic scalar-square parameter gradients do not establish native loss activity.",
                 "No independence package, native GPU resources, admission or efficacy claim.",
                 "No validation, graph-parameter, feature-subset, loss-weight or threshold search."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
