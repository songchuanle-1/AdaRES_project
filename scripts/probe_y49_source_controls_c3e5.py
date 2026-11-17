"""Actual-source CPU controls for global topology beyond normalized summaries."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
import torch
from torch import nn

assert P["physical_gpu"] is None
torch.set_num_threads(4)
base = Path(P["root"]) / "ipdn/model_paper5"
specs = {
    "W2": ("adares_w2_pointmass_competition", ["QueryCompetitionIndicator"]),
    "Y31": ("adares_y31_mask_shape", ["MaskShapeReadout"]),
    "Y36": ("adares_y36_local_adjacency", ["MaskLocalAdjacencyReadout"]),
    "Y41": ("adares_y41_boundary_probability", ["MaskBoundaryProbabilityReadout"]),
    "Y46": ("adares_y46_component_coherence", ["component_inputs", "AllComponentReadout"]),
}
namespaces, source_hashes = {}, {}
for key, (package, names) in specs.items():
    source = base / package / "dec.py"
    source_hashes[str(source.relative_to(Path(P["root"])))] = hashlib.sha256(source.read_bytes()).hexdigest()
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    namespace = dict(torch=torch, nn=nn, np=np, csr_matrix=csr_matrix,
                     connected_components=connected_components)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    namespaces[key] = namespace


def nearest(points):
    distance = torch.cdist(points, points)
    distance.fill_diagonal_(float("inf"))
    return csr_matrix((distance == distance.min(-1, keepdim=True).values).numpy())


def trace_resolvent(graph):
    graph = graph.maximum(graph.T).astype(np.float64)
    matrix = eye(graph.shape[0], format="csc") + diags(np.asarray(graph.sum(1)).ravel()) - graph
    inverse = splu(matrix.tocsc()).solve(np.eye(graph.shape[0]))
    dense = np.linalg.solve(matrix.toarray(), np.eye(graph.shape[0]))
    assert np.max(np.abs(inverse - dense)) <= 8 * np.finfo(float).eps * graph.shape[0]
    return float(np.trace(inverse) / graph.shape[0])


def envelope(a, b, nodes):
    return 8 * torch.finfo(torch.double).eps * nodes * max(
        1.0, float(a.abs().max()), float(b.abs().max()))


# Both integer-coordinate polyhedra are isotropic, connected and regular.
tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
truncated = np.array([2 * left + right for left in tetra for right in tetra
                      if not np.array_equal(left, right)])
cubocta = np.array([[0, a, b] for a in (-1, 1) for b in (-1, 1)]
                   + [[a, 0, b] for a in (-1, 1) for b in (-1, 1)]
                   + [[a, b, 0] for a in (-1, 1) for b in (-1, 1)])
positions = [torch.from_numpy(value.astype(np.float64)) for value in (truncated, cubocta)]
graphs = [nearest(value) for value in positions]
assert [graph.shape for graph in graphs] == [(12, 12), (12, 12)]
assert [connected_components(graph, directed=False)[0] for graph in graphs] == [1, 1]
assert [set(np.asarray(graph.sum(1)).ravel()) for graph in graphs] == [{3}, {4}]
physical_traces = [trace_resolvent(graph) for graph in graphs]
assert abs(physical_traces[0] - physical_traces[1]) > 8 * np.finfo(float).eps * 12

# A separate equal-degree witness rules out explaining every difference by degree.
cube = np.zeros((8, 8), bool)
mobius = np.zeros((8, 8), bool)
for i in range(8):
    for bit in (1, 2, 4):
        cube[i, i ^ bit] = True
    for j in ((i - 1) % 8, (i + 1) % 8, (i + 4) % 8):
        mobius[i, j] = True
assert np.array_equal(cube.sum(1), mobius.sum(1))
assert (cube.sum(1) == 3).all()
abstract_graphs = [csr_matrix(value) for value in (cube, mobius)]
assert all(connected_components(graph, directed=False)[0] == 1 for graph in abstract_graphs)
abstract_traces = [trace_resolvent(graph) for graph in abstract_graphs]
assert abs(abstract_traces[0] - abstract_traces[1]) > 8 * np.finfo(float).eps * 8
cases = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    query = torch.randn(1, 128, 256, dtype=torch.double)
    features = torch.randn(1, 1, 256, dtype=torch.double).expand(1, 12, 256).clone()
    masks = torch.full((1, 128, 12), 8.0, dtype=torch.double)
    padding = torch.zeros(1, 12, dtype=torch.bool)
    masses = torch.ones(1, 12, dtype=torch.double)
    w2 = namespaces["W2"]["QueryCompetitionIndicator"](256).double()
    shape = namespaces["Y31"]["MaskShapeReadout"](256).double()
    local = namespaces["Y36"]["MaskLocalAdjacencyReadout"](256).double()
    boundary = namespaces["Y41"]["MaskBoundaryProbabilityReadout"](256).double()
    components = namespaces["Y46"]["AllComponentReadout"](256).double()
    readings = dict(W2=[], Y31=[], Y36=[], Y41=[], Y46=[])
    component_rows = []
    with torch.no_grad():
        for points, graph in zip(positions, graphs):
            readings["W2"].append(w2(query, masks, padding, masses))
            readings["Y31"].append(shape(query, masks, padding, masses, points.unsqueeze(0)))
            readings["Y36"].append(local(query, masks, padding, masses, features, points.unsqueeze(0)))
            readings["Y41"].append(boundary(query, masks, padding, masses, features, points.unsqueeze(0)))
            inputs = namespaces["Y46"]["component_inputs"](
                graph, masks[0].numpy() > 0, masses[0].numpy(), features[0].numpy())
            component_rows.append(inputs)
            readings["Y46"].append(components(query[0], inputs))
    gaps = {}
    for key, (left, right) in readings.items():
        gap = float((left - right).abs().max())
        assert gap <= envelope(left, right, 12), (key, gap)
        gaps[key] = gap
    assert all(torch.equal(a, b) for a, b in zip(*component_rows))
    order = np.random.RandomState(1999 + seed).permutation(12)
    permuted_trace = [trace_resolvent(graph[order][:, order]) for graph in graphs]
    assert np.max(np.abs(np.array(permuted_trace) - physical_traces)) <= 8 * np.finfo(float).eps * 12
    assert [trace_resolvent(graph) for graph in graphs] == physical_traces
    with torch.no_grad():
        permuted_shape = shape(query, masks[:, :, order], padding[:, order], masses[:, order], positions[0][order].unsqueeze(0))
        permuted_local = local(query, masks[:, :, order], padding[:, order], masses[:, order], features[:, order], positions[0][order].unsqueeze(0))
        assert float((permuted_shape - readings["Y31"][0]).abs().max()) <= envelope(permuted_shape, readings["Y31"][0], 12)
        assert float((permuted_local - readings["Y36"][0]).abs().max()) <= envelope(permuted_local, readings["Y36"][0], 12)
    cases.append(dict(seed=seed, actual_source_parent_gaps=gaps,
                      Y46_component_rows_weights_groups_exact=True,
                      physical_resolvent_traces=physical_traces,
                      abstract_equal_degree_resolvent_traces=abstract_traces,
                      joint_permutation_error=float(np.max(np.abs(np.array(permuted_trace) - physical_traces))),
                      exact_repeat=True))
result = dict(
    complete=True, id="AdaRES-Y49", CPU_cases=32, source_hashes=source_hashes,
    physical_geometry="Integer truncated tetrahedron versus cuboctahedron,12 nodes, one connected component, isotropic normalized covariance, constant tokens and point masses.",
    physical_nearest_graph_degrees=[3, 4],
    physical_resolvent_traces=physical_traces,
    abstract_equal_degree_graphs="8-node cube versus Mobius ladder; both connected3regular, not claimed to be the inherited Euclidean graph.",
    abstract_equal_degree_resolvent_traces=abstract_traces,
    maximum_parent_gap=max(value for case in cases for value in case["actual_source_parent_gaps"].values()),
    maximum_permutation_error=max(case["joint_permutation_error"] for case in cases),
    actual_Y46_component_control_exact=True, cases=cases,
    scientific_status="constructed_actual_source_information_distinction_not_native_task_diagnosis",
    limitations=[
        "Physical polyhedra differ in degree; the separate abstract equal-degree witness isolates global spectral structure but is not a native geometry claim.",
        "Only explicit readouts under controlled constant inputs are compared; no universal backbone/attention blindness, efficacy or novelty conclusion.",
        "No new module, native TRAIN diagnosis/backward, optimizer, GPU/resource/independence/admission evidence.",
        "ClosedY8 acts on query-overlap scores; this probes point-graph topology without diffusing scores. ClosedY41/Y31 are controls, not reopened candidates.",
        "Exact unit resolvent fixed before tests; no radius/k/time/weight/threshold or validation search."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
