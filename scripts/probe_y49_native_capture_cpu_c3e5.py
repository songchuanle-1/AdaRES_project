"""Prediction-blind native-input global-resolvent CPU prototype and controls."""

import base64
import hashlib
import json
from pathlib import Path
import time
import zlib

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
import torch
from torch import nn

assert P["physical_gpu"] is None
torch.set_num_threads(4)
capture_path = Path("/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json")
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == "5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89"
capture = json.loads(capture_path.read_text())


def unpack(item):
    raw = zlib.decompress(base64.b64decode(item["zlib_base64"]))
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    return np.frombuffer(raw, dtype=np.dtype(item["dtype"])).reshape(item["shape"]).copy()


def inverse_diagonal(graph, block_size=64):
    matrix = (eye(graph.shape[0], format="csc")
              + diags(np.asarray(graph.sum(1)).ravel()) - graph).tocsc()
    solver = splu(matrix)
    diagonal = np.empty(graph.shape[0], dtype=np.float64)
    for start in range(0, graph.shape[0], block_size):
        indices = np.arange(start, min(start + block_size, graph.shape[0]))
        rhs = np.zeros((graph.shape[0], len(indices)), dtype=np.float64)
        rhs[indices, np.arange(len(indices))] = 1
        solved = solver.solve(rhs)
        diagonal[indices] = solved[indices, np.arange(len(indices))]
    assert np.isfinite(diagonal).all() and (diagonal > 0).all()
    return diagonal


def descriptors(graph, masks, masses, block_size=64):
    graph = graph.maximum(graph.T).astype(np.float64)
    cache, rows, partitions = {}, [], []
    for mask in masks:
        row, partition = [], []
        for selected in (mask, ~mask):
            key = np.packbits(selected).tobytes()
            if key not in cache:
                indices = np.flatnonzero(selected)
                if len(indices):
                    subgraph = graph[indices][:, indices]
                    diagonal = inverse_diagonal(subgraph, block_size)
                    score = float(diagonal @ masses[indices] / masses[indices].sum())
                    count, labels = connected_components(subgraph, directed=False)
                    component_masses = np.bincount(labels, weights=masses[indices], minlength=count)
                    cache[key] = (score, tuple(sorted(component_masses.tolist())))
                else:
                    cache[key] = (0.0, ())
            value, component_partition = cache[key]
            row.append(value)
            partition.append(component_partition)
        rows.append(row)
        partitions.append(tuple(partition))
    return torch.tensor(rows, dtype=torch.double), partitions, len(cache)


class GlobalResolventReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, query, descriptor):
        normalized = nn.functional.layer_norm(query, (query.shape[-1],))
        return self.head(torch.cat((normalized, descriptor), -1)).flatten()


def bound(left, right, nodes):
    return 8 * torch.finfo(torch.double).eps * max(1, nodes) * max(
        1.0, float(left.abs().max()), float(right.abs().max()))


torch.manual_seed(1999)
head = GlobalResolventReadout(256).double()
rows = []
overall_start = time.monotonic()
for scene in capture["rows"]:
    for expression in scene["expressions"]:
        arrays = {key: unpack(value) for key, value in expression["packed_native_inputs"].items()}
        masses, masks = arrays["masses"].astype(np.float64), arrays["masks"] > 0
        query = torch.from_numpy(arrays["query"].astype(np.float64))
        edges = arrays["graph_edges"]
        graph = csr_matrix((np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])),
                           shape=(len(masses), len(masses)))
        start = time.monotonic()
        values, partitions, unique_sets = descriptors(graph, masks, masses)
        descriptor_seconds = time.monotonic() - start
        assert values.shape == (128, 2)
        assert (values >= 0).all() and (values <= 1 + 8 * torch.finfo(torch.double).eps * len(masses)).all()
        head.zero_grad(set_to_none=True)
        scores = head(query, values)
        targets = torch.arange(128).remainder(2)
        logits = torch.stack((torch.zeros_like(scores), scores), -1)
        nn.functional.cross_entropy(logits, targets).backward()
        gradients = {name: float(value.grad.norm()) for name, value in head.named_parameters()}
        assert len(gradients) == 4 and all(np.isfinite(value) and value > 0 for value in gradients.values())
        with torch.no_grad():
            repeated = head(query, values)
            wrong = head(query, values.roll(1, dims=0))
            query_only = head(query, torch.zeros_like(values))
        assert torch.equal(scores, repeated)
        wrong_gap = float((scores - wrong).abs().max())
        reachable_gap = float((scores - query_only).abs().max())
        partition_groups = {}
        for index, key in enumerate(partitions):
            partition_groups.setdefault(key, []).append(index)
        component_only_collisions = []
        for indices in partition_groups.values():
            if len(indices) < 2:
                continue
            differences = float((values[indices] - values[indices[0]]).abs().max())
            if differences > bound(values[indices], values[indices[0]], len(masses)):
                component_only_collisions.append(dict(query_indices=indices,
                                                       descriptor_difference=differences))
        numerical = None
        # Fixed first four expressions, chosen before reading any result.
        if len(rows) < 4:
            order = np.arange(len(masses) - 1, -1, -1)
            permuted, _, _ = descriptors(graph[order][:, order], masks[:, order], masses[order])
            error = float((values - permuted).abs().max())
            assert error <= bound(values, permuted, len(masses))
            first_indices = np.flatnonzero(masks[0])
            if len(first_indices):
                undirected = graph.maximum(graph.T).astype(np.float64)
                subgraph = undirected[first_indices][:, first_indices]
                one = inverse_diagonal(subgraph, block_size=1)
                blocked = inverse_diagonal(subgraph, block_size=64)
                solve_error = float(np.max(np.abs(one - blocked)))
                assert solve_error <= 8 * np.finfo(float).eps * len(first_indices)
            else:
                solve_error = 0.0
            numerical = dict(joint_permutation_error=error, block1_vs64_diagonal_error=solve_error,
                             prospective8eps_nodes_bound=bound(values, permuted, len(masses)))
        rows.append(dict(
            dataset_index=scene["dataset_index"], scan_id=scene["scan_id"], ann_id=expression["ann_id"],
            nodes=len(masses), unique_selected_sets=unique_sets, descriptor_seconds=descriptor_seconds,
            descriptor_query_ranges=(values.max(0).values - values.min(0).values).tolist(),
            wrong_correspondence_gap=wrong_gap, graph_path_vs_query_only_gap=reachable_gap,
            new_CPU_CE_gradients=gradients, exact_cached_repeat=True,
            component_mass_groups_with_distinct_global_values=component_only_collisions,
            numerical_controls=numerical))
        print("Y49_CPU_EXPRESSION", len(rows), "seconds", round(descriptor_seconds, 3), flush=True)
assert len(rows) == 32
assert any(row["wrong_correspondence_gap"] > 0 for row in rows)
result = dict(
    complete=True, id="AdaRES-Y49", native_scenes=8, native_captured_expressions=32, queries=128,
    graph="Frozen inherited nearest-tie edges, weak undirected support; hard masks>0; every FG/BG vertex retained.",
    descriptor="For each FG/BG induced graph, point-mass-weighted diagonal mean of (I+D-A)^-1; empty side0. Unit resolvent fixed, no time/weight sweep.",
    input_capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
    new_parameter_tensors=4, prototype_parameters=sum(value.numel() for value in head.parameters()),
    minimum_CPU_CE_gradient=min(value for row in rows for value in row["new_CPU_CE_gradients"].values()),
    wrong_correspondence_separated_expressions=sum(row["wrong_correspondence_gap"] > 0 for row in rows),
    expressions_with_distinct_global_values_at_equal_component_masses=sum(
        bool(row["component_mass_groups_with_distinct_global_values"]) for row in rows),
    maximum_descriptor_seconds=max(row["descriptor_seconds"] for row in rows),
    total_CPU_seconds=time.monotonic() - overall_start, rows=rows,
    scientific_status="native_input_CPU_prototype_not_native_backward_or_utility",
    limitations=[
        "Native tensors come from the existing prediction-blind TRAIN8 manifest, not new GPU capture or trained candidate.",
        "Synthetic alternating CE targets test four prototype tensors only, not native loss reachability or task efficacy.",
        "No universal parent information-blindness, novelty, superiority or necessity inference.",
        "Fixed four-expression permutation/solver controls, not all32; block size is an exact-solver execution parameter, not a model search.",
        "CPU descriptor times are measured, not paired native latency or GPU resource admission.",
        "No independent production package, optimizer/checkpoint, admission or training launch."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
