"""Actual native matching/loss CPU controls and frozen exact-mask incidence."""

import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib
import numpy as np
import torch
from torch import nn
import yaml

assert P["physical_gpu"] is None
torch.set_num_threads(4)
loss_path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/loss.py")
loss_tree = ast.parse(loss_path.read_text())
original = next(node for node in loss_tree.body if isinstance(node, ast.ClassDef) and node.name == "Criterion")
methods = ("__init__", "_get_src_permutation_idx", "loss_indi", "match_sampled_ins")
selected = ast.ClassDef(name="ActualCriterion", bases=[ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())],
                        keywords=[], body=[node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods],
                        decorator_list=[])
module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
namespace = dict(torch=torch, nn=nn, np=np)
exec(compile(module, str(loss_path), "exec"), namespace)
config_path = Path("configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml")
cfg = yaml.safe_load(config_path.read_text())
criterion = namespace["ActualCriterion"](**cfg["model"]["criterion"])
assert criterion.eos_coef == .1
indicator_path = Path("ipdn/model_paper5/adares_w2_pointmass_competition/dec.py")
indicator = next(node for node in ast.parse(indicator_path.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == "QueryCompetitionIndicator")
exec(compile(ast.Module(body=[indicator], type_ignores=[]), str(indicator_path), "exec"), namespace)
native_negative_weight = float(torch.tensor(criterion.eos_coef, dtype=torch.get_default_dtype()))
cases = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    count = (1, 2, 4, 8)[seed % 4]
    labels = torch.zeros(128, dtype=torch.long)
    labels[:count] = 7
    positions = torch.stack((torch.arange(128), torch.zeros(128), torch.zeros(128)), -1).double()
    objects = torch.zeros(1, 3, dtype=torch.double)
    top, in_object, indices, counts = criterion.match_sampled_ins(labels, [7], positions, objects)
    assert torch.equal(indices[0].sort().values, torch.arange(count))
    assert int(counts[0]) == count
    # All masks and query states are explicitly indistinguishable in this fixture.
    model = namespace["QueryCompetitionIndicator"](256).double()
    query = torch.randn(1, 1, 256, dtype=torch.double).expand(1, 128, 256)
    masks = torch.tensor([8.] * 4 + [-8.] * 4, dtype=torch.double).reshape(1, 1, 8).expand(1, 128, 8)
    with torch.no_grad():
        scores = model(query, masks, torch.zeros(1, 8, dtype=torch.bool), torch.ones(1, 8, dtype=torch.double))
    output_gap = float((scores - scores[:, :1]).abs().max())
    output_bound = 8 * torch.finfo(torch.double).eps * 128 * max(1., float(scores.abs().max()))
    assert output_gap <= output_bound
    logits = torch.zeros(1, 128, 2, dtype=torch.float32, requires_grad=True)
    num_boxes = torch.tensor(float(count), dtype=torch.float32)
    loss = criterion.loss_indi(logits, [], [indices], num_boxes)
    gradient = torch.autograd.grad(loss, logits)[0]
    targets = torch.zeros(1, 128, dtype=torch.long)
    targets[:, indices[0]] = 1
    weights = torch.full((1, 128), native_negative_weight, dtype=torch.float32)
    weights[:, indices[0]] = 1
    literal = (logits.softmax(-1) - nn.functional.one_hot(targets, 2)) * weights.unsqueeze(-1) / num_boxes
    gradient_error = float((gradient - literal).abs().max())
    gradient_bound = 8 * torch.finfo(logits.dtype).eps * 128 * max(1., float(gradient.abs().max()))
    assert gradient_error <= gradient_bound
    assert (gradient[0, indices[0], 1] < 0).all()
    assert (gradient[0, count:, 1] > 0).all()
    repeated = criterion.loss_indi(logits, [], [indices], num_boxes)
    assert torch.equal(loss, repeated)
    optimum = count / (count + native_negative_weight * (128 - count))
    shared = torch.tensor(np.log(optimum / (1 - optimum)), dtype=torch.float32, requires_grad=True)
    tied = torch.stack((shared * 0, shared)).reshape(1, 1, 2).expand(1, 128, 2)
    optimum_loss = criterion.loss_indi(tied, [], [indices], num_boxes)
    derivative = float(torch.autograd.grad(optimum_loss, shared)[0])
    assert abs(derivative) <= gradient_bound
    reverse = torch.arange(127, -1, -1)
    mapped = (127 - indices[0], indices[1])
    permuted = criterion.loss_indi(logits[:, reverse], [], [mapped], num_boxes)
    permutation_error = float((loss - permuted).abs())
    permutation_bound = 8 * torch.finfo(logits.dtype).eps * 128 * max(1., float(loss.abs()))
    assert permutation_error <= permutation_bound
    cases.append(dict(seed=seed, native_positive_queries=count,
                      native_matching_already_one_to_many=count > 1,
                      equal_actual_indicator_output_gap=output_gap,
                      prospective_output_bound=output_bound,
                      actual_gradient_error=gradient_error,
                      prospective_gradient_bound=gradient_bound,
                      exact_repeat=True, joint_query_permutation_error=permutation_error,
                      prospective_permutation_bound=permutation_bound,
                      analytical_tied_foreground_optimum=optimum,
                      observed_optimum_derivative=derivative,
                      fixture_has_opposing_positive_negative_gradients=True))

OLD_TOKEN = "d179d824a64943e0b06bfa90de0caf52"
capture_path = Path("/tmp/adares_y46_native_capture_d179_" + OLD_TOKEN + ".json")
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == "5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89"
capture = json.loads(capture_path.read_text())


def unpack(item):
    raw = zlib.decompress(base64.b64decode(item["zlib_base64"]))
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    return np.frombuffer(raw, dtype=np.dtype(item["dtype"])).reshape(item["shape"]).copy()


rows = []
for scene in capture["rows"]:
    for expression in scene["expressions"]:
        arrays = expression["packed_native_inputs"]
        binary = unpack(arrays["masks"]) > 0
        masses = unpack(arrays["masses"]).astype(np.float64)
        assert binary.shape == (128, len(masses))
        packed = np.packbits(binary, axis=1)
        keys, inverse, frequencies = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
        groups = []
        for group in np.flatnonzero(frequencies > 1):
            indices = np.flatnonzero(inverse == group)
            assert (binary[indices] == binary[indices[0]]).all()
            selected_mass = float(masses[binary[indices[0]]].sum())
            assert all(float(masses[binary[i]].sum()) == selected_mass for i in indices)
            groups.append(dict(query_indices=indices.tolist(), size=len(indices),
                               selected_point_mass=selected_mass, nonempty=selected_mass > 0))
        rows.append(dict(dataset_index=scene["dataset_index"], scan_id=scene["scan_id"],
                         ann_id=expression["ann_id"], duplicate_groups=groups,
                         nonempty_duplicate_groups=sum(g["nonempty"] for g in groups),
                         nonempty_duplicate_queries=sum(g["size"] for g in groups if g["nonempty"]),
                         empty_duplicate_queries=sum(g["size"] for g in groups if not g["nonempty"]),
                         native_matching_indices_available=False, mixed_positive_label_groups=None))
result = dict(
    complete=True, id="AdaRES-Y48", cases=cases, rows=rows, CPU_cases=32,
    native_scenes=8, native_expressions=len(rows), queries=128,
    initial_single_winner_premise_disconfirmed=True,
    actual_native_loss_source_sha256=hashlib.sha256(loss_path.read_bytes()).hexdigest(),
    actual_native_indicator_source_sha256=hashlib.sha256(indicator_path.read_bytes()).hexdigest(),
    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    inherited_eos_coef=criterion.eos_coef, native_weight_tensor_value=native_negative_weight,
    native_loss_dtype="float32", native_loss_source_functions_unchanged=True,
    maximum_actual_gradient_error=max(c["actual_gradient_error"] for c in cases),
    nonempty_duplicate_groups=sum(r["nonempty_duplicate_groups"] for r in rows),
    nonempty_duplicate_queries=sum(r["nonempty_duplicate_queries"] for r in rows),
    expressions_with_nonempty_duplicates=sum(r["nonempty_duplicate_groups"] > 0 for r in rows),
    empty_duplicate_queries=sum(r["empty_duplicate_queries"] for r in rows),
    actual_native_mixed_label_groups=None,
    input_capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
    scientific_status="native_source_CPU_controls_and_exact_mask_incidence_not_native_label_conflict",
    limitations=["Actual native matching already one-to-many; original single-winner interpretation disconfirmed.",
                 "CPU identical-state fixtures exhibit ownership-gradient opposition, not observed native mixed-label incidence.",
                 "Frozen actual TRAIN masks contain no corresponding matching indices; no cross-capture query-label inference.",
                 "Empty-mask duplicates alone do not justify a useful confidence intervention.",
                 "No production target/weight/model/selector/checkpoint/queue/admission or GPU change."])
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
