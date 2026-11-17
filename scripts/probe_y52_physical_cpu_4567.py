"""Test exact isolated Y52 bytes, count plumbing and complete loss component."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import copy
import torch
from torch import nn

assert P['physical_gpu'] is None
torch.set_num_threads(2)
directory = Path(tempfile.mkdtemp(prefix='adares_y52_' + P['token'] + '_'))
for name, row in PAYLOAD['files'].items():
    data = base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    (directory / name).write_bytes(data)
spec = importlib.util.spec_from_file_location('y52_physical_loss', str(directory / 'loss.py'))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
parent_path = Path(P['root']) / 'ipdn/model_paper5/adares_w2_pointmass_competition/loss.py'
parent_spec = importlib.util.spec_from_file_location('y52_parent_loss', str(parent_path))
parent = importlib.util.module_from_spec(parent_spec)
parent_spec.loader.exec_module(parent)
tree = ast.parse((directory / 'ipdn.py').read_text())
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'expression_point_targets')
exec(compile(ast.Module(body=[helper], type_ignores=[]), '<exact-candidate-point-plumbing>', 'exec'), globals())


def expanded_reference(inputs, total, positive):
    mapping = torch.repeat_interleave(torch.arange(len(total)), total.long())
    target = torch.cat([torch.cat((torch.ones(int(y)), torch.zeros(int(n-y)))) for n, y in zip(total, positive)])
    return parent.get_iou(inputs[:, mapping], target.unsqueeze(0))


rows = []
for seed in range(1999, 2031):
    g = torch.Generator().manual_seed(seed)
    total = torch.randint(1, 30, (19,), generator=g).float()
    positive = torch.floor(torch.rand(19, generator=g) * (total + 1))
    masks = torch.randn((128, 19), generator=g)
    expected = expanded_reference(masks, total, positive)
    observed = candidate.get_point_iou(masks, total, positive)
    assert torch.equal(expected, observed)
    rows.append(dict(seed=seed, point_quality_error=0.))

membership = torch.tensor([0, 0, 1, 2, 2, 3, 4, 4, 4, 5])
offsets = torch.tensor([0, 3, 6])
scenes_len = [2, 3]
point_targets = [torch.tensor([1, 0, 1, 0, 1]), torch.ones(5),
                 torch.tensor([0, 1, 0, 1, 0]), torch.zeros(5), torch.ones(5)]
plumbed = expression_point_targets(membership, offsets, scenes_len, point_targets)
assert len(plumbed) == 5
for i, (total, positive) in enumerate(plumbed):
    local = membership[:5] if i < 2 else membership[5:] - 3
    assert torch.equal(total, torch.bincount(local, minlength=3).float())
    assert torch.equal(positive, torch.bincount(local, weights=point_targets[i].float(), minlength=3))

weights = [1., 1., .5, 5., .1, .1, .1, 1.]
parent_criterion = parent.Criterion(loss_weight=weights)
candidate_criterion = candidate.Criterion(loss_weight=weights)
hard = torch.tensor([1., 1., 0., 0., 1., 0.])


def pred(seed):
    torch.manual_seed(seed)
    masks = (torch.randn((1, 128, 6)) * .01 + (hard * 2 - 1) * 4).requires_grad_(True)
    scores = torch.randn((1, 128, 1), requires_grad=True)
    indis = torch.randn((1, 128, 2), requires_grad=True)
    ref = torch.randn((1, 128), requires_grad=True)
    auxiliary_scores = torch.randn((1, 128, 1), requires_grad=True)
    return dict(masks=masks, scores=scores, indis=indis, ref_scores=ref,
                batch_mask=torch.zeros((1, 6), dtype=torch.bool), proj_tokens=None, proj_queries=None,
                sample_inds=torch.arange(128).reshape(1, -1),
                aux_outputs=[dict(masks=masks, scores=auxiliary_scores, indis=indis, proj_tokens=None, proj_queries=None)]), [masks, scores, indis, ref, auxiliary_scores]


mapping = dict(positive_map=torch.ones(1, 2), modify_positive_map=torch.ones(1, 2),
               pron_positive_map=torch.ones(1, 2), other_entity_map=torch.ones(1, 2), rel_positive_map=torch.ones(1, 2))
arguments = dict(gt_spmasks=[hard], sp_ref_masks=[hard], object_idss=[[1]],
                 sp_ins_labels=[hard.long()], dense_maps=[mapping], lang_masks=torch.ones((1, 2), dtype=torch.bool),
                 fps_seed_sp=torch.arange(128).remainder(6).reshape(1, -1),
                 sp_coords_float=torch.arange(18).float().reshape(6, 3) / 10, batch_offsets=torch.tensor([0, 6]))
quality_function = candidate.get_point_iou
component_results = []
for label, total, positive in (
    ('uniform_pure', torch.ones(6), hard),
    ('nonuniform_mixed', torch.tensor([20., 2., 1., 3., 8., 2.]), torch.tensor([18., 1., 0., 1., 6., 0.]))):
    actual_pred, actual_leaves = pred(1999)
    ref_pred, ref_leaves = pred(1999)
    parent_pred, parent_leaves = pred(1999)
    candidate.get_point_iou = quality_function
    actual, actual_parts = candidate_criterion(actual_pred, point_targets=[(total, positive)], **arguments)
    candidate.get_point_iou = expanded_reference
    expected, expected_parts = candidate_criterion(ref_pred, point_targets=[(total, positive)], **arguments)
    candidate.get_point_iou = quality_function
    old, old_parts = parent_criterion(parent_pred, **arguments)
    assert torch.equal(actual, expected)
    for name in actual_parts:
        assert torch.equal(actual_parts[name], expected_parts[name])
        if 'score_loss' not in name and name != 'loss':
            assert torch.equal(actual_parts[name], old_parts[name]), 'Unrelated loss changed: ' + name
    actual.sum().backward()
    expected.sum().backward()
    repeat_pred, repeat_leaves = pred(1999)
    repeat_loss, _ = candidate_criterion(repeat_pred, point_targets=[(total, positive)], **arguments)
    assert torch.equal(actual, repeat_loss)
    repeat_loss.sum().backward()
    max_error = 0.
    gradient_controls = []
    for index, (left, right, repeat) in enumerate(zip(actual_leaves, ref_leaves, repeat_leaves)):
        assert left.grad is not None and right.grad is not None and repeat.grad is not None
        assert torch.isfinite(left.grad).all() and torch.isfinite(right.grad).all()
        error = float((left.grad - right.grad).abs().max())
        repeat_error = float((left.grad - repeat.grad).abs().max())
        bound = 8 * torch.finfo(left.dtype).eps * max(1., float(left.grad.abs().max()), float(right.grad.abs().max()))
        if index in (1, 4):
            assert error == 0., 'Auxiliary score target gradient must remain exact'
        assert error <= max(repeat_error, bound), 'Unexplained gradient discrepancy'
        gradient_controls.append(dict(tensor=index, expanded_error=error, identical_repeat_error=repeat_error,
                                      preregistered_v2_8eps_roundoff_bound=bound))
        max_error = max(max_error, error)
    if label == 'uniform_pure':
        assert torch.equal(actual, old)
    component_results.append(dict(label=label, full_Criterion_loss=float(actual), parent_loss=float(old),
        expanded_reference_loss_error=0., gradient_error=max_error, gradient_controls=gradient_controls,
        unchanged_non_score_losses=True, original_strict_gradient_assertion_failure_preserved=True,
        final_score_loss=float(actual_parts['score_loss']), auxiliary_score_loss=float(actual_parts['layer_0_score_loss']),
        final_score_gradient_norm=float(actual_leaves[1].grad.norm()), auxiliary_score_gradient_norm=float(actual_leaves[-1].grad.norm())))

log = Path(P['root']) / 'logs/train_adares_y50_joint_incidence_res_20261016_145628.log'
with log.open('rb') as stream:
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(max(0, size - 8 * 1024 * 1024))
    text = stream.read(8 * 1024 * 1024).decode('utf-8', errors='replace')
progress = re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[', text[-16384:])
metrics = [line[:600] for line in re.split(r'[\r\n]', text) if any(marker in line for marker in ('mIoU :', 'mIOU:', 'Evaluate referring segmentation:', 'save best pth', 'save last pth'))][-12:]
result = dict(complete=True, id='AdaRES-Y52', no_GPU=True, source_files={k: v['sha256'] for k, v in PAYLOAD['files'].items()},
    physical_module_import=True, exact_count_fixtures=len(rows), ragged_scene_expression_plumbing=5,
    complete_Criterion_component_tests=component_results, new_trainable_parameters=0,
    native_full_model_backward='not executed: actual sparse-backbone/FPS TRAIN matching requires exclusive GPU',
    scientific_status='physical_CPU_component_pass_not_native_gate_PASS',
    running_Y50=dict(pid=31539, native_progress=progress[-2:], bounded_metric_lines=metrics, max_log_read_bytes=8*1024*1024),
    resources=dict(gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode().splitlines(),
        processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits']).decode().splitlines()),
    rows=rows)
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
