"""Unit-test the frozen capture observer, never the sparse model or matcher."""
import ast
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import torch

assert P['physical_gpu'] is None and torch.cuda.device_count() == 0
torch.set_num_threads(2)


def decode(name):
    row = PAYLOAD[name]
    value = base64.b64decode(row['base64'])
    assert hashlib.sha256(value).hexdigest() == row['sha256']
    return value


quality_tree = ast.parse(decode('loss'))
functions = [node for node in quality_tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('get_iou', 'get_point_iou')]
assert len(functions) == 2
quality_namespace = dict(torch=torch, Union=Union)
exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen-quality-helpers>', 'exec'), quality_namespace)
observer_tree = ast.parse(decode('observer'))
observer = next(node for node in observer_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'observe')
active, rows, delegated = {}, [], []
sentinel = object()


def original(*args):
    delegated.append(args)
    return sentinel


namespace = dict(torch=torch, active=active, rows=rows, original=original,
                 quality=SimpleNamespace(**{name: quality_namespace[name] for name in ('get_iou', 'get_point_iou')}))
exec(compile(ast.Module(body=[observer], type_ignores=[]), '<exact-frozen-observer>', 'exec'), namespace)
observe = namespace['observe']
masks = [torch.tensor([[-1., -1., -1.], [1., 1., -1.], [1., -1., 1.], [-1., 1., 1.]], requires_grad=True),
         torch.tensor([[1., -1., 1., -1.], [-1., -1., -1., -1.], [-2., 2., -2., 2.]], requires_grad=True)]
scores = [torch.zeros(4, 1, requires_grad=True), torch.zeros(3, 1, requires_grad=True)]
targets = [dict(masks=torch.tensor([[1., 1., 0.]]), point_counts=(torch.tensor([1., 3., 2.]), torch.tensor([1., 1., 0.]))),
           dict(masks=torch.tensor([[0., 1., 0., 1.]]), point_counts=(torch.ones(4), torch.tensor([0., 1., 0., 1.])))]
indices = [(torch.tensor([3, 1]), torch.tensor([0, 0])), (torch.tensor([2]), torch.tensor([0]))]
expanded = [(torch.tensor([0, 1, 1, 1, 2, 2]), torch.tensor([1., 1., 0., 0., 0., 0.])),
            (torch.arange(4), torch.tensor([0., 1., 0., 1.]))]
num_boxes = torch.tensor([3.])


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)


before = [(value, value.detach().clone()) for value in tensors((masks, scores, targets, indices, expanded, num_boxes))]
active.update(batch=0, layer=0, expanded=expanded, ann_ids=['fixture-A', 'fixture-B'])
arguments = (masks, scores, targets, indices, num_boxes)
for _ in range(2):
    assert observe(*arguments) is sentinel, 'Observer replaced the original return value'
assert len(rows) == 4 and len(delegated) == 2
assert [row['actual_query_indices'] for row in rows] == [[3, 1], [2], [3, 1], [2]]
assert [row['actual_target_indices'] for row in rows] == [[0, 0], [0], [0, 0], [0]]
assert [row['layer'] for row in rows] == ['final', 'final', 'aux_0', 'aux_0']
assert [row['existing_half_filter_disagreements'] for row in rows] == [1, 0, 1, 0]
for call in delegated:
    assert all(actual is expected for actual, expected in zip(call, arguments)), 'Observer replaced criterion inputs'
assert all(torch.equal(value.detach(), saved) for value, saved in before), 'Observer mutated an input'
assert all(value.grad is None for value, _ in before), 'Observer unexpectedly ran backward'
active.update(batch=1, layer=0, expanded=expanded[:1], ann_ids=['empty-fixture'])
empty = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
assert observe(masks[:1], scores[:1], targets[:1], empty, torch.tensor([0.])) is sentinel
assert rows[-1]['matched_queries'] == 0 and rows[-1]['point_vs_superpoint_max_abs'] is None
assert rows[-1]['expanded_point_reference_max_abs_error'] is None
active.update(batch=2, layer=0, expanded=[(expanded[0][0], torch.zeros_like(expanded[0][1]))], ann_ids=['bad-reference'])
row_count, call_count = len(rows), len(delegated)
try:
    observe(masks[:1], scores[:1], targets[:1], indices[:1], num_boxes)
except AssertionError as error:
    assert str(error) == 'Native matched point-IoU reference disagreement'
else:
    raise AssertionError('Mismatched expanded reference was silently accepted')
assert len(rows) == row_count and len(delegated) == call_count
result = dict(complete=True, id='AdaRES-Y52', no_GPU=True,
    test_scope='synthetic unit controls for exact frozen observer function only',
    actual_frozen_observer_function_executed_on_CPU=True,
    matched_index_order_preserved=True, original_input_identity_preserved=True,
    original_return_identity_preserved=True, inputs_unchanged=True,
    final_auxiliary_labels_correct=True, empty_matches_recorded_without_fake_zeros=True,
    inconsistent_expanded_reference_rejected=True, synthetic_rows_recorded=len(rows),
    native_model_constructed=False, native_sparse_forward_executed=False,
    actual_TRAIN_match_frequency_measured=False, native_backward_executed=False,
    resources_or_restore_passed=False, scientific_gate_passed=False,
    limitation='Synthetic observer instrumentation test, not a real Criterion empty-case test, TRAIN/FPS/matcher witness, model gradient test or trained result. GPU native gates and resource lease remain pending.', rows=rows)
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
