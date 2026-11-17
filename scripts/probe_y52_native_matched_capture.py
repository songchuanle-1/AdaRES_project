"""Capture real Y52 TRAIN matches only after the guarded runner grants a GPU."""
import ast
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time

assert P['physical_gpu'] in (0, 1), 'Exclusive authorized GPU runner required'
assert os.environ['CUDA_VISIBLE_DEVICES'] == str(P['physical_gpu'])
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert torch.cuda.device_count() == 1
torch.set_num_threads(2)
torch.cuda.set_device(0)
root = Path(P['root'])
for row in PAYLOAD['parent_files']:
    assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256']
train_source = (root / 'tools/train.py').read_bytes()
assert hashlib.sha256(train_source).hexdigest() == PAYLOAD['train_source_sha256']
loader = next(node for node in ast.parse(train_source).body
              if isinstance(node, ast.FunctionDef) and node.name == 'load_shape_compatible_pretrain')
exec(compile(ast.Module(body=[loader], type_ignores=[]), '<exact-native-pretrain-loader>', 'exec'), globals())
directory = Path(tempfile.mkdtemp(prefix='adares_y52_native_capture_' + P['token'] + '_'))
package = directory / 'adares_y52_point_target'
package.mkdir()
for name, row in PAYLOAD['files'].items():
    data = base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    destination = package / name if name.endswith('.py') else directory / name
    destination.write_bytes(data)
cfg = gorilla.Config.fromfile(str(directory / 'adares_y52_point_target_res.yaml'))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.model.task_type == 'res' and cfg.dataloader.train.batch_size == 2
assert cfg.model.sampling_module.num_proposal == 128
importlib.import_module('ipdn.model_paper5')
namespace = 'ipdn.model_paper5.adares_y52_point_target'
assert namespace not in sys.modules
spec = importlib.util.spec_from_file_location(namespace, str(package / '__init__.py'),
                                            submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[namespace] = module
spec.loader.exec_module(module)
Candidate = importlib.import_module(namespace + '.ipdn').IPDN
quality = importlib.import_module(namespace + '.loss')
gorilla.set_random_seed(1999)
model = Candidate(**cfg.model).cuda()
logger = get_root_logger()
load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)
model.train()
capture_path = Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == PAYLOAD['capture_sha256']
frozen = json.loads(capture_path.read_text())['rows']
assert len(frozen) == 8
dataset = build_dataset(cfg.data.train.copy(), logger)
rows = []
batch_rows = []
active = {}
original = model.criterion.loss_masks


def observe(pred_masks, pred_scores, targets, indices, num_boxes):
    # Observe the actual criterion inputs; never synthesize FPS or match indices.
    layer = active['layer']
    active['layer'] += 1
    for expression, (mask, target, match) in enumerate(zip(pred_masks, targets, indices)):
        query_ids, target_ids = match
        selected = mask[query_ids].detach()
        membership, point_mask = active['expanded'][expression]
        if query_ids.numel():
            native = quality.get_iou(selected, target['masks'][target_ids])
            corrected = quality.get_point_iou(selected, *target['point_counts'])
            expanded = quality.get_iou(selected[:, membership], point_mask.unsqueeze(0))
            error = float((corrected - expanded).abs().max().item())
            assert torch.equal(corrected, expanded), 'Native matched point-IoU reference disagreement'
            support = int(((native > .5) != (corrected > .5)).sum().item())
            difference = float((corrected - native).abs().max().item())
        else:
            error, support, difference = None, 0, None
        rows.append(dict(batch=active['batch'], expression=expression,
            ann_id=active['ann_ids'][expression], layer='final' if layer == 0 else 'aux_' + str(layer - 1),
            actual_query_indices=query_ids.detach().cpu().tolist(),
            actual_target_indices=target_ids.detach().cpu().tolist(),
            queries=int(mask.shape[0]), matched_queries=int(query_ids.numel()),
            point_vs_superpoint_max_abs=difference, existing_half_filter_disagreements=support,
            expanded_point_reference_max_abs_error=error))
    return original(pred_masks, pred_scores, targets, indices, num_boxes)


model.criterion.loss_masks = observe
try:
    torch.cuda.reset_peak_memory_stats()
    for pair in range(0, 8, 2):
        items = []
        for scene in frozen[pair:pair + 2]:
            gorilla.set_random_seed(1999)
            item = dataset[scene['dataset_index']]
            single = dataset.collate_fn([copy.deepcopy(item)])
            assert single['scan_ids'][0] == scene['scan_id']
            assert [str(value) for value in single['ann_ids']] == [value['ann_id'] for value in scene['expressions']]
            assert single['object_idss'] == [value['object_ids'] for value in scene['expressions']]
            items.append(item)
        batch = dataset.collate_fn(copy.deepcopy(items))
        expanded = []
        expression = 0
        for scene_index, count in enumerate(batch['scenes_len']):
            start, end = [int(v) for v in batch['batch_offsets'][scene_index:scene_index + 2]]
            membership = batch['superpoints']
            membership = membership[(membership >= start) & (membership < end)].long() - start
            for _ in range(int(count)):
                target = batch['gt_pmasks'][expression].reshape(-1)
                assert target.numel() == membership.numel()
                expanded.append((membership.cuda(), target.cuda()))
                expression += 1
        active.update(batch=pair // 2, layer=0, expanded=expanded,
                      ann_ids=[str(value) for value in batch['ann_ids']])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            loss, log_vars = model(batch, mode='loss')
        torch.cuda.synchronize()
        assert torch.isfinite(loss).all(), 'Nonfinite native TRAIN loss'
        batch_rows.append(dict(batch=pair // 2, expressions=expression,
            criterion_layer_calls=active['layer'], native_train_loss=float(loss.item()),
            instrumented_elapsed_seconds=time.perf_counter() - started))
        del batch, expanded, loss, log_vars
        active.clear()
    assert sum(row['expressions'] for row in batch_rows) == 32
    assert all(row['criterion_layer_calls'] > 1 for row in batch_rows)
    result = dict(complete=True, id='AdaRES-Y52', frozen_TRAIN_scenes=8, expressions=32,
        actual_sparse_TRAIN_forward=True, actual_match_indices_observed=True,
        native_final_and_auxiliary_capture=True, fixed_seed=1999, trained_weights=False,
        native_backward_executed=False, optimizer_steps=0, restore_executed=False,
        model_or_checkpoint_weights_saved=False, admission=False,
        matched_queries_across_layers=sum(row['matched_queries'] for row in rows),
        half_filter_disagreements_across_layers=sum(row['existing_half_filter_disagreements'] for row in rows),
        diagnostic_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        diagnostic_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        batch_rows=batch_rows, rows=rows,
        limitation='Initial released-SPFormer TRAIN forward only. Layer counts are correlated, not independent samples. No trained efficacy, native backward/resource-restore gate PASS, inference latency, Pareto result or Git authorization. Preserve zero witnesses as inconclusive, not a fabricated PASS.')
    Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
finally:
    model.criterion.loss_masks = original
