"""Exercise frozen Y52 point-target plumbing on real two-scene TRAIN batches."""
import ast
import base64
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert P['physical_gpu'] is None and torch.cuda.device_count() == 0
torch.set_num_threads(2)
directory = Path(tempfile.mkdtemp(prefix='adares_y52_collate_' + P['token'] + '_'))
for name, row in PAYLOAD['files'].items():
    data = base64.b64decode(row['base64'])
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    (directory / name).write_bytes(data)
tree = ast.parse((directory / 'ipdn.py').read_text())
helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'expression_point_targets')
exec(compile(ast.Module(body=[helper], type_ignores=[]), '<frozen-Y52-point-plumbing>', 'exec'), globals())
spec = importlib.util.spec_from_file_location('y52_collate_exact_loss', str(directory / 'loss.py'))
loss_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loss_module)
cfg = gorilla.Config.fromfile(str(directory / 'adares_y52_point_target_res.yaml'))
parent_cfg = gorilla.Config.fromfile(str(Path(P['root']) / 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'))
assert cfg.data.train == parent_cfg.data.train and cfg.dataloader.train.batch_size == 2
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
capture_path = Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
capture = json.loads(capture_path.read_text())
frozen = capture['rows']
assert len(frozen) == 8
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
rows = []
expression_count = 0
mixed_count = 0
for pair in range(0, 8, 2):
    items, singles = [], []
    for scene in frozen[pair:pair + 2]:
        gorilla.set_random_seed(1999)
        item = dataset[scene['dataset_index']]
        single = dataset.collate_fn([copy.deepcopy(item)])
        assert single['scan_ids'][0] == scene['scan_id']
        assert len(single['object_idss']) == len(scene['expressions'])
        for i, expected in enumerate(scene['expressions']):
            assert str(single['ann_ids'][i]) == expected['ann_id']
            assert single['object_idss'][i] == expected['object_ids']
        items.append(item)
        singles.append(single)
    batch = dataset.collate_fn(copy.deepcopy(items))
    repeated = dataset.collate_fn(copy.deepcopy(items))
    assert len(batch['scenes_len']) == 2 and len(batch['batch_offsets']) == 3
    assert batch['ann_ids'] == repeated['ann_ids']
    for key in ('superpoints', 'batch_offsets', 'coords_float'):
        assert torch.equal(batch[key], repeated[key]), 'Collate repeat mismatch: ' + key
    actual = expression_point_targets(batch['superpoints'], batch['batch_offsets'], batch['scenes_len'], batch['gt_pmasks'])
    expected_targets = []
    expected_annotations = []
    expected_objects = []
    for single in singles:
        expected_targets.extend(expression_point_targets(single['superpoints'], single['batch_offsets'], single['scenes_len'], single['gt_pmasks']))
        expected_annotations.extend(single['ann_ids'])
        expected_objects.extend(single['object_idss'])
    assert batch['ann_ids'] == expected_annotations and batch['object_idss'] == expected_objects
    assert len(actual) == len(expected_targets) == len(batch['gt_pmasks'])
    expression = 0
    pair_mixed = 0
    for scene_index, scene_length in enumerate(batch['scenes_len']):
        start, end = [int(v) for v in batch['batch_offsets'][scene_index:scene_index + 2]]
        selected = (batch['superpoints'] >= start) & (batch['superpoints'] < end)
        membership = batch['superpoints'][selected].long() - start
        for _ in range(int(scene_length)):
            total, positive = actual[expression]
            reference_total, reference_positive = expected_targets[expression]
            assert torch.equal(total, reference_total) and torch.equal(positive, reference_positive)
            target = batch['gt_pmasks'][expression].reshape(-1)
            assert target.numel() == membership.numel()
            assert int(total.sum()) == target.numel()
            assert int(positive.sum()) == int((target > .5).sum())
            assert bool(((positive >= 0) & (positive <= total)).all())
            logits = torch.stack((torch.full_like(total, -10.), torch.full_like(total, 10.), (torch.arange(total.numel()).remainder(2).float() * 2 - 1) * 10.))
            observed = loss_module.get_point_iou(logits, total, positive)
            expanded = loss_module.get_iou(logits[:, membership], target.unsqueeze(0))
            assert torch.equal(observed, expanded), 'Point-IoU disagreement after real two-scene collate'
            pair_mixed += int(((positive > 0) & (positive < total)).sum())
            expression += 1
    assert expression == len(actual)
    expression_count += expression
    mixed_count += pair_mixed
    rows.append(dict(dataset_indices=[scene['dataset_index'] for scene in frozen[pair:pair + 2]],
        scan_ids=[scene['scan_id'] for scene in frozen[pair:pair + 2]],
        scene_expression_counts=[int(value) for value in batch['scenes_len']],
        expressions=expression, points=int(batch['superpoints'].numel()),
        superpoints=int(batch['batch_offsets'][-1]), mixed_expression_superpoints=pair_mixed,
        single_vs_two_scene_counts_exact=True, collate_repeat_exact=True, expanded_point_quality_error=0.))
assert expression_count == 32
dataset_source = Path(inspect.getsourcefile(type(dataset)))
result = dict(complete=True, id='AdaRES-Y52', no_GPU=True, frozen_TRAIN_scenes=8,
    native_two_scene_batches=len(rows), expressions=expression_count,
    fixed_mask_controls_per_expression=3, mixed_expression_superpoints=mixed_count,
    actual_production_TRAIN_dataset_collate_executed=True, fixed_seed=1999,
    train_dataset_source_sha256=hashlib.sha256(dataset_source.read_bytes()).hexdigest(),
    frozen_capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
    single_vs_two_scene_counts_exact=True, expanded_point_quality_error=0.,
    candidate_model_constructed=False, native_full_model_forward_executed=False,
    native_full_model_backward_executed=False, model_weights_saved=False,
    scientific_status='real_TRAIN_batchsize2_CPU_dataflow_control_only',
    limitation='Fixed synthetic mask controls on real collated TRAIN inputs test count/annotation/offset plumbing; not learned native predictions, matching frequency, shared-query gradients, GPU resources, restore, admission or trained efficacy.', rows=rows)
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
