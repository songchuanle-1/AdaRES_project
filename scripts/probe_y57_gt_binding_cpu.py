"""Reconstruct exact TRAIN labels before describing fixed-cache associations."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import ast
import base64
import gc
import hashlib
import inspect
import json
from pathlib import Path
from typing import Union
import zlib

import gorilla
import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch_scatter import scatter_mean
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert P['physical_gpu'] is None and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
torch.set_num_threads(1)
available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith('MemAvailable:'))
assert available >= 8 * 1024 ** 3
root = Path(P['root'])


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def unpack(item):
    data = zlib.decompress(base64.b64decode(item['zlib_base64'], validate=True))
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    value = np.frombuffer(data, dtype=np.dtype(item['dtype'])).reshape(item['shape']).copy()
    assert np.isfinite(value).all()
    return torch.from_numpy(value)


def native_sources():
    for row in FROZEN['sources']:
        assert hashlib.sha256((root / row['path']).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == row['lf_sha256']


def rank_correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(x, y)[0])
    return value if np.isfinite(value) else None


native_sources()
cache_path = Path(FROZEN['remote_cache'])
assert digest_file(cache_path) == FROZEN['capture_sha256']
capture = json.loads(cache_path.read_text())
assert [row['dataset_index'] for row in capture['rows']] == [0, 44, 60, 101, 115, 126, 137, 171]
assert sum(len(row['expressions']) for row in capture['rows']) == 32
cfg = gorilla.Config.fromfile(str(root / FROZEN['config_path']))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30 and cfg.data.train.prefix == 'train'
assert cfg.data.train.training and cfg.data.train.aug and cfg.data.train.lang_num_max == 4
checkpoint = Path(FROZEN['checkpoint'])
assert digest_file(checkpoint) == FROZEN['checkpoint_sha256']
namespace = dict(torch=torch, nn=nn, Union=Union)
exec(compile(FROZEN['class_source'], '<frozen-native-W2-head>', 'exec'), namespace)
exec(compile(FROZEN['iou_source'], '<frozen-native-point-IoU>', 'exec'), namespace)
native_iou = namespace['get_iou']
model = namespace['QueryCompetitionIndicator'](256).eval()
stored = torch.load(str(checkpoint), map_location='cpu')
prefix = 'dec.indi_embedding.'
model.load_state_dict({key[len(prefix):]: value for key, value in stored['model'].items()
                       if key.startswith(prefix)}, strict=True)
del stored
gc.collect()
state_before = {key: tensor_hash(value) for key, value in model.state_dict().items()}
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
source_paths = {Path(inspect.getsourcefile(build_dataset)), Path(inspect.getsourcefile(type(dataset)))}
source_paths.update(Path(inspect.getsourcefile(cls)) for cls in type(dataset).__mro__
                    if cls.__module__ not in ('builtins', 'torch.utils.data.dataset') and inspect.getsourcefile(cls))
dataset_sources = {str(path): digest_file(path) for path in source_paths}
rows = []
for scene in capture['rows']:
    gorilla.set_random_seed(1999)
    batch = dataset.collate_fn([dataset[scene['dataset_index']]])
    gorilla.set_random_seed(1999)
    repeated = dataset.collate_fn([dataset[scene['dataset_index']]])
    repeat_ok = batch['ann_ids'] == repeated['ann_ids'] and batch['object_idss'] == repeated['object_idss']
    repeat_ok = repeat_ok and all(torch.equal(batch[key], repeated[key]) for key in ('superpoints', 'coords_float', 'batch_offsets'))
    repeat_ok = repeat_ok and all(torch.equal(a, b) for a, b in zip(batch['gt_pmasks'], repeated['gt_pmasks']))
    del repeated
    membership = batch['superpoints'].long()
    n = int(batch['batch_offsets'][-1])
    total = torch.bincount(membership, minlength=n)
    coordinates = scatter_mean(batch['coords_float'], membership, dim=0)
    assert len(batch['object_idss']) == len(scene['expressions']) == 4
    for index, expression in enumerate(scene['expressions']):
        packed = expression['packed_native_inputs']
        masks, query, mass, cached_coordinates = (unpack(packed[key]).float()
                                                  for key in ('masks', 'query', 'masses', 'coordinates'))
        assert query.shape == (128, 256) and masks.shape == (128, n)
        point_target = batch['gt_pmasks'][index].reshape(-1).float()
        assert point_target.numel() == membership.numel()
        positive = torch.bincount(membership, weights=(point_target > .5).float(), minlength=n)
        coordinate_error = float((coordinates - cached_coordinates).abs().max())
        coordinate_bound = 32 * torch.finfo(coordinates.dtype).eps * max(1., float(coordinates.abs().max()))
        annotation_ok = (batch['scan_ids'][0] == scene['scan_id'] and
            str(batch['ann_ids'][index]) == expression['ann_id'] and
            batch['object_idss'][index] == expression['object_ids'])
        mass_ok = torch.equal(total.float(), mass)
        target_ok = (int(total.sum()) == point_target.numel() and int(positive.sum()) == int((point_target > .5).sum())
                     and bool(((positive >= 0) & (positive <= total)).all()))
        binary = (masks.sigmoid() >= .5).float()
        intersection = (binary * positive.unsqueeze(0)).sum(-1)
        quality = intersection / ((binary * total.unsqueeze(0)).sum(-1) + positive.sum() - intersection + 1e-6)
        reference = native_iou(masks[:4, membership], point_target.unsqueeze(0))
        reference_error = float((quality[:4] - reference).abs().max())
        binding_ok = repeat_ok and annotation_ok and mass_ok and target_ok and coordinate_error <= coordinate_bound and reference_error == 0
        row = dict(scene=scene['scan_id'], dataset_index=scene['dataset_index'], ann_id=expression['ann_id'],
            object_ids=expression['object_ids'], points=int(len(membership)), superpoints=n,
            annotation_exact=annotation_ok, mass_exact=mass_ok, collate_repeat_exact=repeat_ok,
            coordinate_error=coordinate_error, coordinate_bound=coordinate_bound,
            point_target_counts_exact=target_ok, expanded_point_IoU_error=reference_error, binding_passed=binding_ok,
            reconstructed_GT_sha256=tensor_hash(point_target), membership_sha256=tensor_hash(membership),
            coordinates_sha256=tensor_hash(coordinates), total_counts_sha256=tensor_hash(total),
            positive_counts_sha256=tensor_hash(positive), task_quality_sha256=tensor_hash(quality),
            mixed_label_superpoints=int(((positive > 0) & (positive < total)).sum()),
            all_query_point_IoU=quality.tolist())
        if binding_ok:
            with torch.no_grad():
                logits = model(query.unsqueeze(0), masks.unsqueeze(0), torch.zeros((1, n), dtype=torch.bool), mass.unsqueeze(0))[0]
                confidence = logits.softmax(-1)[:, 1]
                probabilities = masks.sigmoid()
                weighted = probabilities * mass.unsqueeze(0)
                areas = weighted.sum(-1)
                intersections = weighted @ probabilities.T
                overlap = intersections / (areas[:, None] + areas[None, :] - intersections).clamp_min(1e-6)
                overlap.fill_diagonal_(0)
                competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
                normalized = model.query_norm(query)
                context = competition @ normalized
                diagonal_variance = (competition @ normalized.square() - context.square()).clamp_min(0)
                dispersion = diagonal_variance.square().sum(-1).sqrt()
                error = (confidence - quality).abs()
            descriptor, errors, truth = dispersion.numpy(), error.numpy(), quality.numpy()
            selected = int(confidence.argmax())
            correlations = dict(dispersion_vs_abs_confidence_error=rank_correlation(descriptor, errors),
                dispersion_vs_point_quality=rank_correlation(descriptor, truth),
                rotated_label_error=rank_correlation(descriptor, np.abs(confidence.numpy() - np.roll(truth, 1))))
            permuted = rank_correlation(descriptor[::-1], errors[::-1])
            reference_corr = correlations['dispersion_vs_abs_confidence_error']
            permutation_error = None if reference_corr is None else abs(reference_corr - permuted)
            row.update(dispersion=descriptor.tolist(), native_confidence=confidence.tolist(),
                all_query_absolute_confidence_error=errors.tolist(), correlations=correlations,
                joint_query_permutation_correlation_error=permutation_error,
                native_argmax_query=selected, selected_cached_point_quality=float(quality[selected]),
                oracle_cached_point_quality=float(quality.max()),
                cached_selection_gap=float(quality.max() - quality[selected]))
        rows.append(row)
    del batch
    gc.collect()

assert len(rows) == 32
valid = all(row['binding_passed'] for row in rows)
scene_rows = []
if valid:
    for scene in capture['rows']:
        selected = [row for row in rows if row['scene'] == scene['scan_id']]
        grouped = {}
        for key in ('dispersion_vs_abs_confidence_error', 'dispersion_vs_point_quality', 'rotated_label_error'):
            values = [row['correlations'][key] for row in selected if row['correlations'][key] is not None]
            grouped[key] = dict(valid_expressions=len(values), mean=float(np.mean(values)) if values else None)
        scene_rows.append(dict(scene=scene['scan_id'], correlations=grouped))
summary = {}
for key in ('dispersion_vs_abs_confidence_error', 'dispersion_vs_point_quality', 'rotated_label_error'):
    values = [row['correlations'][key]['mean'] for row in scene_rows if row['correlations'][key]['mean'] is not None]
    if len(values) >= 2:
        generator = np.random.RandomState(1999)
        bootstrap = np.asarray(values)[generator.randint(0, len(values), size=(10000, len(values)))].mean(1)
        summary[key] = dict(valid_scenes=len(values), mean=float(np.mean(values)),
            descriptive_scene_bootstrap95=[float(value) for value in np.percentile(bootstrap, [2.5, 97.5])])
    else:
        summary[key] = dict(valid_scenes=len(values), mean=float(np.mean(values)) if values else None,
                            descriptive_scene_bootstrap95=None)

native_sources()
assert dataset_sources == {path: digest_file(path) for path in dataset_sources}
assert digest_file(cache_path) == FROZEN['capture_sha256']
assert digest_file(checkpoint) == FROZEN['checkpoint_sha256']
assert state_before == {key: tensor_hash(value) for key, value in model.state_dict().items()}
assert not torch.cuda.is_initialized()
result = dict(complete=True, id='AdaRES-Y57', device='cpu', threads=1,
    scientific_status='matched_TRAIN_label_reconstruction_and_descriptive_association_only',
    cache_sha256=FROZEN['capture_sha256'], checkpoint_sha256=FROZEN['checkpoint_sha256'],
    scenes=8, expressions=len(rows), native_queries_per_expression=128, all_bindings_passed=valid,
    binding_passed_expressions=sum(row['binding_passed'] for row in rows),
    max_coordinate_error=max(row['coordinate_error'] for row in rows),
    max_expanded_point_IoU_error=max(row['expanded_point_IoU_error'] for row in rows),
    mixed_expression_superpoints=sum(row['mixed_label_superpoints'] for row in rows),
    associations=summary, scene_summaries=scene_rows,
    dataset_sources=dataset_sources, source_cache_checkpoint_head_unchanged=True,
    gpu_tests=0, full_native_forward=False, full_native_backward=False, training_launches=0, admission=False,
    limitations=[
        'Frozen8TRAIN scenes/32expressions are not validation, a benchmark mIoU, or a new trained model.',
        'Source/annotation/mass/geometry/repeat checks bind current reconstructed labels to old cache geometry; original capture did not hash raw point-GT or checkpoint bytes at capture time.',
        'Descriptor is fixed all-channel diagonal-variance L2, not selected channels, a fitted predictor, threshold search or a changed selector.',
        'Scene-bootstrap intervals with8scenes describe this fixed cache; correlations do not establish incremental information conditional on the full native query/context, causality, reachable counterfactuals or task harm.',
        'One-step rotated labels are a fixed broken-correspondence control, not independent task labels or a formal randomization significance test.',
        'Native softmax foreground argmax is replayed on TRAIN loss-forward inputs, not fresh inference; fixed-mask feature interventions also preserve true IoU.',
        'No implementation, native batch2/all7layer capture, resources/backward gate, trained efficacy or novelty established.',
    ], rows=rows)
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
