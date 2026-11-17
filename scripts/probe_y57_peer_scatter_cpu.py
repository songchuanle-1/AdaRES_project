"""Trained-head representation witness, not a task-benefit or native gate."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import base64
import gc
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
import torch
from torch import nn

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert P['physical_gpu'] is None
torch.set_num_threads(1)
available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                 if line.startswith('MemAvailable:'))
assert available >= 8 * 1024 ** 3, 'Insufficient spare host RAM'


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def unpack(item):
    raw = zlib.decompress(base64.b64decode(item['zlib_base64'], validate=True))
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    value = np.frombuffer(raw, dtype=np.dtype(item['dtype'])).reshape(item['shape']).copy()
    assert np.isfinite(value).all()
    return torch.from_numpy(value)


def state_hash(model):
    value = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        value.update(key.encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


source = Path(P['root']) / FROZEN['source_path']
assert hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == FROZEN['source_lf_sha256']
cache_path = Path(FROZEN['remote_cache'])
assert digest_file(cache_path) == FROZEN['capture_sha256']
checkpoint = Path(FROZEN['checkpoint'])
assert digest_file(checkpoint) == FROZEN['checkpoint_sha256']
namespace = dict(torch=torch, nn=nn)
exec(compile(FROZEN['class_source'], '<frozen-native-W2-indicator>', 'exec'), namespace)
Indicator = namespace['QueryCompetitionIndicator']
stored = torch.load(str(checkpoint), map_location='cpu')
prefix = 'dec.indi_embedding.'
head_state = {key[len(prefix):]: value.detach().clone() for key, value in stored['model'].items()
              if key.startswith(prefix)}
assert head_state
del stored
gc.collect()
cache = json.loads(cache_path.read_text())
assert len(cache['rows']) == 8 and sum(len(row['expressions']) for row in cache['rows']) == 32


def native(model, query, masks, mass):
    captured = []
    hook = model.evidence_norm.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        output = model(query.unsqueeze(0), masks.unsqueeze(0),
                       torch.zeros((1, len(mass)), dtype=torch.bool), mass.unsqueeze(0))[0]
    finally:
        hook.remove()
    return output, captured[0][0, :, 256:512]


def focal_weights(masks, mass):
    probabilities = masks.sigmoid()
    weighted = probabilities * mass.unsqueeze(0)
    area = weighted.sum(-1)
    intersections = weighted @ probabilities.T
    overlap = intersections / (area[:, None] + area[None, :] - intersections).clamp_min(1e-6)
    overlap.fill_diagonal_(0)
    return (overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6))[0]


def reflection(query, weights, eps):
    centered = query - query.mean(-1, keepdim=True)
    standardized = centered / (centered.square().mean(-1, keepdim=True) + eps).sqrt()
    mean = weights @ standardized
    direction = torch.zeros(256, dtype=torch.float64)
    direction[0] = 1
    one = torch.ones(256, dtype=torch.float64)
    one /= one.norm()
    mean = mean.double() - mean.double().dot(one) * one
    direction -= direction.dot(one) * one
    if float(mean.norm()) > torch.finfo(torch.float64).eps:
        mean /= mean.norm()
        direction -= direction.dot(mean) * mean
    assert float(direction.norm()) > 128 * torch.finfo(torch.float64).eps
    direction = (direction / direction.norm()).to(query.dtype)
    transformed = query.mean(-1, keepdim=True) + centered - 2 * (centered @ direction)[:, None] * direction
    transformed[0] = query[0]
    return transformed, direction


def measure(model, query, masks, mass, controls=True):
    weights = focal_weights(masks, mass)
    assert float(weights.sum()) > .99 and float(weights[0]) == 0
    changed, direction = reflection(query, weights, model.query_norm.eps)
    with torch.no_grad():
        original, context = native(model, query, masks, mass)
        other, other_context = native(model, changed, masks, mass)
        normalized = model.query_norm(query)
        other_normalized = model.query_norm(changed)
        second = weights @ normalized.square()
        other_second = weights @ other_normalized.square()
        bound = 128 * torch.finfo(query.dtype).eps * (1 + max(
            float(original.abs().max()), float(other.abs().max()), float(context.abs().max())))
        moment_bound = 128 * torch.finfo(query.dtype).eps * (1 + max(
            float(second.abs().max()), float(other_second.abs().max())))
        repeat_error = permutation_error = None
        if controls:
            repeated, _ = native(model, query, masks, mass)
            order = torch.arange(127, -1, -1)
            permuted, _ = native(model, query[order], masks[order], mass)
            repeat_error = float((original - repeated).abs().max())
            permutation_error = float((original - permuted[order]).abs().max())
        mean_error = float((context[0] - other_context[0]).abs().max())
        logit_error = float((original[0] - other[0]).abs().max())
        row_mean_error = float((query.mean(-1) - changed.mean(-1)).abs().max())
        row_variance_error = float((query.var(-1, unbiased=False) - changed.var(-1, unbiased=False)).abs().max())
        norm_bound = 128 * torch.finfo(query.dtype).eps * (1 + float(query.square().mean(-1).max()))
        second_difference = float((second - other_second).abs().max())
        result = dict(dtype=str(query.dtype), focal_query=0, query_count=128,
            source_unique_queries=int(len(np.unique(query.numpy(), axis=0))),
            transformed_unique_queries=int(len(np.unique(changed.numpy(), axis=0))),
            source_unique_masks=int(len(np.unique(masks.numpy(), axis=0))),
            focal_query_bitwise_preserved=bool(torch.equal(query[0], changed[0])),
            masks_mass_query_count_unchanged=True, weights_sum=float(weights.sum()),
            peer_feature_change=float((query[1:] - changed[1:]).abs().max()),
            row_mean_error=row_mean_error, row_variance_error=row_variance_error,
            row_arithmetic_bound=norm_bound, focal_context_error=mean_error,
            focal_logit_error=logit_error, arithmetic_bound=bound,
            weighted_diagonal_second_moment_difference=second_difference,
            second_moment_arithmetic_bound=moment_bound,
            repeat_error=repeat_error, permutation_error=permutation_error)
        result['matched_control_valid'] = (result['focal_query_bitwise_preserved'] and
            max(mean_error, logit_error, repeat_error or 0, permutation_error or 0) <= bound and
            max(row_mean_error, row_variance_error) <= norm_bound)
        result['representation_witness'] = (result['matched_control_valid'] and second_difference > moment_bound)
        return result


rows, negatives = [], []
for dtype in (torch.float32, torch.float64):
    model = Indicator(256)
    model.load_state_dict(head_state, strict=True)
    model = model.to(dtype=dtype).eval()
    before = state_hash(model)
    for scene in cache['rows']:
        for expression in scene['expressions']:
            packed = expression['packed_native_inputs']
            query, masks, mass = (unpack(packed[key]).to(dtype=dtype)
                                  for key in ('query', 'masks', 'masses'))
            assert query.shape == (128, 256) and masks.shape[0] == 128 and masks.shape[1] == len(mass)
            assert bool((mass >= 0).all())
            row = measure(model, query, masks, mass)
            row.update(scene=scene['scan_id'], dataset_index=scene['dataset_index'], ann_id=expression['ann_id'])
            rows.append(row)
    torch.manual_seed(1999)
    query = torch.randn(128, 256, dtype=dtype)
    query[1:] = query[1].clone()
    coordinate = torch.linspace(-1, 1, 64, dtype=dtype)
    masks = coordinate.unsqueeze(0).expand(128, -1).clone()
    negative = measure(model, query, masks, torch.ones(64, dtype=dtype))
    negative['expected_no_scatter_witness'] = not negative['representation_witness']
    negatives.append(negative)
    assert state_hash(model) == before

assert digest_file(cache_path) == FROZEN['capture_sha256']
assert digest_file(checkpoint) == FROZEN['checkpoint_sha256']
assert hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == FROZEN['source_lf_sha256']
assert not torch.cuda.is_initialized()
result = dict(complete=True, id='AdaRES-Y57', device='cpu', threads=1,
    scientific_status='representation_capacity_witness_only_task_relevance_unproven',
    cached_TRAIN_scenes=8, cached_TRAIN_expressions=32, dtype_cases=len(rows),
    valid_matched_cases=sum(row['matched_control_valid'] for row in rows),
    representation_witness_cases=sum(row['representation_witness'] for row in rows),
    negative_controls_passed=all(row['matched_control_valid'] and row['expected_no_scatter_witness'] for row in negatives),
    max_focal_context_error=max(row['focal_context_error'] for row in rows),
    max_focal_logit_error=max(row['focal_logit_error'] for row in rows),
    max_repeat_error=max(row['repeat_error'] for row in rows),
    max_permutation_error=max(row['permutation_error'] for row in rows),
    min_second_moment_difference=min(row['weighted_diagonal_second_moment_difference'] for row in rows),
    max_second_moment_difference=max(row['weighted_diagonal_second_moment_difference'] for row in rows),
    checkpoint_sha256=FROZEN['checkpoint_sha256'], capture_sha256=FROZEN['capture_sha256'],
    source_lf_sha256=FROZEN['source_lf_sha256'], checkpoint_source_cache_unchanged=True,
    training_launches=0, gpu_tests=0, admission=False, full_native_backward=False,
    rows=rows, cases=negatives,
    limitations=[
        'Eight fixed cached TRAIN scenes/32 expressions; focal index0 fixed without prediction-based selection.',
        'Reflection is an algebraically valid indicator-input intervention, not a known reachable decoder/data intervention.',
        'Same masks, masses, query counts and focal query; peer feature values deliberately differ. This is NOT Y56 exact-copy multiplicity invariance.',
        'Nonlinear feature scatter is not the mask joint-coverage readout of rejected Y50, nor within-superpoint point-feature covariance.',
        'A representation witness is neither a task defect nor mIoU benefit. No labels, thresholds, weights or training protocol were selected here.',
        'Original cache lacks checkpoint blob SHA at capture time; strict current replay SHA does not retroactively bind old capture bytes.',
        'No formal model correction, native batch2/all-layer forward/backward, resource gate, novelty review or admission performed.',
    ])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
