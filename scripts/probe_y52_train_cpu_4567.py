"""Replay frozen TRAIN masks and test exact point-count supervision on CPU."""
import ast
import base64
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import zlib
import gorilla
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_mean
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert P['physical_gpu'] is None
torch.set_num_threads(2)
root = Path(P['root'])
package = root / 'ipdn/model_paper5/adares_w2_pointmass_competition'
spec = importlib.util.spec_from_file_location('y52_native_loss_cpu', str(package / 'loss.py'))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
capture_path = Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
capture = json.loads(capture_path.read_text())


def unpack(item):
    raw = zlib.decompress(base64.b64decode(item['zlib_base64']))
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    return np.frombuffer(raw, dtype=np.dtype(item['dtype'])).reshape(item['shape']).copy()


def exact_point_quality(mask_logits, total_counts, positive_counts):
    """Native threshold and epsilon with exact point rather than SP counts."""
    with torch.no_grad():
        binary = (mask_logits.sigmoid() >= .5).to(mask_logits.dtype)
        intersection = (binary * positive_counts).sum(-1)
        union = (binary * total_counts).sum(-1) + positive_counts.sum(-1) - intersection
        return intersection / (union + 1e-6)


def existing_score_mse(scores, quality, num_boxes):
    valid = quality > .5
    return (torch.nn.functional.mse_loss(scores[valid, 0], quality[valid]) / num_boxes
            if valid.any() else scores.sum() * 0.)


fixture_rows = []
for seed in range(1999, 2031):
    g = torch.Generator().manual_seed(seed)
    total = torch.randint(1, 30, (19,), generator=g)
    membership = torch.repeat_interleave(torch.arange(19), total)
    point_target = torch.randint(0, 2, (len(membership),), generator=g).float()
    positive = torch.bincount(membership, weights=point_target, minlength=19)
    masks = torch.randn((128, 19), generator=g)
    scores_a = torch.randn((128, 1), generator=g, requires_grad=True)
    scores_b = scores_a.detach().clone().requires_grad_(True)
    efficient = exact_point_quality(masks, total, positive)
    expanded = native.get_iou(masks[:, membership], point_target.unsqueeze(0))
    assert torch.equal(efficient, expanded)
    la, lb = [existing_score_mse(s, q, 128.) for s, q in ((scores_a, efficient), (scores_b, expanded))]
    la.backward()
    lb.backward()
    assert torch.equal(la, lb) and torch.equal(scores_a.grad, scores_b.grad)
    fixture_rows.append(dict(seed=seed, quality_error=0., loss_error=0., gradient_error=0.,
                             mixed_label_superpoints=int(((positive > 0) & (positive < total)).sum())))
for name, masks, total, positive in (
    ('empty_prediction', torch.full((3, 4), -100.), torch.tensor([20, 1, 1, 1]), torch.tensor([20., 1., 1., 0.])),
    ('empty_target', torch.full((3, 4), 100.), torch.tensor([20, 1, 1, 1]), torch.zeros(4)),
    ('empty_both', torch.full((3, 4), -100.), torch.ones(4, dtype=torch.long), torch.zeros(4))):
    values = exact_point_quality(masks, total, positive)
    assert torch.equal(values, torch.zeros(3))
    fixture_rows.append(dict(name=name, quality_error=0.))

# This tests the ACTUAL unchanged score-head component, not full native backward.
tree = ast.parse((package / 'dec.py').read_text())
dec = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DEC')
init = next(n for n in dec.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign) and any(
    isinstance(t, ast.Attribute) and t.attr == 'out_score' for t in n.targets))
head = eval(compile(ast.Expression(body=assignment.value), '<native-score-head-constructor>', 'eval'), dict(nn=nn, d_model=256))
checkpoint_path = Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth')
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
saved = torch.load(str(checkpoint_path), map_location='cpu')
prefix = 'dec.out_score.'
head.load_state_dict({k[len(prefix):]: v for k, v in saved['model'].items() if k.startswith(prefix)}, strict=True)
del saved
gc.collect()
torch.manual_seed(1999)
query = torch.randn((128, 256), requires_grad=True)
component_loss = existing_score_mse(head(query), torch.full((128,), .75), 128.)
component_loss.backward()
component_gradients = {name: dict(finite=bool(torch.isfinite(p.grad).all()), nonzero=int(torch.count_nonzero(p.grad)), norm=float(p.grad.norm())) for name, p in head.named_parameters()}
assert len(component_gradients) == 4 and all(v['finite'] and v['nonzero'] > 0 for v in component_gradients.values())
assert query.grad is not None and torch.isfinite(query.grad).all() and query.grad.norm() > 0

gorilla.set_random_seed(1999)
cfg = gorilla.Config.fromfile(str(root / 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'))
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
rows = []
for captured_scene in capture['rows']:
    index = captured_scene['dataset_index']
    gorilla.set_random_seed(1999)
    batch = dataset.collate_fn([dataset[index]])
    gorilla.set_random_seed(1999)
    repeated = dataset.collate_fn([dataset[index]])
    assert batch['ann_ids'] == repeated['ann_ids'] and batch['object_idss'] == repeated['object_idss']
    for name in ('superpoints', 'coords_float', 'batch_offsets'):
        assert torch.equal(batch[name], repeated[name]), 'Nonrepeatable dataset: ' + name
    for left, right in zip(batch['gt_pmasks'], repeated['gt_pmasks']):
        assert torch.equal(left, right)
    del repeated
    assert batch['scan_ids'][0] == captured_scene['scan_id']
    assert len(batch['scenes_len']) == 1 and len(batch['object_idss']) == len(captured_scene['expressions'])
    membership = batch['superpoints'].long()
    n = int(batch['batch_offsets'][-1])
    total = torch.bincount(membership, minlength=n)
    coordinates = scatter_mean(batch['coords_float'], membership, dim=0)
    for b, expression in enumerate(captured_scene['expressions']):
        assert str(batch['ann_ids'][b]) == expression['ann_id']
        assert batch['object_idss'][b] == expression['object_ids'] and len(expression['object_ids']) == 1
        packed = expression['packed_native_inputs']
        assert np.array_equal(total.numpy(), unpack(packed['masses']))
        cached_coordinates = torch.from_numpy(unpack(packed['coordinates']))
        coordinate_error = float((coordinates - cached_coordinates).abs().max())
        coordinate_roundoff = 32 * torch.finfo(coordinates.dtype).eps * max(1., float(coordinates.abs().max()))
        assert coordinate_error <= coordinate_roundoff, 'Different augmented geometry'
        masks = torch.from_numpy(unpack(packed['masks']))
        assert masks.shape == (128, n)
        point_target = batch['gt_pmasks'][b].reshape(-1).float()
        assert len(point_target) == len(membership)
        positive = torch.bincount(membership, weights=(point_target > .5).float(), minlength=n)
        hard_target = (batch['sp_ins_labels'][0] == expression['object_ids'][0]).float()
        supplied_target = (batch['gt_spmasks'][b].reshape(-1) > .5).float()
        parent = native.get_iou(masks, hard_target.unsqueeze(0))
        weighted_hard = exact_point_quality(masks, total, hard_target * total)
        exact = exact_point_quality(masks, total, positive)
        reference = native.get_iou(masks[:8, membership], point_target.unsqueeze(0))
        assert torch.equal(exact[:8], reference)
        delta = exact - parent
        rows.append(dict(dataset_index=index, scan_id=captured_scene['scan_id'], ann_id=expression['ann_id'],
            object_ids=expression['object_ids'], queries=128, superpoints=n,
            point_count=len(membership), mixed_label_superpoints=int(((positive > 0) & (positive < total)).sum()),
            hard_target_vs_supplied_target_disagreement=int(torch.count_nonzero(hard_target - supplied_target)),
            target_mass_difference=int((hard_target * total).sum() - positive.sum()),
            coordinate_replay_error=coordinate_error, prospective_coordinate_roundoff=coordinate_roundoff,
            mean_abs_quality_difference=float(delta.abs().mean()), max_abs_quality_difference=float(delta.abs().max()),
            half_filter_disagreements=int(torch.count_nonzero((parent > .5) != (exact > .5))),
            count_kernel_reference_error=0.,
            hard_mass_only_mean_abs_difference=float((weighted_hard - parent).abs().mean()),
            within_superpoint_mixing_mean_abs_difference=float((exact - weighted_hard).abs().mean()),
            native_matched_query_frequency='unknown: matching indices were not in frozen capture'))
    del batch
    gc.collect()

assert len(rows) == 32 and len({r['scan_id'] for r in rows}) == 8
log_path = root / 'logs/train_adares_y50_joint_incidence_res_20261016_145628.log'
with log_path.open('rb') as stream:
    stream.seek(0, 2)
    length = stream.tell()
    stream.seek(max(0, length - 16384))
    tail = stream.read(16384).decode('utf-8', errors='replace')
progress = re.findall(r'Train\[(\d+)/(\d+)\]:[^\r\n]*?(\d+)/(\d+)\s*\[', tail)
result = dict(complete=True, id='AdaRES-Y52', no_GPU=True, fixture_count=len(fixture_rows),
    native_frozen_scenes=8, native_frozen_expressions=len(rows), native_queries_per_expression=128,
    actual_corrected_loss_on_native_TRAIN_NOT_executed=True,
    task_quality_difference_expressions=sum(r['max_abs_quality_difference'] > 1e-7 for r in rows),
    total_half_filter_disagreements=sum(r['half_filter_disagreements'] for r in rows),
    component_gradients=component_gradients, shared_query_component_gradient_norm=float(query.grad.norm()),
    kernel_source_binding='Exact tested function is bound by executor source_sha256; no candidate model deployed.',
    scientific_status='CPU_component_and_frozen_mask_diagnosis_only',
    limitations=['Native matched-query frequency, full loss reachability, full backward, resource/restore gates and trained efficacy remain unknown.',
                 'The cache is from trained W2, not Y50 or a new trained candidate.',
                 'A nonzero component gradient and all-query target discrepancy do not prove final indicator improvement.'],
    training=dict(pid=31539, progress_tail=progress[-2:], bounded_log_bytes=16384,
                  crash_marker_in_bounded_tail=any(v in tail for v in ('Traceback (most recent call last)', 'CUDA out of memory'))),
    resources=dict(gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode().splitlines(),
                   processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits']).decode().splitlines()),
    rows=rows, cases=fixture_rows)
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
