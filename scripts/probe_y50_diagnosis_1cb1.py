"""Pre-registered matched source information test, followed by frozen TRAIN controls."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import time
import zlib
import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn

assert P['physical_gpu'] is None
torch.set_num_threads(4)
root = Path(P['root'])
parent_source = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
capture_path = Path('/tmp/adares_y49_train24_capture_v2_1cb1_1cb15474bfd346dcbc34837f3e9b79d3.json')
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == '277a11859224110feec8819de03f3d1aa3ae6bc37074d42f9fa5b29b4e842c11'
capture = json.loads(capture_path.read_text())
parent_sha = hashlib.sha256(parent_source.read_bytes()).hexdigest()
assert parent_sha == capture['source_hashes']['ipdn/model_paper5/adares_w2_pointmass_competition/dec.py']
parent_node = next(n for n in ast.parse(parent_source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'QueryCompetitionIndicator')
exec(compile(ast.Module(body=[parent_node], type_ignores=[]), str(parent_source), 'exec'), globals())
extension = Path('/tmp/adares_y50_joint_incidence_1cb1.py')
extension_sha = hashlib.sha256(extension.read_bytes()).hexdigest()
expected = json.loads(Path('/tmp/adares_y50_source_freeze_1cb1.json').read_text())
assert extension_sha == expected['source_sha256']
exec(compile(extension.read_text(), str(extension), 'exec'), globals())
rows = []
bits = torch.tensor([[0,0,0],[0,1,1],[1,0,1],[1,1,0]], dtype=torch.double).T
other = 1 - bits
probabilities = [torch.cat((.25 + .5 * item, torch.full((125,4), .5, dtype=torch.double)), 0).unsqueeze(0) for item in (bits, other)]
masks = [torch.logit(p) for p in probabilities]
padding = torch.zeros((1,4), dtype=torch.bool)
mass = torch.ones((1,4), dtype=torch.double)
areas = [(p * mass.unsqueeze(1)).sum(-1) for p in probabilities]
grams = [torch.einsum('bim,bjm->bij', p * mass.unsqueeze(1), p) for p in probabilities]
assert torch.equal(areas[0], areas[1]) and torch.equal(grams[0], grams[1])
descriptors = [joint_incidence_descriptor(m, padding, mass) for m in masks]
bound = 8 * torch.finfo(torch.double).eps * 4 * 128
gap = float((descriptors[0] - descriptors[1]).abs().max())
assert gap > bound
start = time.monotonic()
for seed in range(1999, 2031):
    torch.manual_seed(seed)
    parent = QueryCompetitionIndicator(256).double()
    query = torch.randn((1,128,256), dtype=torch.double)
    parent_outputs = [parent(query, m, padding, mass) for m in masks]
    parent_error = float((parent_outputs[0] - parent_outputs[1]).abs().max())
    envelope = bound * max(1., *(float(v.abs().max()) for v in parent_outputs))
    assert parent_error <= envelope
    head = JointIncidenceReadout(256).double()
    scores = [head(query, d) for d in descriptors]
    score_gap = float((scores[0] - scores[1]).abs().max())
    head.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(torch.stack((torch.zeros_like(scores[0]), scores[0]), -1).reshape(-1,2), torch.arange(128).remainder(2)).backward()
    gradients = {n: float(v.grad.norm()) for n, v in head.named_parameters()}
    assert all(np.isfinite(v) and v > 0 for v in gradients.values())
    rows.append(dict(seed=seed, parent_indicator_max_error=parent_error, existing_envelope=envelope, descriptor_gap=gap,
                     new_score_gap=score_gap, proxy_gradients=gradients))
assert all(r['new_score_gap'] > r['existing_envelope'] for r in rows)


def unpack(item):
    raw = zlib.decompress(base64.b64decode(item['zlib_base64']))
    assert hashlib.sha256(raw).hexdigest() == item['sha256']
    return np.frombuffer(raw, dtype=np.dtype(item['dtype'])).reshape(item['shape']).copy()


original = Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(original.read_bytes()).hexdigest() == '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'
cohorts = [('original_TRAIN8', json.loads(original.read_text())), ('new_TRAIN24', capture)]
native = []
for label, cohort in cohorts:
    for scene in cohort['rows']:
        for expression in scene['expressions']:
            arrays = {k: unpack(v) for k, v in expression['packed_native_inputs'].items()}
            m = torch.from_numpy(arrays['masks'].astype(np.float64)).unsqueeze(0)
            w = torch.from_numpy(arrays['masses'].astype(np.float64)).unsqueeze(0)
            pad = torch.zeros((1, m.shape[-1]), dtype=torch.bool)
            value = joint_incidence_descriptor(m, pad, w)
            assert torch.isfinite(value).all()
            reverse = torch.arange(m.shape[-1]-1, -1, -1)
            permuted = joint_incidence_descriptor(m[:,:,reverse], pad[:,reverse], w[:,reverse])
            point_error = float((value - permuted).abs().max())
            query_permuted = joint_incidence_descriptor(m.flip(1), pad, w)
            query_error = float((value.flip(1) - query_permuted).abs().max())
            extended_masks = torch.cat((m, torch.full((1,128,3), 1000., dtype=torch.double)), -1)
            extended_mass = torch.cat((w, torch.full((1,3), 1e8, dtype=torch.double)), -1)
            extended_pad = torch.cat((pad, torch.ones((1,3), dtype=torch.bool)), -1)
            extended = joint_incidence_descriptor(extended_masks, extended_pad, extended_mass)
            padding_error = float((value - extended).abs().max())
            envelope = 8 * torch.finfo(torch.double).eps * m.shape[-1] * 128
            assert max(point_error, query_error, padding_error) <= envelope
            gt = arrays.get('TRAIN_gt_spmask')
            associations = None
            if gt is not None:
                hard = arrays['masks'] > 0
                masses = arrays['masses'].astype(np.float64)
                intersection = (hard & gt) @ masses
                union = (hard | gt) @ masses
                iou = intersection / np.maximum(union, 1e-6)
                correlations = []
                for j in range(2):
                    values = value[0,:,j].numpy()
                    correlations.append(float(spearmanr(values, iou).correlation) if np.ptp(values) > 0 and np.ptp(iou) > 0 else None)
                associations = dict(TRAIN_IoU_range=float(np.ptp(iou)), descriptive_spearman_FG_BG=correlations)
            native.append(dict(cohort=label, scan_id=scene['scan_id'], ann_id=expression['ann_id'], nodes=m.shape[-1],
                               descriptor_min=value.min(1).values[0].tolist(), descriptor_max=value.max(1).values[0].tolist(),
                               descriptor_ranges=(value.max(1).values-value.min(1).values)[0].tolist(),
                               point_permutation_error=point_error, query_permutation_error=query_error,
                               padded_point_error=padding_error, existing_envelope=envelope, TRAIN_associations=associations))
print('Y50_NATIVE_DESCRIPTOR_EXPRESSIONS', len(native), flush=True)
assert len(native) == 128
result = dict(complete=True, id='AdaRES-Y50', parent_decoder_sha256=parent_sha, extension_sha256=extension_sha,
              source_fixture_count=32, full128_matched_areas_and_Gram_exact=True,
              maximum_actual_parent_indicator_error=max(r['parent_indicator_max_error'] for r in rows), descriptor_gap=gap,
              minimum_new_score_gap=min(r['new_score_gap'] for r in rows),
              native_expressions=128, native_nonconstant_expressions=sum(max(r['descriptor_ranges']) > r['existing_envelope'] for r in native),
              CPU_seconds=time.monotonic()-start, rows=rows, native_rows=native,
              scientific_status='indicator_information_controls_only_not_native_backward_or_trained_utility',
              limitations=['The actual parent indicator, not the entire upstream decoder, is blind on this matched source fixture.',
                           'A higher-order statistic is not proof of generalization, novelty or mIoU improvement.',
                           'Native feature ranges and TRAIN associations are descriptive only; no association-based selection or sign fitting.',
                           'No validation set, optimizer update, admission receipt or training launch.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
