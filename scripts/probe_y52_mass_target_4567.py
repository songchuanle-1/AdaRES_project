"""Execute native W2 quality-target and selector controls on CPU only."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

assert P['physical_gpu'] is None
torch.set_num_threads(2)
root = Path(P['root'])
package = root / 'ipdn/model_paper5/adares_w2_pointmass_competition'
spec = importlib.util.spec_from_file_location('y52_exact_native_loss', str(package / 'loss.py'))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
get_iou = native.get_iou
criterion = native.Criterion()
source = (package / 'ipdn.py').read_text()
model = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'IPDN')
predict_node = next(copy.deepcopy(n) for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'predict_by_feat')
exec(compile(ast.Module(body=[predict_node], type_ignores=[]), '<exact-native-RES-selector>', 'exec'), globals())
dummy = SimpleNamespace(task_type='res')


def one_case(binary, target, masses, label):
    logits = (binary * 2 - 1).float() * 4
    gt = target.float()
    native_quality = get_iou(logits, gt.unsqueeze(0))
    membership = torch.repeat_interleave(torch.arange(len(masses)), masses)
    expanded_quality = get_iou(logits[:, membership], gt[membership].unsqueeze(0))
    weighted_intersection = (binary * gt * masses).sum(-1)
    weighted_union = ((binary + gt - binary * gt) * masses).sum(-1)
    independent_quality = weighted_intersection / (weighted_union + 1e-6)
    assert torch.allclose(expanded_quality, independent_quality, atol=1e-7, rtol=0)
    scores = torch.zeros((len(binary), 1), requires_grad=True)
    indices = [(torch.arange(len(binary)), torch.arange(len(binary)))]
    targets = [{'masks': gt.unsqueeze(0).repeat(len(binary), 1)}]
    _, _, score_loss = criterion.loss_masks([logits], [scores], targets, indices, float(len(binary)))
    if score_loss.requires_grad:
        score_loss.sum().backward()
    gradients = None if scores.grad is None else scores.grad[:, 0].tolist()
    return dict(label=label, masses=masses.tolist(), native_quality=native_quality.tolist(),
                expanded_point_quality=expanded_quality.tolist(),
                max_quality_mismatch=float((native_quality - expanded_quality).abs().max()),
                native_gt_half_filter=(native_quality > .5).tolist(),
                expanded_gt_half_filter=(expanded_quality > .5).tolist(),
                native_score_gradients=gradients,
                reference_error=float((expanded_quality - independent_quality).abs().max()))


target = torch.tensor([1, 1, 1, 0])
binary = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
fixtures = [one_case(binary, target, torch.tensor(m), name) for m, name in (
    ([1, 1, 1, 1], 'uniform'), ([20, 1, 1, 1], 'heavy_first'),
    ([1, 1, 20, 1], 'heavy_third'))]
for seed in range(1999, 2031):
    g = torch.Generator().manual_seed(seed)
    masses = torch.randint(1, 40, (19,), generator=g)
    gt = torch.randint(0, 2, (19,), generator=g)
    masks = torch.randint(0, 2, (128, 19), generator=g)
    fixtures.append(one_case(masks, gt, masses, 'seed' + str(seed)))
assert fixtures[0]['max_quality_mismatch'] < 1e-7
assert fixtures[1]['max_quality_mismatch'] > .5
assert fixtures[1]['native_gt_half_filter'] == [True, True, False]
assert fixtures[1]['expanded_gt_half_filter'] == [True, False, False]

# Direct selection consumes indis, not the independent auxiliary scores branch.
logits = (binary * 2 - 1).float() * 4
indis = torch.tensor([[0., 1.], [0., 2.], [0., -2.]])
membership = torch.repeat_interleave(torch.arange(4), torch.tensor([20, 1, 1, 1]))
outs = []
for value in (-1000., 0., 1000.):
    out = dict(masks=[logits], indis=[indis], scores=[torch.full((3, 1), value)], sample_inds=[torch.arange(3)])
    outs.append(predict_by_feat(dummy, ['fixture'], [[0]], ['0'], out, membership,
        [target[membership]], [target], [torch.arange(3)]))
selector_invariant = all(torch.equal(outs[0]['pred_pmask'][0], v['pred_pmask'][0]) for v in outs[1:])
assert selector_invariant
indicator_logits = indis.clone().unsqueeze(0).requires_grad_(True)
loss = criterion.loss_indi(indicator_logits, None,
    [(torch.tensor([0]), torch.tensor([0]))], 1.)
loss.backward()
assert torch.isfinite(indicator_logits.grad).all()
selection_names = sorted({n.id for n in ast.walk(predict_node) if isinstance(n, ast.Name)})
assert 'pred_scores' not in selection_names

capture_path = Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
capture_schema = None
if capture_path.exists():
    captured = json.loads(capture_path.read_text())
    capture_schema = dict(sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                          top_keys=sorted(captured),
                          containers={k: len(v) for k, v in captured.items() if isinstance(v, list)})
    for k, v in captured.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            capture_schema[k + '_first_keys'] = sorted(v[0])

result = dict(complete=True, id='AdaRES-Y52', no_GPU=True,
    files={name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in ('loss.py', 'ipdn.py')},
    fixture_count=len(fixtures), nonuniform_mismatch_cases=sum(v['max_quality_mismatch'] > 1e-7 for v in fixtures),
    auxiliary_quality_mass_mismatch_confirmed=True,
    direct_indicator_quality_regression_premise=False,
    auxiliary_scores_not_consumed_by_native_selector=selector_invariant,
    native_indicator_is_matched_one_hot_CE=True,
    scientific_status='auxiliary_target_mismatch_confirmed_not_trained_benefit',
    limitation='Synthetic exact controls; changing the auxiliary target can affect shared queries through training, so selector invariance does NOT falsify that indirect mechanism. No native TRAIN frequency, efficacy or gate PASS is claimed.',
    rows=fixtures, cached_native_schema=capture_schema,
    resources=dict(gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits']).decode().splitlines(),
        processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits']).decode().splitlines()))
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
