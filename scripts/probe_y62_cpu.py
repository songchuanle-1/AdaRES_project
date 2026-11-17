"""CPU construction, real TRAIN collate and matched density-path controls."""

import ast
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import gorilla
import torch
from torch import nn
import yaml

assert P['physical_gpu'] is None and not torch.cuda.is_initialized()
torch.set_num_threads(2)
root = Path(P['root'])
stage = Path(P['output']).with_suffix('')
assert not stage.exists()
stage.mkdir()
row = PAYLOAD['variant']
package = stage / row['package']
package.mkdir()
for name, value in row['python_files'].items():
    data = base64.b64decode(value['base64'])
    assert hashlib.sha256(data).hexdigest() == value['sha256']
    (package / name).write_bytes(data)
config_path = stage / Path(row['config']).name
data = base64.b64decode(row['config_data']['base64'])
assert hashlib.sha256(data).hexdigest() == row['config_data']['sha256']
config_path.write_bytes(data)
assert yaml.safe_load(config_path.read_text()) == yaml.safe_load((root / PAYLOAD['parent_config']).read_text())
native = importlib.import_module('train')
assert hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['train_source_lf_sha256']
cfg = gorilla.Config.fromfile(str(config_path))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.dataloader.train.batch_size == 2 and cfg.model.sampling_module.num_proposal == 128
assert cfg.model.dec.num_layer == 6 and cfg.model.dec.iter_pred
assert cfg.data.train.training and cfg.data.train.prefix == 'train'
namespace = 'ipdn.model_paper5.' + row['package']
importlib.import_module('ipdn.model_paper5')
assert namespace not in sys.modules
spec = importlib.util.spec_from_file_location(namespace, str(package / '__init__.py'), submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[namespace] = module
spec.loader.exec_module(module)
Candidate = native.resolve_model_class(str(config_path))
assert Candidate is module.IPDN
gorilla.set_random_seed(cfg.train.seed)
model = Candidate(**cfg.model)
assert all(p.device.type == 'cpu' for p in model.parameters())
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)
dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
gorilla.set_random_seed(cfg.train.seed)
batch = dataset.collate_fn([dataset[0], dataset[1]])
assert len(batch['scan_ids']) == 2 and len(batch['ann_ids']) >= 2
assert torch.isfinite(batch['feats']).all()
model.train(True)
assert model.training and model.text_encoder.training
assert all(not p.requires_grad for p in model.text_encoder.parameters())
assert all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm1d))
parameters = sum(p.numel() for p in model.parameters())
assert parameters == 164629639
head = model.dec.indi_embedding
assert torch.equal(head.density_weight, torch.zeros(2))
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
Parent = parent_module.QueryCompetitionIndicator


def summaries(masks, padded, mass):
    probabilities = masks.sigmoid() * (~padded).unsqueeze(1).to(masks.dtype)
    weighted = probabilities * mass.unsqueeze(1)
    area = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (area.unsqueeze(-1) + area.unsqueeze(1) - intersection).clamp_min(1e-6)
    overlap = overlap.masked_fill(torch.eye(masks.shape[1], dtype=torch.bool).unsqueeze(0), 0)
    return overlap.max(-1).values, overlap.sum(-1) / max(masks.shape[1] - 1, 1)


controls = []
for dtype in (torch.float32, torch.float64):
    gorilla.set_random_seed(1999)
    parent = Parent(256).to(dtype=dtype)
    candidate = copy.deepcopy(head).to(dtype=dtype)
    candidate.load_state_dict(dict(parent.state_dict(), density_weight=torch.zeros(2, dtype=dtype)))
    query = torch.randn(2, 5, 256, dtype=dtype, requires_grad=True)
    masks = torch.randn(2, 5, 11, dtype=dtype, requires_grad=True)
    padded = torch.zeros(2, 11, dtype=torch.bool)
    padded[0, -3:] = True
    mass = torch.arange(1, 12, dtype=dtype).repeat(2, 1)
    a = parent(query, masks, padded, mass)
    b = candidate(query, masks, padded, mass)
    assert torch.equal(a, b)
    parent_grad = torch.autograd.grad(a.square().sum(), (query, masks), retain_graph=True)
    candidate_grad = torch.autograd.grad(b.square().sum(), (query, masks), retain_graph=True)
    assert all(torch.equal(x, y) for x, y in zip(parent_grad, candidate_grad))
    b[..., 1].sum().backward()
    assert candidate.density_weight.grad.abs().max() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
    with torch.no_grad():
        candidate.density_weight.copy_(torch.tensor([-0.25, 0.25], dtype=dtype))
    changed = candidate(query.detach(), masks.detach(), padded, mass)
    order = torch.tensor([3, 1, 4, 0, 2])
    permuted = candidate(query.detach()[:, order], masks.detach()[:, order], padded, mass)
    tolerance = 64 * torch.finfo(dtype).eps
    torch.testing.assert_allclose(changed[:, order], permuted, rtol=tolerance, atol=tolerance)
    padded_masks = masks.detach().clone()
    padded_masks[0, :, -3:] = 100
    assert torch.equal(changed, candidate(query.detach(), padded_masks, padded, mass))
    singleton = candidate(query.detach()[:, :1], masks.detach()[:, :1], padded, mass)
    singleton_base = parent(query.detach()[:, :1], masks.detach()[:, :1], padded, mass)
    assert torch.equal(singleton, singleton_base)
    empty = torch.ones_like(padded)
    assert torch.equal(candidate(query.detach(), masks.detach(), empty, mass), parent(query.detach(), masks.detach(), empty, mass))
    probability_a = torch.tensor([[.9,.9,.1,.1,.1,.1], [.8,.8,.1,.1,.1,.1],
        [.1,.1,.8,.1,.1,.1], [.1,.1,.1,.8,.1,.1], [.1,.1,.1,.1,.8,.1]], dtype=dtype).unsqueeze(0)
    probability_b = probability_a.clone()
    probability_b[0, 2:] = torch.tensor([[.7,.6,.1,.1,.1,.1], [.6,.7,.1,.1,.1,.1], [.65,.65,.15,.1,.1,.1]], dtype=dtype)
    witness_masks = [p.log() - (1 - p).log() for p in (probability_a, probability_b)]
    witness_query = torch.randn(1, 1, 256, dtype=dtype).repeat(1, 5, 1)
    witness_padding = torch.zeros(1, 6, dtype=torch.bool)
    witness_mass = torch.ones(1, 6, dtype=dtype)
    assert all(torch.unique(p[0], dim=0).shape[0] == 5 for p in (probability_a, probability_b))
    maximum_a, density_a = summaries(witness_masks[0], witness_padding, witness_mass)
    maximum_b, density_b = summaries(witness_masks[1], witness_padding, witness_mass)
    assert torch.equal(maximum_a[:, 0], maximum_b[:, 0])
    density_difference = float((density_b[:, 0] - density_a[:, 0]).item())
    assert density_difference > .05
    base = [parent(witness_query, m, witness_padding, witness_mass)[:, 0] for m in witness_masks]
    torch.testing.assert_allclose(base[0], base[1], rtol=tolerance, atol=tolerance)
    outputs = [candidate(witness_query, m, witness_padding, witness_mass)[:, 0] for m in witness_masks]
    expected = (density_b[:, :1] - density_a[:, :1]) * candidate.density_weight
    torch.testing.assert_allclose(outputs[1] - outputs[0], expected, rtol=tolerance, atol=tolerance)
    controls.append(dict(dtype=str(dtype), zero_anchor_output_and_input_gradients_exact=True,
        finite_parameter_gradients=True, density_weight_gradient_nonzero=True,
        permutation_padding_singleton_empty_controls=True, unique_masks_in_both_arms=True,
        focal_max_overlap_matched=True, focal_W2_output_matched_within_arithmetic=True,
        focal_density_difference=density_difference, explicit_path_response_observed=True))

# Exercise all seven native decoder stages on synthetic inputs without CUDA.
candidate = model.dec
parent = parent_module.DEC(**cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
state = candidate.state_dict()
parent.load_state_dict({k: v for k, v in state.items() if k != 'indi_embedding.density_weight'})
gorilla.set_random_seed(1999)
lengths = (512, 513)
total = sum(lengths)
inputs = dict(x=torch.randn(total, cfg.model.media), fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2, 1),
    batch_offsets=torch.tensor([0, lengths[0], total], dtype=torch.int32),
    lang_feats=torch.randn(2, 9, candidate.lang_proj.in_features), lang_masks=torch.ones(2, 9, dtype=torch.long),
    sp_pos=torch.randn(total, 3), feats_2d=torch.randn(total, candidate.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float() + 1)
inputs['lang_masks'][0, -2:] = 0
inputs['lang_feats'][0, -2:] = 0
parent.eval()
candidate.eval()
with torch.no_grad():
    baseline = parent(**inputs)
    output = candidate(**inputs)
parts = lambda value: list(value['aux_outputs']) + [value]
assert len(parts(output)) == 7
assert all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(output)) for k in ('masks', 'scores', 'indis'))
calls = []
handle = candidate.indi_embedding.register_forward_hook(lambda m, a, o: calls.append(o))
try:
    candidate.train(True)
    gorilla.set_random_seed(1999)
    output = candidate(**inputs)
    assert len(calls) == 7
    labels = torch.arange(256).reshape(2, 128).remainder(2).long()
    loss = sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1, 2), labels.reshape(-1)) + p['masks'].square().mean() for p in parts(output))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
    assert candidate.indi_embedding.density_weight.grad.abs().max() > 0
finally:
    handle.remove()
model.eval()
assert all(not m.training for m in model.modules())
model.train(True)
assert model.training and model.text_encoder.training
assert not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'],
    goal_sha256=PAYLOAD['goal_sha256'], implementation_test_passed=True,
    full_model_CPU_constructed=True, configured_TRAIN_data_collated=True,
    released_backbone_loaded=True, native_YAML_semantically_unchanged=True,
    train_eval_train_roundtrip=True, points=int(batch['feats'].shape[0]), expressions=len(batch['ann_ids']),
    parameters=parameters, additional_parameters=2, component_controls=controls,
    full_decoder_CPU_executed=True, decoder_seven_calls=7, decoder_anchor_all_stages_exact=True,
    decoder_backward_finite=True, decoder_density_gradient_nonzero=True,
    CUDA_initialized=False, native_smoke_passed=False, native_task_backward=False,
    native_sparse_forward=False, validation_used=False, admission=False,
    limitation='Actual native YAML/full CPU model/released pretrain/real TRAIN collate, plus synthetic '
        'component and full decoder tests only. Not native sparse task criterion/optimizer steps, '
        'GPU resources, population evidence or efficacy.')),
    encoding='utf-8')
