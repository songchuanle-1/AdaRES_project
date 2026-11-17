"""CPU construction, real TRAIN collate and directed-containment controls."""

import ast
import base64
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
assert parameters == 164630149
head = model.dec.indi_embedding
assert torch.equal(head.containment_weight, torch.zeros(256, 2))
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
Parent = parent_module.QueryCompetitionIndicator


controls = []
for dtype in (torch.float32, torch.float64):
    gorilla.set_random_seed(1999)
    parent = Parent(256).to(dtype=dtype)
    parent_rng = torch.get_rng_state().clone()
    gorilla.set_random_seed(1999)
    candidate = type(head)(256).to(dtype=dtype)
    assert torch.equal(torch.get_rng_state(), parent_rng)
    assert all(torch.equal(candidate.state_dict()[k], v) for k, v in parent.state_dict().items())
    candidate.load_state_dict(dict(parent.state_dict(), containment_weight=torch.zeros(256, 2, dtype=dtype)))
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
    assert candidate.containment_weight.grad.abs().max() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
    with torch.no_grad():
        candidate.containment_weight.zero_()
        candidate.containment_weight[0, 0] = .25
        candidate.containment_weight[1, 1] = -.25
    changed = candidate(query.detach(), masks.detach(), padded, mass)
    tolerance = 64 * torch.finfo(dtype).eps
    order = torch.tensor([3, 1, 4, 0, 2])
    permuted = candidate(query.detach()[:, order], masks.detach()[:, order], padded, mass)
    torch.testing.assert_allclose(changed[:, order], permuted, rtol=tolerance, atol=tolerance)
    padded_masks = masks.detach().clone()
    padded_masks[0, :, -3:] = 100
    assert torch.equal(changed, candidate(query.detach(), padded_masks, padded, mass))
    for singleton, empty, zero_mass in ((True, False, False), (False, True, False), (False, False, True)):
        q = query.detach()[:, :1] if singleton else query.detach()
        m = masks.detach()[:, :1] if singleton else masks.detach()
        pad = torch.ones_like(padded) if empty else padded
        weight = torch.zeros_like(mass) if zero_mass else mass
        assert torch.equal(candidate(q, m, pad, weight), parent(q, m, pad, weight))

    # Literal scalar loops are independent of the vectorized implementation.
    probabilities = masks.detach().sigmoid() * (~padded).unsqueeze(1).to(dtype)
    normalized = parent.query_norm(query.detach())
    literal_edges = torch.zeros(2, 5, 5, dtype=dtype)
    for batch_id in range(2):
        area = [(probabilities[batch_id, i] * mass[batch_id]).sum() for i in range(5)]
        for i in range(5):
            for j in range(5):
                if i != j:
                    inter = (probabilities[batch_id, i] * probabilities[batch_id, j] * mass[batch_id]).sum()
                    literal_edges[batch_id, i, j] = inter / area[j].clamp_min(1e-6) - inter / area[i].clamp_min(1e-6)
    literal_context = sum(literal_edges[:, :, j:j+1] * normalized[:, j:j+1] for j in range(5)) / 4
    literal = torch.stack((.25 * literal_context[..., 0], -.25 * literal_context[..., 1]), dim=-1)
    reference = parent(query.detach(), masks.detach(), padded, mass)
    torch.testing.assert_allclose(changed - reference, literal, rtol=tolerance, atol=tolerance)
    assert literal.abs().max() > 1e-6
    assert not torch.allclose(literal, -literal, rtol=tolerance, atol=tolerance)

    # Nested hard-mask fixture: focal area 10 contains two disjoint area-3 peers.
    fixtures = torch.tensor([[[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 0]]], dtype=dtype)
    fixture_masks = (fixtures * 2 - 1) * 1000
    fixture_mass = torch.tensor([[1, 2, 3, 4]], dtype=dtype)
    fixture_q = query.detach()[:1, :3]
    fixture_pad = torch.zeros(1, 4, dtype=torch.bool)
    fixture_norm = parent.query_norm(fixture_q)
    expected_edges = torch.tensor([[[0, .7, .7], [-.7, 0, 0], [-.7, 0, 0]]], dtype=dtype)
    expected_context = torch.einsum('bij,bjd->bid', expected_edges, fixture_norm) / 2
    expected = torch.stack((.25 * expected_context[..., 0], -.25 * expected_context[..., 1]), dim=-1)
    observed = candidate(fixture_q, fixture_masks, fixture_pad, fixture_mass) - parent(fixture_q, fixture_masks, fixture_pad, fixture_mass)
    torch.testing.assert_allclose(observed, expected, rtol=tolerance, atol=tolerance)
    uniform = candidate(fixture_q, fixture_masks, fixture_pad, torch.ones_like(fixture_mass)) - parent(fixture_q, fixture_masks, fixture_pad, torch.ones_like(fixture_mass))
    assert not torch.allclose(expected, uniform, rtol=tolerance, atol=tolerance)
    scaled = candidate(query.detach(), masks.detach(), padded, mass * 7)
    torch.testing.assert_allclose(changed, scaled, rtol=tolerance, atol=tolerance)
    split = candidate(query.detach(), masks.detach().repeat_interleave(2, -1),
                      padded.repeat_interleave(2, -1), mass.repeat_interleave(2, -1) / 2)
    torch.testing.assert_allclose(changed, split, rtol=tolerance, atol=tolerance)
    extreme = masks.detach().sign() * 1000
    out = candidate(query.detach(), extreme, padded, mass)
    assert torch.isfinite(out).all()
    controls.append(dict(dtype=str(dtype), zero_anchor_output_and_input_gradients_exact=True,
        finite_parameter_gradients=True, containment_weight_gradient_nonzero=True,
        permutation_padding_singleton_empty_zero_mass_controls=True, initial_RNG_matches_parent=True,
        literal_directed_formula_matched=True, reversed_direction_negative_control_detected=True,
        nested_containment_sign_and_point_mass_controls=True, mass_scale_and_superpoint_split_invariance=True,
        extreme_logit_outputs_finite=True, nonzero_residual_path_observed=True))

# Exercise all seven native decoder stages on synthetic inputs without CUDA.
candidate = model.dec
parent = parent_module.DEC(**cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
state = candidate.state_dict()
parent.load_state_dict({k: v for k, v in state.items() if k != 'indi_embedding.containment_weight'})
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
    assert candidate.indi_embedding.containment_weight.grad.abs().max() > 0
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
    parameters=parameters, additional_parameters=512, component_controls=controls,
    full_decoder_CPU_executed=True, decoder_seven_calls=7, decoder_anchor_all_stages_exact=True,
    decoder_backward_finite=True, decoder_containment_gradient_nonzero=True,
    CUDA_initialized=False, native_smoke_passed=False, native_task_backward=False,
    native_sparse_forward=False, validation_used=False, admission=False,
    limitation='Actual native YAML/full CPU model/released pretrain/real TRAIN collate, plus synthetic '
        'component and full decoder tests only. Not native sparse task criterion/optimizer steps, '
        'GPU resources, population evidence or efficacy.')),
    encoding='utf-8')
