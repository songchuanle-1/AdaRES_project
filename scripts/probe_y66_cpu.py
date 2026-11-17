"""CPU model/TRAIN loading and explicit peer-state feedback execution controls."""
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
assert parameters == 164695173
assert torch.equal(model.dec.peer_feedback.weight, torch.zeros(256, 256))
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
Feedback = dec_module.PeerStateFeedback

controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    before = torch.get_rng_state().clone()
    branch = Feedback(256).to(dtype=dtype)
    assert torch.equal(before, torch.get_rng_state())
    q = torch.randn(2, 5, 256, dtype=dtype, requires_grad=True)
    masks = torch.randn(2, 5, 7, dtype=dtype, requires_grad=True)
    padding = torch.zeros(2, 7, dtype=torch.bool)
    padding[0, -2:] = True
    masses = torch.arange(1, 15, dtype=dtype).reshape(2, 7)
    output = branch(q, masks, padding, masses)
    assert torch.equal(output, q)
    qgrad, mgrad = torch.autograd.grad(output.square().sum(), (q, masks), retain_graph=True)
    assert torch.equal(qgrad, 2 * q) and torch.equal(mgrad, torch.zeros_like(masks))
    output.square().sum().backward()
    assert torch.isfinite(branch.weight.grad).all() and branch.weight.grad.abs().max() > 0
    with torch.no_grad():
        branch.weight.copy_(.02 * torch.eye(256, dtype=dtype))
    changed = branch(q.detach(), masks.detach(), padding, masses)
    # Independent scalar point-mass IoU and peer-sum formula.
    probs = masks.detach().sigmoid()
    literal = q.detach().clone()
    for b in range(2):
        area = [sum(probs[b, i, n] * masses[b, n] for n in range(7) if not padding[b, n])
                for i in range(5)]
        for i in range(5):
            edges = []
            for j in range(5):
                inter = sum(probs[b, i, n] * probs[b, j, n] * masses[b, n]
                            for n in range(7) if not padding[b, n])
                edges.append(inter / (area[i] + area[j] - inter).clamp_min(1e-6) if i != j
                             else torch.zeros((), dtype=dtype))
            context = sum(edges[j] * q.detach()[b, j] for j in range(5)) / sum(edges).clamp_min(1e-6)
            literal[b, i] += context @ branch.weight.detach()
    tol = 64 * torch.finfo(dtype).eps
    torch.testing.assert_allclose(changed, literal, rtol=tol, atol=tol)
    assert not torch.equal(changed, q.detach())
    order = torch.tensor([3, 1, 4, 0, 2])
    torch.testing.assert_allclose(branch(q.detach()[:, order], masks.detach()[:, order], padding, masses),
                                  changed[:, order], rtol=tol, atol=tol)
    altered = masks.detach().clone()
    altered[0, :, -2:] = 1000.
    assert torch.equal(branch(q.detach(), altered, padding, masses), changed)
    torch.testing.assert_allclose(branch(q.detach(), masks.detach(), padding, masses * 11),
                                  changed, rtol=tol, atol=tol)
    split_masks = torch.cat([masks.detach()[:, :, :1], masks.detach()], dim=-1)
    split_padding = torch.cat([padding[:, :1], padding], dim=-1)
    split_mass = torch.cat([masses[:, :1] / 2, masses], dim=-1)
    split_mass[:, 1] /= 2
    torch.testing.assert_allclose(branch(q.detach(), split_masks, split_padding, split_mass),
                                  changed, rtol=tol, atol=tol)
    assert torch.equal(branch(q.detach()[:, :1], masks.detach()[:, :1], padding, masses), q.detach()[:, :1])
    assert torch.equal(branch(q.detach(), masks.detach(), padding, torch.zeros_like(masses)), q.detach())
    assert torch.equal(branch(q.detach(), masks.detach(), torch.ones_like(padding), masses), q.detach())
    wrong = branch(q.detach(), masks.detach().roll(1, 1), padding, masses)
    assert not torch.allclose(changed, wrong, rtol=tol, atol=tol)
    q2, m2 = q.detach().requires_grad_(), masks.detach().requires_grad_()
    objective = branch(q2, m2, padding, masses).square().sum()
    grads = torch.autograd.grad(objective, (q2, m2, branch.weight))
    assert all(torch.isfinite(g).all() for g in grads) and grads[1].abs().max() > 0
    extreme = masks.detach().sign() * 1000.
    assert torch.isfinite(branch(q.detach(), extreme, padding, masses)).all()
    controls.append(dict(dtype=str(dtype), zero_output_and_input_gradients_exact=True,
        initial_RNG_unchanged=True, feedback_weight_gradient_finite_nonzero=True,
        scalar_peer_context_formula_matched=True, query_permutation_equivariant=True,
        padded_points_ignored_exactly=True, point_mass_scale_invariant=True,
        superpoint_split_invariant=True, singleton_zero_mass_all_padding_noop=True,
        wrong_query_mask_correspondence_detected=True,
        nonzero_response_and_mask_gradient=True, extreme_logits_finite=True))

# Full decoder construction: zero-only additional parameters consume no RNG.
kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999)
parent = parent_module.DEC(**kwargs)
parent_rng = torch.get_rng_state().clone()
torch.manual_seed(1999)
decoder = dec_module.DEC(**kwargs)
assert torch.equal(torch.get_rng_state(), parent_rng)
assert all(torch.equal(v, decoder.state_dict()[k]) for k, v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k: v for k, v in decoder.state_dict().items() if k != 'peer_feedback.weight'})
torch.manual_seed(1999)
lengths = (512, 513)
total = sum(lengths)
inputs = dict(x=torch.randn(total, cfg.model.media), fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2, 1),
    batch_offsets=torch.tensor([0, lengths[0], total], dtype=torch.int32),
    lang_feats=torch.randn(2, 9, decoder.lang_proj.in_features), lang_masks=torch.ones(2, 9, dtype=torch.long),
    sp_pos=torch.randn(total, 3), feats_2d=torch.randn(total, decoder.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float() + 1)
inputs['lang_masks'][0, -2:] = 0
inputs['lang_feats'][0, -2:] = 0
parts = lambda value: list(value['aux_outputs']) + [value]
parent.eval()
decoder.eval()
with torch.no_grad():
    baseline = parent(**inputs)
    output = decoder(**inputs)
assert len(parts(output)) == 7
assert all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(output)) for k in ('masks', 'scores', 'indis'))
def input_gradients(net):
    feed = dict(inputs)
    names = ('x', 'lang_feats', 'feats_2d')
    for key in names: feed[key] = inputs[key].clone().requires_grad_()
    value = net(**feed)
    loss = sum(part[k].square().mean() for part in parts(value) for k in ('masks', 'scores', 'indis'))
    return torch.autograd.grad(loss, tuple(feed[k] for k in names))
g1, g2 = input_gradients(parent), input_gradients(decoder)
for a, b in zip(g1, g2): torch.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
feedback_calls, indicator_calls = [], []
h1 = decoder.peer_feedback.register_forward_hook(lambda m, a, o: feedback_calls.append(o))
h2 = decoder.indi_embedding.register_forward_hook(lambda m, a, o: indicator_calls.append(o))
try:
    decoder.train(True)
    torch.manual_seed(1999)
    output = decoder(**inputs)
    assert len(feedback_calls) == 6 and len(indicator_calls) == 7
    labels = torch.arange(256).reshape(2, 128).remainder(2).long()
    loss = sum(torch.nn.functional.cross_entropy(part['indis'].reshape(-1, 2), labels.reshape(-1))
               + part['masks'].square().mean() for part in parts(output))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    grad = decoder.peer_feedback.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0
finally:
    h1.remove()
    h2.remove()
decoder.eval()
with torch.no_grad():
    decoder.peer_feedback.weight.copy_(.02 * torch.eye(256))
    changed = decoder(**inputs)
    assert torch.equal(parts(changed)[0]['masks'], parts(baseline)[0]['masks'])
    assert not torch.allclose(changed['masks'], baseline['masks'])
    hook = decoder.peer_feedback.register_forward_hook(lambda m, a, o: a[0])
    try: wrong = decoder(**inputs)
    finally: hook.remove()
    assert all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(wrong)) for k in ('masks', 'scores', 'indis'))
model.eval()
assert all(not m.training for m in model.modules())
model.train(True)
assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'],
    goal_sha256=PAYLOAD['goal_sha256'], implementation_test_passed=True,
    full_model_CPU_constructed=True, configured_TRAIN_data_collated=True,
    released_backbone_loaded=True, native_YAML_semantically_unchanged=True,
    train_eval_train_roundtrip=True, points=int(batch['feats'].shape[0]), expressions=len(batch['ann_ids']),
    parameters=parameters, additional_parameters=65536, component_controls=controls,
    decoder_zero_anchor_all_stages_exact=True, decoder_initial_RNG_and_parent_weights_exact=True,
    decoder_input_gradients_anchor_close=True, decoder_feedback_calls=6, decoder_indicator_calls=7,
    decoder_backward_finite=True, decoder_feedback_gradient_nonzero=True,
    nonzero_feedback_changes_refinement=True, bypassed_feedback_negative_control_exact=True,
    CUDA_initialized=False, native_smoke_passed=False, native_task_backward=False,
    native_sparse_forward=False, validation_used=False, admission=False,
    limitation='Actual CPU YAML/full model/released backbone/real TRAIN collation plus '
        'synthetic component/decoder execution and gradients. No native sparse task criterion/'
        'optimizer steps, GPU admission or trained efficacy.')), encoding='utf-8')
