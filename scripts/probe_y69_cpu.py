"""CPU model/TRAIN loading and direct language mask modulation execution controls."""
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
parameters=sum(p.numel() for p in model.parameters());assert parameters==164629661
parent_path=root/'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==PAYLOAD['parent_dec_lf_sha256']
parent_module=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')

dec_module = importlib.import_module(namespace + '.dec')
Geometry = dec_module.BoundedGeometryBias
assert torch.equal(model.dec.coarse_geometry_bias.weight, torch.zeros(8, 3))
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999); rng = torch.get_rng_state().clone(); branch = Geometry(8).to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    q = torch.randn(2, 5, 3, dtype=dtype, requires_grad=True)
    s = torch.randn(2, 11, 3, dtype=dtype, requires_grad=True)
    padded = torch.zeros(2, 11, dtype=torch.bool); padded[0, -2:] = True
    zero = branch(q, s, padded); assert torch.equal(zero, torch.zeros_like(zero))
    qg, sg, wg = torch.autograd.grad(zero.sum(), (q, s, branch.weight))
    assert torch.equal(qg, torch.zeros_like(q)) and torch.equal(sg, torch.zeros_like(s))
    assert torch.isfinite(wg).all() and wg.abs().max() > 0
    with torch.no_grad():
        branch.weight.copy_(torch.arange(24, dtype=dtype).reshape(8, 3) / 24 - .5)
        value = branch(q, s, padded).reshape(2, 8, 5, 11)
        reference = torch.zeros(2, 8, 5, 11, dtype=torch.float64)
        for b in range(2):
            for h in range(8):
                for i in range(5):
                    for j in range(11):
                        if not padded[b, j]:
                            delta = (q[b, i].double() - s[b, j].double()).abs()
                            reference[b, h, i, j] = ((delta / (1 + delta)).square() * branch.weight[h].double().tanh()).sum()
        tol = 128 * torch.finfo(dtype).eps
        torch.testing.assert_allclose(value.double(), reference, rtol=tol, atol=tol)
        assert value.abs().max() > 0 and value.abs().max() < 3
        shifted = branch(q + 3., s + 3., padded).reshape_as(value)
        torch.testing.assert_allclose(shifted, value, rtol=tol, atol=tol)
        assert torch.equal(branch(-q, -s, padded).reshape_as(value), value)
        order = torch.tensor([3, 1, 6, 0, 2, 5, 4, 7, 10, 8, 9])
        torch.testing.assert_allclose(branch(q, s[:, order], padded[:, order]).reshape_as(value), value[..., order], rtol=tol, atol=tol)
        perm = torch.tensor([4, 1, 0, 3, 2])
        torch.testing.assert_allclose(branch(q[:, perm], s, padded).reshape_as(value), value[:, :, perm], rtol=tol, atol=tol)
        noisy = s.clone(); noisy[padded] = float('nan')
        assert torch.equal(branch(q, noisy, padded).reshape_as(value), value)
        assert torch.equal(branch(q, noisy, torch.ones_like(padded)), torch.zeros_like(zero))
        assert torch.isfinite(branch(q * 1e10, s * 1e10, padded)).all()
        branch.weight.zero_(); branch.weight[0, 0] = .3
        isolated = branch(q, s, padded).reshape_as(value)
        assert isolated[:, 0].abs().max() > 0 and torch.equal(isolated[:, 1:], torch.zeros_like(isolated[:, 1:]))
    q2 = q.detach().requires_grad_(); s2 = s.detach().requires_grad_()
    value = branch(q2, s2, padded)
    grads = torch.autograd.grad((value * torch.randn_like(value)).sum(), (q2, s2, branch.weight))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads)
    assert torch.equal(grads[1][padded], torch.zeros_like(grads[1][padded]))
    controls.append(dict(dtype=str(dtype),zero_bias_and_coordinate_gradients_exact=True,
        initial_RNG_unchanged=True,new_gradient_finite_nonzero=True,independent_axis_formula=True,
        translation_and_reflection_invariant=True,query_key_permutation_equivariant=True,
        padding_NaNs_ignored_and_all_padding_zero=True,head_specific_parameter_isolation=True,
        nonzero_coordinate_gradients_finite=True,extreme_finite_coordinates_bounded=True))

# Exercise the actual custom attention, including its hard exclusions.
torch.manual_seed(1999); old_attention = parent_module.CrossAttentionLayer(16, 4)
torch.manual_seed(1999); new_attention = dec_module.GeometryCrossAttentionLayer(16, 4)
assert all(torch.equal(v, new_attention.state_dict()[k]) for k, v in old_attention.state_dict().items())
src = torch.randn(2, 11, 16); query = torch.randn(2, 5, 16)
padding = torch.zeros(2, 11, dtype=torch.bool); padding[0, -2:] = True
gate = torch.zeros(2, 5, 11, dtype=torch.bool); gate[:, :, 2] = True
zero = torch.zeros(8, 5, 11)
base_attention = old_attention(src, query, padding, gate)
anchor_attention = new_attention(src, query, padding, gate, attn_bias=zero)
assert all(torch.equal(a, b) for a, b in zip(base_attention, anchor_attention))
bias = torch.randn_like(zero) * .1
changed_attention = new_attention(src, query, padding, gate, attn_bias=bias)
excluded = gate | padding[:, None]
assert torch.equal(changed_attention[1][excluded], torch.zeros_like(changed_attention[1][excluded]))
expected = base_attention[2] + bias
assert torch.equal(torch.isneginf(expected), torch.isneginf(changed_attention[2]))
finite = torch.isfinite(expected)
torch.testing.assert_allclose(changed_attention[2][finite], expected[finite], rtol=1e-6, atol=1e-6)
assert not torch.allclose(changed_attention[0], base_attention[0])

kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999); parent = parent_module.DEC(**kwargs); rng = torch.get_rng_state().clone()
torch.manual_seed(1999); decoder = dec_module.DEC(**kwargs)
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, decoder.state_dict()[k]) for k, v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k: v for k, v in decoder.state_dict().items() if k != 'coarse_geometry_bias.weight'})
torch.manual_seed(1999); lengths = (512, 513); total = sum(lengths)
inputs = dict(x=torch.randn(total, cfg.model.media), fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2, 1),
    batch_offsets=torch.tensor([0, lengths[0], total], dtype=torch.int32),
    lang_feats=torch.randn(2, 9, decoder.lang_proj.in_features), lang_masks=torch.ones(2, 9, dtype=torch.long),
    sp_pos=torch.randn(total, 3), feats_2d=torch.randn(total, decoder.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float() + 1)
inputs['lang_masks'][0, -2:] = 0; inputs['lang_feats'][0, -2:] = 0
parts = lambda v: list(v['aux_outputs']) + [v]
parent.eval(); decoder.eval()
with torch.no_grad(): baseline = parent(**inputs); output = decoder(**inputs)
assert len(parts(output)) == 7
assert all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(output)) for k in ('masks', 'scores', 'indis'))
def input_gradients(net):
    feed = dict(inputs); names = ('x', 'lang_feats', 'feats_2d')
    for key in names: feed[key] = inputs[key].clone().requires_grad_()
    value = net(**feed)
    loss = sum(part[k].square().mean() for part in parts(value) for k in ('masks', 'scores', 'indis'))
    return torch.autograd.grad(loss, tuple(feed[k] for k in names))
for a, b in zip(input_gradients(parent), input_gradients(decoder)):
    torch.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
bias_outputs = []; consumed = []; indicators = []; originals = []
hooks = [decoder.coarse_geometry_bias.register_forward_hook(lambda m, a, o: bias_outputs.append(o)),
    decoder.indi_embedding.register_forward_hook(lambda m, a, o: indicators.append(o))]
for index, layer in enumerate(decoder.qia_layers):
    original = layer.forward; originals.append(original)
    def checked(*args, _original=original, _index=index, **kwargs):
        assert kwargs['attn_bias'] is bias_outputs[-1]
        assert args[3] is None or (args[3].dtype == torch.bool and not args[3].requires_grad)
        consumed.append(_index)
        return _original(*args, **kwargs)
    layer.forward = checked
try:
    decoder.train(True); torch.manual_seed(1999); value = decoder(**inputs)
    assert len(bias_outputs) == 1 and consumed == list(range(6)) and len(indicators) == 7
    labels = torch.arange(256).reshape(2, 128).remainder(2).long()
    loss = sum(torch.nn.functional.cross_entropy(part['indis'].reshape(-1, 2), labels.reshape(-1)) +
        part['masks'].square().mean() for part in parts(value))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    gradient = decoder.coarse_geometry_bias.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
finally:
    for layer, original in zip(decoder.qia_layers, originals): layer.forward = original
    for h in hooks: h.remove()
decoder.eval()
with torch.no_grad():
    decoder.coarse_geometry_bias.weight.copy_(torch.arange(24).reshape(8, 3) / 24 - .5)
    changed = decoder(**inputs)
    assert all(torch.equal(parts(changed)[0][k], parts(baseline)[0][k]) for k in ('masks', 'scores', 'indis'))
    assert not torch.allclose(changed['masks'], baseline['masks'])
    h = decoder.coarse_geometry_bias.register_forward_hook(lambda m, a, o: torch.zeros_like(o))
    try: bypassed = decoder(**inputs)
    finally: h.remove()
    assert all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(bypassed)) for k in ('masks', 'scores', 'indis'))
model.eval(); assert all(not m.training for m in model.modules())
model.train(True); assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'], goal_sha256=PAYLOAD['goal_sha256'],
    implementation_test_passed=True, full_model_CPU_constructed=True, configured_TRAIN_data_collated=True, released_backbone_loaded=True,
    native_YAML_semantically_unchanged=True, train_eval_train_roundtrip=True, points=int(batch['feats'].shape[0]), expressions=len(batch['ann_ids']),
    parameters=parameters, additional_parameters=24, component_controls=controls,
    actual_attention_zero_anchor_exact=True, actual_hard_exclusions_preserved=True, actual_attention_bias_formula_verified=True,
    decoder_initial_RNG_and_parent_weights_exact=True, decoder_all_stages_anchor_exact=True,
    decoder_input_gradients_anchor_close=True, decoder_bias_calls=1, decoder_coarse_consumption_calls=6, decoder_indicator_calls=7,
    decoder_backward_finite=True, decoder_geometry_gradient_finite_nonzero=True,
    nonzero_geometry_changes_final_masks=True, initial_prediction_unchanged=True, bypassed_geometry_negative_control_exact=True,
    CUDA_initialized=False, native_smoke_passed=False, native_task_backward=False, native_sparse_forward=False, validation_used=False, admission=False,
    limitation='CPU native YAML/full model/pretrain/real TRAIN collate plus synthetic component/custom-attention/decoder controls only. Actual sparse TRAIN loss/backward/optimizer and trained efficacy remain untested.')), encoding='utf-8')
