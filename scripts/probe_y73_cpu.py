"""CPU native-model/TRAIN loading and initial-query anchor residual controls."""
import base64
import hashlib
import importlib
import importlib.util
import json
from functools import partial
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
assert len(batch['scan_ids']) == 2 and len(batch['ann_ids']) >= 2 and torch.isfinite(batch['feats']).all()
model.train(True)
assert model.training and model.text_encoder.training
assert all(not p.requires_grad for p in model.text_encoder.parameters())
assert all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm1d))
parameters = sum(p.numel() for p in model.parameters()); assert parameters == 164695173
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec'); Residual = dec_module.QueryAnchorResidual
assert torch.equal(model.dec.query_anchor_residual.weight, torch.zeros(256, 256))
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999); rng = torch.get_rng_state().clone(); branch = Residual(7).to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    query = torch.randn(2, 11, 7, dtype=dtype, requires_grad=True)
    anchor = torch.randn(2, 11, 7, dtype=dtype, requires_grad=True)
    zero = branch(query, anchor); assert torch.equal(zero, query)
    grads = torch.autograd.grad(zero.sum(), (query, anchor, branch.weight))
    assert torch.equal(grads[0], torch.ones_like(query)) and torch.equal(grads[1], torch.zeros_like(anchor))
    assert torch.isfinite(grads[2]).all() and grads[2].abs().max() > 0
    with torch.no_grad():
        branch.weight.copy_(torch.arange(49, dtype=dtype).reshape(7, 7) / 49 - .5)
        changed = branch(query, anchor); reference = torch.empty_like(changed)
        for b in range(2):
            for i in range(11):
                for d in range(7):
                    residual = sum((anchor[b, i, k].double() - query[b, i, k].double()) * branch.weight[k, d].double() for k in range(7))
                    reference[b, i, d] = query[b, i, d].double() + residual
        tol = 128 * torch.finfo(dtype).eps
        torch.testing.assert_allclose(changed, reference, rtol=tol, atol=tol)
        assert not torch.allclose(changed, query, rtol=tol, atol=tol)
        order = torch.tensor([3, 1, 10, 0, 2, 5, 4, 6, 7, 8, 9])
        assert torch.equal(branch(query[:, order], anchor[:, order]), changed[:, order])
        assert torch.equal(branch(query.flip(0), anchor.flip(0)), changed.flip(0))
        perm = torch.tensor([3, 1, 6, 0, 2, 5, 4]); other = Residual(7).to(dtype=dtype)
        other.weight.copy_(branch.weight[perm][:, perm])
        torch.testing.assert_allclose(other(query[:, :, perm], anchor[:, :, perm]), changed[:, :, perm], rtol=tol, atol=tol)
        assert torch.equal(branch(query, query), query)
        assert not torch.allclose(branch(query, anchor.flip(0)), changed, rtol=tol, atol=tol)
        assert torch.isfinite(branch(query * 1e4, anchor * 1e4)).all()
    q = query.detach().requires_grad_(); a = anchor.detach().requires_grad_()
    grads = torch.autograd.grad(branch(q, a).square().mean(), (q, a, branch.weight))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads)
    controls.append(dict(dtype=str(dtype), initial_RNG_unchanged=True, zero_output_and_input_gradients_exact=True,
        weight_gradient_finite_nonzero=True, independent_scalar_residual_formula=True,
        joint_query_batch_channel_permutations=True, equal_anchor_noop=True,
        wrong_anchor_correspondence_changes_output=True, large_finite_inputs_finite=True,
        both_input_gradients_finite_nonzero=True))

kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999); parent = parent_module.DEC(**kwargs); rng = torch.get_rng_state().clone()
torch.manual_seed(1999); decoder = dec_module.DEC(**kwargs)
assert torch.equal(rng, torch.get_rng_state()) and all(torch.equal(v, decoder.state_dict()[k]) for k, v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k: v for k, v in decoder.state_dict().items() if k != 'query_anchor_residual.weight'})
torch.manual_seed(1999); lengths = (512, 513); total = sum(lengths)
inputs = dict(x=torch.randn(total, cfg.model.media), fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2, 1),
    batch_offsets=torch.tensor([0, lengths[0], total], dtype=torch.int32),
    lang_feats=torch.randn(2, 9, decoder.lang_proj.in_features), lang_masks=torch.ones(2, 9, dtype=torch.long),
    sp_pos=torch.randn(total, 3), feats_2d=torch.randn(total, decoder.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float() + 1)
inputs['lang_masks'][0, -2:] = 0; inputs['lang_feats'][0, -2:] = 0
parts = lambda value: list(value['aux_outputs']) + [value]
parent.eval(); decoder.eval()
with torch.no_grad(): baseline = parent(**inputs); output = decoder(**inputs)
assert len(parts(output)) == 7 and all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(output)) for k in ('masks', 'scores', 'indis'))


def input_gradients(net):
    feed = dict(inputs); names = ('x', 'lang_feats', 'feats_2d')
    for key in names: feed[key] = inputs[key].clone().requires_grad_()
    value = net(**feed); loss = sum(p[k].square().mean() for p in parts(value) for k in ('masks', 'scores', 'indis'))
    return torch.autograd.grad(loss, tuple(feed[k] for k in names))


for a, b in zip(input_gradients(parent), input_gradients(decoder)):
    torch.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
current = []; outputs = []; consumed = []; indicators = []; anchors = []; masks = []; padding = []


def capture_current(index, module, args, value):
    assert index == len(current); current.append(value)


def capture_residual(module, args, value):
    index = len(outputs)
    assert args[0] is current[index] and args[1] is anchors[0] and args[1].requires_grad
    assert torch.equal(value, args[0]); outputs.append(value)


prediction = decoder.prediction_head


def checked_prediction(query, features, padded):
    index = len(consumed)
    if index == 0:
        assert not outputs; anchors.append(query); masks.append(features); padding.append(padded)
    else:
        assert query is outputs[index - 1] and features is masks[0] and padded is padding[0]
    consumed.append(index)
    return prediction(query, features, padded)


hooks = [decoder.query_anchor_residual.register_forward_hook(capture_residual),
         decoder.indi_embedding.register_forward_hook(lambda m, a, o: indicators.append(o))]
for index, layer in enumerate(decoder.qqa_ffn_layers):
    hooks.append(layer.register_forward_hook(partial(capture_current, index)))
decoder.prediction_head = checked_prediction
try:
    decoder.train(True); torch.manual_seed(1999); value = decoder(**inputs)
    assert len(outputs) == len(current) == 6 and consumed == list(range(7)) and len(indicators) == 7
    labels = torch.arange(256).reshape(2, 128).remainder(2).long()
    loss = sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1, 2), labels.reshape(-1)) + p['masks'].square().mean() for p in parts(value))
    loss.backward(); assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    grad = decoder.query_anchor_residual.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0
finally:
    decoder.prediction_head = prediction
    for hook in hooks: hook.remove()
decoder.eval()
with torch.no_grad():
    decoder.query_anchor_residual.weight.copy_(.003 * torch.eye(256)); changed = decoder(**inputs)
    assert all(torch.equal(parts(changed)[0][k], parts(baseline)[0][k]) for k in ('masks', 'scores', 'indis'))
    assert not torch.allclose(changed['masks'], baseline['masks'])
    hook = decoder.query_anchor_residual.register_forward_hook(lambda m, a, o: a[0])
    try: bypassed = decoder(**inputs)
    finally: hook.remove()
    assert all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(bypassed)) for k in ('masks', 'scores', 'indis'))
model.eval(); assert all(not m.training for m in model.modules())
model.train(True); assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'], goal_sha256=PAYLOAD['goal_sha256'],
    implementation_test_passed=True, full_model_CPU_constructed=True, configured_TRAIN_data_collated=True,
    released_backbone_loaded=True, native_YAML_semantically_unchanged=True, train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]), expressions=len(batch['ann_ids']), parameters=parameters,
    additional_parameters=65536, component_controls=controls, decoder_initial_RNG_and_parent_weights_exact=True,
    decoder_all_stages_anchor_exact=True, decoder_input_gradients_anchor_close=True, decoder_residual_calls=6,
    decoder_current_qqa_ffn_identity_verified=True, same_undetached_initial_query_identity_verified=True,
    decoder_prediction_consumption_stages=consumed, original_static_masks_and_padding_preserved=True,
    decoder_indicator_calls=7, decoder_backward_finite=True, decoder_residual_gradient_finite_nonzero=True,
    nonzero_residual_preserves_initial_changes_final_masks=True, bypassed_residual_negative_control_exact=True,
    CUDA_initialized=False, native_smoke_passed=False, native_task_backward=False, native_sparse_forward=False,
    validation_used=False, admission=False,
    limitation='CPU native full-model/pretrain/TRAIN collation plus synthetic component/decoder checks only; no sparse native task loss/backward/optimizer, GPU PASS or trained efficacy.')), encoding='utf-8')
