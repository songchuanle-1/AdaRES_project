"""CPU model/TRAIN loading and query-conditioned scale fusion controls."""
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
parameters=sum(p.numel() for p in model.parameters());assert parameters==164695173
parent_path=root/'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==PAYLOAD['parent_dec_lf_sha256']
parent_module=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
Fusion = dec_module.QueryScaleFusion
assert torch.equal(model.dec.query_scale_fusion.weight, torch.zeros(256, 256))
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999); rng = torch.get_rng_state().clone(); branch = Fusion(7).to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    q, c, f = [torch.randn(2, 5, 7, dtype=dtype, requires_grad=True) for _ in range(3)]
    zero = branch(q, c, f); anchor = (c + f) / 2
    assert torch.equal(zero, anchor)
    grads = torch.autograd.grad(zero.sum(), (q, c, f, branch.weight))
    assert torch.equal(grads[0], torch.zeros_like(q))
    assert torch.equal(grads[1], .5 * torch.ones_like(c)) and torch.equal(grads[2], .5 * torch.ones_like(f))
    assert torch.isfinite(grads[3]).all() and grads[3].abs().max() > 0
    with torch.no_grad():
        branch.weight.copy_(torch.arange(49, dtype=dtype).reshape(7, 7) / 49 - .5)
        value = branch(q, c, f)
        reference = torch.empty_like(value)
        for b in range(2):
            for i in range(5):
                for d in range(7):
                    z = sum(q[b, i, k].double() * branch.weight[k, d].double() for k in range(7))
                    alpha = (1 + z.tanh()) / 2
                    reference[b, i, d] = (1-alpha)*c[b, i, d].double() + alpha*f[b, i, d].double()
        tol = 128 * torch.finfo(dtype).eps
        torch.testing.assert_allclose(value, reference, rtol=tol, atol=tol)
        assert not torch.allclose(value, anchor, rtol=tol, atol=tol)
        assert (value >= torch.minimum(c, f) - tol).all() and (value <= torch.maximum(c, f) + tol).all()
        order = torch.tensor([3, 1, 4, 0, 2])
        assert torch.equal(branch(q[:, order], c[:, order], f[:, order]), value[:, order])
        assert torch.equal(branch(q.flip(0), c.flip(0), f.flip(0)), value.flip(0))
        perm = torch.tensor([3, 1, 6, 0, 2, 5, 4])
        rotated = Fusion(7).to(dtype=dtype)
        rotated.weight.copy_(branch.weight[perm][:, perm])
        torch.testing.assert_allclose(rotated(q[:, :, perm], c[:, :, perm], f[:, :, perm]), value[:, :, perm], rtol=tol, atol=tol)
        torch.testing.assert_allclose(branch(q, c, c), c, rtol=tol, atol=tol)
        reverse = Fusion(7).to(dtype=dtype); reverse.weight.copy_(-branch.weight)
        torch.testing.assert_allclose(reverse(q, f, c), value, rtol=tol, atol=tol)
        wrong = branch(q.flip(1), c, f)
        assert not torch.allclose(wrong, value, rtol=tol, atol=tol)
        assert torch.isfinite(branch(q * 1e6, c, f)).all()
    q2, c2, f2 = [v.detach().requires_grad_() for v in (q, c, f)]
    live = torch.autograd.grad(branch(q2, c2, f2).square().mean(), (q2, c2, f2, branch.weight))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in live)
    controls.append(dict(dtype=str(dtype),initial_RNG_unchanged=True,zero_average_and_input_gradients_exact=True,
        weight_gradient_finite_nonzero=True,independent_scalar_convex_formula=True,bounded_between_branches=True,
        joint_query_batch_channel_permutations=True,equal_branch_noop=True,branch_swap_with_negative_weight=True,
        wrong_query_correspondence_changes_output=True,extreme_query_finite=True,all_input_gradients_finite_nonzero=True))

kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999); parent = parent_module.DEC(**kwargs); rng = torch.get_rng_state().clone()
torch.manual_seed(1999); decoder = dec_module.DEC(**kwargs)
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if k != 'query_scale_fusion.weight'})
torch.manual_seed(1999); lengths = (512, 513); total = sum(lengths)
inputs = dict(x=torch.randn(total,cfg.model.media),fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2,1),
    batch_offsets=torch.tensor([0,lengths[0],total],dtype=torch.int32),
    lang_feats=torch.randn(2,9,decoder.lang_proj.in_features),lang_masks=torch.ones(2,9,dtype=torch.long),
    sp_pos=torch.randn(total,3),feats_2d=torch.randn(total,decoder.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float()+1)
inputs['lang_masks'][0,-2:]=0;inputs['lang_feats'][0,-2:]=0
parts = lambda v: list(v['aux_outputs']) + [v]
parent.eval(); decoder.eval()
with torch.no_grad(): baseline = parent(**inputs); output = decoder(**inputs)
assert len(parts(output)) == 7
assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(output)) for k in ('masks','scores','indis'))

def input_gradients(net):
    feed = dict(inputs); names = ('x','lang_feats','feats_2d')
    for key in names: feed[key] = inputs[key].clone().requires_grad_()
    value = net(**feed)
    loss = sum(p[k].square().mean() for p in parts(value) for k in ('masks','scores','indis'))
    return torch.autograd.grad(loss,tuple(feed[k] for k in names))
for a,b in zip(input_gradients(parent),input_gradients(decoder)):
    torch.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
coarse=[]; fine=[]; previous_queries=[]; fused=[]; consumed=[]; indicators=[]; hooks=[]
def capture_coarse(module,args,value):
    previous_queries.append(args[1]); coarse.append(value[0])
def capture_fine(module,args,value): fine.append(value[0])
def capture_fusion(module,args,value):
    i=len(fused)
    assert args[0] is previous_queries[i] and args[1] is coarse[i] and args[2] is fine[i]
    assert torch.equal(value,(coarse[i]+fine[i])/2)
    fused.append(value)
def consume(module,args):
    assert args[1] is fused[-1]; consumed.append(len(fused)-1)
for layer in decoder.qia_layers: hooks.append(layer.register_forward_hook(capture_coarse))
for layer in decoder.qia_layers_fine: hooks.append(layer.register_forward_hook(capture_fine))
for layer in decoder.qla_layers: hooks.append(layer.register_forward_pre_hook(consume))
hooks.extend([decoder.query_scale_fusion.register_forward_hook(capture_fusion),decoder.indi_embedding.register_forward_hook(lambda m,a,o: indicators.append(o))])
try:
    decoder.train(True); torch.manual_seed(1999); value=decoder(**inputs)
    assert len(fused)==6 and consumed==list(range(6)) and len(indicators)==7
    labels=torch.arange(256).reshape(2,128).remainder(2).long()
    loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    grad=decoder.query_scale_fusion.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
finally:
    for h in hooks: h.remove()
decoder.eval()
with torch.no_grad():
    decoder.query_scale_fusion.weight.copy_(.003*torch.eye(256)); changed=decoder(**inputs)
    assert all(torch.equal(parts(changed)[0][k],parts(baseline)[0][k]) for k in ('masks','scores','indis'))
    assert not torch.allclose(changed['masks'],baseline['masks'])
    h=decoder.query_scale_fusion.register_forward_hook(lambda m,a,o:(a[1]+a[2])/2)
    try: bypassed=decoder(**inputs)
    finally: h.remove()
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(bypassed)) for k in ('masks','scores','indis'))
model.eval(); assert all(not m.training for m in model.modules())
model.train(True); assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],
    implementation_test_passed=True,full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,
    native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),
    parameters=parameters,additional_parameters=65536,component_controls=controls,decoder_initial_RNG_and_parent_weights_exact=True,
    decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,decoder_fusion_calls=6,
    decoder_consumption_stage_calls=consumed,decoder_indicator_calls=7,decoder_backward_finite=True,
    decoder_fusion_gradient_finite_nonzero=True,nonzero_fusion_preserves_initial_and_changes_final_masks=True,
    bypassed_fusion_negative_control_exact=True,CUDA_initialized=False,native_smoke_passed=False,
    native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU native YAML/full model/released backbone/actual TRAIN collate plus synthetic fusion and decoder controls only; no sparse TRAIN criterion/optimizer, GPU native PASS or efficacy.')),encoding='utf-8')
