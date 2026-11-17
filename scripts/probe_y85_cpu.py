"""CPU original model/TRAIN loading and shared visual mask-feature normalization controls."""
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


parameters=sum(p.numel() for p in model.parameters());assert parameters==164630149
parent_path=root/'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==PAYLOAD['parent_dec_lf_sha256']
parent_module=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module=importlib.import_module(namespace+'.dec')
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();norm=nn.LayerNorm(7).to(dtype=dtype)
    assert torch.equal(rng,torch.get_rng_state()) and norm.eps==model.dec.mask_feature_norm.eps==1e-5
    x=torch.randn(2,5,7,dtype=dtype,requires_grad=True);value=norm(x);xd=x.double()
    mean=xd.mean(-1,keepdim=True);var=((xd-mean)**2).mean(-1,keepdim=True)
    reference=(xd-mean)/(var+norm.eps).sqrt()*norm.weight.double()+norm.bias.double()
    tol=128*torch.finfo(dtype).eps;torch.testing.assert_allclose(value.double(),reference,rtol=tol,atol=tol)
    order=torch.tensor([3,1,4,0,2]);torch.testing.assert_allclose(norm(x[:,order]),value[:,order],rtol=tol,atol=tol)
    perm=torch.tensor([3,1,6,0,2,5,4]);torch.testing.assert_allclose(norm(x[:,:,perm]),value[:,:,perm],rtol=tol,atol=tol)
    torch.testing.assert_allclose(norm(x+3.),value,rtol=tol,atol=tol)
    assert torch.isfinite(norm(x*1e4)).all()
    ((value*torch.arange(1,8,dtype=dtype))**2).mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().max()>0 for t in (x,norm.weight,norm.bias))
    with torch.no_grad():
        original=norm.bias.clone();norm.bias[0]+=.25;changed=norm(x)
        assert not torch.equal(changed,value);norm.bias.copy_(original);assert torch.equal(norm(x),value)
    controls.append(dict(dtype=str(dtype),no_RNG_and_original_epsilon=True,independent_scalar_affine_LayerNorm=True,point_and_channel_permutations=True,channel_shift_invariance=True,large_finite_inputs_finite=True,input_and_affine_gradients_finite_nonzero=True,affine_intervention_and_restoration=True))
kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999); parent = parent_module.DEC(**kwargs); rng = torch.get_rng_state().clone()
torch.manual_seed(1999); decoder = dec_module.DEC(**kwargs)
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if not k.startswith('mask_feature_norm.')})
torch.manual_seed(1999); lengths = (512, 513); total = sum(lengths)
inputs = dict(x=torch.randn(total,cfg.model.media),fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2,1),
    batch_offsets=torch.tensor([0,lengths[0],total],dtype=torch.int32),
    lang_feats=torch.randn(2,9,decoder.lang_proj.in_features),lang_masks=torch.ones(2,9,dtype=torch.long),
    sp_pos=torch.randn(total,3),feats_2d=torch.randn(total,decoder.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float()+1)
inputs['lang_masks'][0,-2:]=0;inputs['lang_feats'][0,-2:]=0
parts = lambda v: list(v['aux_outputs']) + [v]
parent.eval(); decoder.eval()
norm_forward=decoder.mask_feature_norm.forward;decoder.mask_feature_norm.forward=lambda x:x
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


parent.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
decoder.mask_feature_norm.forward=norm_forward
with torch.no_grad():
    active=decoder(**inputs)
    assert torch.equal(active['aux_outputs'][0]['scores'],baseline['aux_outputs'][0]['scores'])
    assert not torch.allclose(active['aux_outputs'][0]['masks'],baseline['aux_outputs'][0]['masks'])
    decoder.mask_feature_norm.forward=lambda x:x;restored=decoder(**inputs)
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(restored)) for k in ('masks','scores','indis'))
    decoder.mask_feature_norm.forward=norm_forward
model.dec=decoder;optimizer=gorilla.build_optimizer(model,cfg.optimizer)
norm=model.dec.mask_feature_norm;head=model.dec.indi_embedding
step_rows=[];path_rows=[];state={};original_prediction=model.dec.prediction_head;original_batch=model.dec.get_batches

def capture_mask_raw(module,args,value):
    assert 'raw' not in state and value.requires_grad;value.retain_grad();state['raw']=value

def capture_mask_norm(module,args,value):
    assert len(args)==1 and args[0] is state['raw'] and value.requires_grad and 'normalized' not in state
    assert module is norm and module.eps==1e-5 and tuple(module.normalized_shape)==(256,)
    reference=torch.nn.functional.layer_norm(args[0].double(),module.normalized_shape,module.weight.double(),module.bias.double(),module.eps)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    value.retain_grad();state['normalized']=value
    path_rows.append(dict(step=len(step_rows),original_x_mask_output_identity=True,all_channels_and_original_epsilon=True,input_undetached=True,float64_LayerNorm_reference=True,shared_module_identity=True))

def tracked_batch(*args,**kwargs):
    if len(state['batches'])==1:assert args[0] is state['normalized']
    result=original_batch(*args,**kwargs);state['batches'].append(result);return result

def checked_prediction(q,features,pad):
    assert len(state['batches'])==4 and features is state['batches'][1][0] and pad is state['batches'][0][1]
    assert torch.count_nonzero(features[pad])==0
    state['query']=q;state['predictions']+=1;value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1);expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad;state['gate']=value[2];return value

def before_indicator(module,args):
    assert args[0] is state['query'] and args[1] is state['mask'];state['indicators']+=1

def capture_query_norm(module,args,value):
    assert args[0] is state['query'] and args[0].requires_grad;state['query_norms']+=1

def before_coarse(module,args):
    assert module is model.dec.qia_layers[state['coarse_calls']] and args[3] is state['gate'];state['coarse_calls']+=1

hooks=[model.dec.x_mask.register_forward_hook(capture_mask_raw),norm.register_forward_hook(capture_mask_norm),head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
hooks.extend(m.register_forward_pre_hook(before_coarse) for m in model.dec.qia_layers)
model.dec.get_batches=tracked_batch;model.dec.prediction_head=checked_prediction
try:
    model.train(True)
    for index in range(2):
        state.clear();state.update(batches=[],predictions=0,indicators=0,query_norms=0,coarse_calls=0)
        optimizer.zero_grad();torch.manual_seed(1999+index);value=model.dec(**inputs)
        assert state['predictions']==state['indicators']==state['query_norms']==7 and state['coarse_calls']==6
        labels=torch.arange(256).reshape(2,128).remainder(2).long()
        loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
        assert torch.isfinite(loss);loss.backward()
        active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
        grads=[p.grad for p in norm.parameters()];assert len(grads)==2 and all(g is not None and torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (state['raw'],state['normalized']))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        before=[p.detach().clone() for p in norm.parameters()];optimizer.step();updates=[float((p-b).abs().max()) for p,b in zip(norm.parameters(),before)];assert all(u>0 for u in updates)
        step_rows.append(dict(step=index,synthetic_CPU_decoder_loss=float(loss),new_gradient_maxima=[float(g.abs().max()) for g in grads],parameter_updates=updates,prediction_calls=7,indicator_calls=7,query_norm_calls=7,coarse_gate_calls=6,raw_and_normalized_gradients_finite=True))
finally:
    model.dec.get_batches=original_batch;model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
model.eval();assert all(not m.training for m in model.modules());model.train(True)
assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=512,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,norm_bypass_all_stages_W2_anchor_exact=True,norm_bypass_input_gradients_W2_anchor_close=True,
    active_transform_changes_intended_initial_masks=True,active_initial_scores_unchanged=True,bypass_restoration_exact=True,
    original_mask_raw_norm_batch_prediction_identities_verified=True,padded_batched_features_zero=True,original_hard_gate_algorithm_and_consumption_verified=True,
    original_shared_head_query_norm_consumption_verified=True,synthetic_CPU_decoder_optimizer_steps=2,CPU_decoder_steps=step_rows,CPU_norm_paths=path_rows,CUDA_initialized=False,native_smoke_passed=False,
    native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU full native model/backbone/TRAIN collation and synthetic component/decoder optimizer checks only. Active normalization intentionally changes masks; only bypass anchors restore W2. Native GPU task and efficacy are untested.')),encoding='utf8')
