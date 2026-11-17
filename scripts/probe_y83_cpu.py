"""CPU original model/TRAIN loading and stage-private recurrent fusion normalization controls."""
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
parameters=sum(p.numel() for p in model.parameters());assert parameters==164632197
parent_path=root/'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==PAYLOAD['parent_dec_lf_sha256']
parent_module=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
import copy
dec_module=importlib.import_module(namespace+'.dec')
active_norms=[model.dec.norm_qqa]+list(model.dec.refinement_query_norms)
assert len(active_norms)==6 and len({p.data_ptr() for n in active_norms for p in n.parameters()})==12
assert all(n.eps==active_norms[0].eps and all(torch.equal(v,n.state_dict()[k]) for k,v in active_norms[0].state_dict().items()) for n in active_norms)
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);base=nn.LayerNorm(7).to(dtype=dtype);rng=torch.get_rng_state().clone();private=[base]+[copy.deepcopy(base) for _ in range(5)]
    assert torch.equal(rng,torch.get_rng_state()) and len({p.data_ptr() for n in private for p in n.parameters()})==12
    x=torch.randn(2,5,7,dtype=dtype,requires_grad=True);values=[n(x) for n in private];assert all(torch.equal(v,values[0]) for v in values)
    xd=x.double();mean=xd.mean(-1,keepdim=True);var=((xd-mean)**2).mean(-1,keepdim=True)
    reference=((xd-mean)/(var+base.eps).sqrt())*base.weight.double()+base.bias.double();tol=128*torch.finfo(dtype).eps
    torch.testing.assert_allclose(values[0].double(),reference,rtol=tol,atol=tol)
    with torch.no_grad():
        private[3].bias[0].add_(.25);changed=[n(x) for n in private]
        assert all(torch.equal(changed[i],values[i]) for i in range(6) if i!=3) and not torch.equal(changed[3],values[3])
        private[3].bias[0].sub_(.25)
        assert torch.equal(private[3](x),values[3])
        order=torch.tensor([3,1,4,0,2]);assert torch.equal(base(x[:,order]),values[0][:,order])
        perm=torch.tensor([3,1,6,0,2,5,4]);rotated=copy.deepcopy(base);rotated.weight.copy_(base.weight[perm]);rotated.bias.copy_(base.bias[perm])
        torch.testing.assert_allclose(rotated(x[:,:,perm]),values[0][:,:,perm],rtol=tol,atol=tol)
        torch.testing.assert_allclose(base(x+3.),values[0],rtol=tol,atol=tol)
        assert torch.isfinite(base(x*1e4)).all()
    loss=(private[4](x)*torch.arange(1,8,dtype=dtype)).square().mean();loss.backward()
    assert all(p.grad is None for i,n in enumerate(private) if i!=4 for p in n.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in private[4].parameters())
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max()>0
    controls.append(dict(dtype=str(dtype),copy_RNG_unchanged=True,six_disjoint_norms=True,identical_initial_values=True,
        independent_scalar_affine_LayerNorm=True,private_intervention_and_restoration=True,private_loss_no_sibling_gradients=True,
        query_channel_permutations_and_shift=True,large_finite_inputs_finite=True,input_and_private_gradients_finite_nonzero=True))
kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999); parent = parent_module.DEC(**kwargs); rng = torch.get_rng_state().clone()
torch.manual_seed(1999); decoder = dec_module.DEC(**kwargs)
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if not k.startswith('refinement_query_norms.')})
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
def fusion_gradients(net):
    net.zero_grad(set_to_none=True);v=net(**inputs)
    loss=sum(p[k].square().mean() for p in parts(v) for k in ('masks','scores','indis'));loss.backward()
    ns=[net.norm_qqa]+list(getattr(net,'refinement_query_norms',[]))
    return [sum(n.state_dict(keep_vars=True)[name].grad for n in ns).detach().clone() for name in ('weight','bias')]
for a,b in zip(fusion_gradients(parent),fusion_gradients(decoder)):torch.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-5)
parent.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
with torch.no_grad():
    changed_norm=decoder.refinement_query_norms[-1];original=changed_norm.bias.clone();changed_norm.bias[0].add_(.25);changed=decoder(**inputs)
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline)[:-1],parts(changed)[:-1]) for k in ('masks','scores','indis'))
    assert not torch.allclose(changed['masks'],baseline['masks'])
    changed_norm.bias.copy_(original);restored=decoder(**inputs)
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(restored)) for k in ('masks','scores','indis'))
model.dec=decoder;optimizer=gorilla.build_optimizer(model,cfg.optimizer)
step_rows=[];path_rows=[];state={};original_prediction=model.dec.prediction_head
head=model.dec.indi_embedding;norms=[model.dec.norm_qqa]+list(model.dec.refinement_query_norms)
"""Exact private-query-normalization callbacks for CPU and native task checks."""
def norm_parameters():
    assert len(norms)==6 and len({p.data_ptr() for n in norms for p in n.parameters()})==12
    assert norms[0] is model.dec.norm_qqa and all(n.eps==norms[0].eps for n in norms)
    return [p for n in norms for p in n.parameters()]


def check_norm_rows(rows):
    assert len(rows)==12
    for i,r in enumerate(rows):
        assert type(r['step']) is int and r['step']==i//6 and type(r['stage']) is int and r['stage']==i%6
        for key in ('original_self_identity','original_language_identity','sum_input_exact','both_inputs_undetached','private_module_identity','original_epsilon','float64_LayerNorm_reference'):
            assert r[key] is True


def check_path_calls():
    assert len(state['qla'])==len(state['self'])==len(state['norm_values'])==6
    assert state['ffn_consumers']==list(range(6)) and state['coarse_calls']==6
    assert len(state['queries'])==7 and state['indicators']==list(range(7)) and state['query_norms']==[True]*7 and len(state['gates'])==7


def checked_prediction(q,features,pad):
    if not state['queries']:state['features'],state['padding']=features,pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(q);value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1)
    expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad
    state['gates'].append(value[2]);return value


def capture_qla(module,args,value):
    i=len(state['qla']);assert module is model.dec.qla_layers[i]
    value[0].retain_grad();state['qla'].append(value[0])


def capture_self(module,args,value):
    i=len(state['self']);assert module is model.dec.qqa_layers[i] and args[0] is state['qla'][i]
    value.retain_grad();state['self'].append(value)


def capture_fusion_norm(module,args,value):
    i=len(state['norm_values']);assert module is norms[i] and len(args)==1
    assert torch.equal(args[0],state['self'][i]+state['qla'][i]) and args[0].requires_grad and value.requires_grad
    args[0].retain_grad();value.retain_grad()
    reference=torch.nn.functional.layer_norm(args[0].double(),module.normalized_shape,module.weight.double(),module.bias.double(),module.eps)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    assert module.eps==model.dec.norm_qqa.eps
    state['sum_inputs'].append(args[0]);state['norm_values'].append(value)
    path_rows.append(dict(step=len(step_rows),stage=i,original_self_identity=True,original_language_identity=True,
        sum_input_exact=True,both_inputs_undetached=True,private_module_identity=True,original_epsilon=True,float64_LayerNorm_reference=True))


def before_ffn(module,args):
    i=len(state['ffn_consumers']);assert module is model.dec.qqa_ffn_layers[i] and args[0] is state['norm_values'][i]
    state['ffn_consumers'].append(i)


def before_coarse(module,args):
    i=state['coarse_calls'];assert module is model.dec.qia_layers[i] and args[3] is state['gates'][-1]
    state['coarse_calls']+=1


def before_indicator(module,args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries'])-1)


def capture_query_norm(module,args,value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad;state['query_norms'].append(True)


def reset_state():
    state.clear();state.update(features=None,padding=None,queries=[],gates=[],qla=[],self=[],norm_values=[],sum_inputs=[],ffn_consumers=[],coarse_calls=0,indicators=[],query_norms=[])


def gradient_record():
    check_path_calls();parameters=norm_parameters()
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in parameters)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['qla']+state['self']+state['sum_inputs']+state['norm_values'])
    heads=[p.grad for p in head.parameters() if p.grad is not None];assert len(heads)==10 and all(torch.isfinite(g).all() for g in heads)
    return dict(all_finite=True,tensors=len(active),indicator_tensors=10,indicator_calls=7,private_norm_calls=6,
        ffn_consumption_calls=6,original_coarse_gate_consumption_calls=6,query_norm_consumption_calls=7,
        norm_gradient_tensors=12,norm_gradient_maxima=[float(p.grad.abs().max()) for p in parameters],both_input_gradients_finite=True)


def install_hooks():
    hooks=[head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(n.register_forward_hook(capture_fusion_norm) for n in norms)
    hooks.extend(n.register_forward_hook(capture_qla) for n in model.dec.qla_layers)
    hooks.extend(n.register_forward_hook(capture_self) for n in model.dec.qqa_layers)
    hooks.extend(n.register_forward_pre_hook(before_ffn) for n in model.dec.qqa_ffn_layers)
    hooks.extend(n.register_forward_pre_hook(before_coarse) for n in model.dec.qia_layers)
    return hooks

hooks=install_hooks();model.dec.prediction_head=checked_prediction
try:
    model.train(True)
    for index in range(2):
        reset_state();optimizer.zero_grad();torch.manual_seed(1999+index);value=model.dec(**inputs)
        labels=torch.arange(256).reshape(2,128).remainder(2).long()
        loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
        assert torch.isfinite(loss);loss.backward();record=gradient_record();params=norm_parameters();before=[p.detach().clone() for p in params]
        optimizer.step();updates=[float((p-b).abs().max()) for p,b in zip(params,before)]
        assert all(u>0 for u in updates) and all(torch.isfinite(p).all() for p in params)
        record.update(step=index,synthetic_CPU_decoder_loss=float(loss),norm_parameter_updates=updates);step_rows.append(record)
    check_norm_rows(path_rows)
finally:
    model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
model.eval();assert all(not m.training for m in model.modules());model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=2560,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,aggregate_shared_norm_gradient_anchor_close=True,
    original_initial_prediction_unchanged=True,private_fusion_norm_identity_and_FFN_consumption_verified=True,static_mask_and_original_hard_gate_preserved=True,
    original_shared_head_query_norm_consumption_verified=True,synthetic_CPU_decoder_optimizer_steps=2,CPU_decoder_steps=step_rows,CPU_norm_paths=path_rows,
    private_final_intervention_only_final_prediction=True,restored_fusion_negative_control_exact=True,CUDA_initialized=False,native_smoke_passed=False,
    native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU full native model/pretrain/TRAIN collation and synthetic affine/decoder optimizer checks only; no native sparse TRAIN task/GPU PASS/admission/efficacy.')),encoding='utf8')
