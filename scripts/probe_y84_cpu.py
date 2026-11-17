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

parameters=sum(p.numel() for p in model.parameters());assert parameters==164695173
parent_path=root/'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==PAYLOAD['parent_dec_lf_sha256']
parent_module=importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module=importlib.import_module(namespace+'.dec')
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();m=dec_module.VisualQueryInitialization(3).to(dtype=dtype)
    assert torch.equal(rng,torch.get_rng_state()) and torch.count_nonzero(m.weight)==0
    q=torch.randn(2,5,3,dtype=dtype,requires_grad=True);f=torch.randn(2,4,3,dtype=dtype,requires_grad=True)
    padding=torch.tensor([[False,False,True,True],[False,False,False,True]])
    mass=torch.tensor([[1.,3.,100.,200.],[2.,1.,4.,100.]],dtype=dtype,requires_grad=True)
    assert torch.equal(m(q,f,padding,mass),q)
    with torch.no_grad():m.weight.copy_(torch.arange(9,dtype=dtype).reshape(3,3)/9)
    value=m(q,f,padding,mass);refs=[]
    for b in range(2):
        denom=sum(float(mass[b,j]) for j in range(4) if not padding[b,j]);ctx=[]
        for c in range(3):ctx.append(sum(float(mass[b,j])*float(f[b,j,c]) for j in range(4) if not padding[b,j])/denom)
        refs.append([sum(ctx[c]*float(m.weight[c,d]) for c in range(3)) for d in range(3)])
    reference=q.detach().double()+torch.tensor(refs,dtype=torch.float64).unsqueeze(1)
    tol=128*torch.finfo(dtype).eps;torch.testing.assert_allclose(value.double(),reference,rtol=tol,atol=tol)
    order=torch.tensor([2,0,3,1]);torch.testing.assert_allclose(m(q,f[:,order],padding[:,order],mass[:,order]),value,rtol=tol,atol=tol)
    perm=torch.tensor([4,0,2,1,3]);torch.testing.assert_allclose(m(q[:,perm],f,padding,mass),value[:,perm],rtol=tol,atol=tol)
    dirty=f.detach().clone();dirty[padding]=float('nan');assert torch.equal(m(q,dirty,padding,mass),value)
    empty=torch.ones_like(padding);assert torch.equal(m(q,f,empty,mass),q)
    assert torch.equal(m(q,f,padding,torch.zeros_like(mass)),q)
    shifted=mass.detach().clone();shifted[0,0]+=2;assert not torch.allclose(m(q,f,padding,shifted),value)
    value.square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max()>0 for x in (q,f,mass,m.weight))
    assert torch.count_nonzero(f.grad[padding])==torch.count_nonzero(mass.grad[padding])==0
    controls.append(dict(dtype=str(dtype),zero_anchor_no_RNG=True,independent_scalar_weighted_visual_mean_projection=True,scene_and_query_permutations=True,padded_NaN_and_mass_ignored=True,empty_and_zero_mass_noop=True,valid_mass_intervention_changes_output=True,all_input_and_weight_gradients_finite_nonzero=True,padded_input_gradients_zero=True))
kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999); parent = parent_module.DEC(**kwargs); rng = torch.get_rng_state().clone()
torch.manual_seed(1999); decoder = dec_module.DEC(**kwargs)
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if not k.startswith('visual_query_init.')})
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

parent.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
with torch.no_grad():
    original=decoder.visual_query_init.weight.clone();decoder.visual_query_init.weight[0,0]=.25;changed=decoder(**inputs)
    assert not torch.allclose(changed['aux_outputs'][0]['masks'],baseline['aux_outputs'][0]['masks'])
    decoder.visual_query_init.weight.copy_(original);restored=decoder(**inputs)
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(restored)) for k in ('masks','scores','indis'))
model.dec=decoder;optimizer=gorilla.build_optimizer(model,cfg.optimizer)
initialization=model.dec.visual_query_init;head=model.dec.indi_embedding
step_rows=[];path_rows=[];state={};original_prediction=model.dec.prediction_head;original_batch=model.dec.get_batches

def tracked_batch(*args,**kwargs):
    result=original_batch(*args,**kwargs);state['batches'].append(result);return result

def capture_generated(module,args,value):
    assert 'generated' not in state;state['generated']=value

def capture_initial(module,args,value):
    assert len(args)==4 and len(state['batches'])==4 and 'initialized' not in state
    q,f,pad,mass=args;assert q is state['generated'] and f is state['batches'][0][0] and pad is state['batches'][0][1]
    assert mass.data_ptr()==state['batches'][2][0].data_ptr() and torch.equal(mass,state['batches'][2][0].squeeze(-1))
    assert q.requires_grad and f.requires_grad and value.requires_grad;q.retain_grad();f.retain_grad();value.retain_grad()
    weights=mass.masked_fill(pad,0).double();clean=f.double().masked_fill(pad.unsqueeze(-1),0)
    totals=weights.sum(1,keepdim=True);context=(clean*weights.unsqueeze(-1)).sum(1)/totals.clamp_min(1)
    context=context.masked_fill(totals<=0,0);reference=q.double()+(context@module.weight.double()).unsqueeze(1)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    state['initialized']=value;state['scene']=f
    nonzero=bool(torch.count_nonzero(module.weight));delta=float((value-q).abs().max())
    assert nonzero==bool(len(step_rows)) and (delta>0 if nonzero else delta==0)
    path_rows.append(dict(step=len(step_rows),original_generated_query_identity=True,original_scene_visual_identity=True,original_padding_and_point_mass_identity=True,all_inputs_undetached=True,float64_weighted_context_reference=True,weight_nonzero=nonzero,query_delta_max=delta))

def checked_prediction(q,features,pad):
    assert features is state['batches'][1][0] and pad is state['batches'][0][1]
    if state['predictions']==0:assert q is state['initialized']
    state['query']=q;state['predictions']+=1;value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1);expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad;return value

def before_indicator(module,args):
    assert args[0] is state['query'] and args[1] is state['mask'];state['indicators']+=1

def capture_norm(module,args,value):
    assert args[0] is state['query'];state['norms']+=1

hooks=[model.dec.query_generator.register_forward_hook(capture_generated),initialization.register_forward_hook(capture_initial),head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_norm)]
model.dec.get_batches=tracked_batch;model.dec.prediction_head=checked_prediction
try:
    model.train(True)
    for index in range(2):
        state.clear();state.update(batches=[],predictions=0,indicators=0,norms=0)
        optimizer.zero_grad();torch.manual_seed(1999+index);value=model.dec(**inputs)
        assert state['predictions']==state['indicators']==state['norms']==7
        labels=torch.arange(256).reshape(2,128).remainder(2).long()
        loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
        assert torch.isfinite(loss);loss.backward()
        active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
        grad=initialization.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (state['generated'],state['scene'],state['initialized']))
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        before=initialization.weight.detach().clone();optimizer.step();update=float((initialization.weight-before).abs().max());assert update>0
        step_rows.append(dict(step=index,synthetic_CPU_decoder_loss=float(loss),new_gradient_max=float(grad.abs().max()),parameter_update_max=update,prediction_calls=7,indicator_calls=7,query_norm_calls=7,both_query_and_scene_gradients_finite=True))
finally:
    model.dec.get_batches=original_batch;model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
model.eval();assert all(not m.training for m in model.modules());model.train(True)
assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=65536,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,
    original_generated_query_and_scene_visual_consumer_verified=True,static_mask_and_original_hard_gate_preserved=True,
    original_shared_head_query_norm_consumption_verified=True,nonzero_initial_query_intervention_and_restoration=True,
    synthetic_CPU_decoder_optimizer_steps=2,CPU_decoder_steps=step_rows,CPU_initialization_paths=path_rows,CUDA_initialized=False,native_smoke_passed=False,
    native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU full native model/backbone/TRAIN collation and synthetic component/decoder optimizer checks only. Native sparse TRAIN/GPU task and efficacy remain untested.')),encoding='utf8')
