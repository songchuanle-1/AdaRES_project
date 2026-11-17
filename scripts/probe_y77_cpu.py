"""CPU native model/TRAIN loading and geometry query-initialization controls."""
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
parameters = sum(p.numel() for p in model.parameters()); assert parameters == 164630405
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')


dec_module=importlib.import_module(namespace+'.dec');Init=dec_module.GeometryQueryInitialization
assert torch.equal(model.dec.geometry_query_init.weight,torch.zeros(3,256))
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();branch=Init(7).to(dtype=dtype);assert torch.equal(rng,torch.get_rng_state())
    query=torch.randn(2,11,7,dtype=dtype,requires_grad=True);query_pos=torch.randn(2,11,3,dtype=dtype,requires_grad=True)
    scene=torch.randn(2,5,3,dtype=dtype,requires_grad=True);mass=(torch.rand(2,5,dtype=dtype)+1).requires_grad_()
    pad=torch.tensor([[False,False,False,True,True],[False,False,False,False,False]])
    zero=branch(query,query_pos,scene,pad,mass);assert torch.equal(zero,query)
    grads=torch.autograd.grad(zero.sum(),(query,query_pos,scene,mass,branch.weight))
    assert torch.equal(grads[0],torch.ones_like(query)) and all(torch.equal(g,torch.zeros_like(g)) for g in grads[1:4])
    assert torch.isfinite(grads[4]).all() and grads[4].abs().max()>0
    tol=128*torch.finfo(dtype).eps
    with torch.no_grad():
        branch.weight.copy_(torch.arange(21,dtype=dtype).reshape(3,7)/21-.5)
        changed=branch(query,query_pos,scene,pad,mass);reference=torch.empty_like(changed)
        for b in range(2):
            valid=[j for j in range(5) if not bool(pad[b,j])];den=sum(mass[b,j].double() for j in valid)
            center=[sum(scene[b,j,a].double()*mass[b,j].double() for j in valid)/den for a in range(3)]
            for q in range(11):
                for d in range(7):reference[b,q,d]=query[b,q,d].double()+sum((query_pos[b,q,a].double()-center[a])*branch.weight[a,d].double() for a in range(3))
        torch.testing.assert_allclose(changed,reference,rtol=tol,atol=tol);assert not torch.allclose(changed,query,rtol=tol,atol=tol)
        order=torch.tensor([3,1,10,0,2,5,4,6,7,8,9]);assert torch.equal(branch(query[:,order],query_pos[:,order],scene,pad,mass),changed[:,order])
        assert torch.equal(branch(query.flip(0),query_pos.flip(0),scene.flip(0),pad.flip(0),mass.flip(0)),changed.flip(0))
        sp_order=torch.tensor([4,1,3,0,2]);torch.testing.assert_allclose(branch(query,query_pos,scene[:,sp_order],pad[:,sp_order],mass[:,sp_order]),changed,rtol=tol,atol=tol)
        shift=torch.tensor([3.25,-1.5,.125],dtype=dtype)
        torch.testing.assert_allclose(branch(query,query_pos+shift,scene+shift,pad,mass),changed,rtol=tol,atol=tol)
        torch.testing.assert_allclose(branch(query,query_pos,scene,pad,mass*3),changed,rtol=tol,atol=tol)
        axis=torch.tensor([2,0,1]);other=Init(7).to(dtype=dtype);other.weight.copy_(branch.weight[axis])
        torch.testing.assert_allclose(other(query,query_pos[:,:,axis],scene[:,:,axis],pad,mass),changed,rtol=tol,atol=tol)
        channel=torch.tensor([3,1,6,0,2,5,4]);other.weight.copy_(branch.weight[:,channel])
        assert torch.equal(other(query[:,:,channel],query_pos,scene,pad,mass),changed[:,:,channel])
        dirty_scene=scene.clone();dirty_mass=mass.clone();dirty_scene[pad]=float('nan');dirty_mass[pad]=float('nan')
        assert torch.equal(branch(query,query_pos,dirty_scene,pad,dirty_mass),changed)
        assert torch.equal(branch(query,query_pos,scene,torch.ones_like(pad),mass),query)
        assert not torch.allclose(branch(query,query_pos,scene,pad,mass.flip(1)),changed,rtol=tol,atol=tol)
    grads=torch.autograd.grad(branch(query,query_pos,scene,pad,mass).square().mean(),(query,query_pos,scene,mass,branch.weight))
    assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    assert torch.equal(grads[2][pad],torch.zeros_like(grads[2][pad])) and torch.equal(grads[3][pad],torch.zeros_like(grads[3][pad]))
    controls.append(dict(dtype=str(dtype),initial_RNG_unchanged=True,zero_output_and_original_input_gradients_exact=True,new_weight_gradient_finite_nonzero=True,
        independent_scalar_point_mass_centroid_and_projection=True,joint_query_batch_scene_axis_channel_permutations=True,joint_translation_invariant=True,
        uniform_mass_scale_invariant=True,padded_coordinate_and_mass_NaNs_ignored_exact=True,empty_scene_branch_noop=True,wrong_mass_correspondence_changes_output=True,
        all_input_gradients_finite_nonzero=True,padded_scene_and_mass_gradients_zero=True))
kwargs=dict(cfg.model.dec,sampling_module=cfg.model.sampling_module,in_channel=cfg.model.media)
torch.manual_seed(1999);parent=parent_module.DEC(**kwargs);rng=torch.get_rng_state().clone()
torch.manual_seed(1999);decoder=dec_module.DEC(**kwargs)
assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict());parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if k!='geometry_query_init.weight'})
torch.manual_seed(1999);lengths=(512,513);total=sum(lengths)
inputs=dict(x=torch.randn(total,cfg.model.media),fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2,1),batch_offsets=torch.tensor([0,lengths[0],total],dtype=torch.int32),
    lang_feats=torch.randn(2,9,decoder.lang_proj.in_features),lang_masks=torch.ones(2,9,dtype=torch.long),sp_pos=torch.randn(total,3),feats_2d=torch.randn(total,decoder.input_proj_2d[0].in_features),point_mass=torch.arange(total).remainder(13).float()+1)
inputs['lang_masks'][0,-2:]=0;inputs['lang_feats'][0,-2:]=0
parts=lambda v:list(v['aux_outputs'])+[v]
parent.eval();decoder.eval()
with torch.no_grad():baseline=parent(**inputs);output=decoder(**inputs)
assert len(parts(output))==7 and all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(output)) for k in ('masks','scores','indis'))
def input_gradients(net):
    feed=dict(inputs);names=('x','lang_feats','feats_2d')
    for key in names:feed[key]=inputs[key].clone().requires_grad_()
    v=net(**feed);loss=sum(p[k].square().mean() for p in parts(v) for k in ('masks','scores','indis'))
    return torch.autograd.grad(loss,tuple(feed[k] for k in names))
for a,b in zip(input_gradients(parent),input_gradients(decoder)):torch.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
state=dict(batches=[],generated=[],samples=[],outputs=[],predictions=[],indicators=[],position_consumers=[])
batching=decoder.get_batches;scg=decoder.get_scg_weight;prediction=decoder.prediction_head

def checked_batches(flat,offsets):
    result=batching(flat,offsets);state['batches'].append((flat,result))
    if len(state['batches'])==1:state['padding']=result[1]
    if flat is inputs['sp_pos']:state['scene_pos']=result[0]
    if flat.data_ptr()==inputs['point_mass'].data_ptr() and flat.shape[-1]==1:state['mass_tensor']=result[0]
    return result

def checked_scg(pos,*args,**kwargs):
    if state['outputs']:
        assert pos is state['query_pos'];state['position_consumers'].append(True)
    return scg(pos,*args,**kwargs)

def capture_generated(module,args,value):state['generated'].append(value)
def capture_sampling(module,args,value):state['samples'].append(value[0])
def capture_initialization(module,args,value):
    query,qpos,scene_pos,pad,mass=args
    assert query is state['generated'][0] and scene_pos is state['scene_pos'] and pad is state['padding']
    assert mass.data_ptr()==state['mass_tensor'].data_ptr() and torch.equal(mass,state['mass_tensor'].squeeze(-1))
    assert query.requires_grad and torch.equal(query,value)
    seeded=scene_pos.gather(1,inputs['fps_seed_sp'].long().unsqueeze(-1).expand(-1,-1,3))
    expected=seeded.gather(1,state['samples'][0].long().unsqueeze(-1).expand(-1,-1,3))
    assert torch.equal(qpos,expected);state['query_pos']=qpos;state['outputs'].append(value)

def checked_prediction(query,features,pad):
    if not state['predictions']:assert query is state['outputs'][0]
    assert pad is state['padding'];state['predictions'].append(query)
    return prediction(query,features,pad)
hooks=[decoder.query_generator.register_forward_hook(capture_generated),decoder.sampling_module.register_forward_hook(capture_sampling),
    decoder.geometry_query_init.register_forward_hook(capture_initialization),decoder.indi_embedding.register_forward_hook(lambda m,a,o:state['indicators'].append(o))]
decoder.get_batches=checked_batches;decoder.get_scg_weight=checked_scg;decoder.prediction_head=checked_prediction
try:
    decoder.train(True);torch.manual_seed(1999);value=decoder(**inputs)
    assert len(state['outputs'])==len(state['generated'])==len(state['samples'])==1
    assert len(state['predictions'])==len(state['indicators'])==7 and len(state['position_consumers'])==12 and len(state['batches'])==4
    labels=torch.arange(256).reshape(2,128).remainder(2).long()
    loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    grad=decoder.geometry_query_init.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
finally:
    decoder.get_batches=batching;decoder.get_scg_weight=scg;decoder.prediction_head=prediction
    for hook in hooks:hook.remove()
decoder.eval()
with torch.no_grad():
    decoder.geometry_query_init.weight.copy_(.03*torch.linspace(-1,1,768).reshape(3,256));changed=decoder(**inputs)
    assert not torch.allclose(parts(changed)[0]['masks'],parts(baseline)[0]['masks']) and not torch.allclose(changed['masks'],baseline['masks'])
    hook=decoder.geometry_query_init.register_forward_hook(lambda m,a,o:a[0])
    try:bypassed=decoder(**inputs)
    finally:hook.remove()
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(bypassed)) for k in ('masks','scores','indis'))
model.eval();assert all(not m.training for m in model.modules());model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=768,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,
    original_generated_query_scene_position_padding_mass_and_sampled_position_identities_verified=True,original_SCG_query_position_consumption_calls=12,
    initial_prediction_consumption_verified=True,decoder_initialization_calls=1,decoder_indicator_calls=7,decoder_backward_finite=True,
    decoder_initialization_gradient_finite_nonzero=True,nonzero_initialization_changes_initial_and_final_masks=True,bypassed_initialization_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='Actual CPU native full model/pretrain/TRAIN collation and synthetic component/decoder controls only; not sparse native TRAIN task/GPU PASS or efficacy.')),encoding='utf-8')
