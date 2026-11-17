"""CPU original model/TRAIN loading and coarse/fine product projection controls."""
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
dec_module=importlib.import_module(namespace+'.dec');Fusion=dec_module.ScaleProductFusion
assert torch.equal(model.dec.scale_product.weight,torch.zeros(256,256))
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();branch=Fusion(7).to(dtype=dtype)
    assert torch.equal(rng,torch.get_rng_state())
    c,f=[torch.randn(2,5,7,dtype=dtype,requires_grad=True) for _ in range(2)]
    zero=branch(c,f);assert torch.equal(zero,(c+f)/2)
    grads=torch.autograd.grad(zero.sum(),(c,f,branch.weight))
    assert torch.equal(grads[0],.5*torch.ones_like(c)) and torch.equal(grads[1],.5*torch.ones_like(f))
    assert torch.isfinite(grads[2]).all() and grads[2].abs().max()>0
    with torch.no_grad():
        branch.weight.copy_(torch.arange(49,dtype=dtype).reshape(7,7)/49-.5)
        value=branch(c,f);reference=torch.empty_like(value)
        for b in range(2):
            for i in range(5):
                for d in range(7):
                    reference[b,i,d]=(c[b,i,d].double()+f[b,i,d].double())/2+sum(c[b,i,k].double()*f[b,i,k].double()*branch.weight[k,d].double() for k in range(7))
        tol=128*torch.finfo(dtype).eps;torch.testing.assert_allclose(value,reference,rtol=tol,atol=tol)
        assert not torch.allclose(value,(c+f)/2,rtol=tol,atol=tol)
        order=torch.tensor([3,1,4,0,2]);assert torch.equal(branch(c[:,order],f[:,order]),value[:,order])
        assert torch.equal(branch(c.flip(0),f.flip(0)),value.flip(0))
        perm=torch.tensor([3,1,6,0,2,5,4]);other=Fusion(7).to(dtype=dtype);other.weight.copy_(branch.weight[perm][:,perm])
        torch.testing.assert_allclose(other(c[:,:,perm],f[:,:,perm]),value[:,:,perm],rtol=tol,atol=tol)
        assert torch.equal(branch(f,c),value)
        witness=Fusion(7).to(dtype=dtype);witness.weight.copy_(torch.eye(7,dtype=dtype))
        a=torch.ones(1,2,7,dtype=dtype);b=2*a;z=torch.zeros_like(a)
        assert torch.equal((a+a)/2,(b+z)/2) and not torch.equal(witness(a,a),witness(b,z))
        assert not torch.allclose(branch(c,f.flip(1)),value,rtol=tol,atol=tol)
        assert torch.isfinite(branch(c*1e4,f*1e4)).all()
    grads=torch.autograd.grad(branch(c,f).square().mean(),(c,f,branch.weight))
    assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    controls.append(dict(dtype=str(dtype),initial_RNG_unchanged=True,zero_average_and_input_gradients_exact=True,
        new_gradient_finite_nonzero=True,independent_scalar_product_projection=True,joint_query_batch_channel_permutations=True,
        branch_swap_exact=True,same_average_different_product_distinguished=True,wrong_correspondence_changes_output=True,
        large_finite_inputs_finite=True,both_input_gradients_finite_nonzero=True))
kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999); parent = parent_module.DEC(**kwargs); rng = torch.get_rng_state().clone()
torch.manual_seed(1999); decoder = dec_module.DEC(**kwargs)
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(v, decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if k != 'scale_product.weight'})
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
with torch.no_grad():
    decoder.scale_product.weight.copy_(.003*torch.eye(256));changed=decoder(**inputs)
    assert all(torch.equal(parts(changed)[0][k],parts(baseline)[0][k]) for k in ('masks','scores','indis'))
    assert not torch.allclose(parts(changed)[1]['masks'],parts(baseline)[1]['masks']) and not torch.allclose(changed['masks'],baseline['masks'])
    hook=decoder.scale_product.register_forward_hook(lambda m,a,o:(a[0]+a[1])/2)
    try:bypassed=decoder(**inputs)
    finally:hook.remove()
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(bypassed)) for k in ('masks','scores','indis'))
    decoder.scale_product.weight.zero_()
model.dec=decoder;optimizer=gorilla.build_optimizer(model,cfg.optimizer)
step_rows=[];path_rows=[];state={};original_prediction=model.dec.prediction_head
head=model.dec.indi_embedding;fusion=model.dec.scale_product
"""Callbacks shared literally by CPU decoder and future native TRAIN checks."""
def check_path_calls(state):
    for key in ('coarse','fine','fused','coarse_inputs','fine_inputs'):
        assert len(state[key])==6
    assert len(state['queries'])==7 and len(state['gates'])==7
    assert state['qla_consumers']==list(range(6)) and all(type(i) is int for i in state['qla_consumers'])
    assert state['indicators']==list(range(7)) and state['norms']==[True]*7


def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==12
    for index,row in enumerate(rows):
        assert type(row['step']) is int and row['step']==index//6
        assert type(row['stage']) is int and row['stage']==index%6
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero']==bool(index>=6)
        for key in ('original_coarse_identity','original_fine_identity','both_inputs_undetached','float64_product_projection_reference'):
            assert row[key] is True
        delta=row['fusion_delta_max'];assert isfinite(delta) and delta>=0
        assert delta==0 if index<6 else delta>0


def checked_prediction(q,features,pad):
    if not state['queries']:state['features'],state['padding']=features,pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(q);value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1)
    expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad
    state['gates'].append(value[2]);return value


def capture_coarse(module,args,value):
    index=len(state['coarse'])
    assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1]
    state['coarse'].append(value[0])


def capture_fine(module,args,value):
    index=len(state['fine']);assert module is model.dec.qia_layers_fine[index]
    state['fine'].append(value[0])


def capture_fusion(module,args,value):
    index=len(state['fused']);assert len(args)==2 and args[0] is state['coarse'][index] and args[1] is state['fine'][index]
    assert all(t.requires_grad for t in args)
    for t in args+(value,):t.retain_grad()
    reference=(args[0].double()+args[1].double())/2+(args[0].double()*args[1].double())@module.weight.double()
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    nonzero=bool(module.weight.abs().max()>0);assert nonzero==bool(len(step_rows))
    delta=float((value-(args[0]+args[1])/2).abs().max());assert (delta>0)==nonzero
    state['coarse_inputs'].append(args[0]);state['fine_inputs'].append(args[1]);state['fused'].append(value)
    path_rows.append(dict(step=len(step_rows),stage=index,weight_nonzero=nonzero,fusion_delta_max=delta,
        original_coarse_identity=True,original_fine_identity=True,both_inputs_undetached=True,float64_product_projection_reference=True))


def before_qla(module,args):
    index=len(state['qla_consumers']);assert module is model.dec.qla_layers[index] and args[1] is state['fused'][index]
    state['qla_consumers'].append(index)


def before_indicator(module,args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries'])-1)


def capture_query_norm(module,args,value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad;state['norms'].append(True)


def reset_state():
    state.clear();state.update(features=None,padding=None,queries=[],gates=[],coarse=[],fine=[],fused=[],coarse_inputs=[],fine_inputs=[],qla_consumers=[],indicators=[],norms=[])


def gradient_record():
    check_path_calls(state)
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None]
    assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['coarse_inputs']+state['fine_inputs']+state['fused'])
    grad=fusion.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    return dict(all_finite=True,tensors=len(active),indicator_tensors=10,indicator_calls=7,fusion_calls=6,
        qla_consumption_calls=6,query_norm_consumption_calls=7,both_input_gradients_finite=True,new_gradient_max=float(grad.abs().max()))


def install_hooks():
    hooks=[fusion.register_forward_hook(capture_fusion),head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(m.register_forward_hook(capture_coarse) for m in model.dec.qia_layers)
    hooks.extend(m.register_forward_hook(capture_fine) for m in model.dec.qia_layers_fine)
    hooks.extend(m.register_forward_pre_hook(before_qla) for m in model.dec.qla_layers)
    return hooks

hooks=install_hooks();model.dec.prediction_head=checked_prediction
try:
    model.train(True)
    for index in range(2):
        reset_state();optimizer.zero_grad();torch.manual_seed(1999+index);value=model.dec(**inputs)
        labels=torch.arange(256).reshape(2,128).remainder(2).long()
        loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
        assert torch.isfinite(loss);loss.backward();record=gradient_record()
        before=fusion.weight.detach().clone();optimizer.step();update=float((fusion.weight.detach()-before).abs().max())
        assert update>0 and torch.isfinite(fusion.weight).all()
        record.update(step=index,synthetic_CPU_decoder_loss=float(loss),parameter_update_max=update);step_rows.append(record)
    check_response_rows(path_rows)
finally:
    model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
model.eval();assert all(not m.training for m in model.modules());model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=65536,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,original_initial_prediction_unchanged=True,
    original_branch_identities_and_qla_consumption_verified=True,static_mask_and_original_hard_gate_preserved=True,original_shared_head_query_norm_consumption_verified=True,
    synthetic_CPU_decoder_optimizer_steps=2,CPU_decoder_steps=step_rows,CPU_fusion_paths=path_rows,
    nonzero_fusion_changes_refined_and_final_masks=True,bypassed_fusion_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU full native model/pretrain/TRAIN collation and synthetic component/decoder optimizer checks only; no native sparse TRAIN task/GPU PASS/admission/efficacy.')),encoding='utf8')
