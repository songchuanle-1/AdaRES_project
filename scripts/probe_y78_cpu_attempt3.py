"""CPU native model/TRAIN loading and language-conditioned query-initialization controls."""
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
parameters = sum(p.numel() for p in model.parameters()); assert parameters == 164629893
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')


dec_module=importlib.import_module(namespace+'.dec');Residual=dec_module.MaskLogitResidual
assert torch.equal(model.dec.mask_logit_residual.weight,torch.zeros(256,1))
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();branch=Residual(7).to(dtype=dtype);assert torch.equal(rng,torch.get_rng_state())
    q=torch.randn(2,11,7,dtype=dtype,requires_grad=True);raw=torch.randn(2,11,5,dtype=dtype,requires_grad=True);prev=torch.randn(2,11,5,dtype=dtype,requires_grad=True)
    pad=torch.tensor([[False,False,False,True,True],[False,False,False,False,False]])
    value=branch(q,raw,prev,pad);assert torch.equal(value,raw)
    grads=torch.autograd.grad(value.square().sum(),(q,raw,prev,branch.weight))
    assert torch.equal(grads[0],torch.zeros_like(q)) and torch.equal(grads[1],2*raw) and torch.equal(grads[2],torch.zeros_like(prev))
    assert torch.isfinite(grads[3]).all() and grads[3].abs().max()>0
    with torch.no_grad():
        branch.weight.copy_(torch.arange(7,dtype=dtype).reshape(7,1)/7-.5)
        changed=branch(q,raw,prev,pad);reference=raw.clone()
        for b in range(2):
            for j in range(11):
                gate=torch.tanh(sum(q[b,j,k].double()*branch.weight[k,0].double() for k in range(7)))
                for m in range(5):
                    if not bool(pad[b,m]):reference[b,j,m]=raw[b,j,m].double()+gate*prev[b,j,m].double()
        tol=128*torch.finfo(dtype).eps;torch.testing.assert_allclose(changed,reference,rtol=tol,atol=tol)
        assert not torch.allclose(changed,raw,rtol=tol,atol=tol)
        order=torch.tensor([3,1,10,0,2,5,4,6,7,8,9]);permuted=branch(q[:,order],raw[:,order],prev[:,order],pad)
        error=float((permuted-changed[:,order]).abs().max())
        # GEMV/GEMM kernels may round differently after row reordering. Require
        # the existing independently checked scalar formula and predeclared
        # dtype-scaled bound, and save the measured error without altering model.
        torch.testing.assert_allclose(permuted,reference[:,order],rtol=tol,atol=tol)
        torch.testing.assert_allclose(permuted,changed[:,order],rtol=tol,atol=tol)
        print(json.dumps(dict(component='query_permutation',dtype=str(dtype),max_absolute_error=error,dtype_epsilon=float(torch.finfo(dtype).eps),rtol_atol=tol,model_unchanged=True)),flush=True)
        torch.testing.assert_allclose(branch(q.flip(0),raw.flip(0),prev.flip(0),pad.flip(0)),changed.flip(0),rtol=tol,atol=tol)
        points=torch.tensor([3,1,4,0,2]);torch.testing.assert_allclose(branch(q,raw[:,:,points],prev[:,:,points],pad[:,points]),changed[:,:,points],rtol=tol,atol=tol)
        channels=torch.tensor([3,1,6,0,2,5,4]);other=Residual(7).to(dtype=dtype);other.weight.copy_(branch.weight[channels])
        torch.testing.assert_allclose(other(q[:,:,channels],raw,prev,pad),changed,rtol=tol,atol=tol)
        polluted=prev.clone();polluted.masked_fill_(pad.unsqueeze(1),float('nan'));assert torch.equal(branch(q,raw,polluted,pad),changed)
        assert torch.equal(branch(q,raw,prev,torch.ones_like(pad)),raw)
        assert not torch.allclose(branch(q,raw,prev.roll(1,1),pad),changed,rtol=tol,atol=tol)
        assert not torch.allclose(branch(q,raw,prev.roll(1,2),pad),changed,rtol=tol,atol=tol)
    grads=torch.autograd.grad(branch(q,raw,prev,pad).square().mean(),(q,raw,prev,branch.weight))
    assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    assert torch.equal(grads[2].masked_select(pad.unsqueeze(1)),torch.zeros_like(grads[2].masked_select(pad.unsqueeze(1))))
    controls.append(dict(dtype=str(dtype),initial_RNG_unchanged=True,zero_output_and_input_gradients_exact=True,new_gradient_finite_nonzero=True,
        independent_scalar_query_gate_residual=True,joint_query_point_batch_channel_permutations=True,padded_previous_NaN_intervention_exact=True,
        all_padding_branch_noop=True,wrong_previous_query_and_point_correspondences_change_output=True,all_input_gradients_finite_nonzero=True,padded_previous_gradients_zero=True))
kwargs=dict(cfg.model.dec,sampling_module=cfg.model.sampling_module,in_channel=cfg.model.media)
torch.manual_seed(1999);parent=parent_module.DEC(**kwargs);rng=torch.get_rng_state().clone()
torch.manual_seed(1999);decoder=dec_module.DEC(**kwargs)
assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict());parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if k!='mask_logit_residual.weight'})
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
with torch.no_grad():
    decoder.mask_logit_residual.weight.fill_(.003);changed=decoder(**inputs)
    assert all(torch.equal(parts(changed)[0][k],parts(baseline)[0][k]) for k in ('masks','scores','indis'))
    assert not torch.allclose(parts(changed)[1]['masks'],parts(baseline)[1]['masks']) and not torch.allclose(changed['masks'],baseline['masks'])
    hook=decoder.mask_logit_residual.register_forward_hook(lambda m,a,o:a[1])
    try:bypassed=decoder(**inputs)
    finally:hook.remove()
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(bypassed)) for k in ('masks','scores','indis'))
    decoder.mask_logit_residual.weight.zero_()

# Execute complete decoder paths on two real CPU optimizer steps. These are
# synthetic decoder inputs, explicitly not native sparse TRAIN task execution.
step_rows=[];path_rows=[]
model.dec=decoder
optimizer=gorilla.build_optimizer(model,cfg.optimizer)
original_prediction=decoder.prediction_head;original_refined=decoder.refined_prediction_head
state={}
def checked_prediction(q,features,pad):
    result=original_prediction(q,features,pad)
    if state['features'] is None:state['features']=features;state['pad']=pad
    assert features is state['features'] and pad is state['pad']
    state['queries'].append(q);state['raw'].append(result[1])
    if len(state['queries'])==1:state['gates'].append(result[2])
    return result

def capture_residual(module,args,value):
    assert len(args)==4 and args[0] is state['queries'][-1] and args[1] is state['raw'][-1]
    assert args[2] is state['emitted'][-1] and args[3] is state['pad']
    assert all(t.requires_grad for t in args[:3])
    args[2].retain_grad();args[0].retain_grad();value.retain_grad()
    gate=torch.tanh(args[0].double()@module.weight.double())
    reference=args[1].double()+gate*args[2].double().masked_fill(args[3].unsqueeze(1),0.)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    nonzero=bool(module.weight.abs().max()>0)
    assert nonzero==bool(len(step_rows))
    delta=float((value-args[1]).abs().max())
    assert (delta>0)==nonzero
    state['values'].append(value);state['previous'].append(args[2])
    path_rows.append(dict(step=len(step_rows),stage=len(state['values']),raw_query_previous_padding_identities=True,previous_undetached=True,
        static_mask_features_preserved=True,independent_formula_checked=True,weight_nonzero=nonzero,residual_delta_max=delta))

def checked_refined(q,features,pad,previous):
    result=original_refined(q,features,pad,previous)
    assert result[1] is state['values'][-1]
    expected=(result[1].sigmoid()<.5)|pad.unsqueeze(1)
    expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(result[2],expected) and not result[2].requires_grad
    state['gates'].append(result[2]);return result

def before_indicator(module,args):
    assert args[0] is state['queries'][-1] and args[2] is state['pad']
    expected=state['raw'][-1] if not state['emitted'] else state['values'][-1]
    assert args[1] is expected
    state['emitted'].append(args[1])

def before_coarse(module,args):
    assert args[3] is state['gates'][-1];state['coarse_calls']+=1

hooks=[decoder.mask_logit_residual.register_forward_hook(capture_residual),decoder.indi_embedding.register_forward_pre_hook(before_indicator)]
hooks.extend(m.register_forward_pre_hook(before_coarse) for m in decoder.qia_layers)
decoder.prediction_head=checked_prediction;decoder.refined_prediction_head=checked_refined
try:
    decoder.train(True)
    for index in range(2):
        state.update(features=None,pad=None,queries=[],raw=[],gates=[],emitted=[],values=[],previous=[],coarse_calls=0)
        optimizer.zero_grad();torch.manual_seed(1999+index);value=decoder(**inputs)
        assert len(state['queries'])==len(state['emitted'])==len(state['gates'])==7 and len(state['values'])==state['coarse_calls']==6
        labels=torch.arange(256).reshape(2,128).remainder(2).long()
        loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
        assert torch.isfinite(loss);loss.backward()
        active=[p.grad for p in decoder.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['previous']+state['values'])
        grad=decoder.mask_logit_residual.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
        headgrads=[p.grad for p in decoder.indi_embedding.parameters() if p.grad is not None];assert len(headgrads)==10
        before=decoder.mask_logit_residual.weight.detach().clone();optimizer.step();update=float((decoder.mask_logit_residual.weight.detach()-before).abs().max());assert update>0
        step_rows.append(dict(step=index,synthetic_CPU_decoder_loss=float(loss),finite_gradient_tensors=len(active),original_head_gradient_tensors=len(headgrads),
            shared_head_calls=7,residual_calls=6,original_hard_gate_consumption_calls=6,previous_mask_gradients_finite=True,new_gradient_max=float(grad.abs().max()),parameter_update_max=update))
finally:
    decoder.prediction_head=original_prediction;decoder.refined_prediction_head=original_refined
    for hook in hooks:hook.remove()
model.eval();assert all(not m.training for m in model.modules());model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=256,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,
    original_initial_prediction_unchanged=True,static_mask_features_preserved=True,original_hard_attention_algorithm_preserved=True,
    exact_raw_current_query_previous_mask_padding_identities_verified=True,previous_mask_path_undetached=True,original_indicator_adjusted_mask_consumption_verified=True,
    synthetic_CPU_decoder_optimizer_steps=2,CPU_decoder_steps=step_rows,CPU_residual_paths=path_rows,
    nonzero_residual_changes_refined_and_final_masks=True,bypassed_residual_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU native full model/pretrain/TRAIN collation and synthetic component/complete decoder optimizer checks only. No sparse native TRAIN task, GPU PASS, efficacy or full-dataset memory guarantee.')),encoding='utf-8')
