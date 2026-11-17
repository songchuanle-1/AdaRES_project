"""CPU native model/TRAIN loading and current-query-set to visual feedback controls."""
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


dec_module=importlib.import_module(namespace+'.dec');Feedback=dec_module.QueryMaskRefinement
assert torch.equal(model.dec.query_mask_refinement.weight,torch.zeros(256,256))
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();branch=Feedback(7).to(dtype=dtype);assert torch.equal(rng,torch.get_rng_state())
    visual=torch.randn(2,5,7,dtype=dtype,requires_grad=True);query=torch.randn(2,11,7,dtype=dtype,requires_grad=True)
    pad=torch.tensor([[False,False,False,True,True],[False,False,False,False,False]])
    value=branch(visual,query,pad);assert torch.equal(value,visual)
    grads=torch.autograd.grad(value.sum(),(visual,query,branch.weight))
    assert torch.equal(grads[0],torch.ones_like(visual)) and torch.equal(grads[1],torch.zeros_like(query)) and torch.isfinite(grads[2]).all() and grads[2].abs().max()>0
    with torch.no_grad():
        branch.weight.copy_(torch.arange(49,dtype=dtype).reshape(7,7)/49-.5)
        changed=branch(visual,query,pad);reference=visual.clone()
        for b in range(2):
            for t in range(5):
                if bool(pad[b,t]):continue
                logits=[sum(visual[b,t,k].double()*query[b,j,k].double() for k in range(7))/(7**.5) for j in range(11)]
                top=max(logits);weights=[torch.exp(s-top) for s in logits];den=sum(weights)
                context=[sum(weights[j]*query[b,j,k].double() for j in range(11))/den for k in range(7)]
                for d in range(7):reference[b,t,d]=visual[b,t,d].double()+sum(context[k]*branch.weight[k,d].double() for k in range(7))
        tol=128*torch.finfo(dtype).eps;torch.testing.assert_allclose(changed,reference,rtol=tol,atol=tol);assert not torch.allclose(changed,visual,rtol=tol,atol=tol)
        order=torch.tensor([3,1,10,0,2,5,4,6,7,8,9]);torch.testing.assert_allclose(branch(visual,query[:,order],pad),changed,rtol=tol,atol=tol)
        words=torch.tensor([3,1,4,0,2]);torch.testing.assert_allclose(branch(visual[:,words],query,pad[:,words]),changed[:,words],rtol=tol,atol=tol)
        torch.testing.assert_allclose(branch(visual.flip(0),query.flip(0),pad.flip(0)),changed.flip(0),rtol=tol,atol=tol)
        channels=torch.tensor([3,1,6,0,2,5,4]);other=Feedback(7).to(dtype=dtype);other.weight.copy_(branch.weight[channels][:,channels])
        torch.testing.assert_allclose(other(visual[:,:,channels],query[:,:,channels],pad),changed[:,:,channels],rtol=tol,atol=tol)
        polluted=visual.clone();polluted[pad]=1e4;observed=branch(polluted,query,pad)
        assert torch.equal(observed[~pad],changed[~pad]) and torch.equal(observed[pad],polluted[pad])
        assert torch.equal(branch(visual,query,torch.ones_like(pad)),visual)
        assert not torch.allclose(branch(visual,query.flip(0),pad),changed,rtol=tol,atol=tol)
        assert torch.isfinite(branch(visual*1e4,query*1e4,pad)).all()
        witness=Feedback(7).to(dtype=dtype);witness.weight.copy_(torch.eye(7,dtype=dtype));la=torch.ones(1,1,7,dtype=dtype);qa=torch.zeros(1,2,7,dtype=dtype);qa[0,0,0]=1;qa[0,1,0]=-1;qb=torch.zeros_like(qa);pa=torch.zeros(1,1,dtype=torch.bool)
        assert torch.equal(qa.mean(1),qb.mean(1)) and not torch.allclose(witness(la,qa,pa),witness(la,qb,pa),rtol=tol,atol=tol)
    grads=torch.autograd.grad(branch(visual,query,pad).square().mean(),(visual,query,branch.weight));assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    controls.append(dict(dtype=str(dtype),initial_RNG_unchanged=True,zero_output_and_input_gradients_exact=True,new_gradient_finite_nonzero=True,
        independent_scalar_softmax_context_projection=True,query_set_token_batch_channel_permutations=True,padding_only_feature_intervention_and_padded_identity=True,
        all_padding_branch_noop=True,wrong_query_batch_changes_output=True,equal_query_mean_different_distribution_distinguished=True,large_finite_inputs_finite=True,both_input_gradients_finite_nonzero=True))
kwargs=dict(cfg.model.dec,sampling_module=cfg.model.sampling_module,in_channel=cfg.model.media)
torch.manual_seed(1999);parent=parent_module.DEC(**kwargs);rng=torch.get_rng_state().clone();torch.manual_seed(1999);decoder=dec_module.DEC(**kwargs)
assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict());parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if k!='query_mask_refinement.weight'})
torch.manual_seed(1999);lengths=(512,513);total=sum(lengths)
inputs=dict(x=torch.randn(total,cfg.model.media),fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2,1),batch_offsets=torch.tensor([0,lengths[0],total],dtype=torch.int32),lang_feats=torch.randn(2,9,decoder.lang_proj.in_features),lang_masks=torch.ones(2,9,dtype=torch.long),sp_pos=torch.randn(total,3),feats_2d=torch.randn(total,decoder.input_proj_2d[0].in_features),point_mass=torch.arange(total).remainder(13).float()+1)
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
    decoder.query_mask_refinement.weight.copy_(.003*torch.eye(256));changed=decoder(**inputs)
    assert all(torch.equal(parts(changed)[0][k],parts(baseline)[0][k]) for k in ('masks','scores','indis'))
    assert not torch.allclose(parts(changed)[1]['masks'],parts(baseline)[1]['masks']) and not torch.allclose(changed['masks'],baseline['masks'])
    hook=decoder.query_mask_refinement.register_forward_hook(lambda m,a,o:a[0])
    try:bypassed=decoder(**inputs)
    finally:hook.remove()
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(bypassed)) for k in ('masks','scores','indis'))
    decoder.query_mask_refinement.weight.zero_()
model.dec=decoder;optimizer=gorilla.build_optimizer(model,cfg.optimizer)
step_rows=[];path_rows=[];state={};original_prediction=model.dec.prediction_head
head=model.dec.indi_embedding;refinement=model.dec.query_mask_refinement
def check_path_calls(queries,values,predictions,coarse,indicators,norms):
    assert type(queries) is int and type(values) is int and type(coarse) is int and (queries,values,coarse)==(7,6,6)
    assert predictions==list(range(1,7)) and all(type(v) is int for v in predictions)
    assert indicators==list(range(7)) and all(type(v) is int for v in indicators)
    assert norms==[True]*7 and all(type(v) is bool for v in norms)

def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==12
    for index,row in enumerate(rows):
        assert type(row['step']) is int and row['step']==index//6 and type(row['stage']) is int and row['stage']==index%6+1
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero']==bool(index>=6)
        for key in ('original_static_mask_base_identity','current_refined_query_identity','original_padding_identity','both_inputs_undetached','padded_output_identity','float64_attention_reference_checked'):
            assert row[key] is True
        delta=row['mask_feature_delta_max'];assert isfinite(delta) and delta>=0
        if index<6:assert delta==0
        else:assert delta>0

"""Exact callbacks shared by synthetic CPU and actual native Y81 task checks."""
def checked_prediction(q,features,pad):
    index=len(state['queries'])
    if index==0:
        assert not state['values'];state['features']=features;state['padding']=pad
    else:
        assert q is state['refinement_queries'][-1] and features is state['values'][-1] and pad is state['padding']
        assert len(state['values'])==index;state['prediction_consumers'].append(index)
    state['queries'].append(q);result=original_prediction(q,features,pad);state['mask']=result[1]
    expected=(result[1].sigmoid()<.5)|pad.unsqueeze(1)
    expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(result[2],expected) and not result[2].requires_grad
    state['gates'].append(result[2]);return result


def capture_refinement(module,args,value):
    index=len(state['values'])
    assert len(args)==3 and args[0] is state['features'] and args[2] is state['padding'] and len(state['queries'])==index+1
    assert all(t.requires_grad for t in args[:2])
    args[0].retain_grad();args[1].retain_grad();value.retain_grad()
    attention=torch.softmax(args[0].double()@args[1].double().transpose(-1,-2)/(args[1].shape[-1]**.5),dim=-1)
    reference=args[0].double()+((attention@args[1].double())@module.weight.double()).masked_fill(args[2].unsqueeze(-1),0.)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    nonzero=bool(module.weight.abs().max()>0);assert nonzero==bool(len(step_rows))
    delta=float((value-args[0]).abs().max());assert (delta>0)==nonzero and torch.equal(value[args[2]],args[0][args[2]])
    state['values'].append(value);state['mask_inputs'].append(args[0]);state['refinement_queries'].append(args[1])
    path_rows.append(dict(step=len(step_rows),stage=index+1,weight_nonzero=nonzero,mask_feature_delta_max=delta,
        original_static_mask_base_identity=True,current_refined_query_identity=True,original_padding_identity=True,
        both_inputs_undetached=True,padded_output_identity=True,float64_attention_reference_checked=True))


def before_coarse(module,args):
    index=state['coarse_calls'];assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1]
    state['coarse_calls']+=1


def before_indicator(module,args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries'])-1)


def capture_query_norm(module,args,value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad;state['norms'].append(True)
hooks=[refinement.register_forward_hook(capture_refinement),head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
hooks.extend(layer.register_forward_pre_hook(before_coarse) for layer in model.dec.qia_layers)
model.dec.prediction_head=checked_prediction
try:
    model.train(True)
    for index in range(2):
        state.clear();state.update(features=None,padding=None,queries=[],values=[],mask_inputs=[],refinement_queries=[],gates=[],prediction_consumers=[],coarse_calls=0,indicators=[],norms=[])
        optimizer.zero_grad();torch.manual_seed(1999+index);value=model.dec(**inputs)
        check_path_calls(len(state['queries']),len(state['values']),state['prediction_consumers'],state['coarse_calls'],state['indicators'],state['norms'])
        labels=torch.arange(256).reshape(2,128).remainder(2).long()
        loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
        assert torch.isfinite(loss);loss.backward()
        active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['mask_inputs']+state['refinement_queries']+state['values'])
        grad=refinement.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
        headgrads=[p.grad for p in head.parameters() if p.grad is not None];assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
        before=refinement.weight.detach().clone();optimizer.step();update=float((refinement.weight.detach()-before).abs().max());assert update>0
        step_rows.append(dict(step=index,synthetic_CPU_decoder_loss=float(loss),finite_gradient_tensors=len(active),original_head_gradient_tensors=10,
            shared_head_calls=7,mask_refinement_calls=6,prediction_consumption_calls=6,original_coarse_gate_consumption_calls=6,
            both_input_gradients_finite=True,new_gradient_max=float(grad.abs().max()),parameter_update_max=update))
    check_response_rows(path_rows)
finally:
    model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
model.eval();assert all(not m.training for m in model.modules());model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=65536,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,original_initial_prediction_unchanged=True,
    exact_static_mask_current_query_padding_identities_verified=True,both_input_paths_undetached=True,padded_output_identity=True,
    original_prediction_and_hard_gate_consumption_verified=True,original_shared_head_query_norm_consumption_verified=True,
    synthetic_CPU_decoder_optimizer_steps=2,CPU_decoder_steps=step_rows,CPU_refinement_paths=path_rows,
    nonzero_refinement_changes_refined_and_final_masks=True,bypassed_refinement_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU full native model/pretrain/TRAIN collation and synthetic component/decoder optimizer checks only; no native sparse TRAIN task/GPU PASS/admission/efficacy.')),encoding='utf8')
