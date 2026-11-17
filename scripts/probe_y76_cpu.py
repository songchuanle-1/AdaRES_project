"""CPU native model/TRAIN loading and confidence-only stage conditioning controls."""
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
parameters = sum(p.numel() for p in model.parameters()); assert parameters == 164631429
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')


dec_module=importlib.import_module(namespace+'.dec');Condition=dec_module.StageConfidenceConditioning
assert torch.equal(model.dec.indi_embedding.stage_conditioning.offset,torch.zeros(7,256))
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();branch=Condition(7).to(dtype=dtype);assert torch.equal(rng,torch.get_rng_state())
    query=torch.randn(2,11,7,dtype=dtype,requires_grad=True)
    for stage_index in range(7):
        value=branch(query,stage_index);assert torch.equal(value,query)
        dq,dw=torch.autograd.grad(value.sum(),(query,branch.offset))
        assert torch.equal(dq,torch.ones_like(query))
        expected=torch.zeros_like(dw);expected[stage_index]=22
        assert torch.equal(dw,expected)
    for invalid in (-1,7,0.,True):
        try:branch(query,invalid)
        except ValueError:pass
        else:raise AssertionError('invalid stage accepted')
    with torch.no_grad():
        branch.offset.copy_(torch.arange(49,dtype=dtype).reshape(7,7)/49-.5)
        changed=branch(query,3);reference=torch.empty_like(query)
        for b in range(2):
            for q in range(11):
                for d in range(7):reference[b,q,d]=query[b,q,d].double()+branch.offset[3,d].double()
        assert torch.equal(changed,reference)
        order=torch.tensor([3,1,10,0,2,5,4,6,7,8,9]);assert torch.equal(branch(query[:,order],3),changed[:,order])
        assert torch.equal(branch(query.flip(0),3),changed.flip(0))
        perm=torch.tensor([3,1,6,0,2,5,4]);other=Condition(7).to(dtype=dtype);other.offset.copy_(branch.offset[:,perm])
        assert torch.equal(other(query[:,:,perm],3),changed[:,:,perm])
        before=branch(query,4).clone();branch.offset[3].add_(.5);assert torch.equal(before,branch(query,4))
        assert not torch.equal(branch(query,3),changed)
    controls.append(dict(dtype=str(dtype),initial_RNG_unchanged=True,zero_output_and_input_gradient_exact=True,
        independent_scalar_stage_formula=True,stage_sparse_offset_gradient_exact=True,invalid_stage_refused=True,
        query_batch_channel_permutations_exact=True,other_stage_intervention_exact=True))
kwargs=dict(cfg.model.dec,sampling_module=cfg.model.sampling_module,in_channel=cfg.model.media)
torch.manual_seed(1999);parent=parent_module.DEC(**kwargs);rng=torch.get_rng_state().clone()
torch.manual_seed(1999);decoder=dec_module.DEC(**kwargs)
assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict());parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if k!='indi_embedding.stage_conditioning.offset'})
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
condition_calls=[];head_calls=[];consumption=[];condition_outputs=[]
def head_input(module,args):
    assert len(args)==5 and args[4]==len(head_calls);head_calls.append(args)
def condition_output(module,args,value):
    query,stage_index=args;assert stage_index==len(condition_calls)
    assert query is head_calls[stage_index][0] and query.requires_grad and torch.equal(query,value)
    condition_calls.append(stage_index);condition_outputs.append(value)
def consume(module,args):
    index=len(consumption);assert args[0] is condition_outputs[index];consumption.append(index)
hooks=[decoder.indi_embedding.register_forward_pre_hook(head_input),decoder.indi_embedding.stage_conditioning.register_forward_hook(condition_output),decoder.indi_embedding.query_norm.register_forward_pre_hook(consume)]
try:
    decoder.train(True);torch.manual_seed(1999);value=decoder(**inputs)
    assert condition_calls==consumption==list(range(7)) and len(head_calls)==7
    labels=torch.arange(256).reshape(2,128).remainder(2).long()
    loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    grad=decoder.indi_embedding.stage_conditioning.offset.grad
    assert grad is not None and torch.isfinite(grad).all() and (grad.abs().amax(1)>0).all()
finally:
    for hook in hooks:hook.remove()
decoder.eval()
with torch.no_grad():
    decoder.indi_embedding.stage_conditioning.offset[3].copy_(.03*torch.linspace(-1,1,256));changed=decoder(**inputs)
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(changed)) for k in ('masks','scores'))
    assert not torch.allclose(parts(changed)[3]['indis'],parts(baseline)[3]['indis'])
    assert all(torch.equal(parts(changed)[i]['indis'],parts(baseline)[i]['indis']) for i in range(7) if i!=3)
    hook=decoder.indi_embedding.stage_conditioning.register_forward_hook(lambda m,a,o:a[0])
    try:bypassed=decoder(**inputs)
    finally:hook.remove()
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(bypassed)) for k in ('masks','scores','indis'))
model.eval();assert all(not m.training for m in model.modules());model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,
    full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),parameters=parameters,additional_parameters=1792,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,
    original_seven_head_query_and_stage_identities_verified=True,original_query_norm_consumption_verified=True,decoder_conditioning_calls=7,decoder_indicator_calls=7,
    decoder_backward_finite=True,all_seven_offset_rows_gradient_finite_nonzero=True,nonzero_stage_only_changes_corresponding_confidence=True,
    every_mask_and_score_unchanged_with_nonzero_offset=True,bypassed_conditioning_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU full native model/pretrain/TRAIN collation and synthetic component/decoder checks only; no sparse native task loss/backward/optimizer or GPU PASS/efficacy.')),encoding='utf-8')
