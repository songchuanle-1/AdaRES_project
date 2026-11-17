"""CPU model/TRAIN loading and stage-specific prediction normalization controls."""
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
parameters = sum(p.numel() for p in model.parameters())
assert parameters == 164632709
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
norms = [model.dec.out_norm] + list(model.dec.refinement_out_norms)
assert len(norms) == 7 and all(isinstance(n, nn.LayerNorm) for n in norms)
assert len({p.data_ptr() for n in norms for p in n.parameters()}) == 14
assert all(all(torch.equal(v, n.state_dict()[k]) for k, v in norms[0].state_dict().items()) for n in norms[1:])
controls=[]
for dtype in (torch.float32, torch.float64):
    model.dec.to(dtype=dtype).eval()
    model.dec.zero_grad(set_to_none=True)
    torch.manual_seed(1999)
    query=torch.randn(2,5,256,dtype=dtype)
    masks=torch.randn(2,9,256,dtype=dtype)
    padding=torch.zeros(2,9,dtype=torch.bool);padding[0,-2:]=True
    values=[model.dec.prediction_head(query,masks,padding,stage=i) for i in range(7)]
    assert all(torch.equal(v[k],values[0][k]) for v in values[1:] for k in (0,1))
    # Independent affine LayerNorm formula and native mask/scoring consumers.
    expected=torch.nn.functional.layer_norm(query,(256,),norms[0].weight,norms[0].bias,norms[0].eps)
    expected_masks=torch.einsum('bqd,bmd->bqm',expected,masks)
    torch.testing.assert_allclose(values[0][1],expected_masks,rtol=1e-6,atol=1e-6)
    torch.testing.assert_allclose(values[0][0],model.dec.out_score(expected),rtol=1e-6,atol=1e-6)
    original=norms[-1].bias.detach().clone()
    with torch.no_grad():
        norms[-1].bias[0].add_(.25)
        changed=[model.dec.prediction_head(query,masks,padding,stage=i) for i in range(7)]
        assert all(torch.equal(changed[i][k],values[i][k]) for i in range(6) for k in (0,1))
        assert not torch.equal(changed[-1][1],values[-1][1])
        norms[-1].bias.copy_(original)
    valid=(~padding).unsqueeze(1).expand(2,5,9)
    private=values[-1][1][valid].square().mean()+values[-1][0].square().mean()
    private.backward()
    assert all(p.grad is None for n in norms[:-1] for p in n.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in norms[-1].parameters())
    model.dec.zero_grad(set_to_none=True)
    for i in range(7):
        scores,pred,_=model.dec.prediction_head(query+i/7.,masks,padding,stage=i)
        (pred[valid].square().mean()+scores.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for n in norms for p in n.parameters())
    controls.append(dict(dtype=str(dtype),seven_disjoint_norms=True,identical_initial_predictions=True,
        independent_normalization_formula=True,one_stage_intervention_isolated=True,
        private_loss_no_sibling_norm_gradients=True,all_stage_norm_gradients_finite_nonzero=True))
model.dec.float()
model.dec.zero_grad(set_to_none=True)
kwargs=dict(cfg.model.dec,sampling_module=cfg.model.sampling_module,in_channel=cfg.model.media)
torch.manual_seed(1999);parent=parent_module.DEC(**kwargs);rng=torch.get_rng_state().clone()
torch.manual_seed(1999);decoder=dec_module.DEC(**kwargs)
assert torch.equal(torch.get_rng_state(),rng)
assert all(torch.equal(v,decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if not k.startswith('refinement_out_norms.')})
torch.manual_seed(1999)
lengths=(512,513);total=sum(lengths)
inputs=dict(x=torch.randn(total,cfg.model.media),fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2,1),
    batch_offsets=torch.tensor([0,lengths[0],total],dtype=torch.int32),
    lang_feats=torch.randn(2,9,decoder.lang_proj.in_features),lang_masks=torch.ones(2,9,dtype=torch.long),
    sp_pos=torch.randn(total,3),feats_2d=torch.randn(total,decoder.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float()+1)
inputs['lang_masks'][0,-2:]=0;inputs['lang_feats'][0,-2:]=0
parts=lambda v:list(v['aux_outputs'])+[v]
parent.eval();decoder.eval()
with torch.no_grad():
    baseline=parent(**inputs);output=decoder(**inputs)
assert len(parts(output))==7
assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(output)) for k in ('masks','scores','indis'))
def input_gradients(net):
    feed=dict(inputs);names=('x','lang_feats','feats_2d')
    for key in names:feed[key]=inputs[key].clone().requires_grad_()
    values=net(**feed)
    loss=sum(part[k].square().mean() for part in parts(values) for k in ('masks','scores','indis'))
    return torch.autograd.grad(loss,tuple(feed[k] for k in names))
for a,b in zip(input_gradients(parent),input_gradients(decoder)):
    torch.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
calls=[0]*7;indicator=[]
active_norms=[decoder.out_norm]+list(decoder.refinement_out_norms)
def observed(i):
    def hook(module,args,value):
        assert value.requires_grad and torch.isfinite(value).all();calls[i]+=1
    return hook
hooks=[n.register_forward_hook(observed(i)) for i,n in enumerate(active_norms)]
hooks.append(decoder.indi_embedding.register_forward_hook(lambda m,a,o:indicator.append(o)))
try:
    decoder.train(True);torch.manual_seed(1999);output=decoder(**inputs)
    assert calls==[1]*7 and len(indicator)==7
    labels=torch.arange(256).reshape(2,128).remainder(2).long()
    loss=sum(torch.nn.functional.cross_entropy(part['indis'].reshape(-1,2),labels.reshape(-1))
        +part['masks'].square().mean()+part['scores'].square().mean() for part in parts(output))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for n in active_norms for p in n.parameters())
finally:
    for hook in hooks:hook.remove()
decoder.eval()
with torch.no_grad():
    original=active_norms[-1].bias.clone();active_norms[-1].bias[0].add_(.25)
    changed=decoder(**inputs)
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline)[:-1],parts(changed)[:-1]) for k in ('masks','scores','indis'))
    assert not torch.allclose(changed['masks'],baseline['masks'])
    active_norms[-1].bias.copy_(original);restored=decoder(**inputs)
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(restored)) for k in ('masks','scores','indis'))
model.eval();assert all(not m.training for m in model.modules())
model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],
    goal_sha256=PAYLOAD['goal_sha256'],implementation_test_passed=True,full_model_CPU_constructed=True,
    configured_TRAIN_data_collated=True,released_backbone_loaded=True,native_YAML_semantically_unchanged=True,
    train_eval_train_roundtrip=True,points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),
    parameters=parameters,additional_parameters=3072,component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True,decoder_all_stages_anchor_exact=True,
    decoder_input_gradients_anchor_close=True,decoder_norm_calls=calls,decoder_indicator_calls=len(indicator),
    decoder_backward_finite=True,decoder_all_norm_gradients_finite_nonzero=True,
    final_stage_intervention_changes_only_final_prediction=True,restoration_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,
    validation_used=False,admission=False,limitation='CPU native YAML/full model/released backbone/real TRAIN collation plus synthetic prediction/decoder controls. No native sparse criterion/optimizer steps, GPU checks or training efficacy.')),encoding='utf-8')
