"""CPU model/TRAIN loading and direct language mask modulation execution controls."""
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
dec_module=importlib.import_module(namespace+'.dec');Modulation=dec_module.LanguageMaskModulation
assert torch.equal(model.dec.language_mask_modulation.weight,torch.zeros(256,256))
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();branch=Modulation(256).to(dtype=dtype)
    assert torch.equal(rng,torch.get_rng_state())
    features=torch.randn(2,11,256,dtype=dtype,requires_grad=True)
    language=torch.randn(2,7,256,dtype=dtype,requires_grad=True)
    padded=torch.zeros(2,7,dtype=torch.bool);padded[0,-2:]=True
    value=branch(features,language,padded);assert torch.equal(value,features)
    fg,lg=torch.autograd.grad(value.square().sum(),(features,language),retain_graph=True)
    assert torch.equal(fg,2*features) and torch.equal(lg,torch.zeros_like(language))
    value.square().sum().backward()
    assert branch.weight.grad is not None and torch.isfinite(branch.weight.grad).all() and branch.weight.grad.abs().max()>0
    with torch.no_grad():
        branch.weight.copy_(.03*torch.eye(256,dtype=dtype))
        changed=branch(features.detach(),language.detach(),padded)
        sentence=torch.stack([sum(language.detach()[b,t] for t in range(7) if not padded[b,t])/int((~padded[b]).sum()) for b in range(2)])
        reference=features.detach()*(1+torch.tanh(sentence@branch.weight.detach()).unsqueeze(1))
        tol=64*torch.finfo(dtype).eps
        torch.testing.assert_allclose(changed,reference,rtol=tol,atol=tol)
        assert not torch.allclose(changed,features.detach(),rtol=tol,atol=tol)
        order=torch.tensor([3,1,6,0,2,5,4])
        torch.testing.assert_allclose(branch(features.detach(),language.detach()[:,order],padded[:,order]),changed,rtol=tol,atol=tol)
        noisy=language.detach().clone();noisy[padded]=float('nan')
        assert torch.equal(branch(features.detach(),noisy,padded),changed)
        repeated=language.detach().repeat_interleave(2,dim=1);repeat_padding=padded.repeat_interleave(2,dim=1)
        torch.testing.assert_allclose(branch(features.detach(),repeated,repeat_padding),changed,rtol=tol,atol=tol)
        assert torch.equal(branch(features.detach(),language.detach(),torch.ones_like(padded)),features.detach())
        wrong=branch(features.detach(),language.detach().flip(0),padded.flip(0));assert not torch.allclose(wrong,changed,rtol=tol,atol=tol)
        empty_features=torch.cat([features.detach(),torch.zeros(2,3,256,dtype=dtype)],1)
        output=branch(empty_features,language.detach(),padded)
        assert torch.equal(output[:,11:],torch.zeros_like(output[:,11:]))
        torch.testing.assert_allclose(output[:,:11],changed,rtol=tol,atol=tol)
        assert torch.isfinite(branch(features.detach(),language.detach()*1e6,padded)).all()
    f2=features.detach().requires_grad_();l2=language.detach().requires_grad_()
    grads=torch.autograd.grad(branch(f2,l2,padded).square().sum(),(f2,l2,branch.weight))
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(g.abs().max()>0 for g in grads) and torch.equal(grads[1][padded],torch.zeros_like(grads[1][padded]))
    controls.append(dict(dtype=str(dtype),initial_RNG_unchanged=True,zero_output_and_input_gradients_exact=True,
        modulation_weight_gradient_finite_nonzero=True,independent_valid_token_formula=True,
        token_permutation_and_duplication_invariant=True,padded_NaN_tokens_ignored_exactly=True,
        all_padding_noop_and_padded_superpoints_zero=True,wrong_language_control_changes_output=True,
        nonzero_modulation_feature_and_valid_language_gradients=True,extreme_valid_tokens_finite=True))
kwargs=dict(cfg.model.dec,sampling_module=cfg.model.sampling_module,in_channel=cfg.model.media)
torch.manual_seed(1999);parent=parent_module.DEC(**kwargs);rng=torch.get_rng_state().clone()
torch.manual_seed(1999);decoder=dec_module.DEC(**kwargs)
assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict());parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if k!='language_mask_modulation.weight'})
torch.manual_seed(1999);lengths=(512,513);total=sum(lengths)
inputs=dict(x=torch.randn(total,cfg.model.media),fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2,1),
    batch_offsets=torch.tensor([0,lengths[0],total],dtype=torch.int32),
    lang_feats=torch.randn(2,9,decoder.lang_proj.in_features),lang_masks=torch.ones(2,9,dtype=torch.long),
    sp_pos=torch.randn(total,3),feats_2d=torch.randn(total,decoder.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float()+1)
inputs['lang_masks'][0,-2:]=0;inputs['lang_feats'][0,-2:]=0
parts=lambda v:list(v['aux_outputs'])+[v]
parent.eval();decoder.eval()
with torch.no_grad():baseline=parent(**inputs);output=decoder(**inputs)
assert len(parts(output))==7 and all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(output)) for k in ('masks','scores','indis'))
def input_gradients(net):
    feed=dict(inputs);names=('x','lang_feats','feats_2d')
    for key in names:feed[key]=inputs[key].clone().requires_grad_()
    value=net(**feed);loss=sum(part[k].square().mean() for part in parts(value) for k in ('masks','scores','indis'))
    return torch.autograd.grad(loss,tuple(feed[k] for k in names))
for a,b in zip(input_gradients(parent),input_gradients(decoder)):torch.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
modulation_outputs=[];consumed=[];indicators=[]
hooks=[decoder.language_mask_modulation.register_forward_hook(lambda m,a,o:modulation_outputs.append(o)),decoder.indi_embedding.register_forward_hook(lambda m,a,o:indicators.append(o))]
original=decoder.prediction_head

def checked_prediction(query,mask_feats,batch_mask):
    assert mask_feats is modulation_outputs[-1];consumed.append(True)
    return original(query,mask_feats,batch_mask)

decoder.prediction_head=checked_prediction
try:
    decoder.train(True);torch.manual_seed(1999);value=decoder(**inputs)
    assert len(modulation_outputs)==1 and len(consumed)==len(indicators)==7
    labels=torch.arange(256).reshape(2,128).remainder(2).long()
    loss=sum(torch.nn.functional.cross_entropy(part['indis'].reshape(-1,2),labels.reshape(-1))+part['masks'].square().mean() for part in parts(value))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    grad=decoder.language_mask_modulation.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
finally:
    decoder.prediction_head=original
    for h in hooks:h.remove()
decoder.eval()
with torch.no_grad():
    decoder.language_mask_modulation.weight.copy_(.03*torch.eye(256));changed=decoder(**inputs)
    assert torch.equal(parts(changed)[0]['scores'],parts(baseline)[0]['scores'])
    assert not torch.allclose(parts(changed)[0]['masks'],parts(baseline)[0]['masks']) and not torch.allclose(changed['masks'],baseline['masks'])
    h=decoder.language_mask_modulation.register_forward_hook(lambda m,a,o:a[0])
    try:bypassed=decoder(**inputs)
    finally:h.remove()
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(bypassed)) for k in ('masks','scores','indis'))
model.eval();assert all(not m.training for m in model.modules())
model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],
    implementation_test_passed=True,full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,
    native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),
    parameters=parameters,additional_parameters=65536,component_controls=controls,decoder_initial_RNG_and_parent_weights_exact=True,
    decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,decoder_modulation_calls=1,decoder_prediction_consumption_calls=7,
    decoder_indicator_calls=7,decoder_backward_finite=True,decoder_modulation_gradient_finite_nonzero=True,
    nonzero_modulation_changes_initial_and_final_masks=True,bypassed_modulation_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU native YAML/full model/pretrain/real TRAIN collate plus synthetic component and decoder controls only. No sparse TRAIN criterion/optimizer steps, GPU native PASS or efficacy result.')),encoding='utf-8')
