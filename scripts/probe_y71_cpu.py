"""CPU native-model/TRAIN loading and visual product interaction controls."""
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
dec_module=importlib.import_module(namespace+'.dec');Fusion=dec_module.VisualProductFusion
assert torch.equal(model.dec.visual_product_fusion.weight,torch.zeros(256,256))
controls=[]
for dtype in (torch.float32,torch.float64):
    torch.manual_seed(1999);rng=torch.get_rng_state().clone();branch=Fusion(7).to(dtype=dtype)
    assert torch.equal(rng,torch.get_rng_state())
    spatial,appearance=[torch.randn(2,5,7,dtype=dtype,requires_grad=True) for _ in range(2)]
    value=branch(spatial,appearance);anchor=spatial+appearance
    assert torch.equal(value,anchor)
    grads=torch.autograd.grad(value.sum(),(spatial,appearance,branch.weight))
    assert torch.equal(grads[0],torch.ones_like(spatial)) and torch.equal(grads[1],torch.ones_like(appearance))
    assert torch.isfinite(grads[2]).all() and grads[2].abs().max()>0
    with torch.no_grad():
        branch.weight.copy_(torch.arange(49,dtype=dtype).reshape(7,7)/49-.5)
        changed=branch(spatial,appearance);reference=torch.empty_like(changed)
        for b in range(2):
            for i in range(5):
                for d in range(7):
                    residual=sum(spatial[b,i,k].double()*appearance[b,i,k].double()*branch.weight[k,d].double() for k in range(7))
                    reference[b,i,d]=spatial[b,i,d].double()+appearance[b,i,d].double()+residual
        tol=128*torch.finfo(dtype).eps
        torch.testing.assert_allclose(changed,reference,rtol=tol,atol=tol)
        assert not torch.allclose(changed,anchor,rtol=tol,atol=tol)
        rows=torch.tensor([3,1,4,0,2])
        assert torch.equal(branch(spatial[:,rows],appearance[:,rows]),changed[:,rows])
        assert torch.equal(branch(spatial.flip(0),appearance.flip(0)),changed.flip(0))
        perm=torch.tensor([3,1,6,0,2,5,4]);rotated=Fusion(7).to(dtype=dtype)
        rotated.weight.copy_(branch.weight[perm][:,perm])
        torch.testing.assert_allclose(rotated(spatial[:,:,perm],appearance[:,:,perm]),changed[:,:,perm],rtol=tol,atol=tol)
        assert torch.equal(branch(torch.zeros_like(spatial),appearance),appearance)
        assert torch.equal(branch(spatial,torch.zeros_like(appearance)),spatial)
        assert torch.equal(branch(appearance,spatial),changed)
        assert not torch.allclose(branch(spatial,appearance.flip(1)),changed,rtol=tol,atol=tol)
        one=torch.ones(2,5,7,dtype=dtype)
        first=branch(.5*one,1.5*one);second=branch(one,one)
        assert torch.equal(.5*one+1.5*one,one+one) and not torch.allclose(first,second,rtol=tol,atol=tol)
        assert torch.isfinite(branch(spatial*1e4,appearance*1e4)).all()
    s=spatial.detach().requires_grad_();a=appearance.detach().requires_grad_()
    grads=torch.autograd.grad(branch(s,a).square().mean(),(s,a,branch.weight))
    assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    controls.append(dict(dtype=str(dtype),initial_RNG_unchanged=True,zero_sum_output_and_input_gradients_exact=True,
        weight_gradient_finite_nonzero=True,independent_scalar_product_formula=True,joint_row_batch_channel_permutations=True,
        zero_factor_no_residual=True,branch_exchange_symmetric=True,wrong_correspondence_changes_output=True,
        same_sum_different_factorization_distinguished=True,large_finite_inputs_finite=True,all_input_gradients_finite_nonzero=True))

kwargs=dict(cfg.model.dec,sampling_module=cfg.model.sampling_module,in_channel=cfg.model.media)
torch.manual_seed(1999);parent=parent_module.DEC(**kwargs);rng=torch.get_rng_state().clone()
torch.manual_seed(1999);decoder=dec_module.DEC(**kwargs)
assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,decoder.state_dict()[k]) for k,v in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({k:v for k,v in decoder.state_dict().items() if k!='visual_product_fusion.weight'})
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
    value=net(**feed);loss=sum(p[k].square().mean() for p in parts(value) for k in ('masks','scores','indis'))
    return torch.autograd.grad(loss,tuple(feed[k] for k in names))
for a,b in zip(input_gradients(parent),input_gradients(decoder)):torch.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
spatial=[];appearance=[];fused=[];consumed=[];indicators=[]
def capture_fusion(module,args,value):
    assert args[0] is spatial[-1] and args[1] is appearance[-1]
    assert torch.equal(value,spatial[-1]+appearance[-1]);fused.append(value)
def consume(module,args):
    assert args[0] is fused[-1];consumed.append(True)
hooks=[decoder.input_proj.register_forward_hook(lambda m,a,o:spatial.append(o)),
    decoder.input_proj_2d.register_forward_hook(lambda m,a,o:appearance.append(o)),
    decoder.visual_product_fusion.register_forward_hook(capture_fusion),
    decoder.sum_norm.register_forward_pre_hook(consume),decoder.indi_embedding.register_forward_hook(lambda m,a,o:indicators.append(o))]
try:
    decoder.train(True);torch.manual_seed(1999);value=decoder(**inputs)
    assert len(fused)==len(consumed)==len(spatial)==len(appearance)==1 and len(indicators)==7
    labels=torch.arange(256).reshape(2,128).remainder(2).long()
    loss=sum(torch.nn.functional.cross_entropy(p['indis'].reshape(-1,2),labels.reshape(-1))+p['masks'].square().mean() for p in parts(value))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    grad=decoder.visual_product_fusion.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
finally:
    for h in hooks:h.remove()
decoder.eval()
with torch.no_grad():
    decoder.visual_product_fusion.weight.copy_(.003*torch.eye(256));changed=decoder(**inputs)
    assert not torch.allclose(parts(changed)[0]['masks'],parts(baseline)[0]['masks']) and not torch.allclose(changed['masks'],baseline['masks'])
    h=decoder.visual_product_fusion.register_forward_hook(lambda m,a,o:a[0]+a[1])
    try:bypassed=decoder(**inputs)
    finally:h.remove()
    assert all(torch.equal(a[k],b[k]) for a,b in zip(parts(baseline),parts(bypassed)) for k in ('masks','scores','indis'))
model.eval();assert all(not m.training for m in model.modules())
model.train(True);assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],goal_sha256=PAYLOAD['goal_sha256'],
    implementation_test_passed=True,full_model_CPU_constructed=True,configured_TRAIN_data_collated=True,released_backbone_loaded=True,
    native_YAML_semantically_unchanged=True,train_eval_train_roundtrip=True,points=int(batch['feats'].shape[0]),expressions=len(batch['ann_ids']),
    parameters=parameters,additional_parameters=65536,component_controls=controls,decoder_initial_RNG_and_parent_weights_exact=True,
    decoder_all_stages_anchor_exact=True,decoder_input_gradients_anchor_close=True,decoder_fusion_calls=1,decoder_normalization_consumers=1,
    decoder_indicator_calls=7,decoder_backward_finite=True,decoder_fusion_gradient_finite_nonzero=True,
    nonzero_fusion_changes_initial_and_final_masks=True,bypassed_fusion_negative_control_exact=True,
    CUDA_initialized=False,native_smoke_passed=False,native_task_backward=False,native_sparse_forward=False,validation_used=False,admission=False,
    limitation='CPU native YAML/full model/released backbone/actual TRAIN collation and synthetic component/decoder backward only; no sparse TRAIN criterion/optimizer or GPU PASS/efficacy.')),encoding='utf-8')
