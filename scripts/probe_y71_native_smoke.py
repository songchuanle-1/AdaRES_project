import os
assert P['physical_gpu'] == FROZEN['runtime_physical_gpu']
assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(P['physical_gpu'])
import subprocess
assert subprocess.check_output(['hostname']).decode().strip() == 'dl01-MS-7C60'
assert float(subprocess.check_output(['nvidia-smi', '-i', str(P['physical_gpu']), '--query-gpu=power.limit',
    '--format=csv,noheader,nounits']).decode().strip()) == FROZEN['runtime_power_limit_watts']
import base64
import hashlib
import importlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import gorilla
import torch
from torch import nn

assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
torch.set_num_threads(2)
stage = Path(P['output']).with_suffix('')
assert not stage.exists()
stage.mkdir()
native = importlib.import_module('train')
assert hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == FROZEN['train_source_lf_sha256']
row = FROZEN['variant']
assert row['id'] == 'AdaRES-Y71'
package = stage / row['package']
package.mkdir()
for name, value in row['python_files'].items():
    raw = base64.b64decode(value['base64'])
    assert hashlib.sha256(raw).hexdigest() == value['sha256']
    (package / name).write_bytes(raw)
config_path = stage / Path(row['config']).name
raw = base64.b64decode(row['config_data']['base64'])
assert hashlib.sha256(raw).hexdigest() == row['config_data']['sha256']
config_path.write_bytes(raw)
cfg = gorilla.Config.fromfile(str(config_path))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.dataloader.train.batch_size == 2
assert cfg.model.sampling_module.num_proposal == 128
assert cfg.model.dec.num_layer == 6 and cfg.model.dec.iter_pred
assert cfg.data.train.training and cfg.data.train.prefix == 'train'
cfg.dist = False
namespace = 'ipdn.model_paper5.' + row['package']
importlib.import_module('ipdn.model_paper5')
assert namespace not in sys.modules
spec = importlib.util.spec_from_file_location(namespace, str(package / '__init__.py'),
    submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[namespace] = module
spec.loader.exec_module(module)
Candidate = native.resolve_model_class(str(config_path))
assert Candidate is module.IPDN
gorilla.set_random_seed(cfg.train.seed)
model = Candidate(**cfg.model).cuda()
assert sum(p.numel() for p in model.parameters()) == 164695173
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)
dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)
losses=[];gradient_checks=[];batches=[];modes=[];decoder_inputs=[];spatial=[];appearance=[];fusion_inputs=[];fused=[];consumed=[];indicators=[];norm_inputs=[];fusion_paths=[]
fusion=model.dec.visual_product_fusion;head=model.dec.indi_embedding
assert len(list(head.parameters()))==10 and torch.equal(fusion.weight.detach(),torch.zeros_like(fusion.weight))

def check_path_calls(fusion_count,consumer_count,indicator_count,norm_calls):
    assert fusion_count==consumer_count==1 and type(fusion_count) is int and type(consumer_count) is int
    assert indicator_count==7 and norm_calls==[True]*7 and all(type(v) is bool for v in norm_calls)

def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==2
    for index,row in enumerate(rows):
        assert type(row['weight_nonzero']) is bool
        assert row['native_projection_identity_verified'] is True and row['normalization_consumption_verified'] is True
        assert type(row['feature_rows']) is int and row['feature_rows']>=3
        assert row['reference_feature_rows']==[0,row['feature_rows']//2,row['feature_rows']-1]
        assert row['reference_channels']==[0,127,255]
        assert all(isfinite(row[k]) and row[k]>=0 for k in ('reference_error_max','fusion_delta_max','spatial_abs_max','appearance_abs_max'))
        if index==0:assert row['weight_nonzero'] is False and row['fusion_delta_max']==0
        else:assert row['weight_nonzero'] is True and row['fusion_delta_max']>0 and row['spatial_abs_max']>0 and row['appearance_abs_max']>0

class FirstTwoBatches:
    sampler=loader.sampler
    def __len__(self):return 2
    def __iter__(self):
        for batch in itertools.islice(loader,2):
            assert len(batch['scan_ids'])==2
            batches.append(dict(scan_ids=[str(s) for s in batch['scan_ids']],expressions=len(batch['ann_ids']),points=int(batch['feats'].shape[0])))
            yield batch
class Writer:
    def add_scalar(self,*args,**kwargs):pass

def before_model(module,args):
    bn=[m for block in (module.input_conv,module.unet,module.output_layer) for m in block.modules() if isinstance(m,nn.BatchNorm1d)]
    assert module.training and bn and all(m.training for m in bn)
    assert module.text_encoder.training and all(not p.requires_grad for p in module.text_encoder.parameters())
    modes.append(True)
def after_model(module,args,value):
    loss=value[0];assert loss.requires_grad and torch.isfinite(loss).all();losses.append(float(loss.detach().cpu()))
def capture_decoder(module,args):assert len(args)==8;decoder_inputs.append(args)
def capture_spatial(module,args,value):spatial.append((args[0],value))
def capture_appearance(module,args,value):appearance.append((args[0],value))
def capture_fusion(module,args,value):
    s,a=args;native=decoder_inputs[-1]
    assert s is spatial[-1][1] and a is appearance[-1][1]
    assert spatial[-1][0] is native[0] and appearance[-1][0] is native[6]
    assert s.shape==a.shape and s.ndim==2 and s.shape[0]==int(native[2][-1])
    assert all(t.requires_grad and torch.isfinite(t).all() for t in (s,a,value))
    for t in (s,a,value):t.retain_grad()
    fusion_inputs.append(args);fused.append(value)
    with torch.no_grad():
        rows=[0,s.shape[0]//2,s.shape[0]-1];channels=[0,127,255];reference=[];actual=[]
        for i in rows:
            for d in channels:
                residual=(s[i].double()*a[i].double()*module.weight[:,d].double()).sum()
                reference.append(s[i,d].double()+a[i,d].double()+residual);actual.append(value[i,d].double())
        expected=torch.stack(reference);observed=torch.stack(actual)
        torch.testing.assert_close(observed,expected,rtol=4e-5,atol=4e-6)
        anchor=s+a;nonzero=bool(module.weight.detach().abs().max()>0)
        if not nonzero:assert torch.equal(value,anchor)
        fusion_paths.append(dict(weight_nonzero=nonzero,feature_rows=s.shape[0],reference_feature_rows=rows,reference_channels=channels,
            reference_error_max=float((observed-expected).abs().max()),fusion_delta_max=float((value-anchor).abs().max()),
            spatial_abs_max=float(s.abs().max()),appearance_abs_max=float(a.abs().max()),
            native_projection_identity_verified=True,normalization_consumption_verified=False))
def consume_norm(module,args):
    assert args[0] is fused[-1];consumed.append(True);fusion_paths[-1]['normalization_consumption_verified']=True
def capture_indicator(module,args,value):
    assert args[0].requires_grad and args[1].requires_grad and torch.isfinite(value).all();indicators.append(True)
def capture_norm(module,args,value):norm_inputs.append(bool(args[0].requires_grad))

step=optimizer.step
def checked_step(*args,**kwargs):
    index=len(gradient_checks);assert index<2
    check_path_calls(len(fusion_inputs[index:]),len(consumed[index:]),len(indicators[index*7:]),norm_inputs[index*7:])
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None]
    assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    grad=fusion.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in fusion_inputs[index])
    assert fused[index].grad is not None and torch.isfinite(fused[index].grad).all()
    before=fusion.weight.detach().clone();gradmax=float(grad.abs().max());result=step(*args,**kwargs)
    update=float((fusion.weight.detach()-before).abs().max());assert update>0 and torch.isfinite(fusion.weight).all()
    gradient_checks.append(dict(all_finite=True,tensors=len(active),indicator_tensors=len(headgrads),indicator_calls=7,
        fusion_calls=1,normalization_consumption_calls=1,projected_branch_gradients_finite=True,
        fusion_weight_gradient_max=gradmax,parameter_update_max=update))
    return result

hooks=[model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),
    model.dec.register_forward_pre_hook(capture_decoder),model.dec.input_proj.register_forward_hook(capture_spatial),
    model.dec.input_proj_2d.register_forward_hook(capture_appearance),fusion.register_forward_hook(capture_fusion),
    model.dec.sum_norm.register_forward_pre_hook(consume_norm),head.register_forward_hook(capture_indicator),
    head.query_norm.register_forward_hook(capture_norm)]
optimizer.step=checked_step;torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradient_checks)==len(batches)==len(modes)==2
    assert len({s for b in batches for s in b['scan_ids']})==4
    check_response_rows(fusion_paths);peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    optimizer.step=step
    for h in hooks:h.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,
    losses=losses,native_data_loaded=True,backward_finite=True,intended_path_observed=True,batches=batches,
    gradient_checks=gradient_checks,fusion_paths=fusion_paths,fusion_calls=len(fusion_inputs),normalization_consumption_calls=len(consumed),
    shared_head_calls=len(indicators),native_modes_verified=True,peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),
    validation_used=False,model_weights_saved=False,admission=False,
    limitation='Actual native two-step TRAIN execution only, not efficacy or a full-dataset memory bound.')),encoding='utf-8')
