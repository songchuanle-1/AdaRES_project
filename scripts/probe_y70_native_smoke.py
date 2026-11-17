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
assert row['id'] == 'AdaRES-Y70'
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
losses=[];gradient_checks=[];batches=[];native_modes=[];fusion_paths=[]
coarse_inputs=[];coarse_outputs=[];fine_outputs=[];fusion_inputs=[];fusion_outputs=[];consumed=[];indicators=[];norm_inputs=[]
fusion=model.dec.query_scale_fusion;head=model.dec.indi_embedding
assert len(list(head.parameters()))==10
assert torch.equal(fusion.weight.detach(),torch.zeros_like(fusion.weight))

def check_path_calls(fusion_stages,consumption_stages,indicator_count,norm_calls):
    assert fusion_stages==consumption_stages==list(range(6))
    assert all(type(i) is int for i in fusion_stages+consumption_stages)
    assert indicator_count==7 and norm_calls==[True]*7 and all(type(v) is bool for v in norm_calls)

def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==12
    for index,row in enumerate(rows):
        assert row['stage']==index%6 and type(row['stage']) is int
        assert type(row['weight_nonzero']) is bool
        assert row['native_branch_identity_verified'] is True and row['downstream_consumption_verified'] is True
        assert type(row['expressions']) is int and row['expressions']>=2 and row['reference_expression_rows']==[0,row['expressions']-1]
        assert row['reference_queries']==[0,127] and row['reference_channels']==[0,127,255]
        assert all(isfinite(row[k]) and row[k]>=0 for k in ('reference_error_max','fusion_delta_max','query_abs_max'))
        if index<6:assert row['weight_nonzero'] is False and row['fusion_delta_max']==0
        else:assert row['weight_nonzero'] is True and row['fusion_delta_max']>0

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
    native_modes.append(True)

def after_model(module,args,value):
    loss=value[0];assert loss.requires_grad and torch.isfinite(loss).all();losses.append(float(loss.detach().cpu()))

def capture_coarse(index,module,args,value):
    assert args[3].dtype==torch.bool and args[2].dtype==torch.bool
    assert args[1].requires_grad and value[0].requires_grad
    coarse_inputs.append((index,args[1]));coarse_outputs.append((index,value[0]))

def capture_fine(index,module,args,value):fine_outputs.append((index,value[0]))

def capture_fusion(module,args,value):
    index=len(fusion_inputs);stage=index%6
    query,coarse,fine=args
    assert coarse_inputs[index][0]==coarse_outputs[index][0]==fine_outputs[index][0]==stage
    assert query is coarse_inputs[index][1] and coarse is coarse_outputs[index][1] and fine is fine_outputs[index][1]
    assert all(t.requires_grad and torch.isfinite(t).all() for t in (*args,value))
    for t in (*args,value):t.retain_grad()
    fusion_inputs.append(args);fusion_outputs.append(value)
    with torch.no_grad():
        expr=[0,query.shape[0]-1];queries=[0,127];channels=[0,127,255]
        references=[];actual=[]
        for b in expr:
            for q in queries:
                for d in channels:
                    z=(query[b,q].double()*module.weight[:,d].double()).sum()
                    alpha=(1+z.tanh())/2
                    references.append((1-alpha)*coarse[b,q,d].double()+alpha*fine[b,q,d].double())
                    actual.append(value[b,q,d].double())
        expected=torch.stack(references);observed=torch.stack(actual)
        torch.testing.assert_close(observed,expected,rtol=4e-5,atol=4e-6)
        anchor=(coarse+fine)/2;nonzero=bool(module.weight.detach().abs().max()>0)
        if not nonzero:assert torch.equal(value,anchor)
        fusion_paths.append(dict(stage=stage,weight_nonzero=nonzero,expressions=query.shape[0],
            reference_expression_rows=expr,reference_queries=queries,reference_channels=channels,
            reference_error_max=float((observed-expected).abs().max()),fusion_delta_max=float((value-anchor).abs().max()),
            query_abs_max=float(query.abs().max()),native_branch_identity_verified=True,downstream_consumption_verified=False))

def capture_consumption(index,module,args):
    assert args[1] is fusion_outputs[-1] and fusion_paths[-1]['stage']==index
    fusion_paths[-1]['downstream_consumption_verified']=True;consumed.append(index)

def capture_indicator(module,args,value):
    assert args[0].requires_grad and args[1].requires_grad and torch.isfinite(value).all()
    indicators.append(True)

def capture_norm(module,args,value):norm_inputs.append(bool(args[0].requires_grad))

step=optimizer.step
def checked_step(*args,**kwargs):
    index=len(gradient_checks);assert index<2
    start=index*6
    check_path_calls([p['stage'] for p in fusion_paths[start:]],consumed[start:],len(indicators[index*7:]),norm_inputs[index*7:])
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    indicator=[p.grad for p in head.parameters() if p.grad is not None]
    assert len(indicator)==10 and all(torch.isfinite(g).all() for g in indicator)
    grad=fusion.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for args in fusion_inputs[start:] for t in args)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in fusion_outputs[start:])
    before=fusion.weight.detach().clone();gradmax=float(grad.abs().max())
    result=step(*args,**kwargs)
    update=float((fusion.weight.detach()-before).abs().max())
    assert update>0 and torch.isfinite(fusion.weight).all()
    gradient_checks.append(dict(all_finite=True,tensors=len(active),indicator_tensors=len(indicator),
        fusion_stage_calls=[p['stage'] for p in fusion_paths[start:]],consumption_stage_calls=consumed[start:],
        indicator_calls=7,fusion_weight_gradient_max=gradmax,parameter_update_max=update,
        all_three_branch_input_gradients_finite=True,original_coarse_Boolean_masks_preserved=True))
    return result

from functools import partial
hooks=[model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),
    fusion.register_forward_hook(capture_fusion),head.register_forward_hook(capture_indicator),
    head.query_norm.register_forward_hook(capture_norm)]
for i,layer in enumerate(model.dec.qia_layers):hooks.append(layer.register_forward_hook(partial(capture_coarse,i)))
for i,layer in enumerate(model.dec.qia_layers_fine):hooks.append(layer.register_forward_hook(partial(capture_fine,i)))
for i,layer in enumerate(model.dec.qla_layers):hooks.append(layer.register_forward_pre_hook(partial(capture_consumption,i)))
optimizer.step=checked_step
torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer())
    torch.cuda.synchronize()
finally:
    optimizer.step=step
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
assert len(losses)==len(gradient_checks)==len(batches)==len(native_modes)==2
assert len({s for b in batches for s in b['scan_ids']})==4
check_response_rows(fusion_paths)
peak=torch.cuda.max_memory_reserved();assert 0<peak<24*2**30
result=dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,
    losses=losses,native_data_loaded=True,backward_finite=True,intended_path_observed=True,batches=batches,
    gradient_checks=gradient_checks,fusion_paths=fusion_paths,fusion_calls=12,consumption_stage_calls=consumed,
    shared_head_calls=len(indicators),native_modes_verified=True,peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),
    validation_used=False,model_weights_saved=False,admission=False,
    limitation='Two actual native TRAIN task steps verify execution and gradients only, not trained efficacy or a full-dataset memory bound.')
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
