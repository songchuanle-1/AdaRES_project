def check_path_calls(calls,mask_calls,score_calls,indicator_calls):
    assert len(calls)==7 and all(type(v) is int for v in calls)
    assert calls==list(range(7))
    assert mask_calls==score_calls==indicator_calls==7


def check_gradient_rows(rows):
    from math import isfinite
    assert len(rows)==7
    for i,row in enumerate(rows):
        assert type(row['stage']) is int and row['stage']==i
        assert row['parameter_tensors']==2 and row['all_finite'] is True
        assert isfinite(row['gradient_max']) and row['gradient_max']>0
        assert isfinite(row['parameter_update_max']) and row['parameter_update_max']>0


"""Two native TRAIN steps with all seven private output normalizations exercised."""

import subprocess
assert P['physical_gpu']==0
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())==280.0

import os
assert P['physical_gpu']==0
assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(P['physical_gpu'])
import base64
import gc
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
import yaml

assert torch.cuda.device_count()==1
torch.cuda.set_device(0)
torch.set_num_threads(2)
root=Path(P['root'])
stage=Path(P['output']).with_suffix('')
assert not stage.exists()
stage.mkdir()
native=importlib.import_module('train')
assert hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==FROZEN['train_source_lf_sha256']
row=FROZEN['variant']
assert row['id']=='AdaRES-Y67'
package=stage/row['package']
package.mkdir()
for name,value in row['python_files'].items():
    data=base64.b64decode(value['base64'])
    assert hashlib.sha256(data).hexdigest()==value['sha256']
    (package/name).write_bytes(data)
config_path=stage/Path(row['config']).name
config_data=base64.b64decode(row['config_data']['base64'])
assert hashlib.sha256(config_data).hexdigest()==row['config_data']['sha256']
config_path.write_bytes(config_data)
cfg=gorilla.Config.fromfile(str(config_path))
assert cfg.train.seed==1999 and cfg.train.epochs==30 and cfg.dataloader.train.batch_size==2
assert cfg.model.sampling_module.num_proposal==128
assert cfg.data.train.training and cfg.data.train.prefix=='train'
cfg.dist=False
namespace='ipdn.model_paper5.'+row['package']
importlib.import_module('ipdn.model_paper5')
assert namespace not in sys.modules
spec=importlib.util.spec_from_file_location(namespace,str(package/'__init__.py'),
    submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec)
sys.modules[namespace]=module
spec.loader.exec_module(module)
Candidate=native.resolve_model_class(str(config_path))
assert Candidate is module.IPDN
gorilla.set_random_seed(cfg.train.seed)
model=Candidate(**cfg.model).cuda()
optimizer=gorilla.build_optimizer(model,cfg.optimizer)
scheduler=gorilla.build_lr_scheduler(optimizer,cfg.lr_scheduler)
logger=importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model,cfg.train.pretrain,logger)
dataset=native.resolve_train_dataset(str(config_path),cfg.data.train.copy(),logger)
loader=native.build_dataloader(dataset,dist=False,**cfg.dataloader.train)
import subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl01-MS-7C60'
assert float(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=power.limit','--format=csv,noheader,nounits']).decode().strip())==280.0
assert cfg.model.dec.num_layer==6 and cfg.model.dec.iter_pred
assert sum(p.numel() for p in model.parameters())==164632709
norms=[model.dec.out_norm]+list(model.dec.refinement_out_norms)
assert len(norms)==7 and len({p.data_ptr() for n in norms for p in n.parameters()})==14
assert all(all(torch.equal(v,n.state_dict()[k]) for k,v in norms[0].state_dict().items()) for n in norms[1:])
assert not hasattr(model.dec,'refinement_indicators')
losses=[];gradients=[];batches=[];paths=[]
norm_calls=[];mask_calls=[];score_calls=[];indicator_calls=[];norm_outputs=[];mask_outputs=[]
formula_errors=[]

class FirstTwoBatches:
    sampler=loader.sampler
    def __len__(self):return 2
    def __iter__(self):
        for batch in itertools.islice(loader,2):
            assert len(batch['scan_ids'])==cfg.dataloader.train.batch_size
            batches.append(dict(scan_ids=[str(s) for s in batch['scan_ids']],expressions=len(batch['ann_ids']),points=int(batch['feats'].shape[0])))
            yield batch

class Writer:
    def add_scalar(self,*args,**kwargs):pass

def before_forward(module,args):
    bn=[m for block in (module.input_conv,module.unet,module.output_layer) for m in block.modules() if isinstance(m,nn.BatchNorm1d)]
    assert module.training and bn and all(m.training for m in bn)
    assert all(m.training for m in module.dec.modules() if isinstance(m,nn.BatchNorm1d))
    assert module.text_encoder.training and all(not p.requires_grad for p in module.text_encoder.parameters())
    paths.append(True)

def after_forward(module,args,value):
    loss=value[0];assert loss.requires_grad and torch.isfinite(loss).all();losses.append(float(loss.detach().cpu()))

def observe_norm(index):
    def hook(module,args,value):
        assert value.requires_grad and args[0].requires_grad and torch.isfinite(value).all()
        norm_calls.append(index);norm_outputs.append(value)
        with torch.no_grad():
            q=args[0][:,[0,64,127]].double()
            reference=(q-q.mean(-1,keepdim=True))*torch.rsqrt(q.var(-1,unbiased=False,keepdim=True)+module.eps)
            reference=reference*module.weight.double()+module.bias.double()
            actual=value[:,[0,64,127]].double()
            torch.testing.assert_close(actual,reference,rtol=3e-5,atol=2e-6)
            formula_errors.append(float((actual-reference).abs().max()))
    return hook

def observe_score(module,args):
    assert args[0] is norm_outputs[-1]
    score_calls.append(True)

original_get_mask=model.dec.get_mask

def checked_get_mask(query,mask_feats,batch_mask):
    assert query is norm_outputs[-1]
    value=original_get_mask(query,mask_feats,batch_mask)
    mask_calls.append(True);mask_outputs.append(value[0])
    return value

model.dec.get_mask=checked_get_mask

def observe_indicator(module,args,value):
    assert args[1] is mask_outputs[-1] and value.requires_grad and torch.isfinite(value).all()
    indicator_calls.append(True)

step=optimizer.step

def checked_step(*args,**kwargs):
    index=len(gradients);assert index<2
    check_path_calls(norm_calls[index*7:],len(mask_calls[index*7:]),len(score_calls[index*7:]),len(indicator_calls[index*7:]))
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    old=[];rows=[]
    for i,norm in enumerate(norms):
        ps=list(norm.parameters());gs=[p.grad for p in ps]
        assert len(gs)==2 and all(g is not None and torch.isfinite(g).all() and g.abs().max()>0 for g in gs)
        old.append([p.detach().clone() for p in ps])
        rows.append(dict(stage=i,parameter_tensors=2,all_finite=True,gradient_max=max(float(g.abs().max()) for g in gs)))
    returned=step(*args,**kwargs)
    for i,norm in enumerate(norms):
        ps=list(norm.parameters());assert all(torch.isfinite(p).all() for p in ps)
        rows[i]['parameter_update_max']=max(float((p.detach()-v).abs().max()) for p,v in zip(ps,old[i]))
    check_gradient_rows(rows)
    gradients.append(dict(tensors=len(active),all_finite=True,normalization_stages=rows,
        native_mask_and_score_consumption=True,shared_indicator_calls=7))
    return returned

optimizer.step=checked_step
hooks=[model.register_forward_pre_hook(before_forward),model.register_forward_hook(after_forward),
    model.dec.out_score.register_forward_pre_hook(observe_score),model.dec.indi_embedding.register_forward_hook(observe_indicator)]
hooks.extend(n.register_forward_hook(observe_norm(i)) for i,n in enumerate(norms))
torch.cuda.reset_peak_memory_stats()
try:
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer())
    torch.cuda.synchronize()
    assert len(losses)==len(gradients)==len(batches)==len(paths)==2
    assert norm_calls==list(range(7))*2
    assert len(mask_calls)==len(score_calls)==len(indicator_calls)==len(formula_errors)==14
    peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    model.dec.get_mask=original_get_mask
    for h in hooks:h.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
output=dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',
    steps=2,optimizer_steps=2,losses=losses,native_data_loaded=True,backward_finite=True,intended_path_observed=all(paths),
    data_source=dict(config_data_root=cfg.data.train.data_root,dataset_class=type(dataset).__module__+'.'+type(dataset).__name__),
    batches=batches,gradient_checks=gradients,stage_norm_calls=norm_calls,native_mask_calls=len(mask_calls),
    native_score_calls=len(score_calls),shared_indicator_calls=len(indicator_calls),formula_errors=formula_errors,
    peak_reserved_bytes=peak,model_weights_saved=False,validation_used=False,admission=False,
    limitation='Native two-step initialization only; no trained efficacy or full-dataset resource guarantee.')
Path(P['output']).write_text(json.dumps(output),encoding='utf-8')
