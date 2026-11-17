"""Two original native TRAIN steps with shared confidence stage-input observation."""
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
from functools import partial
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
assert row['id'] == 'AdaRES-Y76'
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
assert sum(p.numel() for p in model.parameters()) == 164631429
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)

dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)



losses=[];gradient_checks=[];batches=[];modes=[];states=[];paths=[]
head=model.dec.indi_embedding;condition=head.stage_conditioning
assert len([p for p in head.parameters() if p is not condition.offset])==10
assert torch.equal(condition.offset.detach(),torch.zeros_like(condition.offset))

def check_path_calls(conditions,consumers,indicators,predictions,norms):
    for v in (conditions,consumers,indicators,predictions):
        assert v==list(range(7)) and all(type(i) is int for i in v)
    assert norms==[True]*7 and all(type(i) is bool for i in norms)

def check_native_inputs(query,head_query,prediction_query,stage_index,expected_stage):
    assert query is head_query is prediction_query
    assert type(stage_index) is int and stage_index==expected_stage

def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==14
    for i,r in enumerate(rows):
        assert type(r['stage']) is int and r['stage']==i%7
        assert type(r['offset_nonzero']) is bool and r['offset_nonzero']==bool(i>=7)
        for k in ('original_head_query_identity_verified','prediction_query_identity_verified','query_undetached_verified','query_norm_consumption_verified','static_mask_features_preserved','padding_identity_verified','original_mask_identity_verified'):
            assert r[k] is True
        n=r['expressions'];assert type(n) is int and n>=2 and r['reference_expression_rows']==[0,n-1]
        assert type(r['queries']) is int and r['queries']==128 and r['reference_queries']==[0,127]
        assert r['reference_channels']==[0,127,255]
        for k in ('reference_error_max','conditioning_delta_max','offset_abs_max'):
            assert isfinite(r[k]) and r[k]>=0
        if i<7:assert r['conditioning_delta_max']==r['offset_abs_max']==0
        else:assert r['conditioning_delta_max']>0 and r['offset_abs_max']>0

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
    assert module.text_encoder.training and all(not p.requires_grad for p in module.text_encoder.parameters());modes.append(True)
def after_model(module,args,value):
    loss=value[0];assert loss.requires_grad and torch.isfinite(loss).all();losses.append(float(loss.detach().cpu()))
def before_decoder(module,args):
    assert len(args)==8 and len(states)==len(gradient_checks)
    states.append(dict(conditions=[],consumers=[],indicators=[],predictions=[],norms=[],queries=[],outputs=[],head_inputs=[]))
prediction=model.dec.prediction_head

def checked_prediction(query,features,padded):
    state=states[-1];index=len(state['predictions']);assert index<7
    if index==0:state.update(mask_features=features,padding=padded)
    else:assert features is state['mask_features'] and padded is state['padding']
    result=prediction(query,features,padded)
    state['predictions'].append(index);state['queries'].append(query);state['mask']=result[1]
    return result

def before_head(module,args):
    state=states[-1];index=len(state['head_inputs']);assert len(args)==5 and args[4]==index
    assert args[0] is state['queries'][index] and args[1] is state['mask'] and args[2] is state['padding']
    state['head_inputs'].append(args)

def capture_condition(module,args,value):
    state=states[-1];index=len(state['conditions']);query,stage_index=args
    check_native_inputs(query,state['head_inputs'][index][0],state['queries'][index],stage_index,index)
    assert query.shape==value.shape and query.shape[1:]==(128,256)
    assert all(t.requires_grad and torch.isfinite(t).all() for t in (query,value));query.retain_grad();value.retain_grad()
    with torch.no_grad():
        n=query.shape[0];reference=[];observed=[]
        for b in (0,n-1):
            for q in (0,127):
                for d in (0,127,255):
                    reference.append(query[b,q,d].double()+module.offset[stage_index,d].double());observed.append(value[b,q,d].double())
        expected=torch.stack(reference);actual=torch.stack(observed)
        torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
        nonzero=bool(module.offset[stage_index].abs().max()>0)
        if not nonzero:assert torch.equal(query,value)
        record=dict(stage=stage_index,offset_nonzero=nonzero,expressions=n,queries=query.shape[1],reference_expression_rows=[0,n-1],reference_queries=[0,127],reference_channels=[0,127,255],
            reference_error_max=float((actual-expected).abs().max()),conditioning_delta_max=float((value-query).abs().max()),offset_abs_max=float(module.offset[stage_index].abs().max()),
            original_head_query_identity_verified=True,prediction_query_identity_verified=True,query_undetached_verified=True,query_norm_consumption_verified=False,
            static_mask_features_preserved=True,padding_identity_verified=True,original_mask_identity_verified=True)
    state['conditions'].append(index);state['outputs'].append(value);paths.append(record)

def capture_norm(module,args,value):
    state=states[-1];index=len(state['consumers']);assert args[0] is state['outputs'][index]
    state['consumers'].append(index);state['norms'].append(bool(args[0].requires_grad));paths[-1]['query_norm_consumption_verified']=True

def capture_indicator(module,args,value):
    state=states[-1];index=len(state['indicators']);assert args[4]==index and torch.isfinite(value).all()
    state['indicators'].append(index)

step=optimizer.step

def checked_step(*args,**kwargs):
    index=len(gradient_checks);assert index<2 and len(states)==index+1;state=states[index]
    check_path_calls(state['conditions'],state['consumers'],state['indicators'],state['predictions'],state['norms'])
    active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p is not condition.offset and p.grad is not None]
    assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    grad=condition.offset.grad;assert grad is not None and torch.isfinite(grad).all()
    row_grad=grad.abs().amax(1);assert (row_grad>0).all()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['queries']+state['outputs'])
    before=condition.offset.detach().clone();gradmax=float(grad.abs().max());result=step(*args,**kwargs)
    row_update=(condition.offset.detach()-before).abs().amax(1);assert (row_update>0).all() and torch.isfinite(condition.offset).all()
    gradient_checks.append(dict(all_finite=True,tensors=len(active),indicator_tensors=len(headgrads),indicator_calls=7,conditioning_calls=7,query_norm_consumption_calls=7,
        original_queries_and_conditioned_queries_gradients_finite=True,stage_offset_gradient_max=gradmax,per_stage_gradient_max=row_grad.tolist(),parameter_update_max=float(row_update.max()),per_stage_update_max=row_update.tolist()))
    return result

hooks=[model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),model.dec.register_forward_pre_hook(before_decoder),
    head.register_forward_pre_hook(before_head),condition.register_forward_hook(capture_condition),head.query_norm.register_forward_hook(capture_norm),head.register_forward_hook(capture_indicator)]
model.dec.prediction_head=checked_prediction;optimizer.step=checked_step;torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradient_checks)==len(batches)==len(modes)==len(states)==2
    assert len({s for b in batches for s in b['scan_ids']})==4
    check_response_rows(paths);peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    optimizer.step=step;model.dec.prediction_head=prediction
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,losses=losses,
    native_data_loaded=True,backward_finite=True,intended_path_observed=True,batches=batches,gradient_checks=gradient_checks,conditioning_paths=paths,
    conditioning_calls=sum(len(s['conditions']) for s in states),query_norm_consumption_calls=sum(len(s['consumers']) for s in states),
    shared_head_calls=sum(len(s['indicators']) for s in states),native_modes_verified=True,peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),validation_used=False,model_weights_saved=False,admission=False,
    limitation='Actual native two-step TRAIN only; no efficacy or full-dataset memory guarantee.')),encoding='utf-8')
