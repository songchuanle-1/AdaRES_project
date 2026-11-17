"""Two original native TRAIN steps observing shared visual mask-feature LayerNorm."""
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
assert row['id'] == 'AdaRES-Y85'
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
assert sum(p.numel() for p in model.parameters()) == 164630149
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)

dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)





losses=[];gradient_checks=[];batches=[];modes=[];step_rows=[];path_rows=[];state={}
norm=model.dec.mask_feature_norm;head=model.dec.indi_embedding
original_prediction=model.dec.prediction_head;original_batch=model.dec.get_batches
assert len(list(head.parameters()))==10 and norm.eps==1e-5
assert torch.equal(norm.weight,torch.ones_like(norm.weight)) and torch.equal(norm.bias,torch.zeros_like(norm.bias))

def check_mask_norm_rows(rows):
    assert len(rows)==2
    for i,row in enumerate(rows):
        assert type(row['step']) is int and row['step']==i
        for key in ('original_x_mask_output_identity','all_channels_and_original_epsilon','input_undetached','float64_LayerNorm_reference','shared_module_identity'):
            assert row[key] is True

def capture_mask_raw(module,args,value):
    assert 'raw' not in state and value.requires_grad;value.retain_grad();state['raw']=value

def capture_mask_norm(module,args,value):
    assert len(args)==1 and args[0] is state['raw'] and value.requires_grad and 'normalized' not in state
    assert module is norm and module.eps==1e-5 and tuple(module.normalized_shape)==(256,)
    reference=torch.nn.functional.layer_norm(args[0].double(),module.normalized_shape,module.weight.double(),module.bias.double(),module.eps)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    value.retain_grad();state['normalized']=value
    path_rows.append(dict(step=len(step_rows),original_x_mask_output_identity=True,all_channels_and_original_epsilon=True,input_undetached=True,float64_LayerNorm_reference=True,shared_module_identity=True))

def tracked_batch(*args,**kwargs):
    if len(state['batches'])==1:assert args[0] is state['normalized']
    result=original_batch(*args,**kwargs);state['batches'].append(result);return result

def checked_prediction(q,features,pad):
    assert len(state['batches'])==4 and features is state['batches'][1][0] and pad is state['batches'][0][1]
    assert torch.count_nonzero(features[pad])==0
    state['query']=q;state['predictions']+=1;value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1);expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad;state['gate']=value[2];return value

def before_indicator(module,args):
    assert args[0] is state['query'] and args[1] is state['mask'];state['indicators']+=1

def capture_query_norm(module,args,value):
    assert args[0] is state['query'] and args[0].requires_grad;state['query_norms']+=1

def before_coarse(module,args):
    assert module is model.dec.qia_layers[state['coarse_calls']] and args[3] is state['gate'];state['coarse_calls']+=1

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
    assert len(args)==8 and len(gradient_checks)==len(step_rows) and len(step_rows)<2
    state.clear();state.update(batches=[],predictions=0,indicators=0,query_norms=0,coarse_calls=0)

step=optimizer.step

def checked_step(*args,**kwargs):
    index=len(step_rows);assert index<2 and len(losses)==len(modes)==index+1
    assert state['predictions']==state['indicators']==state['query_norms']==7 and state['coarse_calls']==6
    active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None];assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    grads=[p.grad for p in norm.parameters()];assert len(grads)==2 and all(g is not None and torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (state['raw'],state['normalized']))
    before=[p.detach().clone() for p in norm.parameters()];value=step(*args,**kwargs);updates=[float((p-b).abs().max()) for p,b in zip(norm.parameters(),before)]
    assert all(u>0 for u in updates)
    record=dict(step=index,all_finite=True,tensors=len(active),indicator_tensors=10,new_gradient_maxima=[float(g.abs().max()) for g in grads],parameter_updates=updates,prediction_calls=7,indicator_calls=7,query_norm_calls=7,coarse_gate_calls=6,raw_and_normalized_gradients_finite=True)
    gradient_checks.append(record);step_rows.append(dict(record));return value

hooks=[model.dec.x_mask.register_forward_hook(capture_mask_raw),norm.register_forward_hook(capture_mask_norm),head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm),model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),model.dec.register_forward_pre_hook(before_decoder)]
hooks.extend(m.register_forward_pre_hook(before_coarse) for m in model.dec.qia_layers)
model.dec.get_batches=tracked_batch;model.dec.prediction_head=checked_prediction;optimizer.step=checked_step;torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradient_checks)==len(batches)==len(modes)==2
    assert len({s for b in batches for s in b['scan_ids']})==4
    check_mask_norm_rows(path_rows);peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    optimizer.step=step;model.dec.get_batches=original_batch;model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,losses=losses,native_data_loaded=True,
    backward_finite=True,intended_path_observed=True,batches=batches,gradient_checks=gradient_checks,norm_paths=path_rows,mask_feature_norm_calls=2,
    query_norm_consumption_calls=14,shared_head_calls=14,original_coarse_gate_consumption_calls=12,static_mask_feature_reuse_preserved=True,original_hard_gate_algorithm_preserved=True,normalization_before_batching=True,
    native_modes_verified=True,peak_reserved_bytes=peak,data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),
    validation_used=False,model_weights_saved=False,admission=False,limitation='Actual native two-step TRAIN only. Active mask-feature normalization changes masks by design; no W2 output anchor, efficacy or full-dataset memory guarantee is claimed.')),encoding='utf8')
