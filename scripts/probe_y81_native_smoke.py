"""Two original native TRAIN steps with current query-set to visual feedback observation."""
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
assert row['id'] == 'AdaRES-Y81'
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





losses=[];gradient_checks=[];batches=[];modes=[];step_rows=[];path_rows=[];state={}
head=model.dec.indi_embedding;refinement=model.dec.query_mask_refinement
assert len(list(head.parameters()))==10 and torch.equal(refinement.weight,torch.zeros_like(refinement.weight))
original_prediction=model.dec.prediction_head
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
    state.clear();state.update(features=None,padding=None,queries=[],values=[],mask_inputs=[],refinement_queries=[],gates=[],prediction_consumers=[],coarse_calls=0,indicators=[],norms=[])

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
step=optimizer.step

def checked_step(*args,**kwargs):
    index=len(gradient_checks);assert index<2 and len(losses)==len(modes)==index+1
    check_path_calls(len(state['queries']),len(state['values']),state['prediction_consumers'],state['coarse_calls'],state['indicators'],state['norms'])
    active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None];assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['mask_inputs']+state['refinement_queries']+state['values'])
    grad=refinement.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    before=refinement.weight.detach().clone();gradient_max=float(grad.abs().max());result=step(*args,**kwargs)
    update=float((refinement.weight.detach()-before).abs().max());assert update>0 and torch.isfinite(refinement.weight).all()
    record=dict(all_finite=True,tensors=len(active),indicator_tensors=10,indicator_calls=7,mask_refinement_calls=6,prediction_consumption_calls=6,
        original_coarse_gate_consumption_calls=6,query_norm_consumption_calls=7,both_input_gradients_finite=True,refinement_weight_gradient_max=gradient_max,parameter_update_max=update)
    gradient_checks.append(record);step_rows.append(dict(record));return result

hooks=[refinement.register_forward_hook(capture_refinement),head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
hooks.extend(layer.register_forward_pre_hook(before_coarse) for layer in model.dec.qia_layers)
model.dec.prediction_head=checked_prediction
hooks.extend([model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),model.dec.register_forward_pre_hook(before_decoder)])
optimizer.step=checked_step;torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradient_checks)==len(batches)==len(modes)==2 and len({scene for batch in batches for scene in batch['scan_ids']})==4
    check_response_rows(path_rows);peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    optimizer.step=step;model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,losses=losses,
    native_data_loaded=True,backward_finite=True,intended_path_observed=True,batches=batches,gradient_checks=gradient_checks,refinement_paths=path_rows,
    mask_refinement_calls=12,prediction_consumption_calls=12,original_coarse_gate_consumption_calls=12,query_norm_consumption_calls=14,shared_head_calls=14,
    original_static_mask_base_preserved=True,original_hard_attention_algorithm_preserved=True,original_initial_prediction_preserved=True,
    native_modes_verified=True,peak_reserved_bytes=peak,data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),
    validation_used=False,model_weights_saved=False,admission=False,limitation='Actual native two-step TRAIN only; no efficacy or full-dataset memory guarantee.')),encoding='utf8')
