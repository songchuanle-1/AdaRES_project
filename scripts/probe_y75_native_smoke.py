"""Two original native TRAIN steps with language-conditioned initial-query path observation."""
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
assert row['id'] == 'AdaRES-Y75'
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



losses=[];gradient_checks=[];batches=[];modes=[];states=[];paths=[]
initialization=model.dec.language_query_init;head=model.dec.indi_embedding
assert len(list(head.parameters()))==10
assert torch.equal(initialization.weight.detach(),torch.zeros_like(initialization.weight))
def check_path_calls(generated_count,language_count,sampling_count,initialization_count,prediction_stages,indicator_count,norm_calls):
    assert all(type(v) is int and v==1 for v in (generated_count,language_count,sampling_count,initialization_count))
    assert prediction_stages==list(range(7)) and all(type(v) is int for v in prediction_stages)
    assert type(indicator_count) is int and indicator_count==7
    assert norm_calls==[True]*7 and all(type(v) is bool for v in norm_calls)
def check_native_inputs(query,generated_query,language,projected_language,padding,sampling_padding):
    assert query is generated_query and language is projected_language and padding is sampling_padding
def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==2
    for index,row in enumerate(rows):
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero']==bool(index)
        for key in ('generated_query_identity_verified','projected_language_identity_verified','sampling_padding_identity_verified','native_padding_values_verified',
                    'both_inputs_undetached_verified','initial_prediction_consumption_verified','static_mask_features_preserved','padding_identity_verified'):
            assert row[key] is True
        count=row['expressions'];assert type(count) is int and count>=2 and row['reference_expression_rows']==[0,count-1]
        assert type(row['queries']) is int and row['queries']==128
        assert type(row['tokens']) is int and row['tokens']>=2
        assert row['reference_queries']==[0,127] and row['reference_channels']==[0,127,255]
        assert len(row['reference_valid_token_counts'])==2 and all(type(v) is int and 0<v<=row['tokens'] for v in row['reference_valid_token_counts'])
        for key in ('reference_error_max','initialization_delta_max','context_abs_max'):
            assert isfinite(row[key]) and row[key]>=0
        if index==0:assert row['initialization_delta_max']==0
        else:assert row['initialization_delta_max']>0 and row['context_abs_max']>0
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
    states.append(dict(generated=[],languages=[],sampling_masks=[],outputs=[],predictions=[],indicators=[],norms=[],expected_padding=~args[4].bool()))
def capture_generated(module,args,value):
    state=states[-1];assert not state['generated'] and value.requires_grad and torch.isfinite(value).all();value.retain_grad();state['generated'].append(value)
def capture_language(module,args,value):
    state=states[-1];assert not state['languages'] and value.requires_grad and torch.isfinite(value).all();value.retain_grad();state['languages'].append(value)
def capture_sampling(module,args):
    state=states[-1];assert not state['sampling_masks'] and args[1] is state['languages'][0]
    assert args[3].dtype==torch.bool and torch.equal(args[3],state['expected_padding']);state['sampling_masks'].append(args[3])
def capture_initialization(module,args,value):
    state=states[-1];assert not state['outputs'];query,language,padding=args
    check_native_inputs(query,state['generated'][0],language,state['languages'][0],padding,state['sampling_masks'][0])
    assert query.shape==value.shape and query.shape[1:]==(128,256)
    assert language.shape[0]==query.shape[0] and language.shape[-1]==256 and padding.shape==language.shape[:2]
    assert torch.equal(padding,state['expected_padding']) and all(t.requires_grad and torch.isfinite(t).all() for t in (query,language,value))
    value.retain_grad()
    with torch.no_grad():
        expressions=query.shape[0];expression_rows=[0,expressions-1];queries=[0,127];channels=[0,127,255];counts=[];reference=[];actual=[]
        for b in expression_rows:
            valid=(~padding[b]).nonzero(as_tuple=False).flatten();assert len(valid)>0;counts.append(len(valid))
            context=language[b,valid].double().mean(0)
            for q in queries:
                for d in channels:
                    reference.append(query[b,q,d].double()+(query[b,q].double()*context*module.weight[:,d].double()).sum())
                    actual.append(value[b,q,d].double())
        expected=torch.stack(reference);observed=torch.stack(actual);torch.testing.assert_close(observed,expected,rtol=4e-5,atol=4e-6)
        nonzero=bool(module.weight.detach().abs().max()>0)
        if not nonzero:assert torch.equal(value,query)
        clean=language.masked_fill(padding.unsqueeze(-1),0.);context=clean.sum(1)/(~padding).sum(1,keepdim=True).clamp_min(1).to(language.dtype)
        record=dict(weight_nonzero=nonzero,expressions=expressions,queries=query.shape[1],tokens=language.shape[1],
            reference_expression_rows=expression_rows,reference_queries=queries,reference_channels=channels,reference_valid_token_counts=counts,
            reference_error_max=float((observed-expected).abs().max()),initialization_delta_max=float((value-query).abs().max()),context_abs_max=float(context.abs().max()),
            generated_query_identity_verified=True,projected_language_identity_verified=True,sampling_padding_identity_verified=True,native_padding_values_verified=True,
            both_inputs_undetached_verified=True,initial_prediction_consumption_verified=False,static_mask_features_preserved=False,padding_identity_verified=False)
    state['outputs'].append(value);state['record']=record;paths.append(record)
prediction=model.dec.prediction_head
def checked_prediction(query,features,padded):
    state=states[-1];index=len(state['predictions']);assert index<7
    if index==0:
        assert query is state['outputs'][0];state['mask_features']=features;state['padding']=padded;state['record']['initial_prediction_consumption_verified']=True
    else:assert features is state['mask_features'] and padded is state['padding']
    state['record'].update(static_mask_features_preserved=True,padding_identity_verified=True);state['predictions'].append(index)
    return prediction(query,features,padded)
def capture_indicator(module,args,value):
    state=states[-1];assert args[0].requires_grad and args[1].requires_grad and torch.isfinite(value).all()
    if not state['indicators']:assert args[0] is state['outputs'][0]
    state['indicators'].append(True)
def capture_norm(module,args,value):states[-1]['norms'].append(bool(args[0].requires_grad))
step=optimizer.step
def checked_step(*args,**kwargs):
    index=len(gradient_checks);assert index<2 and len(states)==index+1;state=states[index]
    check_path_calls(len(state['generated']),len(state['languages']),len(state['sampling_masks']),len(state['outputs']),state['predictions'],len(state['indicators']),state['norms'])
    active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None];assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    grad=initialization.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['generated']+state['languages']+state['outputs'])
    before=initialization.weight.detach().clone();gradmax=float(grad.abs().max());result=step(*args,**kwargs)
    update=float((initialization.weight.detach()-before).abs().max());assert update>0 and torch.isfinite(initialization.weight).all()
    gradient_checks.append(dict(all_finite=True,tensors=len(active),indicator_tensors=len(headgrads),indicator_calls=7,initialization_calls=1,
        initial_prediction_consumption_calls=1,generated_query_and_projected_language_gradients_finite=True,initialization_weight_gradient_max=gradmax,parameter_update_max=update))
    return result
hooks=[model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),model.dec.register_forward_pre_hook(before_decoder),
    model.dec.query_generator.register_forward_hook(capture_generated),model.dec.lang_norm.register_forward_hook(capture_language),
    model.dec.sampling_module.register_forward_pre_hook(capture_sampling),initialization.register_forward_hook(capture_initialization),
    head.register_forward_hook(capture_indicator),head.query_norm.register_forward_hook(capture_norm)]
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
    native_data_loaded=True,backward_finite=True,intended_path_observed=True,batches=batches,gradient_checks=gradient_checks,initialization_paths=paths,
    initialization_calls=sum(len(s['outputs']) for s in states),initial_prediction_consumption_calls=sum(bool(s['record']['initial_prediction_consumption_verified']) for s in states),
    shared_head_calls=sum(len(s['indicators']) for s in states),native_modes_verified=True,peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),validation_used=False,model_weights_saved=False,admission=False,
    limitation='Actual native two-step TRAIN only; not efficacy or full-dataset memory guarantee.')),encoding='utf-8')
