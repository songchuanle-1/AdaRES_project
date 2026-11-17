"""Two original native TRAIN steps with scene-centered geometry query-initialization path observation."""
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
assert row['id'] == 'AdaRES-Y77'
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
assert sum(p.numel() for p in model.parameters()) == 164630405
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)

dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)



losses=[];gradient_checks=[];batches=[];modes=[];states=[];paths=[]
initialization=model.dec.geometry_query_init;head=model.dec.indi_embedding
assert len(list(head.parameters()))==10
assert torch.equal(initialization.weight.detach(),torch.zeros_like(initialization.weight))

def check_path_calls(generated,sampling,initialized,predictions,indicators,norms,batching_count,position_consumers):
    assert all(type(v) is int and v==1 for v in (generated,sampling,initialized))
    assert predictions==list(range(7)) and all(type(v) is int for v in predictions)
    assert type(indicators) is int and indicators==7
    assert norms==[True]*7 and all(type(v) is bool for v in norms)
    assert type(batching_count) is int and batching_count==4
    assert type(position_consumers) is int and position_consumers==12

def check_native_inputs(query,generated,scene,original_scene,pad,original_pad,mass_ptr,original_mass_ptr,mass_equal):
    assert query is generated and scene is original_scene and pad is original_pad
    assert type(mass_ptr) is int and type(original_mass_ptr) is int and mass_ptr==original_mass_ptr and mass_ptr>0
    assert mass_equal is True

def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==2
    for index,r in enumerate(rows):
        assert type(r['weight_nonzero']) is bool and r['weight_nonzero']==bool(index)
        for key in ('generated_query_identity_verified','original_scene_position_identity_verified','original_padding_identity_verified','original_mass_storage_and_values_verified',
                    'native_sampled_coordinates_verified','original_SCG_query_position_consumption_verified','query_inputs_undetached_verified','initial_prediction_consumption_verified',
                    'static_mask_features_preserved','padding_identity_verified'):
            assert r[key] is True
        n=r['expressions'];assert type(n) is int and n>=2 and r['reference_expression_rows']==[0,n-1]
        assert type(r['queries']) is int and r['queries']==128 and r['reference_queries']==[0,127] and r['reference_channels']==[0,127,255]
        for key in ('reference_error_max','initialization_delta_max','relative_position_abs_max','point_mass_total_min'):
            assert isfinite(r[key]) and r[key]>=0
        assert r['point_mass_total_min']>0
        if index==0:assert r['initialization_delta_max']==0
        else:assert r['initialization_delta_max']>0 and r['relative_position_abs_max']>0

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
    states.append(dict(generated=[],samples=[],outputs=[],predictions=[],indicators=[],norms=[],batching=[],position_consumers=[],
        raw_scene=args[5],raw_mass=args[7],fps_seed_sp=args[1]))
batching=model.dec.get_batches;scg=model.dec.get_scg_weight;prediction=model.dec.prediction_head

def checked_batches(flat,offsets):
    result=batching(flat,offsets);state=states[-1];state['batching'].append(True)
    if len(state['batching'])==1:state['padding']=result[1]
    if flat is state['raw_scene']:state['scene_pos']=result[0]
    if flat.data_ptr()==state['raw_mass'].data_ptr() and flat.shape[-1]==1:state['mass_tensor']=result[0]
    return result

def checked_scg(pos,*args,**kwargs):
    state=states[-1]
    if state['outputs']:
        assert pos is state['query_pos'];state['position_consumers'].append(True)
        if len(state['position_consumers'])==12:state['record']['original_SCG_query_position_consumption_verified']=True
    return scg(pos,*args,**kwargs)

def capture_generated(module,args,value):
    assert value.requires_grad and torch.isfinite(value).all();value.retain_grad();states[-1]['generated'].append(value)
def capture_sampling(module,args,value):states[-1]['samples'].append(value[0])

def capture_initialization(module,args,value):
    state=states[-1];assert not state['outputs'];query,qpos,scene_pos,pad,mass=args
    check_native_inputs(query,state['generated'][0],scene_pos,state['scene_pos'],pad,state['padding'],mass.data_ptr(),state['mass_tensor'].data_ptr(),bool(torch.equal(mass,state['mass_tensor'].squeeze(-1))))
    assert query.shape==value.shape and query.shape[1:]==(128,256) and qpos.shape==query.shape[:2]+(3,)
    assert all(t.requires_grad and torch.isfinite(t).all() for t in (query,value));value.retain_grad()
    seeded=scene_pos.gather(1,state['fps_seed_sp'].long().unsqueeze(-1).expand(-1,-1,3))
    expected_pos=seeded.gather(1,state['samples'][0].long().unsqueeze(-1).expand(-1,-1,3))
    assert torch.equal(qpos,expected_pos);state['query_pos']=qpos
    with torch.no_grad():
        n=query.shape[0];reference=[];observed=[]
        for b in (0,n-1):
            valid=(~pad[b]).nonzero(as_tuple=False).flatten();weights=mass[b,valid].double();assert weights.sum()>0
            center=(scene_pos[b,valid].double()*weights.unsqueeze(-1)).sum(0)/weights.sum()
            for q in (0,127):
                for d in (0,127,255):
                    reference.append(query[b,q,d].double()+((qpos[b,q].double()-center)*module.weight[:,d].double()).sum())
                    observed.append(value[b,q,d].double())
        expected=torch.stack(reference);actual=torch.stack(observed);torch.testing.assert_close(actual,expected,rtol=4e-5,atol=4e-6)
        weights=mass.masked_fill(pad,0.);totals=weights.sum(1,keepdim=True)
        center=(scene_pos.masked_fill(pad.unsqueeze(-1),0.)*weights.unsqueeze(-1)).sum(1)/totals.clamp_min(1.)
        nonzero=bool(module.weight.abs().max()>0)
        if not nonzero:assert torch.equal(query,value)
        record=dict(weight_nonzero=nonzero,expressions=n,queries=query.shape[1],reference_expression_rows=[0,n-1],reference_queries=[0,127],reference_channels=[0,127,255],
            reference_error_max=float((actual-expected).abs().max()),initialization_delta_max=float((value-query).abs().max()),relative_position_abs_max=float((qpos-center.unsqueeze(1)).abs().max()),point_mass_total_min=float(totals.min()),
            generated_query_identity_verified=True,original_scene_position_identity_verified=True,original_padding_identity_verified=True,original_mass_storage_and_values_verified=True,
            native_sampled_coordinates_verified=True,original_SCG_query_position_consumption_verified=False,query_inputs_undetached_verified=True,
            initial_prediction_consumption_verified=False,static_mask_features_preserved=False,padding_identity_verified=False)
    state['outputs'].append(value);state['record']=record;paths.append(record)

def checked_prediction(query,features,pad):
    state=states[-1];index=len(state['predictions']);assert index<7 and pad is state['padding']
    if index==0:
        assert query is state['outputs'][0];state['mask_features']=features;state['record']['initial_prediction_consumption_verified']=True
    else:assert features is state['mask_features']
    state['record'].update(static_mask_features_preserved=True,padding_identity_verified=True);state['predictions'].append(index)
    state['prediction_query']=query
    return prediction(query,features,pad)

def capture_indicator(module,args,value):
    state=states[-1];assert args[0] is state['prediction_query'] and args[0].requires_grad and args[1].requires_grad and torch.isfinite(value).all()
    state['indicators'].append(True)
def capture_norm(module,args,value):states[-1]['norms'].append(bool(args[0].requires_grad))
step=optimizer.step

def checked_step(*args,**kwargs):
    index=len(gradient_checks);assert index<2 and len(states)==index+1;state=states[index]
    check_path_calls(len(state['generated']),len(state['samples']),len(state['outputs']),state['predictions'],len(state['indicators']),state['norms'],len(state['batching']),len(state['position_consumers']))
    active=[p.grad for p in model.parameters() if p.grad is not None];assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None];assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    grad=initialization.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['generated']+state['outputs'])
    before=initialization.weight.detach().clone();gradmax=float(grad.abs().max());result=step(*args,**kwargs)
    update=float((initialization.weight.detach()-before).abs().max());assert update>0 and torch.isfinite(initialization.weight).all()
    gradient_checks.append(dict(all_finite=True,tensors=len(active),indicator_tensors=len(headgrads),indicator_calls=7,initialization_calls=1,initial_prediction_consumption_calls=1,
        original_SCG_query_position_consumption_calls=12,original_query_and_initialized_query_gradients_finite=True,initialization_weight_gradient_max=gradmax,parameter_update_max=update))
    return result

hooks=[model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),model.dec.register_forward_pre_hook(before_decoder),
    model.dec.query_generator.register_forward_hook(capture_generated),model.dec.sampling_module.register_forward_hook(capture_sampling),
    initialization.register_forward_hook(capture_initialization),head.register_forward_hook(capture_indicator),head.query_norm.register_forward_hook(capture_norm)]
model.dec.get_batches=checked_batches;model.dec.get_scg_weight=checked_scg;model.dec.prediction_head=checked_prediction;optimizer.step=checked_step;torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradient_checks)==len(batches)==len(modes)==len(states)==2
    assert len({s for b in batches for s in b['scan_ids']})==4
    check_response_rows(paths);peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    optimizer.step=step;model.dec.get_batches=batching;model.dec.get_scg_weight=scg;model.dec.prediction_head=prediction
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,losses=losses,
    native_data_loaded=True,backward_finite=True,intended_path_observed=True,batches=batches,gradient_checks=gradient_checks,initialization_paths=paths,
    initialization_calls=sum(len(s['outputs']) for s in states),initial_prediction_consumption_calls=sum(bool(s['record']['initial_prediction_consumption_verified']) for s in states),
    original_SCG_query_position_consumption_calls=sum(len(s['position_consumers']) for s in states),shared_head_calls=sum(len(s['indicators']) for s in states),native_modes_verified=True,peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),validation_used=False,model_weights_saved=False,admission=False,
    limitation='Actual native two-step TRAIN only; not efficacy or full-dataset memory guarantee.')),encoding='utf-8')
