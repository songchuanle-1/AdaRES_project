"""Two original native TRAIN steps with initial-query residual path observation."""
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
assert row['id'] == 'AdaRES-Y73'
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

losses=[]; gradient_checks=[]; batches=[]; modes=[]; states=[]; paths=[]
residual=model.dec.query_anchor_residual;head=model.dec.indi_embedding
assert len(list(head.parameters()))==10
assert torch.equal(residual.weight.detach(),torch.zeros_like(residual.weight))


def check_path_calls(current_stages,residual_stages,prediction_stages,indicator_count,norm_calls):
    assert current_stages==residual_stages==list(range(6))
    assert prediction_stages==list(range(7))
    assert all(type(i) is int for i in current_stages+residual_stages+prediction_stages)
    assert type(indicator_count) is int and indicator_count==7
    assert norm_calls==[True]*7 and all(type(v) is bool for v in norm_calls)


def check_native_inputs(query,current_query,anchor,initial_anchor):
    assert query is current_query
    assert anchor is initial_anchor


def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==12
    for index,row in enumerate(rows):
        assert type(row['stage']) is int and row['stage']==index%6
        assert type(row['weight_nonzero']) is bool
        for key in ('current_qqa_ffn_identity_verified','initial_query_identity_verified','undetached_anchor_verified',
                    'prediction_consumption_verified','static_mask_features_preserved','padding_identity_verified'):
            assert row[key] is True
        count=row['expressions'];assert type(count) is int and count>=2
        assert row['reference_expression_rows']==[0,count-1]
        assert type(row['queries']) is int and row['queries']==128
        assert row['reference_queries']==[0,127] and row['reference_channels']==[0,127,255]
        for key in ('reference_error_max','query_delta_max','anchor_current_difference_max'):
            assert isfinite(row[key]) and row[key]>=0
        if index<6:assert row['weight_nonzero'] is False and row['query_delta_max']==0
        else:
            assert row['weight_nonzero'] is True and row['query_delta_max']>0
            assert row['anchor_current_difference_max']>0


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


def before_decoder(module,args):
    assert len(args)==8 and len(states)==len(gradient_checks)
    states.append(dict(current_stages=[],currents=[],residual_stages=[],outputs=[],predictions=[],indicators=[],norms=[],records=[]))


def capture_current(index,module,args,value):
    state=states[-1];assert index==len(state['currents'])
    assert value.requires_grad and torch.isfinite(value).all();value.retain_grad()
    state['current_stages'].append(index);state['currents'].append(value)


def capture_residual(module,args,value):
    state=states[-1];index=len(state['outputs'])
    assert index<6 and len(state['currents'])==index+1
    query,anchor=args
    check_native_inputs(query,state['currents'][index],anchor,state['anchor'])
    assert query.shape==anchor.shape==value.shape and query.shape[1:]==(128,256)
    assert all(t.requires_grad and torch.isfinite(t).all() for t in (query,anchor,value))
    value.retain_grad()
    with torch.no_grad():
        expressions=query.shape[0];expression_rows=[0,expressions-1];queries=[0,127];channels=[0,127,255]
        reference=[];actual=[]
        for b in expression_rows:
            for q in queries:
                for d in channels:
                    delta=((anchor[b,q].double()-query[b,q].double())*module.weight[:,d].double()).sum()
                    reference.append(query[b,q,d].double()+delta);actual.append(value[b,q,d].double())
        expected=torch.stack(reference);observed=torch.stack(actual)
        torch.testing.assert_close(observed,expected,rtol=4e-5,atol=4e-6)
        nonzero=bool(module.weight.detach().abs().max()>0)
        if not nonzero:assert torch.equal(value,query)
        record=dict(stage=index,weight_nonzero=nonzero,expressions=expressions,queries=query.shape[1],
            reference_expression_rows=expression_rows,reference_queries=queries,reference_channels=channels,
            reference_error_max=float((observed-expected).abs().max()),query_delta_max=float((value-query).abs().max()),
            anchor_current_difference_max=float((anchor-query).abs().max()),current_qqa_ffn_identity_verified=True,
            initial_query_identity_verified=True,undetached_anchor_verified=True,prediction_consumption_verified=False,
            static_mask_features_preserved=False,padding_identity_verified=False)
    state['residual_stages'].append(index);state['outputs'].append(value);state['records'].append(record);paths.append(record)


prediction=model.dec.prediction_head


def checked_prediction(query,features,padded):
    state=states[-1];index=len(state['predictions']);assert index<7
    if index==0:
        assert not state['outputs'] and not state['currents'] and query.requires_grad
        query.retain_grad();state['anchor']=query;state['mask_features']=features;state['padding']=padded
    else:
        assert query is state['outputs'][index-1] and features is state['mask_features'] and padded is state['padding']
        state['records'][index-1].update(prediction_consumption_verified=True,static_mask_features_preserved=True,padding_identity_verified=True)
    state['predictions'].append(index)
    return prediction(query,features,padded)


def capture_indicator(module,args,value):
    assert args[0].requires_grad and args[1].requires_grad and torch.isfinite(value).all();states[-1]['indicators'].append(True)


def capture_norm(module,args,value):states[-1]['norms'].append(bool(args[0].requires_grad))


step=optimizer.step


def checked_step(*args,**kwargs):
    index=len(gradient_checks);assert index<2 and len(states)==index+1
    state=states[index]
    check_path_calls(state['current_stages'],state['residual_stages'],state['predictions'],len(state['indicators']),state['norms'])
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads=[p.grad for p in head.parameters() if p.grad is not None]
    assert len(headgrads)==10 and all(torch.isfinite(g).all() for g in headgrads)
    grad=residual.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    tensors=[state['anchor']]+state['currents']+state['outputs']
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)
    before=residual.weight.detach().clone();gradmax=float(grad.abs().max());result=step(*args,**kwargs)
    update=float((residual.weight.detach()-before).abs().max());assert update>0 and torch.isfinite(residual.weight).all()
    gradient_checks.append(dict(all_finite=True,tensors=len(active),indicator_tensors=len(headgrads),indicator_calls=7,
        residual_calls=6,prediction_consumption_calls=6,current_query_and_initial_anchor_gradients_finite=True,
        residual_weight_gradient_max=gradmax,parameter_update_max=update))
    return result


hooks=[model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),
    model.dec.register_forward_pre_hook(before_decoder),residual.register_forward_hook(capture_residual),
    head.register_forward_hook(capture_indicator),head.query_norm.register_forward_hook(capture_norm)]
for index,layer in enumerate(model.dec.qqa_ffn_layers):hooks.append(layer.register_forward_hook(partial(capture_current,index)))
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
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,
    losses=losses,native_data_loaded=True,backward_finite=True,intended_path_observed=True,batches=batches,
    gradient_checks=gradient_checks,residual_paths=paths,residual_calls=sum(len(s['outputs']) for s in states),
    prediction_consumption_calls=sum(len(s['outputs']) for s in states),shared_head_calls=sum(len(s['indicators']) for s in states),
    native_modes_verified=True,peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),
    validation_used=False,model_weights_saved=False,admission=False,
    limitation='Actual native two-step TRAIN execution only; not efficacy or a full-dataset memory bound.')),encoding='utf-8')
