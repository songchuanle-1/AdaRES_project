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
assert row['id'] == 'AdaRES-Y83'
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
assert sum(p.numel() for p in model.parameters()) == 164632197
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)

dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)





losses=[];gradient_checks=[];batches=[];modes=[];step_rows=[];path_rows=[];state={}
head=model.dec.indi_embedding;norms=[model.dec.norm_qqa]+list(model.dec.refinement_query_norms);original_prediction=model.dec.prediction_head
assert len(list(head.parameters()))==10 and len(norms)==6
assert all(all(torch.equal(v,n.state_dict()[k]) for k,v in norms[0].state_dict().items()) for n in norms[1:])
"""Exact private-query-normalization callbacks for CPU and native task checks."""
def norm_parameters():
    assert len(norms)==6 and len({p.data_ptr() for n in norms for p in n.parameters()})==12
    assert norms[0] is model.dec.norm_qqa and all(n.eps==norms[0].eps for n in norms)
    return [p for n in norms for p in n.parameters()]


def check_norm_rows(rows):
    assert len(rows)==12
    for i,r in enumerate(rows):
        assert type(r['step']) is int and r['step']==i//6 and type(r['stage']) is int and r['stage']==i%6
        for key in ('original_self_identity','original_language_identity','sum_input_exact','both_inputs_undetached','private_module_identity','original_epsilon','float64_LayerNorm_reference'):
            assert r[key] is True


def check_path_calls():
    assert len(state['qla'])==len(state['self'])==len(state['norm_values'])==6
    assert state['ffn_consumers']==list(range(6)) and state['coarse_calls']==6
    assert len(state['queries'])==7 and state['indicators']==list(range(7)) and state['query_norms']==[True]*7 and len(state['gates'])==7


def checked_prediction(q,features,pad):
    if not state['queries']:state['features'],state['padding']=features,pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(q);value=original_prediction(q,features,pad);state['mask']=value[1]
    expected=(value[1].sigmoid()<.5)|pad.unsqueeze(1)
    expected[expected.all(-1)]=False;expected=expected|pad.unsqueeze(1)
    assert torch.equal(value[2],expected) and not value[2].requires_grad
    state['gates'].append(value[2]);return value


def capture_qla(module,args,value):
    i=len(state['qla']);assert module is model.dec.qla_layers[i]
    value[0].retain_grad();state['qla'].append(value[0])


def capture_self(module,args,value):
    i=len(state['self']);assert module is model.dec.qqa_layers[i] and args[0] is state['qla'][i]
    value.retain_grad();state['self'].append(value)


def capture_fusion_norm(module,args,value):
    i=len(state['norm_values']);assert module is norms[i] and len(args)==1
    assert torch.equal(args[0],state['self'][i]+state['qla'][i]) and args[0].requires_grad and value.requires_grad
    args[0].retain_grad();value.retain_grad()
    reference=torch.nn.functional.layer_norm(args[0].double(),module.normalized_shape,module.weight.double(),module.bias.double(),module.eps)
    torch.testing.assert_allclose(value.double(),reference,rtol=1e-5,atol=1e-6)
    assert module.eps==model.dec.norm_qqa.eps
    state['sum_inputs'].append(args[0]);state['norm_values'].append(value)
    path_rows.append(dict(step=len(step_rows),stage=i,original_self_identity=True,original_language_identity=True,
        sum_input_exact=True,both_inputs_undetached=True,private_module_identity=True,original_epsilon=True,float64_LayerNorm_reference=True))


def before_ffn(module,args):
    i=len(state['ffn_consumers']);assert module is model.dec.qqa_ffn_layers[i] and args[0] is state['norm_values'][i]
    state['ffn_consumers'].append(i)


def before_coarse(module,args):
    i=state['coarse_calls'];assert module is model.dec.qia_layers[i] and args[3] is state['gates'][-1]
    state['coarse_calls']+=1


def before_indicator(module,args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries'])-1)


def capture_query_norm(module,args,value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad;state['query_norms'].append(True)


def reset_state():
    state.clear();state.update(features=None,padding=None,queries=[],gates=[],qla=[],self=[],norm_values=[],sum_inputs=[],ffn_consumers=[],coarse_calls=0,indicators=[],query_norms=[])


def gradient_record():
    check_path_calls();parameters=norm_parameters()
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in parameters)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in state['qla']+state['self']+state['sum_inputs']+state['norm_values'])
    heads=[p.grad for p in head.parameters() if p.grad is not None];assert len(heads)==10 and all(torch.isfinite(g).all() for g in heads)
    return dict(all_finite=True,tensors=len(active),indicator_tensors=10,indicator_calls=7,private_norm_calls=6,
        ffn_consumption_calls=6,original_coarse_gate_consumption_calls=6,query_norm_consumption_calls=7,
        norm_gradient_tensors=12,norm_gradient_maxima=[float(p.grad.abs().max()) for p in parameters],both_input_gradients_finite=True)


def install_hooks():
    hooks=[head.register_forward_pre_hook(before_indicator),head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(n.register_forward_hook(capture_fusion_norm) for n in norms)
    hooks.extend(n.register_forward_hook(capture_qla) for n in model.dec.qla_layers)
    hooks.extend(n.register_forward_hook(capture_self) for n in model.dec.qqa_layers)
    hooks.extend(n.register_forward_pre_hook(before_ffn) for n in model.dec.qqa_ffn_layers)
    hooks.extend(n.register_forward_pre_hook(before_coarse) for n in model.dec.qia_layers)
    return hooks

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
    reset_state()

step=optimizer.step

def checked_step(*args,**kwargs):
    index=len(step_rows);assert index<2 and len(losses)==len(modes)==index+1
    record=gradient_record();params=norm_parameters();before=[p.detach().clone() for p in params];value=step(*args,**kwargs)
    updates=[float((p-b).abs().max()) for p,b in zip(params,before)]
    assert all(u>0 for u in updates) and all(torch.isfinite(p).all() for p in params)
    record.update(step=index,norm_parameter_updates=updates);gradient_checks.append(record);step_rows.append(dict(record));return value

hooks=install_hooks()+[model.register_forward_pre_hook(before_model),model.register_forward_hook(after_model),model.dec.register_forward_pre_hook(before_decoder)]
model.dec.prediction_head=checked_prediction;optimizer.step=checked_step;torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradient_checks)==len(batches)==len(modes)==2
    assert len({s for b in batches for s in b['scan_ids']})==4
    check_norm_rows(path_rows);peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    optimizer.step=step;model.dec.prediction_head=original_prediction
    for hook in hooks:hook.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,losses=losses,native_data_loaded=True,
    backward_finite=True,intended_path_observed=True,batches=batches,gradient_checks=gradient_checks,norm_paths=path_rows,private_norm_calls=12,ffn_consumption_calls=12,
    original_coarse_gate_consumption_calls=12,query_norm_consumption_calls=14,shared_head_calls=14,static_mask_features_preserved=True,original_initial_prediction_preserved=True,
    original_hard_gate_algorithm_preserved=True,norm_parameters_disjoint=True,original_epsilon_preserved=True,native_modes_verified=True,peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,config_data_root=cfg.data.train.data_root),
    validation_used=False,model_weights_saved=False,admission=False,limitation='Actual native two-step TRAIN only; no efficacy or full-dataset memory guarantee.')),encoding='utf8')
