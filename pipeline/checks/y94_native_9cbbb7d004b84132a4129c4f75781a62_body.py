def check_path_calls(stages, indicators, geometry_calls, attention_calls):
    assert stages == list(range(6)) and all(type(v) is int for v in stages)
    assert all(type(v) is int for v in (indicators, geometry_calls, attention_calls))
    assert indicators == 7 and geometry_calls == 1 and attention_calls == 6


def check_response_rows(rows):
    from math import isfinite
    assert len(rows) == 2
    for row in rows:
        assert row['native_positions_verified'] is True and row['hard_masks_verified'] is True
        assert row['consumed_by_all_query_self_attentions'] is True
        assert type(row['weight_nonzero']) is bool
        assert row['heads'] == 8 and row['queries'] == 128
        assert type(row['expressions']) is int and row['expressions'] >= 2
        assert row['reference_expression_rows'] == [0, row['expressions']-1]
        assert row['reference_head_rows'] == [0, 7] and row['reference_query_rows'] == [0, 127]
        assert all(isfinite(row[k]) and row[k] >= 0 for k in ('bias_max', 'reference_error_max'))
    assert rows[0]['weight_nonzero'] is False and rows[0]['bias_max'] == 0
    assert rows[1]['weight_nonzero'] is True and rows[1]['bias_max'] > 0


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
assert row['id'] == 'AdaRES-Y94'
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
assert sum(p.numel() for p in model.parameters()) == 164629661
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)
dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)
losses=[]; gradients=[]; batches=[]; modes=[]; decoder_inputs=[]; sampled=[]
bias_inputs=[]; bias_outputs=[]; geometry_paths=[]; query_self_calls=[]; attention_checks=[]; indicator_calls=[]
prediction_gates=[]; query_inputs=[]
branch=model.dec.query_geometry_bias; head=model.dec.indi_embedding
assert torch.equal(branch.weight.detach(),torch.zeros_like(branch.weight)) and branch.weight.shape==(8,3)
assert len(list(head.parameters()))==10
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
def before_forward(module,args):
    bn=[m for block in (module.input_conv,module.unet,module.output_layer) for m in block.modules() if isinstance(m,nn.BatchNorm1d)]
    assert module.training and bn and all(m.training for m in bn)
    assert module.text_encoder.training and all(not p.requires_grad for p in module.text_encoder.parameters())
    modes.append(True)
def after_forward(module,args,value):
    loss=value[0];assert loss.requires_grad and torch.isfinite(loss).all();losses.append(float(loss.detach().cpu()))
def observe_decoder(module,args):
    assert len(args)==8;decoder_inputs.append(args)

def observe_geometry(module,args,value):
    query_pos,source_pos,padding=args
    raw=decoder_inputs[-1];offsets=raw[2]
    assert query_pos is source_pos and query_pos.shape[1:]==(128,3)
    assert padding.dtype==torch.bool and not bool(padding.any())
    assert query_pos.shape[0]==len(offsets)-1
    original_positions,_=model.dec.get_batches(raw[5],offsets)
    seeds=original_positions.gather(1,raw[1].long().unsqueeze(-1).expand(-1,-1,3))
    expected_queries=seeds.gather(1,sampled[-1].long().unsqueeze(-1).expand(-1,-1,3))
    assert torch.equal(query_pos,expected_queries)
    assert value.shape==(query_pos.shape[0]*8,128,128) and value.requires_grad and torch.isfinite(value).all()
    value.retain_grad();bias_inputs.append(args);bias_outputs.append(value)
    with torch.no_grad():
        actual=value.detach().reshape(query_pos.shape[0],8,128,128);references=[];observed=[]
        for b in (0,query_pos.shape[0]-1):
            for h in (0,7):
                for q in (0,127):
                    for k in (0,127):
                        delta=(query_pos[b,q].double()-source_pos[b,k].double()).abs()
                        references.append(((delta/(1+delta)).square()*module.weight.detach()[h].double().tanh()).sum())
                        observed.append(actual[b,h,q,k].double())
        expected=torch.stack(references);observed=torch.stack(observed)
        torch.testing.assert_close(observed,expected,rtol=3e-5,atol=2e-6)
        nonzero=bool(module.weight.detach().abs().max()>0)
        if not nonzero:assert torch.equal(value.detach(),torch.zeros_like(value.detach()))
        geometry_paths.append(dict(weight_nonzero=nonzero,bias_max=float(value.detach().abs().max()),
            reference_error_max=float((observed-expected).abs().max()),native_positions_verified=True,
            hard_masks_verified=False,consumed_by_all_query_self_attentions=False,
            expressions=query_pos.shape[0],heads=8,queries=128,
            reference_expression_rows=[0,query_pos.shape[0]-1],reference_head_rows=[0,7],reference_query_rows=[0,127]))
def observe_indicator(module,args,value):
    assert args[0].requires_grad and args[1].requires_grad and value.requires_grad and torch.isfinite(value).all()
    indicator_calls.append(True)
prediction=model.dec.prediction_head
def checked_prediction(*args,**kwargs):
    output=prediction(*args,**kwargs);prediction_gates.append(output[2]);return output
model.dec.prediction_head=checked_prediction
original_layers=[];original_attention=[]
for index,layer in enumerate(model.dec.qqa_layers):
    original=layer.forward;original_layers.append(original)

    def checked_layer(*args,_original=original,_index=index,**kwargs):
        assert kwargs['attn_bias'] is bias_outputs[-1]
        gate=kwargs['attn_mask']
        _,neighbors=model.dec.get_scg_weight(bias_inputs[-1][0],k=model.dec.num_proposal_query[_index])
        assert torch.equal(gate,~neighbors) and gate.dtype==torch.bool and not gate.requires_grad
        assert args[0].requires_grad;args[0].retain_grad()
        query_self_calls.append(_index);query_inputs.append((args[0],))
        return _original(*args,**kwargs)
    layer.forward=checked_layer
    original_att=layer.attn.forward;original_attention.append(original_att)

    def checked_attention(*args,_original=original_att,_index=index,**kwargs):
        _,neighbors=model.dec.get_scg_weight(bias_inputs[-1][0],k=model.dec.num_proposal_query[_index])
        excluded=(~neighbors)[:,None].expand(-1,8,-1,-1).reshape_as(bias_outputs[-1])
        expected=bias_outputs[-1].masked_fill(excluded,float('-inf'))
        assert torch.equal(kwargs['attn_mask'],expected) and kwargs['key_padding_mask'] is None
        attention_checks.append(_index)
        return _original(*args,**kwargs)
    layer.attn.forward=checked_attention
step=optimizer.step
def checked_step(*args,**kwargs):
    index=len(gradients);assert index<2
    check_path_calls(query_self_calls[index*6:],len(indicator_calls[index*7:]),len(bias_outputs[index:]),len(attention_checks[index*6:]))
    assert attention_checks[index*6:]==list(range(6))
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    indicator=[p.grad for p in head.parameters() if p.grad is not None]
    assert len(indicator)==10 and all(torch.isfinite(g).all() for g in indicator)
    grad=branch.weight.grad;assert grad is not None and grad.shape==(8,3) and torch.isfinite(grad).all() and grad.abs().max()>0
    assert bias_outputs[index].grad is not None and torch.isfinite(bias_outputs[index].grad).all()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for pair in query_inputs[index*6:] for t in pair)
    geometry_paths[index].update(hard_masks_verified=True,consumed_by_all_query_self_attentions=True)
    old=branch.weight.detach().clone();returned=step(*args,**kwargs)
    update=float((branch.weight.detach()-old).abs().max());assert update>0 and torch.isfinite(branch.weight).all()
    gradients.append(dict(tensors=len(active),all_finite=True,indicator_tensors=10,indicator_calls=7,
        geometry_calls=1,query_self_stage_calls=query_self_calls[index*6:],attention_calls=6,
        native_hard_masks_preserved=True,query_gradients_finite=True,
        geometry_weight_gradient_max=float(grad.abs().max()),parameter_update_max=update))
    return returned
optimizer.step=checked_step
hooks=[model.register_forward_pre_hook(before_forward),model.register_forward_hook(after_forward),
    model.dec.register_forward_pre_hook(observe_decoder),branch.register_forward_hook(observe_geometry),
    model.dec.sampling_module.register_forward_hook(lambda m,a,o:sampled.append(o[0])),head.register_forward_hook(observe_indicator)]
torch.cuda.reset_peak_memory_stats()
try:
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradients)==len(batches)==len(modes)==len(geometry_paths)==2
    assert query_self_calls==attention_checks==list(range(6))*2 and len(indicator_calls)==14
    check_response_rows(geometry_paths)
    peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    model.dec.prediction_head=prediction
    for layer,original,original_att in zip(model.dec.qqa_layers,original_layers,original_attention):
        layer.forward=original;layer.attn.forward=original_att
    for h in hooks:h.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
output=dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,losses=losses,
    native_data_loaded=True,backward_finite=True,intended_path_observed=all(modes),
    data_source=dict(config_data_root=cfg.data.train.data_root,dataset_class=type(dataset).__module__+'.'+type(dataset).__name__),
    batches=batches,gradient_checks=gradients,geometry_paths=geometry_paths,
    geometry_calls=len(bias_outputs),query_self_stage_calls=query_self_calls,attention_stage_calls=attention_checks,shared_head_calls=len(indicator_calls),
    peak_reserved_bytes=peak,model_weights_saved=False,validation_used=False,admission=False,
    limitation='Two actual native TRAIN optimizer steps only; not trained efficacy or a full-dataset memory guarantee.')
Path(P['output']).write_text(json.dumps(output),encoding='utf-8')
