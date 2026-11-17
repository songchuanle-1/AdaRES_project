def check_path_calls(norm_inputs,indicator_calls,modulation_calls,consumed_calls):
    assert norm_inputs==[True]*7 and all(type(v) is bool for v in norm_inputs)
    assert indicator_calls==consumed_calls==7 and modulation_calls==1


def check_language_identity(language,projected_language):
    assert language is projected_language


def check_response_rows(rows):
    from math import isfinite
    assert len(rows)==2
    for row in rows:
        assert type(row['projection_nonzero']) is bool
        assert type(row['native_input_correspondence_verified']) is bool and row['native_input_correspondence_verified']
        assert type(row['consumed_by_all_predictions']) is bool and row['consumed_by_all_predictions']
        assert type(row['expressions']) is int
        assert row['reference_expression_rows']==[0,row['expressions']-1] and row['expressions']>=2
        assert len(row['valid_token_counts'])==row['expressions'] and all(type(n) is int and n>0 for n in row['valid_token_counts'])
        assert all(isfinite(row[k]) and row[k]>=0 for k in ('sentence_max','feature_delta_max','reference_error_max'))
    assert not rows[0]['projection_nonzero'] and rows[0]['feature_delta_max']==0
    assert rows[1]['projection_nonzero'] and rows[1]['feature_delta_max']>0 and rows[1]['sentence_max']>0


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
assert row['id'] == 'AdaRES-Y68'
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
losses=[];gradients=[];batches=[];paths=[]
projected_language=[];projected_features=[];decoder_inputs=[];modulation_inputs=[];modulation_outputs=[]
modulation_paths=[];consumed=[];norm_inputs=[];indicator_inputs=[]
head=model.dec.indi_embedding;modulator=model.dec.language_mask_modulation
assert len(list(head.parameters()))==10
assert torch.equal(modulator.weight.detach(),torch.zeros_like(modulator.weight))

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

def observe_decoder(module,args):
    assert len(args)==8;decoder_inputs.append(args)

def observe_norm(module,args,value):
    assert value.requires_grad and torch.isfinite(value).all();norm_inputs.append(bool(args[0].requires_grad))

def observe_indicator(module,args,value):
    assert args[0].requires_grad and args[1].requires_grad and value.requires_grad and torch.isfinite(value).all()
    indicator_inputs.append(args)

def observe_modulation(module,args,value):
    features,language,padded=args
    check_language_identity(language,projected_language[-1])
    native=decoder_inputs[-1];offsets=native[2]
    assert torch.equal(padded,~native[4].bool()) and features.shape[0]==language.shape[0]==len(offsets)-1
    for i in (0,features.shape[0]-1):
        start,end=int(offsets[i]),int(offsets[i+1]);count=end-start
        assert torch.equal(features[i,:count],projected_features[-1][start:end])
        assert torch.equal(features[i,count:],torch.zeros_like(features[i,count:]))
    assert features.requires_grad and language.requires_grad and value.requires_grad and torch.isfinite(value).all()
    for tensor in (features,language,value):tensor.retain_grad()
    modulation_inputs.append(args);modulation_outputs.append(value)
    with torch.no_grad():
        expr_rows=[0,features.shape[0]-1];reference=[];sentences=[]
        for i in expr_rows:
            indices=[j for j in range(language.shape[1]) if not bool(padded[i,j])]
            assert indices
            sentence=sum(language.detach()[i,j].double() for j in indices)/len(indices)
            sentences.append(sentence)
            reference.append(features.detach()[i,:3].double()*(1+torch.tanh(sentence@module.weight.detach().double())))
        expected=torch.stack(reference);actual=value.detach()[expr_rows,:3].double()
        torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-6)
        nonzero=bool(module.weight.detach().abs().max()>0)
        if not nonzero:assert torch.equal(value.detach(),features.detach())
        modulation_paths.append(dict(projection_nonzero=nonzero,sentence_max=float(torch.stack(sentences).abs().max()),
            feature_delta_max=float((value.detach()-features.detach()).abs().max()),reference_error_max=float((actual-expected).abs().max()),
            reference_expression_rows=expr_rows,expressions=features.shape[0],valid_token_counts=[int(v) for v in (~padded).sum(-1).tolist()],
            native_input_correspondence_verified=True,consumed_by_all_predictions=False))

prediction=model.dec.prediction_head

def checked_prediction(query,mask_feats,batch_mask):
    assert mask_feats is modulation_outputs[-1]
    consumed.append(True)
    return prediction(query,mask_feats,batch_mask)

model.dec.prediction_head=checked_prediction
step=optimizer.step

def checked_step(*args,**kwargs):
    index=len(gradients);assert index<2
    check_path_calls(norm_inputs[index*7:],len(indicator_inputs[index*7:]),len(modulation_inputs[index:]),len(consumed[index*7:]))
    active=[p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    indicator=[p.grad for p in head.parameters() if p.grad is not None]
    assert len(indicator)==10 and all(torch.isfinite(g).all() for g in indicator)
    grad=modulator.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in modulation_inputs[index][:2])
    assert modulation_outputs[index].grad is not None and torch.isfinite(modulation_outputs[index].grad).all()
    modulation_paths[index]['consumed_by_all_predictions']=True
    old=modulator.weight.detach().clone();returned=step(*args,**kwargs)
    update=float((modulator.weight.detach()-old).abs().max());assert update>0 and torch.isfinite(modulator.weight).all()
    gradients.append(dict(tensors=len(active),all_finite=True,indicator_tensors=10,indicator_calls=7,
        query_norm_input_requires_grad=norm_inputs[index*7:],modulation_calls=1,prediction_consumption_calls=7,
        feature_and_language_gradients_finite=True,modulation_weight_gradient_max=float(grad.abs().max()),parameter_update_max=update))
    return returned

optimizer.step=checked_step
hooks=[model.register_forward_pre_hook(before_forward),model.register_forward_hook(after_forward),
    model.dec.register_forward_pre_hook(observe_decoder),head.query_norm.register_forward_hook(observe_norm),head.register_forward_hook(observe_indicator),
    model.dec.lang_norm.register_forward_hook(lambda m,a,o:projected_language.append(o)),
    model.dec.x_mask.register_forward_hook(lambda m,a,o:projected_features.append(o)),modulator.register_forward_hook(observe_modulation)]
torch.cuda.reset_peak_memory_stats()
try:
    native.train(1,model,FirstTwoBatches(),optimizer,scheduler,cfg,logger,Writer());torch.cuda.synchronize()
    assert len(losses)==len(gradients)==len(batches)==len(paths)==2
    assert len(modulation_inputs)==len(modulation_outputs)==2 and len(consumed)==len(indicator_inputs)==14
    check_response_rows(modulation_paths)
    peak=int(torch.cuda.max_memory_reserved());assert 0<peak<24*2**30
finally:
    model.dec.prediction_head=prediction
    for h in hooks:h.remove()
    if getattr(loader,'_iterator',None) is not None:loader._iterator._shutdown_workers()
output=dict(complete=True,id=row['id'],config=row['config'],code_commit=FROZEN['code_commit'],goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'],native_entry='tools/train.py::train',split='train',steps=2,optimizer_steps=2,losses=losses,
    native_data_loaded=True,backward_finite=True,intended_path_observed=all(paths),
    data_source=dict(config_data_root=cfg.data.train.data_root,dataset_class=type(dataset).__module__+'.'+type(dataset).__name__),
    batches=batches,gradient_checks=gradients,modulation_paths=modulation_paths,shared_head_calls=len(indicator_inputs),
    modulation_calls=len(modulation_inputs),prediction_consumption_calls=len(consumed),query_norm_input_requires_grad=norm_inputs,
    peak_reserved_bytes=peak,model_weights_saved=False,validation_used=False,admission=False,
    limitation='Two native TRAIN optimizer steps only; not trained efficacy or a full-dataset resource guarantee.')
Path(P['output']).write_text(json.dumps(output),encoding='utf-8')
