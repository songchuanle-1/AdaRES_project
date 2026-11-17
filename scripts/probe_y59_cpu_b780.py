"""Actual frozen Roberta mode/RNG controls and native learned language projection."""

import copy
import hashlib
import inspect
import json
from pathlib import Path
import gorilla
import torch
from torch import nn
from transformers import RobertaModel
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert P['physical_gpu'] is None
torch.set_num_threads(2)
root=Path(P['root'])
frozen=PAYLOAD['frozen']
assert (root/'.codex_source_commit').read_text().strip()==frozen['commit']
for row in frozen['sources']:
    assert hashlib.sha256((root/row['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==row['lf_sha256']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


cfg=gorilla.Config.fromfile(str(root/'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'))
assert list(cfg.model.fix_module)==['text_encoder'] and not cfg.model.norm_eval
dataset=build_dataset(cfg.data.train.copy(),get_root_logger())
assert digest(inspect.getfile(type(dataset)))==PAYLOAD['dataset']['sha256']
gorilla.set_random_seed(1999)
item=dataset[frozen['scenes'][0]['dataset_index']]
batch=dataset.collate_fn([copy.deepcopy(item)])
assert str(batch['scan_ids'][0])==frozen['scenes'][0]['scan_id']
assert [str(v) for v in batch['ann_ids']]==frozen['scenes'][0]['ann_ids']
ids,mask=batch['lang_tokenss'],batch['lang_masks']
assert ids.device.type=='cpu' and mask.device.type=='cpu'
assert digest(frozen['checkpoint'])==frozen['checkpoint_sha256']
checkpoint=torch.load(frozen['checkpoint'],map_location='cpu')['model']
encoder=RobertaModel.from_pretrained(str(root/'backbones/roberta-base'),local_files_only=True)
state={k[len('text_encoder.'):]:v for k,v in checkpoint.items() if k.startswith('text_encoder.')}
extras=set(state)-set(encoder.state_dict())
assert extras.issubset({'embeddings.position_ids','embeddings.token_type_ids'})
for key in extras:
    assert torch.equal(state[key].cpu(),dict(encoder.named_buffers())[key].cpu())
    del state[key]
encoder.load_state_dict(state,strict=True)
projection=nn.Linear(768,256)
projection.load_state_dict({k:checkpoint['dec.lang_proj.'+k] for k in ('weight','bias')},strict=True)
for parameter in list(encoder.parameters())+list(projection.parameters()):
    parameter.requires_grad_(False)
del checkpoint,state
scope=dict(nn=nn)
exec(PAYLOAD['exact_train_method_shim'],scope)
wrapper=scope['IPDN']()
wrapper.norm_eval=False
wrapper.text_encoder=encoder
wrapper.text_encoder.eval()
versions={n:p._version for n,p in encoder.named_parameters()}
valid=mask.to(dtype=torch.bool)


def output():
    with torch.no_grad():
        hidden=wrapper.text_encoder(ids,attention_mask=mask)[0]
        projected=projection(hidden)
    assert torch.isfinite(hidden).all() and torch.isfinite(projected).all()
    return hidden[valid].clone(),projected[valid].clone()


constructor_mode=encoder.training
gorilla.set_random_seed(1999)
constructor=output()
returned=wrapper.train(True)
assert returned is None
reactivated=encoder.training
active_dropout=sum(isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.training for m in encoder.modules())
assert not constructor_mode and reactivated and active_dropout>0
assert all(not p.requires_grad for p in encoder.parameters())
gorilla.set_random_seed(1999)
train_first=output()
train_next=output()
gorilla.set_random_seed(1999)
train_seed_repeat=output()
raw_drift=float((train_first[0]-train_next[0]).abs().max())
projection_drift=float((train_first[1]-train_next[1]).abs().max())
raw_bound=128*torch.finfo(torch.float32).eps*max(1.0,float(train_first[0].abs().max()))
projection_bound=128*torch.finfo(torch.float32).eps*max(1.0,float(train_first[1].abs().max()))
assert raw_drift>raw_bound and projection_drift>projection_bound
assert all(torch.equal(a,b) for a,b in zip(train_first,train_seed_repeat))
wrapper.train(True)
wrapper.text_encoder.eval()
assert wrapper.training and not wrapper.text_encoder.training
restored=output()
restored_repeat=output()
assert all(torch.equal(a,b) for a,b in zip(restored,restored_repeat))
assert all(torch.equal(a,b) for a,b in zip(constructor,restored))
assert versions=={n:p._version for n,p in encoder.named_parameters()}
assert all(p.grad is None for p in encoder.parameters())
assert not torch.cuda.is_initialized() and digest(frozen['checkpoint'])==frozen['checkpoint_sha256']
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y59',
    actual_frozen_Roberta_and_trained_projection=True,actual_TRAIN_scene=frozen['scenes'][0]['scan_id'],
    expressions=len(batch['ann_ids']),valid_tokens=int(valid.sum()),
    exact_original_train_method=True,constructor_encoder_training=constructor_mode,
    outer_train_reactivated_frozen_encoder=reactivated,active_encoder_dropout_modules=active_dropout,
    all_encoder_parameters_frozen=True,uncontrolled_RNG_valid_hidden_drift=raw_drift,
    uncontrolled_RNG_projected_language_drift=projection_drift,
    hidden_arithmetic_bound=raw_bound,projection_arithmetic_bound=projection_bound,
    reset_RNG_repeat_error=0,restored_fixed_mode_repeat_error=0,
    restored_fixed_mode_matches_constructor_exactly=True,parent_remains_training=True,
    parameters_unchanged=True,encoder_gradients_absent=True,CUDA_initialized=False,
    full_sparse_decoder_executed=False,native_loss_backward=False,optimizer_steps=0,weights_saved=False,
    admission=False,limitation='Actual frozen-language mode contract and projected-feature stochasticity only; '
        'no native confidence/task benefit or harm, no validation tuning or changes to running experiments.')),
    encoding='utf-8')
