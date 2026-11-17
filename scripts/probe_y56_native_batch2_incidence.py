"""Pending exclusive-GPU capture of all native indicator layers on frozen TRAIN pairs."""
import os
assert P['physical_gpu'] in (0, 1) and os.environ.get('CUDA_VISIBLE_DEVICES') == str(P['physical_gpu'])

import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import time

import gorilla
import numpy as np
import torch
import yaml
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert subprocess.check_output(['hostname']).decode().strip() == 'dl01-MS-7C60'
assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
torch.set_num_threads(2)
root = Path(P['root'])
assert (root/'.codex_source_commit').read_text().strip() == FROZEN['commit']
for row in FROZEN['sources']:
    assert hashlib.sha256((root/row['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest() == row['lf_sha256']
config = root/'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
assert yaml.safe_load(config.read_text()) == FROZEN['config_values']
cfg = gorilla.Config.fromfile(str(config))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.dataloader.train.batch_size == 2 and cfg.model.sampling_module.num_proposal == 128


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


checkpoint_path = Path(FROZEN['checkpoint'])
assert digest(checkpoint_path) == FROZEN['checkpoint_sha256']
gorilla.set_random_seed(1999)
native = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.ipdn')
model = native.IPDN(**cfg.model).cuda()
checkpoint = torch.load(str(checkpoint_path),map_location='cpu')
weights = dict(checkpoint['model'])
extras = set(weights)-set(model.state_dict())
assert extras.issubset({'text_encoder.embeddings.position_ids','text_encoder.embeddings.token_type_ids'})
for key in extras:
    assert key in dict(model.named_buffers())
    assert torch.equal(weights[key].cpu(),dict(model.named_buffers())[key].cpu())
    del weights[key]
model.load_state_dict(weights,strict=True)
del weights, checkpoint
for parameter in model.parameters():
    parameter.requires_grad_(False)
model.train()
parameter_versions = {name:value._version for name,value in model.named_parameters()}
buffers = {name:value.detach().clone() for name,value in model.named_buffers()}
dataset = build_dataset(cfg.data.train.copy(),get_root_logger())
namespace = dict(np=np,torch=torch)
exec(compile(FROZEN['incidence_function'],'<frozen-CPU-tested-incidence>','exec'),namespace)
incidence = namespace['incidence']
records = []
started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
try:
    for pair in range(0,8,2):
        items=[]
        expression_ids=[]
        for scene in FROZEN['scenes'][pair:pair+2]:
            gorilla.set_random_seed(1999)
            item=dataset[scene['dataset_index']]
            single=dataset.collate_fn([copy.deepcopy(item)])
            assert single['scan_ids'][0]==scene['scan_id']
            assert [str(v) for v in single['ann_ids']]==scene['ann_ids']
            items.append(item)
            expression_ids.extend([(scene['scan_id'],ann) for ann in scene['ann_ids']])
        batch=dataset.collate_fn(copy.deepcopy(items))
        calls=[]

        def observe(_module,inputs):
            query,masks,padding,mass=inputs
            assert query.shape==(8,128,256) and padding.shape==mass.shape
            layer=len(calls)
            for index,(scene,ann) in enumerate(expression_ids):
                valid=~padding[index]
                q=query[index].detach().cpu()
                m=masks[index,:,valid].detach().cpu()
                assert q.dtype==torch.float32 and m.dtype==torch.float32
                assert torch.isfinite(q).all() and torch.isfinite(m).all()
                stats,_,_=incidence(q,m)
                records.append(dict(pair=pair//2,layer=layer,scene=scene,ann_id=ann,
                    valid_superpoints=int(valid.sum()),**stats))
            calls.append(layer)

        hook=model.dec.indi_embedding.register_forward_pre_hook(observe)
        try:
            gorilla.set_random_seed(1999)
            with torch.no_grad():
                loss,_=model(copy.deepcopy(batch),mode='loss')
            assert torch.isfinite(loss).all()
            assert len(calls)==7
        finally:
            hook.remove()
            for name,value in model.named_buffers():
                value.copy_(buffers[name])
        del loss,batch,items
finally:
    for name,value in model.named_buffers():
        value.copy_(buffers[name])
assert parameter_versions=={name:value._version for name,value in model.named_parameters()}
assert all(torch.equal(value,buffers[name]) for name,value in model.named_buffers())
assert len(records)==224
assert digest(checkpoint_path)==FROZEN['checkpoint_sha256']
assert (root/'.codex_source_commit').read_text().strip()==FROZEN['commit']
result=dict(complete=True,id='AdaRES-Y56',native_TRAIN_batch2=True,actual_sparse_FPS=True,
    fixed_scenes=8,expressions=32,indicator_calls=28,expression_layer_records=224,
    checkpoint_sha256=FROZEN['checkpoint_sha256'],parameters_unchanged=True,buffers_restored=True,
    exact_joint_duplicate_excess=sum(row['joint_duplicate_excess'] for row in records),
    hard_mask_duplicate_excess=sum(row['hard_mask_duplicate_excess'] for row in records),
    rows=records,elapsed_seconds=time.perf_counter()-started,
    peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),optimizer_steps=0,weights_saved=False,
    model_or_config_changes=False,training_launches=0,admission=False,
    limitation='Actual fixed TRAIN batch2/all7indicator-layer incidence only. Layers and queries are correlated, not independent prevalence samples; no efficacy, native backward/resource gate or novelty claim. Exact duplication and binary-mask equivalence remain distinct.')
Path(P['output']).write_text(json.dumps(result),encoding='utf-8')
