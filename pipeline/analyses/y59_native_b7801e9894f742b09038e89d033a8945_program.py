FROZEN={'scenes': [{'dataset_index': 0, 'scan_id': 'scene0000_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 44, 'scan_id': 'scene0001_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 60, 'scan_id': 'scene0002_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 101, 'scan_id': 'scene0003_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 115, 'scan_id': 'scene0004_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 126, 'scan_id': 'scene0005_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 137, 'scan_id': 'scene0006_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 171, 'scan_id': 'scene0007_00', 'ann_ids': ['0', '1', '2', '3']}], 'sources': [{'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/__init__.py', 'lf_sha256': '8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/backbone.py', 'lf_sha256': '1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'lf_sha256': '7e60c1cecfc8fa7809f7c836fa6fef1366806bd23c08e2890508ea26a711acac'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'lf_sha256': '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/loss.py', 'lf_sha256': '4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/sample_model.py', 'lf_sha256': '03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b'}], 'commit': '05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea', 'config_values': {'work_dir': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'train': {'seed': 1999, 'epochs': 30, 'interval': 1, 'pretrain': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/sp_unet_backbone.pth', 'save_epoch': 100, 'grad_clip': 0.1}, 'test': {'seed': 1999, 'test_topk_per_scene': 100, 'test_score_thresh': 0.0, 'test_npoint_thresh': 100}, 'dataloader': {'train': {'batch_size': 2, 'num_workers': 4, 'persistent_workers': True}, 'val': {'batch_size': 1, 'num_workers': 2, 'persistent_workers': True}, 'test': {'batch_size': 1, 'num_workers': 1, 'persistent_workers': True}}, 'data': {'train': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/', 'prefix': 'train', 'suffix': '.pth', 'training': True, 'mode': 4, 'aug': True, 'with_elastic': True, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 4}, 'val': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/', 'prefix': 'val', 'suffix': '.pth', 'training': False, 'aug': False, 'with_elastic': False, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 32}}, 'model': {'input_channel': 6, 'blocks': 5, 'block_reps': 2, 'media': 32, 'normalize_before': True, 'return_blocks': True, 'pool': 'mean', 'infer_mode': 'sem', 'fps_num': 256, 'task_type': 'res', 'sampling_module': {'num_proposal': 128, 'pc_dim': 256, 'lang_dim': 256, 'd_model': 256}, 'dec': {'num_layer': 6, 'd_model': 256, 'nhead': 8, 'hidden_dim': 1024, 'dropout': 0.0, 'activation_fn': 'gelu', 'iter_pred': True, 'attn_mask': True, 'kernel': 'top1', 'global_feat': 'mean', 'lang_att': True, 'contrastive_align_loss': True, 'num_proposal_fine': 512, 'num_proposal_query': [128, 64, 32, 16, 8, 4], 'num_proposal_query_inst': [128, 64, 32, 16, 8, 4]}, 'criterion': {'loss_weight': [1.0, 1.0, 0.5, 5.0, 0.1, 0.1, 0.1, 1.0], 'cost_weight': [1.0, 1.0, 0.1], 'loss_fun': 'focal', 'match_last_layer': True, 'layer_differ_weight': True}, 'test_cfg': {'topk_insts': 100, 'score_thr': 0.0, 'npoint_thr': 100}, 'norm_eval': False, 'fix_module': ['text_encoder']}, 'optimizer': {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0005, 'paramwise_cfg': {'text_encoder': {'lr_mult': 0.001}, 'input_conv': {'lr_mult': 0.01}, 'unet': {'lr_mult': 0.01}, 'output_layer': {'lr_mult': 0.01}, 'dec': {'lr_mult': 1.0}}}, 'lr_scheduler': {'type': 'PolyLR', 'max_iters': 70, 'power': 4.0, 'constant_ending': 0.0}}, 'incidence_function': 'def incidence(query, masks):\n    q, m = query.detach().cpu().numpy(), masks.detach().cpu().numpy()\n    _, inverse, counts = np.unique(np.concatenate([q, m], axis=1), axis=0,\n                                   return_inverse=True, return_counts=True)\n    return dict(query_unique=int(len(np.unique(q, axis=0))),\n        soft_mask_unique=int(len(np.unique(m, axis=0))),\n        joint_unique=int(len(counts)), hard_mask_unique=int(len(np.unique(m > 0, axis=0))),\n        joint_duplicate_excess=int(len(q) - len(counts)),\n        hard_mask_duplicate_excess=int(len(q) - len(np.unique(m > 0, axis=0)))), inverse, counts', 'checkpoint': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', 'checkpoint_sha256': 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52', 'capture_sha256': '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'}
SCOPE={'dataset_source': {'path': '/bak_home/songcl/Project/AdaRES_project/AdaRES/ipdn/dataset/scannetv2_test.py', 'sha256': '1328bad8384392f6a8dac2a6a46126b82000132283706f9398030be1cf0f0cc8'}}
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

"""Pending task-facing frozen-mode diagnosis, preserving original controls."""

import gc
import inspect

assert digest(Path(inspect.getfile(type(dataset))))==SCOPE['dataset_source']['sha256']
items=[]
for scene in FROZEN['scenes'][:2]:
    gorilla.set_random_seed(1999)
    item=dataset[scene['dataset_index']]
    single=dataset.collate_fn([copy.deepcopy(item)])
    assert str(single['scan_ids'][0])==scene['scan_id']
    assert [str(v) for v in single['ann_ids']]==scene['ann_ids']
    items.append(item)
batch=dataset.collate_fn(copy.deepcopy(items))
for name,parameter in model.named_parameters():
    parameter.requires_grad_(not name.startswith('text_encoder.'))
rows=[]
focal_logits={}
sampling={}
for role in ('original','original_repeat','restored_frozen_mode','restored_repeat'):
    for name,value in model.named_buffers():value.copy_(buffers[name])
    model.train()
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.dropout._DropoutNd) and not name.startswith('text_encoder.'):
            module.eval()
    if role.startswith('restored'):
        model.text_encoder.eval()
    model.zero_grad(set_to_none=True)
    gorilla.set_random_seed(1999)
    outputs=[]
    rng_state={}
    decoded=[]
    def before_encoder(module,args):
        rng_state['cpu']=torch.get_rng_state()
        rng_state['cuda']=torch.cuda.get_rng_state_all()
    def after_encoder(module,args,value):
        # Dropout must not shift the random stream used by downstream operators.
        torch.set_rng_state(rng_state['cpu'])
        torch.cuda.set_rng_state_all(rng_state['cuda'])
    def observe_decoder(module,args,value):
        decoded.append(value['sample_inds'].detach().cpu().clone())
    def observe(module,args,value):
        value.retain_grad()
        outputs.append(value)
    hooks=[model.dec.indi_embedding.register_forward_hook(observe),
           model.text_encoder.register_forward_pre_hook(before_encoder),
           model.text_encoder.register_forward_hook(after_encoder),
           model.dec.register_forward_hook(observe_decoder)]
    torch.cuda.reset_peak_memory_stats()
    try:
        loss,logs=model(copy.deepcopy(batch),mode='loss')
        assert torch.isfinite(loss).all() and len(outputs)==7 and len(decoded)==1
        loss.backward()
        gradients={}
        for name,parameter in model.dec.indi_embedding.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            gradients[name]=float(parameter.grad.abs().max())
            assert gradients[name]>0
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in outputs)
        active=[p.grad for p in model.parameters() if p.grad is not None]
        assert active and all(torch.isfinite(g).all() for g in active)
        assert all(p.grad is None and not p.requires_grad for p in model.text_encoder.parameters())
        torch.cuda.synchronize()
        peak=int(torch.cuda.max_memory_reserved())
        assert peak<24*2**30
        focal_logits[role]=outputs[-1][:4].detach().cpu().clone()
        sampling[role]=decoded[0][:4]
        rows.append(dict(role=role,native_loss=float(loss.item()),indicator_gradients=gradients,
            all7_native_task_logits_finite=True,encoder_training=model.text_encoder.training,
            active_encoder_dropout=sum(isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.training
                                       for m in model.text_encoder.modules()),
            encoder_parameters_frozen=True,peak_reserved_bytes=peak))
    finally:
        for hook in hooks:hook.remove()
    outputs.clear()
    model.zero_grad(set_to_none=True)
    del loss,logs,active
    gc.collect()
comparisons={}
for first,second,label in (('original','original_repeat','original_repeat'),
                           ('restored_frozen_mode','restored_repeat','restored_repeat'),
                           ('original','restored_frozen_mode','mode_intervention')):
    a,b=focal_logits[first],focal_logits[second]
    comparisons[label]=dict(logit_max=float((a-b).abs().max()),
        confidence_slot_max=float((a.softmax(-1)[...,1]-b.softmax(-1)[...,1]).abs().max()),
        same_focal_sampling_indices=torch.equal(sampling[first],sampling[second]),
        interpretation='Slot-paired only if learned sampling differs; not fixed-query confidence.')
for name,value in model.named_buffers():value.copy_(buffers[name])
assert parameter_versions=={name:value._version for name,value in model.named_parameters()}
assert digest(checkpoint_path)==FROZEN['checkpoint_sha256']
assert (root/'.codex_source_commit').read_text().strip()==FROZEN['commit']
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y59',rows=rows,cases=comparisons,
    actual_sparse_TRAIN_loss_backward=True,trained_W2_diagnostic_only=True,
    exact_same_batch_and_seed=True,other_dropout_held_disabled=True,BN_mode_unchanged=True,
    encoder_RNG_consumption_isolated=True,
    model_weights_unchanged=True,buffers_restored=True,optimizer_steps=0,weights_saved=False,
    admission=False,limitation='Native mode-contract mechanism diagnosis only. No independent candidate '
        'package, released-init candidate gate, optimizer test, resource Pareto or trained benefit/harm.')),
    encoding='utf-8')
