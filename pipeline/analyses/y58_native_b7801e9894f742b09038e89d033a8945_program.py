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

"""Pending exclusive native batch-mate/BN diagnosis; never a training launch."""

import gc
import inspect

assert digest(Path(inspect.getfile(type(dataset))))==SCOPE['dataset_source']['sha256']
items=[]
for scene in FROZEN['scenes'][:3]:
    gorilla.set_random_seed(1999)
    item=dataset[scene['dataset_index']]
    single=dataset.collate_fn([copy.deepcopy(item)])
    assert str(single['scan_ids'][0])==scene['scan_id']
    assert [str(v) for v in single['ann_ids']]==scene['ann_ids']
    items.append(item)
single=dataset.collate_fn([copy.deepcopy(items[0])])
focal_points=int(single['coords_float'].shape[0])
focal_expressions=len(single['ann_ids'])
captures={}
rows=[]
for mode in ('pooled_training_BN','fixed_running_BN'):
    for label,mate in (('A',1),('B',2),('repeatA',1)):
        for name,value in model.named_buffers():
            value.copy_(buffers[name])
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.dropout._DropoutNd):
                module.eval()
            if mode=='fixed_running_BN' and isinstance(module,torch.nn.BatchNorm1d):
                module.eval()
        batch=dataset.collate_fn(copy.deepcopy([items[0],items[mate]]))
        assert torch.equal(batch['coords_float'][:focal_points],single['coords_float'])
        assert torch.equal(batch['feats'][:focal_points],single['feats'])
        assert [str(v) for v in batch['ann_ids'][:focal_expressions]]==FROZEN['scenes'][0]['ann_ids']
        activation=[]
        outputs=[]
        def observe_backbone(module,args,value):
            chosen=value.indices[:,0]==0
            coords=value.indices[chosen].detach().cpu().numpy()
            features=value.features[chosen].detach().cpu()
            order=np.lexsort((coords[:,3],coords[:,2],coords[:,1]))
            activation.append((coords[order].copy(),features[torch.from_numpy(order).long()].clone()))
        def observe_decoder(module,args,value):
            seeds=value.get('sample_inds')
            seeds=seeds[:focal_expressions].detach().cpu().clone() if torch.is_tensor(seeds) else None
            outputs.append((value['indis'][:focal_expressions].detach().cpu().clone(),seeds))
        hooks=[model.output_layer.register_forward_hook(observe_backbone),model.dec.register_forward_hook(observe_decoder)]
        gorilla.set_random_seed(1999)
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                loss,_=model(copy.deepcopy(batch),mode='loss')
            assert torch.isfinite(loss).all() and len(activation)==len(outputs)==1
            assert torch.isfinite(activation[0][1]).all() and torch.isfinite(outputs[0][0]).all()
            torch.cuda.synchronize()
            peak=int(torch.cuda.max_memory_reserved())
            assert peak<24*2**30
            captures[mode,label]=dict(coords=activation[0][0],features=activation[0][1],
                                     logits=outputs[0][0],seeds=outputs[0][1])
            rows.append(dict(mode=mode,mate=label,mate_scene=FROZEN['scenes'][mate]['scan_id'],
                total_native_loss=float(loss.item()),focal_voxels=len(activation[0][0]),
                peak_reserved_bytes=peak,dropout_modules_training=sum(
                    isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.training for m in model.modules()),
                BN_training_modules=sum(isinstance(m,torch.nn.BatchNorm1d) and m.training for m in model.modules())))
        finally:
            for hook in hooks:hook.remove()
        del loss,batch
        gc.collect()
comparisons=[]
for mode in ('pooled_training_BN','fixed_running_BN'):
    a,b,repeat=(captures[mode,label] for label in ('A','B','repeatA'))
    assert np.array_equal(a['coords'],b['coords']) and np.array_equal(a['coords'],repeat['coords'])
    same_seeds=(torch.equal(a['seeds'],b['seeds']) if a['seeds'] is not None and b['seeds'] is not None else None)
    delta=a['features']-b['features']
    comparisons.append(dict(mode=mode,focal_voxel_coordinates_identical=True,
        feature_mate_max=float(delta.abs().max()),feature_mate_RMS=float(delta.square().mean().sqrt()),
        feature_repeat_max=float((a['features']-repeat['features']).abs().max()),
        same_focal_sampling_indices=same_seeds,
        confidence_slot_mate_max=float((a['logits'].softmax(-1)[...,1]-b['logits'].softmax(-1)[...,1]).abs().max()),
        confidence_slot_repeat_max=float((a['logits'].softmax(-1)[...,1]-repeat['logits'].softmax(-1)[...,1]).abs().max()),
        confidence_interpretation='Slot-paired only; not a matched fixed-query confidence contrast if sampling changed.',
        total_loss_interpretation='Includes a different mate target; not a signed task benefit/harm contrast.'))
for name,value in model.named_buffers():value.copy_(buffers[name])
assert parameter_versions=={name:value._version for name,value in model.named_parameters()}
assert all(torch.equal(value,buffers[name]) for name,value in model.named_buffers())
assert digest(checkpoint_path)==FROZEN['checkpoint_sha256']
assert (root/'.codex_source_commit').read_text().strip()==FROZEN['commit']
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y58',rows=rows,cases=comparisons,
    actual_sparse_model=True,actual_TRAIN_batch2=True,model_weights_unchanged=True,buffers_restored=True,
    dropout_held_disabled_in_all_arms=True,optimizer_steps=0,weights_saved=False,admission=False,
    limitation='Native trained-parent context diagnosis only, not released-init candidate backward, '
               'population harm/benefit or permission to replace normalization.')),
    encoding='utf-8')
