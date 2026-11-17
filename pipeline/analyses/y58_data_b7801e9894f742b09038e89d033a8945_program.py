FROZEN={'scenes': [{'dataset_index': 0, 'scan_id': 'scene0000_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 44, 'scan_id': 'scene0001_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 60, 'scan_id': 'scene0002_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 101, 'scan_id': 'scene0003_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 115, 'scan_id': 'scene0004_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 126, 'scan_id': 'scene0005_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 137, 'scan_id': 'scene0006_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 171, 'scan_id': 'scene0007_00', 'ann_ids': ['0', '1', '2', '3']}], 'sources': [{'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/__init__.py', 'lf_sha256': '8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/backbone.py', 'lf_sha256': '1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'lf_sha256': '7e60c1cecfc8fa7809f7c836fa6fef1366806bd23c08e2890508ea26a711acac'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'lf_sha256': '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/loss.py', 'lf_sha256': '4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/sample_model.py', 'lf_sha256': '03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b'}], 'commit': '05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea', 'config_values': {'work_dir': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'train': {'seed': 1999, 'epochs': 30, 'interval': 1, 'pretrain': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/sp_unet_backbone.pth', 'save_epoch': 100, 'grad_clip': 0.1}, 'test': {'seed': 1999, 'test_topk_per_scene': 100, 'test_score_thresh': 0.0, 'test_npoint_thresh': 100}, 'dataloader': {'train': {'batch_size': 2, 'num_workers': 4, 'persistent_workers': True}, 'val': {'batch_size': 1, 'num_workers': 2, 'persistent_workers': True}, 'test': {'batch_size': 1, 'num_workers': 1, 'persistent_workers': True}}, 'data': {'train': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/', 'prefix': 'train', 'suffix': '.pth', 'training': True, 'mode': 4, 'aug': True, 'with_elastic': True, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 4}, 'val': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/', 'prefix': 'val', 'suffix': '.pth', 'training': False, 'aug': False, 'with_elastic': False, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 32}}, 'model': {'input_channel': 6, 'blocks': 5, 'block_reps': 2, 'media': 32, 'normalize_before': True, 'return_blocks': True, 'pool': 'mean', 'infer_mode': 'sem', 'fps_num': 256, 'task_type': 'res', 'sampling_module': {'num_proposal': 128, 'pc_dim': 256, 'lang_dim': 256, 'd_model': 256}, 'dec': {'num_layer': 6, 'd_model': 256, 'nhead': 8, 'hidden_dim': 1024, 'dropout': 0.0, 'activation_fn': 'gelu', 'iter_pred': True, 'attn_mask': True, 'kernel': 'top1', 'global_feat': 'mean', 'lang_att': True, 'contrastive_align_loss': True, 'num_proposal_fine': 512, 'num_proposal_query': [128, 64, 32, 16, 8, 4], 'num_proposal_query_inst': [128, 64, 32, 16, 8, 4]}, 'criterion': {'loss_weight': [1.0, 1.0, 0.5, 5.0, 0.1, 0.1, 0.1, 1.0], 'cost_weight': [1.0, 1.0, 0.1], 'loss_fun': 'focal', 'match_last_layer': True, 'layer_differ_weight': True}, 'test_cfg': {'topk_insts': 100, 'score_thr': 0.0, 'npoint_thr': 100}, 'norm_eval': False, 'fix_module': ['text_encoder']}, 'optimizer': {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0005, 'paramwise_cfg': {'text_encoder': {'lr_mult': 0.001}, 'input_conv': {'lr_mult': 0.01}, 'unet': {'lr_mult': 0.01}, 'output_layer': {'lr_mult': 0.01}, 'dec': {'lr_mult': 1.0}}}, 'lr_scheduler': {'type': 'PolyLR', 'max_iters': 70, 'power': 4.0, 'constant_ending': 0.0}}, 'incidence_function': 'def incidence(query, masks):\n    q, m = query.detach().cpu().numpy(), masks.detach().cpu().numpy()\n    _, inverse, counts = np.unique(np.concatenate([q, m], axis=1), axis=0,\n                                   return_inverse=True, return_counts=True)\n    return dict(query_unique=int(len(np.unique(q, axis=0))),\n        soft_mask_unique=int(len(np.unique(m, axis=0))),\n        joint_unique=int(len(counts)), hard_mask_unique=int(len(np.unique(m > 0, axis=0))),\n        joint_duplicate_excess=int(len(q) - len(counts)),\n        hard_mask_duplicate_excess=int(len(q) - len(np.unique(m > 0, axis=0)))), inverse, counts', 'checkpoint': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', 'checkpoint_sha256': 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52', 'capture_sha256': '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'}
"""CPU native-dataset focal identity checks, without constructing a CUDA model."""

import copy
import hashlib
import inspect
import json
from pathlib import Path
import gorilla
import numpy as np
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert P['physical_gpu'] is None
torch.set_num_threads(2)
root=Path(P['root'])
assert (root/'.codex_source_commit').read_text().strip()==FROZEN['commit']
for row in FROZEN['sources']:
    assert hashlib.sha256((root/row['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==row['lf_sha256']
cfg=gorilla.Config.fromfile(str(root/'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'))
assert cfg.train.seed==1999 and cfg.train.epochs==30 and cfg.dataloader.train.batch_size==2
assert not cfg.model.norm_eval
dataset=build_dataset(cfg.data.train.copy(),get_root_logger())
source=Path(inspect.getfile(type(dataset)))
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
items=[]
for scene in FROZEN['scenes'][:3]:
    gorilla.set_random_seed(1999)
    item=dataset[scene['dataset_index']]
    single=dataset.collate_fn([copy.deepcopy(item)])
    assert str(single['scan_ids'][0])==scene['scan_id']
    assert [str(v) for v in single['ann_ids']]==scene['ann_ids']
    items.append(item)
focal=dataset.collate_fn([copy.deepcopy(items[0])])
n_points=int(focal['coords_float'].shape[0])
n_expression=len(focal['ann_ids'])


def identical(first,second):
    if torch.is_tensor(first):
        return torch.is_tensor(second) and torch.equal(first,second)
    if isinstance(first,np.ndarray):
        return isinstance(second,np.ndarray) and np.array_equal(first,second)
    if isinstance(first,(list,tuple)):
        return isinstance(second,(list,tuple)) and len(first)==len(second) and all(identical(a,b) for a,b in zip(first,second))
    if isinstance(first,dict):
        return isinstance(second,dict) and first.keys()==second.keys() and all(identical(first[k],second[k]) for k in first)
    return first==second


rows=[]
for mate in (1,2,1):
    batch=dataset.collate_fn(copy.deepcopy([items[0],items[mate]]))
    assert torch.equal(batch['coords_float'][:n_points],focal['coords_float'])
    assert torch.equal(batch['feats'][:n_points],focal['feats'])
    assert batch['scan_ids'][0]==focal['scan_ids'][0]
    assert [str(v) for v in batch['ann_ids'][:n_expression]]==[str(v) for v in focal['ann_ids']]
    identities={}
    for key in ('gt_pmasks','gt_spmasks','sp_ref_masks','object_idss','lang_tokenss','lang_masks'):
        if key not in focal:
            identities[key]='unavailable'
            continue
        first,second=focal[key],batch[key]
        if isinstance(first,(list,tuple)):
            identities[key]=identical(first,second[:len(first)])
        elif torch.is_tensor(first) and torch.is_tensor(second):
            identities[key]=identical(first,second[:first.shape[0]])
        else:
            identities[key]='unsupported layout'
    rows.append(dict(focal=FROZEN['scenes'][0]['scan_id'],mate=FROZEN['scenes'][mate]['scan_id'],
        focal_points=n_points,total_points=int(batch['coords_float'].shape[0]),expressions=n_expression,
        focal_coords_and_features_bitwise_equal=True,focal_annotation_ids_equal=True,
        target_and_language_identity=identities))
assert all(all(value is True for value in r['target_and_language_identity'].values()) for r in rows)
assert not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True,id='AdaRES-Y58',rows=rows,
    dataset_source=dict(path=str(source),sha256=source_sha),
    dataset_class=type(dataset).__module__+'.'+type(dataset).__name__,
    collate_source_sha256=hashlib.sha256(inspect.getsource(type(dataset).collate_fn).encode('utf-8')).hexdigest(),
    CPU_native_dataset=True,CUDA_initialized=False,native_model_executed=False,
    optimizer_steps=0,weights_saved=False,admission=False,
    limitation='Actual fixed augmented TRAIN focal item/GT/language invariant across two batch mates; '
               'sparse backbone and confidence response still require a free exclusive GPU.')),
    encoding='utf-8')
