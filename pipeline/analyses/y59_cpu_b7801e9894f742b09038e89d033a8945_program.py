PAYLOAD={'frozen': {'scenes': [{'dataset_index': 0, 'scan_id': 'scene0000_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 44, 'scan_id': 'scene0001_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 60, 'scan_id': 'scene0002_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 101, 'scan_id': 'scene0003_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 115, 'scan_id': 'scene0004_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 126, 'scan_id': 'scene0005_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 137, 'scan_id': 'scene0006_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 171, 'scan_id': 'scene0007_00', 'ann_ids': ['0', '1', '2', '3']}], 'sources': [{'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/__init__.py', 'lf_sha256': '8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/backbone.py', 'lf_sha256': '1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'lf_sha256': '7e60c1cecfc8fa7809f7c836fa6fef1366806bd23c08e2890508ea26a711acac'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'lf_sha256': '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/loss.py', 'lf_sha256': '4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/sample_model.py', 'lf_sha256': '03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b'}], 'commit': '05dc2c0424e33f460f96a4ab4938b0d05e7dc6ea', 'config_values': {'work_dir': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'train': {'seed': 1999, 'epochs': 30, 'interval': 1, 'pretrain': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/sp_unet_backbone.pth', 'save_epoch': 100, 'grad_clip': 0.1}, 'test': {'seed': 1999, 'test_topk_per_scene': 100, 'test_score_thresh': 0.0, 'test_npoint_thresh': 100}, 'dataloader': {'train': {'batch_size': 2, 'num_workers': 4, 'persistent_workers': True}, 'val': {'batch_size': 1, 'num_workers': 2, 'persistent_workers': True}, 'test': {'batch_size': 1, 'num_workers': 1, 'persistent_workers': True}}, 'data': {'train': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/', 'prefix': 'train', 'suffix': '.pth', 'training': True, 'mode': 4, 'aug': True, 'with_elastic': True, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 4}, 'val': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/', 'prefix': 'val', 'suffix': '.pth', 'training': False, 'aug': False, 'with_elastic': False, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 32}}, 'model': {'input_channel': 6, 'blocks': 5, 'block_reps': 2, 'media': 32, 'normalize_before': True, 'return_blocks': True, 'pool': 'mean', 'infer_mode': 'sem', 'fps_num': 256, 'task_type': 'res', 'sampling_module': {'num_proposal': 128, 'pc_dim': 256, 'lang_dim': 256, 'd_model': 256}, 'dec': {'num_layer': 6, 'd_model': 256, 'nhead': 8, 'hidden_dim': 1024, 'dropout': 0.0, 'activation_fn': 'gelu', 'iter_pred': True, 'attn_mask': True, 'kernel': 'top1', 'global_feat': 'mean', 'lang_att': True, 'contrastive_align_loss': True, 'num_proposal_fine': 512, 'num_proposal_query': [128, 64, 32, 16, 8, 4], 'num_proposal_query_inst': [128, 64, 32, 16, 8, 4]}, 'criterion': {'loss_weight': [1.0, 1.0, 0.5, 5.0, 0.1, 0.1, 0.1, 1.0], 'cost_weight': [1.0, 1.0, 0.1], 'loss_fun': 'focal', 'match_last_layer': True, 'layer_differ_weight': True}, 'test_cfg': {'topk_insts': 100, 'score_thr': 0.0, 'npoint_thr': 100}, 'norm_eval': False, 'fix_module': ['text_encoder']}, 'optimizer': {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0005, 'paramwise_cfg': {'text_encoder': {'lr_mult': 0.001}, 'input_conv': {'lr_mult': 0.01}, 'unet': {'lr_mult': 0.01}, 'output_layer': {'lr_mult': 0.01}, 'dec': {'lr_mult': 1.0}}}, 'lr_scheduler': {'type': 'PolyLR', 'max_iters': 70, 'power': 4.0, 'constant_ending': 0.0}}, 'incidence_function': 'def incidence(query, masks):\n    q, m = query.detach().cpu().numpy(), masks.detach().cpu().numpy()\n    _, inverse, counts = np.unique(np.concatenate([q, m], axis=1), axis=0,\n                                   return_inverse=True, return_counts=True)\n    return dict(query_unique=int(len(np.unique(q, axis=0))),\n        soft_mask_unique=int(len(np.unique(m, axis=0))),\n        joint_unique=int(len(counts)), hard_mask_unique=int(len(np.unique(m > 0, axis=0))),\n        joint_duplicate_excess=int(len(q) - len(counts)),\n        hard_mask_duplicate_excess=int(len(q) - len(np.unique(m > 0, axis=0)))), inverse, counts', 'checkpoint': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', 'checkpoint_sha256': 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52', 'capture_sha256': '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'}, 'exact_train_method_shim': 'class IPDN(nn.Module):\n\n    def train(self, mode=True):\n        super(IPDN, self).train(mode)\n        if mode and self.norm_eval:\n            for m in self.modules():\n                if isinstance(m, nn.BatchNorm1d):\n                    m.eval()', 'dataset': {'path': '/bak_home/songcl/Project/AdaRES_project/AdaRES/ipdn/dataset/scannetv2_test.py', 'sha256': '1328bad8384392f6a8dac2a6a46126b82000132283706f9398030be1cf0f0cc8'}}
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
