P={'id': 'AdaRES-Y92', 'pid': 1305468, 'root': '/data1/songcl/Project/AdaRES_project/AdaRES', 'work': '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'config': 'configs_song/model_paper5/adares_y92_mask_feature_quadratic_res.yaml', 'log': '/data1/songcl/Project/AdaRES_project/AdaRES/logs/train_adares_y92_mask_feature_quadratic_res_20261108_090155.log', 'basename': 'adares_y92_mask_feature_quadratic_res', 'code_commit': '27d6f67e7121a5168bbfeca9db745377e0bd49e5', 'expected_config': {'work_dir': '/data1/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'train': {'seed': 1999, 'epochs': 30, 'interval': 1, 'pretrain': '/data1/songcl/Project/AdaRES_project/data/sp_unet_backbone.pth', 'save_epoch': 100, 'grad_clip': 0.1}, 'test': {'seed': 1999, 'test_topk_per_scene': 100, 'test_score_thresh': 0.0, 'test_npoint_thresh': 100}, 'dataloader': {'train': {'batch_size': 2, 'num_workers': 4, 'persistent_workers': True}, 'val': {'batch_size': 1, 'num_workers': 2, 'persistent_workers': True}, 'test': {'batch_size': 1, 'num_workers': 1, 'persistent_workers': True}}, 'data': {'train': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/data1/songcl/Project/AdaRES_project/data/', 'prefix': 'train', 'suffix': '.pth', 'training': True, 'mode': 4, 'aug': True, 'with_elastic': True, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 4}, 'val': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/data1/songcl/Project/AdaRES_project/data/', 'prefix': 'val', 'suffix': '.pth', 'training': False, 'aug': False, 'with_elastic': False, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 32}}, 'model': {'input_channel': 6, 'blocks': 5, 'block_reps': 2, 'media': 32, 'normalize_before': True, 'return_blocks': True, 'pool': 'mean', 'infer_mode': 'sem', 'fps_num': 256, 'task_type': 'res', 'sampling_module': {'num_proposal': 128, 'pc_dim': 256, 'lang_dim': 256, 'd_model': 256}, 'dec': {'num_layer': 6, 'd_model': 256, 'nhead': 8, 'hidden_dim': 1024, 'dropout': 0.0, 'activation_fn': 'gelu', 'iter_pred': True, 'attn_mask': True, 'kernel': 'top1', 'global_feat': 'mean', 'lang_att': True, 'contrastive_align_loss': True, 'num_proposal_fine': 512, 'num_proposal_query': [128, 64, 32, 16, 8, 4], 'num_proposal_query_inst': [128, 64, 32, 16, 8, 4]}, 'criterion': {'loss_weight': [1.0, 1.0, 0.5, 5.0, 0.1, 0.1, 0.1, 1.0], 'cost_weight': [1.0, 1.0, 0.1], 'loss_fun': 'focal', 'match_last_layer': True, 'layer_differ_weight': True}, 'test_cfg': {'topk_insts': 100, 'score_thr': 0.0, 'npoint_thr': 100}, 'norm_eval': False, 'fix_module': ['text_encoder']}, 'optimizer': {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0005, 'paramwise_cfg': {'text_encoder': {'lr_mult': 0.001}, 'input_conv': {'lr_mult': 0.01}, 'unet': {'lr_mult': 0.01}, 'output_layer': {'lr_mult': 0.01}, 'dec': {'lr_mult': 1.0}}}, 'lr_scheduler': {'type': 'PolyLR', 'max_iters': 70, 'power': 4.0, 'constant_ending': 0.0}}, 'source_files': {'ipdn/model_paper5/adares_y92_mask_feature_quadratic/__init__.py': '27662befed0f9d4f20f8902263f0323651746ea3019ff44a34ee54e7b11532ec', 'ipdn/model_paper5/adares_y92_mask_feature_quadratic/backbone.py': '57db8747d3c38136f568837d70e75ceea23e101900679db4931724ecb57665ca', 'ipdn/model_paper5/adares_y92_mask_feature_quadratic/dec.py': 'fd831dc87147ad797bb32079c5210c6e259a0c3d7f871ba6a417c84200248ca1', 'ipdn/model_paper5/adares_y92_mask_feature_quadratic/ipdn.py': '55160dfb062353054f6d2d5a09d971a6dc779be2e8fe34e6906187576a623ad9', 'ipdn/model_paper5/adares_y92_mask_feature_quadratic/loss.py': '9e6e5925523114caba6213d181b11181abd2820b292730999986b068816046d1', 'ipdn/model_paper5/adares_y92_mask_feature_quadratic/sample_model.py': '674e7ef698c6e81ed1c35f14680e1ec748f5c783ed879a6ca72f099b6450bc36'}, 'deployments': {'27d6f67e7121a5168bbfeca9db745377e0bd49e5': {'source_files': {}, 'configs': {}}}, 'allowed_markers': ['27d6f67e7121a5168bbfeca9db745377e0bd49e5']}

import json,pathlib,re,subprocess
assert subprocess.check_output(['hostname']).decode().strip()=='dl02-MS-7B73'
log=pathlib.Path(P['log']).resolve()
log.relative_to(pathlib.Path(P['root'])/'logs')
last=None
with log.open('r',encoding='utf-8',errors='replace') as stream:
 for line in stream:
  paths=re.findall(r'save last pth to ([^\s]+/last\.pth)',line)
  if paths:last=paths[-1]
assert last,'No native checkpoint save announcement yet'
path=pathlib.Path(last).resolve()
path.relative_to(pathlib.Path(P['work'])/P['basename'])
assert re.fullmatch(r'\d{8}_\d{6}',path.parent.name)
assert path.is_file()
import hashlib,yaml
root=pathlib.Path(P['root'])
for relative,digest in P['source_files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
assert yaml.safe_load((root/P['config']).read_text())==P['expected_config']
marker=(root/'.codex_source_commit').read_text().strip()
assert marker in P['allowed_markers']
for relative,digest in P['deployments'][marker]['source_files'].items():
 assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
for relative,config in P['deployments'][marker]['configs'].items():
 assert yaml.safe_load((root/relative).read_text())==config

process=pathlib.Path('/proc')/str(P['pid'])
assert process.is_dir() and (process/'cwd').resolve()==pathlib.Path(P['root'])
args=(process/'cmdline').read_bytes().split(b'\0')
assert b'tools/train.py' in args and P['config'].encode() in args
print(json.dumps(dict(complete=True,checkpoint_directory=path.parent.name,
 declared_last_path=str(path),log=str(log),checkpoint_loaded=False,remote_mutations=False)))
