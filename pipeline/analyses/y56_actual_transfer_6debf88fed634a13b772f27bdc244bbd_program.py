FROZEN={'scenes': [{'dataset_index': 0, 'scan_id': 'scene0000_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 44, 'scan_id': 'scene0001_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 60, 'scan_id': 'scene0002_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 101, 'scan_id': 'scene0003_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 115, 'scan_id': 'scene0004_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 126, 'scan_id': 'scene0005_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 137, 'scan_id': 'scene0006_00', 'ann_ids': ['0', '1', '2', '3']}, {'dataset_index': 171, 'scan_id': 'scene0007_00', 'ann_ids': ['0', '1', '2', '3']}], 'sources': [{'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/__init__.py', 'lf_sha256': '8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/backbone.py', 'lf_sha256': '1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'lf_sha256': '7e60c1cecfc8fa7809f7c836fa6fef1366806bd23c08e2890508ea26a711acac'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'lf_sha256': '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/loss.py', 'lf_sha256': '4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/sample_model.py', 'lf_sha256': '03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b'}], 'commit': '8943c8c460501b200a8dea958964579ffea65f76', 'config_values': {'work_dir': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints', 'train': {'seed': 1999, 'epochs': 30, 'interval': 1, 'pretrain': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/sp_unet_backbone.pth', 'save_epoch': 100, 'grad_clip': 0.1}, 'test': {'seed': 1999, 'test_topk_per_scene': 100, 'test_score_thresh': 0.0, 'test_npoint_thresh': 100}, 'dataloader': {'train': {'batch_size': 2, 'num_workers': 4, 'persistent_workers': True}, 'val': {'batch_size': 1, 'num_workers': 2, 'persistent_workers': True}, 'test': {'batch_size': 1, 'num_workers': 1, 'persistent_workers': True}}, 'data': {'train': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/', 'prefix': 'train', 'suffix': '.pth', 'training': True, 'mode': 4, 'aug': True, 'with_elastic': True, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 4}, 'val': {'type': 'scannetv2_sample_graph_edge', 'dataset': 'scanrefer', 'data_root': '/bak_home/songcl/3D_visual_grounding/data/ipdn_data/data/', 'prefix': 'val', 'suffix': '.pth', 'training': False, 'aug': False, 'with_elastic': False, 'use_xyz': True, 'voxel_cfg': {'scale': 50, 'spatial_shape': [128, 512], 'max_npoint': 2500000}, 'max_des_len': 78, 'lang_num_max': 32}}, 'model': {'input_channel': 6, 'blocks': 5, 'block_reps': 2, 'media': 32, 'normalize_before': True, 'return_blocks': True, 'pool': 'mean', 'infer_mode': 'sem', 'fps_num': 256, 'task_type': 'res', 'sampling_module': {'num_proposal': 128, 'pc_dim': 256, 'lang_dim': 256, 'd_model': 256}, 'dec': {'num_layer': 6, 'd_model': 256, 'nhead': 8, 'hidden_dim': 1024, 'dropout': 0.0, 'activation_fn': 'gelu', 'iter_pred': True, 'attn_mask': True, 'kernel': 'top1', 'global_feat': 'mean', 'lang_att': True, 'contrastive_align_loss': True, 'num_proposal_fine': 512, 'num_proposal_query': [128, 64, 32, 16, 8, 4], 'num_proposal_query_inst': [128, 64, 32, 16, 8, 4]}, 'criterion': {'loss_weight': [1.0, 1.0, 0.5, 5.0, 0.1, 0.1, 0.1, 1.0], 'cost_weight': [1.0, 1.0, 0.1], 'loss_fun': 'focal', 'match_last_layer': True, 'layer_differ_weight': True}, 'test_cfg': {'topk_insts': 100, 'score_thr': 0.0, 'npoint_thr': 100}, 'norm_eval': False, 'fix_module': ['text_encoder']}, 'optimizer': {'type': 'AdamW', 'lr': 0.0001, 'weight_decay': 0.0005, 'paramwise_cfg': {'text_encoder': {'lr_mult': 0.001}, 'input_conv': {'lr_mult': 0.01}, 'unet': {'lr_mult': 0.01}, 'output_layer': {'lr_mult': 0.01}, 'dec': {'lr_mult': 1.0}}}, 'lr_scheduler': {'type': 'PolyLR', 'max_iters': 70, 'power': 4.0, 'constant_ending': 0.0}}, 'incidence_function': 'def incidence(query, masks):\n    q, m = query.detach().cpu().numpy(), masks.detach().cpu().numpy()\n    _, inverse, counts = np.unique(np.concatenate([q, m], axis=1), axis=0,\n                                   return_inverse=True, return_counts=True)\n    return dict(query_unique=int(len(np.unique(q, axis=0))),\n        soft_mask_unique=int(len(np.unique(m, axis=0))),\n        joint_unique=int(len(counts)), hard_mask_unique=int(len(np.unique(m > 0, axis=0))),\n        joint_duplicate_excess=int(len(q) - len(counts)),\n        hard_mask_duplicate_excess=int(len(q) - len(np.unique(m > 0, axis=0)))), inverse, counts', 'checkpoint': '/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', 'checkpoint_sha256': 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52', 'capture_sha256': '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89'}
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

"""Append to native W2 setup: fixed-focal transfer in discovered duplicate cases."""

capture_path = Path("/tmp/adares_y56_broader_unique_6debf88fed634a13b772f27bdc244bbd.json")
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == (
    "c02ac3d48780e9b5557e9edacaecdcc8a98b9c27d0b35074fb4ee7e14a607cee")
capture = json.loads(capture_path.read_text())
pairs = sorted({r["pair"] for r in capture["rows"] if r["joint_duplicate_excess"]})
assert pairs == [8]
cases = []
started = time.perf_counter()


def corrected(module, query, masks, mass, inverse, counts):
    probabilities = masks.sigmoid()
    weighted = probabilities * mass.unsqueeze(0)
    areas = weighted.sum(-1)
    intersection = weighted @ probabilities.T
    overlap = intersection / (
        areas[:, None] + areas[None, :] - intersection).clamp_min(1e-6)
    overlap.fill_diagonal_(0)
    weights = overlap / counts[inverse].to(query.dtype).unsqueeze(0)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    normalized = module.query_norm(query)
    context = weights @ normalized
    evidence = torch.cat((
        normalized, context, normalized - context,
        overlap.max(-1).values.unsqueeze(-1)), -1)
    return module.head(module.evidence_norm(evidence))


for pair in pairs:
    items, identities = [], []
    for scene in capture["cohort"][pair * 2:pair * 2 + 2]:
        gorilla.set_random_seed(1999)
        item = dataset[scene["dataset_index"]]
        single = dataset.collate_fn([copy.deepcopy(item)])
        assert str(single["scan_ids"][0]) == scene["scene"]
        assert [str(v) for v in single["ann_ids"]] == scene["ann_ids"]
        items.append(item)
        identities.extend((scene["scene"], ann) for ann in scene["ann_ids"])
    batch = dataset.collate_fn(copy.deepcopy(items))
    targets = []
    expression = 0
    for scene_index, length in enumerate(batch["scenes_len"]):
        start, end = [int(v) for v in batch["batch_offsets"][scene_index:scene_index + 2]]
        selected = (batch["superpoints"] >= start) & (batch["superpoints"] < end)
        membership = batch["superpoints"][selected].long() - start
        total = torch.bincount(membership, minlength=end - start)
        for _ in range(int(length)):
            gt = batch["gt_pmasks"][expression].reshape(-1)
            assert gt.numel() == membership.numel()
            positive = torch.bincount(
                membership, weights=(gt > 0.5).float(), minlength=len(total))
            targets.append((total, positive, hashlib.sha256(
                gt.contiguous().numpy().tobytes()).hexdigest()))
            expression += 1
    calls = []

    def observe(module, inputs, output):
        layer = len(calls)
        calls.append(layer)
        for index, (scene, ann) in enumerate(identities):
            q = inputs[0][index]
            valid = ~inputs[2][index]
            m = inputs[1][index, :, valid]
            mass = inputs[3][index, valid]
            unique, inverse, counts = torch.unique(
                torch.cat((q, m), dim=1), dim=0, return_inverse=True, return_counts=True)
            row = dict(scene=scene, ann_id=ann, layer=layer, focal=0,
                       duplicate_excess=int(128 - len(unique)))
            if len(unique) == 128:
                row["transfer"] = "not_applicable_no_exact_joint_copy"
                cases.append(row)
                continue
            if int(counts[inverse[0]]) != 1:
                row["transfer"] = "not_applicable_preregistered_focal_not_unique"
                cases.append(row)
                continue
            removable = [i for i in range(1, 128) if int(counts[inverse[i]]) > 1]
            assert removable
            removed = removable[-1]
            recipients = [i for i in range(1, 128) if inverse[i] != inverse[removed]]
            assert recipients
            recipient = recipients[0]
            changed_q, changed_m = q.clone(), m.clone()
            changed_q[removed], changed_m[removed] = q[recipient], m[recipient]
            changed_unique, changed_inverse, changed_counts = torch.unique(
                torch.cat((changed_q, changed_m), 1), dim=0,
                return_inverse=True, return_counts=True)
            assert torch.equal(unique, changed_unique)
            assert torch.equal(q[0], changed_q[0]) and torch.equal(m[0], changed_m[0])
            padding = torch.zeros((1, m.shape[-1]), dtype=torch.bool, device=q.device)

            def native(query, masks):
                return module.forward(query.unsqueeze(0), masks.unsqueeze(0),
                                      padding, mass.unsqueeze(0))[0]

            original = native(q, m)
            repeated = native(q, m)
            transferred = native(changed_q, changed_m)
            perm = torch.roll(torch.arange(128, device=q.device), 1)
            permuted = native(q[perm], m[perm])[torch.argsort(perm)]
            dedup = corrected(module, q, m, mass, inverse, counts)
            dedup_transferred = corrected(
                module, changed_q, changed_m, mass, changed_inverse, changed_counts)
            total, positive, gt_hash = targets[index]
            total, positive = total.to(q.device), positive.to(q.device)
            assert torch.equal(total.to(mass.dtype), mass)
            binary = (m[0].sigmoid() >= 0.5).to(q.dtype)
            numerator = (binary * positive).sum()
            quality = numerator / (
                (binary * total).sum() + positive.sum() - numerator + 1e-6)
            bound = 128 * torch.finfo(q.dtype).eps * max(
                1.0, float(original.abs().max()), float(dedup.abs().max()))
            row.update(
                transfer="executed", removed_copy_index=removed,
                recipient_prototype_index=recipient,
                unique_set_and_focal_unchanged=True,
                focal_point_quality_unchanged=float(quality),
                actual_point_GT_sha256=gt_hash,
                native_repeat_max_abs=float((original - repeated).abs().max()),
                native_permutation_max_abs=float((original - permuted).abs().max()),
                native_focal_logit_change=float((original[0] - transferred[0]).abs().max()),
                native_focal_confidence_change=float(abs(
                    original[0].softmax(-1)[1] - transferred[0].softmax(-1)[1])),
                inverse_multiplicity_focal_change=float(
                    (dedup[0] - dedup_transferred[0]).abs().max()),
                arithmetic_bound=bound)
            assert row["inverse_multiplicity_focal_change"] <= bound
            cases.append(row)

    handle = model.dec.indi_embedding.register_forward_hook(observe)
    try:
        gorilla.set_random_seed(1999)
        with torch.no_grad():
            loss, _ = model(copy.deepcopy(batch), mode="loss")
        assert torch.isfinite(loss).all() and len(calls) == 7
    finally:
        handle.remove()
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    del loss, batch, items
assert parameter_versions == {name: value._version for name, value in model.named_parameters()}
assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
assert digest(checkpoint_path) == FROZEN["checkpoint_sha256"]
executed = [r for r in cases if r["transfer"] == "executed"]
result = dict(
    complete=True, id="AdaRES-Y56", fixed_focal=0, discovery_followup_pairs=pairs,
    native_sparse_TRAIN_batch2=True, expressions=8, expression_layer_records=len(cases),
    actual_duplicate_records=sum(r["duplicate_excess"] > 0 for r in cases),
    transfer_cases=len(executed), cases=cases,
    parameters_unchanged=True, buffers_restored=True, optimizer_steps=0,
    weights_saved=False, admission=False, elapsed_seconds=time.perf_counter() - started,
    limitations="Result-conditioned case follow-up, NOT new prevalence evidence. Same unique "
                "competitor set and unchanged focal point IoU do not establish prediction harm "
                "or trained efficacy. Inverse multiplicity is a diagnostic reference only; "
                "no candidate implementation, new layers/thresholds or training.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
