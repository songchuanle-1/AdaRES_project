PAYLOAD={'parent_files': [{'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/__init__.py', 'sha256': 'c540732d6e3334c85008035cc37b8537b0f85383d5f5a6bf663359fbf2b134bb'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/backbone.py', 'sha256': '57db8747d3c38136f568837d70e75ceea23e101900679db4931724ecb57665ca'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'sha256': '6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'sha256': '7cb4b1c5490d0839fcdaf18553983b4c991f69e50474d07e83af9f0f8c937a5a'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/loss.py', 'sha256': '836ff790a0a8b9c55cf465a472c9462699f0753a7e87d0ef74a8946edf31565a'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/sample_model.py', 'sha256': '674e7ef698c6e81ed1c35f14680e1ec748f5c783ed879a6ca72f099b6450bc36'}], 'capture_sha256': '5716585e9624d06ff5ecd90ddddec1faf95db9333199daf1b5eaa86907219d89', 'config_sha256': 'd4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb', 'checkpoint_sha256': 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'}
"""Replay cached actual TRAIN decoder inputs, retaining native sparse/FPS origin."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

assert P['physical_gpu'] in (0, 1) and os.environ['CUDA_VISIBLE_DEVICES'] == str(P['physical_gpu'])
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger

assert torch.cuda.device_count() == 1
torch.set_num_threads(2)
torch.cuda.set_device(0)
root = Path(P['root'])
for row in PAYLOAD['parent_files']:
    assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256']
config = root / 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
assert hashlib.sha256(config.read_bytes()).hexdigest() == PAYLOAD['config_sha256']
cfg = gorilla.Config.fromfile(str(config))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.model.sampling_module.num_proposal == 128 and cfg.dataloader.train.batch_size == 2
native = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.ipdn')
gorilla.set_random_seed(1999)
model = native.IPDN(**cfg.model).cuda()
model.eval()
checkpoint_path = Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth')
assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == PAYLOAD['checkpoint_sha256']
checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
weights = dict(checkpoint['model'])
extras = set(weights) - set(model.state_dict())
assert extras.issubset({'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'})
buffers = dict(model.named_buffers())
for key in extras:
    assert key in buffers and not buffers[key].requires_grad
    assert torch.equal(weights[key].cpu(), buffers[key].cpu())
    del weights[key]
model.load_state_dict(weights, strict=True)
del weights, checkpoint
for parameter in model.parameters():
    parameter.requires_grad_(False)
capture_path = Path('/tmp/adares_y46_native_capture_d179_d179d824a64943e0b06bfa90de0caf52.json')
assert hashlib.sha256(capture_path.read_bytes()).hexdigest() == PAYLOAD['capture_sha256']
frozen = json.loads(capture_path.read_text())['rows']
assert len(frozen) == 8
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())

"""Append to fixed native W2 setup: observe real fine-support validity and controls."""
import gc

rows = []
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
for pair in range(0, 8, 2):
    items = []
    for scene in frozen[pair:pair+2]:
        gorilla.set_random_seed(1999)
        item = dataset[scene['dataset_index']]
        single = dataset.collate_fn([copy.deepcopy(item)])
        assert single['scan_ids'][0] == scene['scan_id']
        assert [str(v) for v in single['ann_ids']] == [v['ann_id'] for v in scene['expressions']]
        items.append(item)
    batch = dataset.collate_fn(copy.deepcopy(items))
    captured = {}

    def observe(module, args):
        assert not captured
        captured['args'] = tuple(v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v) for v in args)

    handle = model.dec.register_forward_pre_hook(observe)
    try:
        with torch.no_grad():
            native_loss, logs = model(copy.deepcopy(batch), mode='loss')
    finally:
        handle.remove()
    assert torch.isfinite(native_loss).all()
    args = captured['args']
    assert len(args) == 8 and args[1].shape == (8, 256)
    decoder = model.dec
    packed_positions, native_mask = decoder.get_batches(args[5], args[2])
    counts = (args[2][1:]-args[2][:-1]).detach().cpu().tolist()
    active = {}
    module_rows = []
    original_geometry = decoder.get_scg_weight
    originals = [layer.forward for layer in decoder.qia_layers_fine]
    handles = []

    def coarse_hook(index):
        def hook(module, inputs, output):
            active['layer'] = index
            active['coarse'] = output[1].detach()
        return hook

    def geometry(pos_q, pos_k=None, k_mask=None, k=None):
        if pos_k is not None:
            coarse = active['coarse'] * (~native_mask.unsqueeze(1)).float()
            scores = coarse.sum(-2)
            if scores.shape[1] < decoder.num_proposal_fine:
                indices = torch.arange(scores.shape[1], device=scores.device).repeat(scores.shape[0], 1)
            else:
                indices = torch.topk(scores, decoder.num_proposal_fine)[1]
            fine_invalid = native_mask.gather(1, indices)
            expected_positions = packed_positions.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))
            assert torch.equal(expected_positions, pos_k), 'Observed native fine positions disagree with exact coarse-derived gather'
            assert k_mask is None
            active.update(fine_invalid=fine_invalid, positions=(pos_q, pos_k), neighbors=k)
            if active['mode'] == 'corrected':
                k_mask = fine_invalid
        return original_geometry(pos_q, pos_k, k_mask, k)

    def fine_forward(index, original):
        def forward(source, query, batch_mask=None, attn_mask=None, pe=None, att_weights=None):
            assert active['layer'] == index and batch_mask is None
            invalid = active['fine_invalid']
            assert invalid.shape == source.shape[:2]
            used_mask = invalid if active['mode'] == 'corrected' else None
            output = original(source, query, used_mask, attn_mask, pe, att_weights)
            if active['mode'] == 'audit':
                repeated = original(source, query, None, attn_mask, pe, att_weights)
                changed = source.clone()
                changed[invalid] = -changed[invalid]
                padding = original(changed, query, None, attn_mask, pe, att_weights)
                valid_changed = source.clone()
                valid_changed[~invalid] = -valid_changed[~invalid]
                pos_q, pos_k = active['positions']
                corrected_weights, corrected_neighbors = original_geometry(pos_q, pos_k, invalid, active['neighbors'])
                corrected = original(source, query, invalid, ~corrected_neighbors, pe, corrected_weights)
                corrected_padding = original(changed, query, invalid, ~corrected_neighbors, pe, corrected_weights)
                corrected_valid = original(valid_changed, query, invalid, ~corrected_neighbors, pe, corrected_weights)
                eligible = invalid.unsqueeze(1) & ~attn_mask
                attention_mass = output[1].masked_fill(~invalid.unsqueeze(1), 0)
                per_expression = []
                for expression in range(source.shape[0]):
                    per_expression.append(dict(expression=expression, valid_superpoints=int(counts[expression]),
                        padded_fine_supports=int(invalid[expression].sum()),
                        eligible_padding_query_key_edges=int(eligible[expression].sum()),
                        native_padding_attention_mass=float(attention_mass[expression].sum()),
                        native_padding_attended_queries=int((attention_mass[expression].sum(-1)>0).sum()),
                        repeat_max_abs=float((output[0][expression]-repeated[0][expression]).abs().max()),
                        original_padding_intervention_max_abs=float((output[0][expression]-padding[0][expression]).abs().max()),
                        corrected_padding_intervention_max_abs=float((corrected[0][expression]-corrected_padding[0][expression]).abs().max()),
                        corrected_valid_positive_control_max_abs=float((corrected[0][expression]-corrected_valid[0][expression]).abs().max()),
                        corrected_vs_original_max_abs=float((corrected[0][expression]-output[0][expression]).abs().max())))
                module_rows.append(dict(layer=index, exact_native_gather_positions_verified=True,
                    parent_omits_fine_batch_mask=True, rows=per_expression))
            return output
        return forward

    def full_difference(left, right):
        layers = [(left, right)] + list(zip(left['aux_outputs'], right['aux_outputs']))
        return [dict(expression=i, valid_superpoints=int(counts[i]),
            max_indicator_logit_abs=max(float((a['indis'][i]-b['indis'][i]).abs().max()) for a,b in layers),
            max_mask_logit_abs=max(float((a['masks'][i, :, :counts[i]]-b['masks'][i, :, :counts[i]]).abs().max()) for a,b in layers),
            selected_query_changed=bool(left['indis'][i].softmax(-1)[:,1].argmax()!=right['indis'][i].softmax(-1)[:,1].argmax())) for i in range(8)]

    try:
        for index, layer in enumerate(decoder.qia_layers):
            handles.append(layer.register_forward_hook(coarse_hook(index)))
        decoder.get_scg_weight = geometry
        for index, layer in enumerate(decoder.qia_layers_fine):
            layer.forward = fine_forward(index, originals[index])
        with torch.no_grad():
            active['mode'] = 'audit'
            gorilla.set_random_seed(1999)
            original = decoder(*args)
            active['mode'] = 'repeat'
            gorilla.set_random_seed(1999)
            repeated = decoder(*args)
            active['mode'] = 'corrected'
            gorilla.set_random_seed(1999)
            corrected = decoder(*args)
            full_repeat = full_difference(original, repeated)
            full_corrected = full_difference(original, corrected)
        assert len(module_rows) == 6
    finally:
        decoder.get_scg_weight = original_geometry
        for layer, original_forward in zip(decoder.qia_layers_fine, originals):
            layer.forward = original_forward
        for handle in handles:
            handle.remove()
    rows.append(dict(pair=pair//2, scenes=[scene['scan_id'] for scene in frozen[pair:pair+2]],
        native_TRAIN_batchsize2=True, actual_sparse_FPS_inputs=True, native_loss=float(native_loss),
        expressions=8, superpoint_counts=counts, module_controls=module_rows,
        full_repeat=full_repeat, full_corrected=full_corrected))
    print(json.dumps(dict(pair=pair//2, counts=counts,
        padded_fine_support_slots=sum(v['padded_fine_supports'] for layer in module_rows for v in layer['rows']),
        positive_padding_attention_mass=sum(v['native_padding_attention_mass'] for layer in module_rows for v in layer['rows']),
        full_indicator_difference=max(v['max_indicator_logit_abs'] for v in full_corrected))), flush=True)
    del batch, captured, args, original, repeated, corrected, native_loss, logs, module_rows
    gc.collect()
all_modules = [value for pair in rows for layer in pair['module_controls'] for value in layer['rows']]
all_full = [value for pair in rows for value in pair['full_corrected']]
result = dict(complete=True, id='AdaRES-Y54', fixed_TRAIN_scenes=8, expressions=32,
    native_two_scene_batches=4, actual_sparse_forward_and_FPS=True, exact_native_support_positions_verified=True,
    actual_parent_weights_and_model_unchanged=True, intervention='Diagnostic-only fine-support key padding and nearest-neighbor validity; source/query/weights otherwise fixed; all hooks restored.',
    padded_fine_support_slots=sum(v['padded_fine_supports'] for v in all_modules),
    eligible_padding_query_key_edges=sum(v['eligible_padding_query_key_edges'] for v in all_modules),
    native_padding_attention_mass=sum(v['native_padding_attention_mass'] for v in all_modules),
    native_block_padding_effect_max=max(v['original_padding_intervention_max_abs'] for v in all_modules),
    corrected_block_padding_effect_max=max(v['corrected_padding_intervention_max_abs'] for v in all_modules),
    original_block_repeat_max=max(v['repeat_max_abs'] for v in all_modules),
    full_decoder_indicator_effect_max=max(v['max_indicator_logit_abs'] for v in all_full),
    all_valid_negative_control_indicator_max=max(v['max_indicator_logit_abs'] for v in all_full if v['valid_superpoints']>=512),
    peak_reserved_bytes=int(torch.cuda.max_memory_reserved()), elapsed_seconds=time.perf_counter()-started,
    rows=rows, model_or_checkpoint_weights_saved=False, optimizer_steps=0, admission=False,
    limitation='Actual fixed TRAIN incidence and matched diagnostic forward controls, not implemented independent Y54, full native backward/resource/restore gates, trained accuracy or a Pareto claim. Correlated layer/support slots are not independent samples.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
