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
model = native.IPDN(**cfg.model).cuda().eval()
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
records = []


def difference(left, right):
    layers = [(left, right)] + list(zip(left['aux_outputs'], right['aux_outputs']))
    return dict(layers=len(layers),
        max_indicator_logit_abs=max(float((a['indis']-b['indis']).abs().max()) for a,b in layers),
        max_mask_logit_abs=max(float((a['masks']-b['masks']).abs().max()) for a,b in layers),
        max_score_abs=max(float((a['scores']-b['scores']).abs().max()) for a,b in layers),
        max_sampling_score_abs=float((left['ref_scores']-right['ref_scores']).abs().max()),
        sampled_query_indices_changed=int((left['sample_inds']!=right['sample_inds']).sum()),
        final_selected_queries_changed=int((left['indis'].softmax(-1)[...,1].argmax(-1)!=right['indis'].softmax(-1)[...,1].argmax(-1)).sum()))


started = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
for index, scene in enumerate(frozen):
    gorilla.set_random_seed(1999)
    item = dataset[scene['dataset_index']]
    batch = dataset.collate_fn([copy.deepcopy(item)])
    assert batch['scan_ids'][0] == scene['scan_id']
    assert [str(value) for value in batch['ann_ids']] == [row['ann_id'] for row in scene['expressions']]
    captured = {}

    def observe(module, args):
        assert not captured
        captured['args'] = tuple(value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value) for value in args)

    handle = model.dec.register_forward_pre_hook(observe)
    try:
        with torch.no_grad():
            original_prediction = model(copy.deepcopy(batch), mode='predict')
    finally:
        handle.remove()
    args = captured['args']
    assert len(args) == 8 and args[3].shape[-1] == 768
    assert args[1].shape[-1] == 256 and args[3].shape[0] == len(batch['ann_ids'])
    language, valid = args[3], args[4].bool()
    assert (~valid).any() and valid.any()
    pad_changed = language.clone()
    pad_changed[~valid] = -pad_changed[~valid]
    word_changed = language.clone()
    word_changed[valid] = -word_changed[valid]

    def run(value):
        gorilla.set_random_seed(1999)
        current = list(args)
        current[3] = value
        return model.dec(*current)

    with torch.no_grad():
        base = run(language)
        repeat = run(language.clone())
        padding = run(pad_changed)
        positive = run(word_changed)
        readout = dict(repeat=difference(base, repeat), padding_only=difference(base, padding),
                       valid_word_positive_control=difference(base, positive))
    live = language.clone().requires_grad_(True)
    output = run(live)
    objective = sum(value['indis'].square().sum() for value in [output] + output['aux_outputs'])
    gradient = torch.autograd.grad(objective, live)[0]
    assert torch.isfinite(gradient).all()
    offsets = args[2].detach().cpu().tolist()
    records.append(dict(scene=scene['scan_id'], dataset_index=scene['dataset_index'],
        ann_ids=[str(value) for value in batch['ann_ids']], expressions=len(batch['ann_ids']),
        native_sparse_forward=True, actual_fps_and_decoder_inputs=True, actual_fps_shape=list(args[1].shape),
        superpoints_per_expression=[b-a for a,b in zip(offsets,offsets[1:])],
        padded_language_positions=int((~valid).sum()), comparisons=readout,
        language_gradient=dict(padding_max_abs=float(gradient[~valid].abs().max()),
                              valid_max_abs=float(gradient[valid].abs().max()), finite=True)))
    print(json.dumps(dict(native_partial=records[-1])), flush=True)
    del original_prediction, batch, captured, args, language, pad_changed, word_changed, base, repeat, padding, positive, live, output, objective, gradient
assert sum(row['expressions'] for row in records) == 32
result = dict(complete=True, id='AdaRES-Y53', source_bindings=PAYLOAD['parent_files'],
    checkpoint_sha256=PAYLOAD['checkpoint_sha256'], capture_sha256=PAYLOAD['capture_sha256'],
    fixed_TRAIN_scenes=8, expressions=32, actual_native_sparse_and_FPS_capture=True,
    conditions='SAME native decoder inputs, unchanged repeat, padding-only and valid-word sign changes; all actual parameters frozen in fixed trained W2 diagnostic state.',
    peak_reserved_bytes=int(torch.cuda.max_memory_reserved()), elapsed_seconds=time.perf_counter()-started,
    records=records, model_or_config_changes=False, model_weights_saved=False, optimizer_steps=0,
    training_launches=0, admission=False,
    limitation='Native sparse/FPS inputs originate from fixed TRAIN scenes; paired interventions isolate decoder language features and skip repeated sparse numerical noise. This is a diagnostic, not trained Y53 efficacy, a changed loss, or an admission receipt.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
