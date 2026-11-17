PAYLOAD={'parent_files': [{'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/__init__.py', 'sha256': 'c540732d6e3334c85008035cc37b8537b0f85383d5f5a6bf663359fbf2b134bb'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/backbone.py', 'sha256': '57db8747d3c38136f568837d70e75ceea23e101900679db4931724ecb57665ca'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py', 'sha256': '6362d882662b0e51594cd4af994359b93f99497affc93600322a34ca6c349b01'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/ipdn.py', 'sha256': '7cb4b1c5490d0839fcdaf18553983b4c991f69e50474d07e83af9f0f8c937a5a'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/loss.py', 'sha256': '836ff790a0a8b9c55cf465a472c9462699f0753a7e87d0ef74a8946edf31565a'}, {'path': 'ipdn/model_paper5/adares_w2_pointmass_competition/sample_model.py', 'sha256': '674e7ef698c6e81ed1c35f14680e1ec748f5c783ed879a6ca72f099b6450bc36'}], 'config_sha256': 'd4c35ff10712bfb9ece3e25daa9c3b63143ce313f43eaf19735abaa31f6693eb'}
"""Test the actual unchanged decoder on fixed component inputs, CPU only."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

assert P['physical_gpu'] is None and os.environ['CUDA_VISIBLE_DEVICES'] == ''
import gorilla
import torch

assert torch.cuda.device_count() == 0
torch.set_num_threads(2)
root = Path(P['root'])
for row in PAYLOAD['parent_files']:
    assert hashlib.sha256((root / row['path']).read_bytes()).hexdigest() == row['sha256']
config = root / 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
assert hashlib.sha256(config.read_bytes()).hexdigest() == PAYLOAD['config_sha256']
cfg = gorilla.Config.fromfile(str(config))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.model.sampling_module.num_proposal == 128 and cfg.model.dec.num_layer == 6
native = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
gorilla.set_random_seed(1999)
decoder = native.DEC(**cfg.model.dec, in_channel=cfg.model.media,
                     sampling_module=cfg.model.sampling_module).cpu().eval()
for parameter in decoder.parameters():
    parameter.requires_grad_(False)
batch_offsets = torch.tensor([0, 512, 1025], dtype=torch.int32)
x = torch.randn(1025, cfg.model.media)
positions = torch.randn(1025, 3)
clip = torch.randn(1025, 1024)
mass = torch.randint(1, 100, (1025,)).float()
fps = torch.arange(256).repeat(2, 1)
mask = torch.zeros(2, 78, dtype=torch.long)
mask[0, :7] = 1
mask[1, :13] = 1
language = torch.randn(2, 78, 768)
padding_changed = language.clone()
padding_changed[mask == 0] = torch.randn_like(padding_changed[mask == 0])
valid_changed = language.clone()
valid_changed[mask != 0] = -valid_changed[mask != 0]


def run(values):
    gorilla.set_random_seed(1999)
    return decoder(x, fps, batch_offsets, values, mask, positions, clip, mass)


def difference(left, right):
    layers = [(left, right)] + list(zip(left['aux_outputs'], right['aux_outputs']))
    return dict(
        layers=len(layers),
        max_indicator_logit_abs=max(float((a['indis'] - b['indis']).abs().max()) for a, b in layers),
        max_mask_logit_abs=max(float((a['masks'] - b['masks']).abs().max()) for a, b in layers),
        max_score_abs=max(float((a['scores'] - b['scores']).abs().max()) for a, b in layers),
        max_sampling_score_abs=float((left['ref_scores'] - right['ref_scores']).abs().max()),
        sampled_query_indices_changed=int((left['sample_inds'] != right['sample_inds']).sum()),
        final_selected_queries_changed=int((left['indis'].softmax(-1)[..., 1].argmax(-1) != right['indis'].softmax(-1)[..., 1].argmax(-1)).sum()),
    )


started = time.perf_counter()
with torch.no_grad():
    base = run(language)
    repeated = run(language.clone())
    padding = run(padding_changed)
    valid = run(valid_changed)
    comparisons = dict(repeat=difference(base, repeated), padding_only=difference(base, padding),
                       valid_token_positive_control=difference(base, valid))
live_language = language.clone().requires_grad_(True)
out = run(live_language)
objective = sum(value['indis'].square().sum() for value in [out] + out['aux_outputs'])
gradient = torch.autograd.grad(objective, live_language)[0]
assert torch.isfinite(gradient).all()
gradient_readout = dict(padding_max_abs=float(gradient[mask == 0].abs().max()),
                        valid_max_abs=float(gradient[mask != 0].abs().max()),
                        valid_l2=float(gradient[mask != 0].norm()), finite=True)
dependency = importlib.import_module('ipdn.torch.nn_new.functional')
dependency_path = Path(dependency.__file__)
result = dict(complete=True, id='AdaRES-Y53', native_class=type(decoder).__module__ + '.DEC',
    fixed_source_files=PAYLOAD['parent_files'], actual_config_sha256=PAYLOAD['config_sha256'],
    actual_custom_attention_dependency=dict(path=str(dependency_path), sha256=hashlib.sha256(dependency_path.read_bytes()).hexdigest()),
    actual_original_decoder=True, model_or_config_changes=False, cpu_only=True,
    input_scope='Component fixtures:2expressions,78language positions with7/13valid,512/513superpoints,256fixed seed indices,128sampled queries,6native layers. Not native sparse TRAIN/FPS inputs.',
    comparisons=comparisons, indicator_gradient_wrt_language=gradient_readout,
    elapsed_seconds=time.perf_counter() - started, training_launches=0, admission=False,
    limitation='Component graph invariance and positive-control sensitivity only. Not native sparse TRAIN evidence, model resource/backward admission, trained efficacy or a change to historical Y23 loss.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
