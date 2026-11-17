"""Two native TRAIN steps tracing the shared Y63 overlap-bilinear evidence path."""


def check_path_calls(norm_inputs, indicator_calls):
    assert norm_inputs == [True] * 7
    assert indicator_calls == 7


import os
assert P['physical_gpu'] == 0
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
import base64
import hashlib
import importlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import subprocess

assert float(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=power.limit',
    '--format=csv,noheader,nounits']).decode().strip()) == 280

import gorilla
import torch
from torch import nn

assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
torch.set_num_threads(2)
stage = Path(P['output']).with_suffix('')
assert not stage.exists()
stage.mkdir()
native = importlib.import_module('train')
assert hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == FROZEN['train_source_lf_sha256']
row = FROZEN['variant']
assert row['id'] == 'AdaRES-Y63'
package = stage / row['package']
package.mkdir()
for name, value in row['python_files'].items():
    raw = base64.b64decode(value['base64'])
    assert hashlib.sha256(raw).hexdigest() == value['sha256']
    (package / name).write_bytes(raw)
config_path = stage / Path(row['config']).name
raw = base64.b64decode(row['config_data']['base64'])
assert hashlib.sha256(raw).hexdigest() == row['config_data']['sha256']
config_path.write_bytes(raw)
cfg = gorilla.Config.fromfile(str(config_path))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.dataloader.train.batch_size == 2
assert cfg.model.sampling_module.num_proposal == 128
assert cfg.model.dec.num_layer == 6 and cfg.model.dec.iter_pred
assert cfg.data.train.training and cfg.data.train.prefix == 'train'
cfg.dist = False
namespace = 'ipdn.model_paper5.' + row['package']
importlib.import_module('ipdn.model_paper5')
assert namespace not in sys.modules
spec = importlib.util.spec_from_file_location(namespace, str(package / '__init__.py'),
    submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[namespace] = module
spec.loader.exec_module(module)
Candidate = native.resolve_model_class(str(config_path))
assert Candidate is module.IPDN
gorilla.set_random_seed(cfg.train.seed)
model = Candidate(**cfg.model).cuda()
assert sum(p.numel() for p in model.parameters()) == 164760709
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)
dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)
losses, gradients, batches, paths = [], [], [], []
norm_inputs, indicator_inputs, normalized_queries = [], [], []
head = model.dec.indi_embedding
assert torch.equal(head.peer_bilinear_weight, torch.zeros_like(head.peer_bilinear_weight))
base_logits = []
bilinear_paths = []


def observe_base(module, args, output):
    base_logits.append(output.detach())


class FirstTwoBatches:
    sampler = loader.sampler

    def __len__(self):
        return 2

    def __iter__(self):
        for batch in itertools.islice(loader, 2):
            assert len(batch['scan_ids']) == cfg.dataloader.train.batch_size
            batches.append(dict(scan_ids=[str(s) for s in batch['scan_ids']],
                expressions=len(batch['ann_ids']), points=int(batch['feats'].shape[0])))
            yield batch


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def before_forward(module, args):
    backbone = [m for block in (module.input_conv, module.unet, module.output_layer)
                for m in block.modules() if isinstance(m, nn.BatchNorm1d)]
    decoder = [m for m in module.dec.modules() if isinstance(m, nn.BatchNorm1d)]
    assert module.training and backbone and decoder
    assert all(m.training for m in backbone + decoder)
    assert module.text_encoder.training
    assert all(not p.requires_grad for p in module.text_encoder.parameters())
    paths.append(True)


def after_forward(module, args, value):
    loss = value[0]
    assert torch.isfinite(loss).all() and loss.requires_grad
    losses.append(float(loss.detach().cpu()))


def observe_norm(module, args, output):
    assert output.requires_grad and torch.isfinite(output).all()
    norm_inputs.append(bool(args[0].requires_grad))
    normalized_queries.append(output.detach())


def observe_indicator(module, args, output):
    query, masks, valid_mask, masses = args
    assert query.requires_grad and masks.requires_grad
    assert output.requires_grad and output.shape[-2:] == (128, 2)
    assert torch.isfinite(output).all()
    query.retain_grad()
    masks.retain_grad()
    indicator_inputs.append((query, masks))
    valid = (~valid_mask).unsqueeze(1).to(masks.dtype)
    probabilities = masks.sigmoid() * valid
    weighted = probabilities * masses.unsqueeze(1)
    area = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (area.unsqueeze(-1) + area.unsqueeze(1) - intersection).clamp_min(1e-6)
    overlap = overlap.masked_fill(torch.eye(query.shape[1], dtype=torch.bool, device=query.device).unsqueeze(0), 0)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    normalized = normalized_queries[-1]
    context = torch.einsum('bij,bjd->bid', competition, normalized)
    residual = torch.einsum('bqd,odc,bqc->bqo',
        normalized, module.peer_bilinear_weight.detach(), context) / (query.shape[-1] ** .5)
    expected = base_logits[-1] + residual.detach()
    assert torch.equal(output.detach(), expected)
    bilinear_paths.append(dict(residual_max=float(residual.detach().abs().max()),
        bilinear_weight_nonzero=bool(module.peer_bilinear_weight.detach().abs().max() > 0),
        logit_delta_max=float((output.detach() - base_logits[-1]).abs().max())))



step = optimizer.step


def checked_step(*args, **kwargs):
    index = len(gradients)
    assert index < 2
    observed_norm = norm_inputs[index * 7:]
    observed_inputs = indicator_inputs[index * 7:]
    check_path_calls(observed_norm, len(observed_inputs))
    active = [p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    indicator = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(indicator) == len(list(head.parameters())) == 11
    assert all(torch.isfinite(g).all() for g in indicator)
    assert head.head[-1].weight.grad.abs().max() > 0
    assert head.query_norm.weight.grad.abs().max() > 0
    assert head.peer_bilinear_weight.grad is not None and torch.isfinite(head.peer_bilinear_weight.grad).all()
    assert head.peer_bilinear_weight.grad.abs().max() > 0
    assert all(t.grad is not None and torch.isfinite(t.grad).all()
               for pair in observed_inputs for t in pair)
    gradients.append(dict(tensors=len(active), indicator_tensors=len(indicator), all_finite=True,
        query_norm_input_requires_grad=observed_norm, indicator_calls=len(observed_inputs),
        query_and_mask_input_gradients_finite=True, trainable_query_norm_gradient_nonzero=True,
        bilinear_weight_gradient_max=float(head.peer_bilinear_weight.grad.abs().max().detach())))
    return step(*args, **kwargs)


optimizer.step = checked_step
hooks = [model.register_forward_pre_hook(before_forward), model.register_forward_hook(after_forward),
         head.query_norm.register_forward_hook(observe_norm), head.head.register_forward_hook(observe_base),
         head.register_forward_hook(observe_indicator)]
torch.cuda.reset_peak_memory_stats()
try:
    native.train(1, model, FirstTwoBatches(), optimizer, scheduler, cfg, logger, Writer())
    torch.cuda.synchronize()
    assert len(losses) == len(gradients) == len(batches) == len(paths) == 2
    assert len(indicator_inputs) == 14 and norm_inputs == [True] * 14
    assert len(bilinear_paths) == 14 and len(base_logits) == 14
    assert all(not p['bilinear_weight_nonzero'] and p['logit_delta_max'] == 0 for p in bilinear_paths[:7])
    assert all(p['bilinear_weight_nonzero'] for p in bilinear_paths[7:])
    assert any(p['logit_delta_max'] > 0 for p in bilinear_paths[7:])
    peak = int(torch.cuda.max_memory_reserved())
    assert peak < 24 * 2**30
finally:
    for hook in hooks:
        hook.remove()
    if getattr(loader, '_iterator', None) is not None:
        loader._iterator._shutdown_workers()
output = dict(complete=True, id=row['id'], config=row['config'], code_commit=FROZEN['code_commit'],
    goal_sha256=FROZEN['goal_sha256'], steps=len(gradients), losses=losses,
    native_data_loaded=True, backward_finite=True, intended_path_observed=all(paths),
    split='train', data_source=dict(config_data_root=cfg.data.train.data_root,
        dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__),
    batches=batches, gradient_checks=gradients, shared_head_calls=len(indicator_inputs),
    query_norm_input_requires_grad=norm_inputs, bilinear_paths=bilinear_paths, peak_reserved_bytes=peak,
    native_entry='tools/train.py::train', native_YAML_sha256=row['config_data']['sha256'],
    optimizer_steps=2, model_weights_saved=False, validation_used=False, admission=False,
    limitation='Two-step initialization smoke only; not trained efficacy, full-dataset resource '
               'proof or trained benefit from added overlap-bilinear evidence.')
Path(P['output']).write_text(json.dumps(output), encoding='utf-8')
