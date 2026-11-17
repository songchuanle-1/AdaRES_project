"""Two native TRAIN steps tracing peer-state feedback into refinement."""


def check_path_calls(norm_inputs, indicator_calls, feedback_calls, consumed_calls):
    assert norm_inputs == [True] * 7
    assert indicator_calls == 7 and feedback_calls == consumed_calls == 6


def check_feedback_alignment(query, masks, valid, masses, indicator_args):
    assert all(a is b for a, b in zip((query, masks, valid, masses), indicator_args))
    assert len(indicator_args) == 4


def check_response_rows(rows):
    from math import isfinite
    assert len(rows) == 12
    for row in rows:
        assert type(row['projection_nonzero']) is bool
        assert type(row['consumed_by_refinement']) is bool and row['consumed_by_refinement']
        assert row['reference_query_rows'] == [0, 64, 127]
        assert all(isfinite(row[k]) and row[k] >= 0
                   for k in ('context_max', 'query_delta_max', 'reference_error_max'))
    assert all(not r['projection_nonzero'] and r['query_delta_max'] == 0 for r in rows[:6])
    assert all(r['projection_nonzero'] for r in rows[6:])
    assert any(r['context_max'] > 0 and r['query_delta_max'] > 0 for r in rows[6:])


def reference_peer_rows(query, masks, padded, mass):
    """Three complete peer rows; compute union directly per point, in double."""
    import torch
    probability = torch.sigmoid(masks.detach().double())
    probability = probability.masked_fill(padded.unsqueeze(1), 0.)
    weights = mass.detach().double().unsqueeze(1)
    peers = query.detach().double()
    result = []
    for index in (0, 64, 127):
        own = probability[:, index:index + 1]
        product = own * probability
        intersection = (product * weights).sum(-1)
        union = ((own + probability - product) * weights).sum(-1)
        overlap = intersection / union.clamp_min(1e-6)
        overlap[:, index] = 0.
        normalized = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
        result.append((normalized.unsqueeze(-1) * peers).sum(1))
    return torch.stack(result, dim=1)


import os
assert P['physical_gpu'] == 0
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
import subprocess
assert subprocess.check_output(['hostname']).decode().strip() == 'dl01-MS-7C60'
assert float(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=power.limit',
    '--format=csv,noheader,nounits']).decode().strip()) == 280.0
import base64
import hashlib
import importlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

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
assert row['id'] == 'AdaRES-Y66'
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
assert sum(p.numel() for p in model.parameters()) == 164695173
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)
dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)
losses, gradients, batches, paths = [], [], [], []
norm_inputs, indicator_inputs, indicator_outputs = [], [], []
feedback_inputs, feedback_outputs, feedback_paths, consumed = [], [], [], []
head = model.dec.indi_embedding
feedback = model.dec.peer_feedback
assert len(list(head.parameters())) == 10
assert len(list(feedback.parameters())) == 1
assert torch.equal(feedback.weight, torch.zeros_like(feedback.weight))


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


def observe_indicator(module, args, output):
    query, masks, valid_mask, masses = args
    assert query.requires_grad and masks.requires_grad
    assert output.requires_grad and output.shape[-2:] == (128, 2)
    assert torch.isfinite(output).all()
    query.retain_grad()
    masks.retain_grad()
    output.retain_grad()
    indicator_inputs.append(args)
    indicator_outputs.append(output)


def observe_feedback(module, args, output):
    query, masks, valid_mask, masses = args
    check_feedback_alignment(query, masks, valid_mask, masses, indicator_inputs[-1])
    assert query.requires_grad and masks.requires_grad and output.requires_grad
    assert torch.isfinite(output).all()
    query.retain_grad()
    masks.retain_grad()
    output.retain_grad()
    feedback_inputs.append((query, masks))
    feedback_outputs.append(output)
    with torch.no_grad():
        context = reference_peer_rows(query, masks, valid_mask, masses)
        reference = query.detach()[:, [0, 64, 127]].double() + context @ module.weight.detach().double()
        actual = output.detach()[:, [0, 64, 127]].double()
        torch.testing.assert_close(actual, reference, rtol=3e-5, atol=2e-6)
        delta = float((output.detach() - query.detach()).abs().max())
        projection_nonzero = bool(module.weight.detach().abs().max() > 0)
        if not projection_nonzero:
            assert torch.equal(output.detach(), query.detach())
        feedback_paths.append(dict(projection_nonzero=projection_nonzero,
            context_max=float(context.abs().max()), query_delta_max=delta,
            reference_error_max=float((actual - reference).abs().max()),
            reference_query_rows=[0, 64, 127], consumed_by_refinement=False))


def observe_refinement_input(module, args):
    assert args[1] is feedback_outputs[-1], 'Feedback result bypassed by refinement'
    assert not feedback_paths[-1]['consumed_by_refinement']
    feedback_paths[-1]['consumed_by_refinement'] = True
    consumed.append(True)


step = optimizer.step


def checked_step(*args, **kwargs):
    index = len(gradients)
    assert index < 2
    observed_norm = norm_inputs[index * 7:]
    observed_inputs = indicator_inputs[index * 7:]
    observed_feedback = feedback_inputs[index * 6:]
    check_path_calls(observed_norm, len(observed_inputs), len(observed_feedback), len(consumed[index * 6:]))
    active = [p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    indicator = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(indicator) == len(list(head.parameters())) == 10
    assert all(torch.isfinite(g).all() for g in indicator)
    assert head.head[-1].weight.grad.abs().max() > 0
    assert head.query_norm.weight.grad.abs().max() > 0
    projection_grad = feedback.weight.grad
    assert projection_grad is not None and torch.isfinite(projection_grad).all()
    assert projection_grad.abs().max() > 0
    assert all(t.grad is not None and torch.isfinite(t.grad).all()
               for item in observed_inputs for t in item[:2])
    assert all(t.grad is not None and torch.isfinite(t.grad).all()
               for pair in observed_feedback for t in pair)
    assert all(t.grad is not None and torch.isfinite(t.grad).all()
               for t in feedback_outputs[index * 6:])
    gradients.append(dict(tensors=len(active), indicator_tensors=len(indicator), all_finite=True,
        query_norm_input_requires_grad=observed_norm, indicator_calls=len(observed_inputs),
        feedback_calls=len(observed_feedback), feedback_consumed_by_refinement=6,
        query_mask_and_feedback_input_gradients_finite=True,
        feedback_projection_gradient_max=float(projection_grad.abs().max().detach())))
    return step(*args, **kwargs)


optimizer.step = checked_step
hooks = [model.register_forward_pre_hook(before_forward), model.register_forward_hook(after_forward),
         head.query_norm.register_forward_hook(observe_norm),
         head.register_forward_hook(observe_indicator),
         feedback.register_forward_hook(observe_feedback)]
hooks.extend(layer.register_forward_pre_hook(observe_refinement_input) for layer in model.dec.qia_layers)
torch.cuda.reset_peak_memory_stats()
try:
    native.train(1, model, FirstTwoBatches(), optimizer, scheduler, cfg, logger, Writer())
    torch.cuda.synchronize()
    assert len(losses) == len(gradients) == len(batches) == len(paths) == 2
    assert len(indicator_inputs) == len(indicator_outputs) == 14 and norm_inputs == [True] * 14
    assert len(feedback_inputs) == len(feedback_outputs) == len(consumed) == 12
    check_response_rows(feedback_paths)
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
    feedback_calls=len(feedback_inputs), query_norm_input_requires_grad=norm_inputs,
    feedback_paths=feedback_paths, peak_reserved_bytes=peak,
    native_entry='tools/train.py::train', native_YAML_sha256=row['config_data']['sha256'],
    optimizer_steps=2, model_weights_saved=False, validation_used=False, admission=False,
    limitation='Two-step initialization smoke only; not trained efficacy, full-dataset resource '
               'proof or trained benefit from peer-state feedback.')
Path(P['output']).write_text(json.dumps(output), encoding='utf-8')
