"""Two original native TRAIN steps with current query-set to visual feedback observation."""
import os
assert P['physical_gpu'] == FROZEN['runtime_physical_gpu']
assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(P['physical_gpu'])
import subprocess
assert subprocess.check_output(['hostname']).decode().strip() == 'dl01-MS-7C60'
assert float(subprocess.check_output(['nvidia-smi', '-i', str(P['physical_gpu']), '--query-gpu=power.limit',
    '--format=csv,noheader,nounits']).decode().strip()) == FROZEN['runtime_power_limit_watts']
import base64
import hashlib
import importlib
import importlib.util
import itertools
import json
from functools import partial
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
assert row['id'] == 'AdaRES-Y91'
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






losses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}
head=model.dec.indi_embedding;gate=head.hidden_gate;original_prediction=model.dec.prediction_head
"""Verify original evidence and all consumers of the internal confidence gate."""
def check_response_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        assert row['weight_nonzero'] is bool(index // 7)
        for key in ('original_query_identity', 'original_mask_identity', 'original_mass_identity',
                    'original_evidence_exact', 'first_and_second_hidden_identity',
                    'both_inputs_undetached', 'float64_gate_reference', 'final_linear_consumes_gated_hidden'):
            assert row[key] is True, key
        for key in ('gate_delta_max', 'hidden_delta_max'):
            assert math.isfinite(row[key])
            assert row[key] > 0 if index // 7 else row[key] == 0


def reset_state():
    state.clear()
    state.update(queries=[], masks=[], gates=[], indicators=[], normalized=[], evidence_inputs=[],
        evidence_outputs=[], first_linear=[], first_hidden=[], second_linear=[], second_hidden=[],
        gated=[], final_consumers=[], coarse=0, features=None, padding=None, mass=None)


def checked_prediction(query, features, pad):
    if not state['queries']:
        state['features'], state['padding'] = features, pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(query)
    value = original_prediction(query, features, pad)
    state['masks'].append(value[1])
    expected = (value[1].sigmoid() < .5) | pad.unsqueeze(1)
    expected[expected.all(-1)] = False
    expected = expected | pad.unsqueeze(1)
    assert torch.equal(value[2], expected) and not value[2].requires_grad
    state['gates'].append(value[2])
    return value


def before_indicator(module, args):
    index = len(state['indicators'])
    assert module is head and len(args) == 4
    assert args[0] is state['queries'][index] and args[1] is state['masks'][index]
    assert args[2] is state['padding'] and args[0].requires_grad and args[1].requires_grad
    if index == 0:
        state['mass'] = args[3]
    assert args[3] is state['mass'] and not args[3].requires_grad
    state['indicators'].append(args)


def capture_query_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['queries'][-1]
    assert args[0].requires_grad and value.requires_grad
    state['normalized'].append(value)


def capture_evidence(module, args, value):
    index = len(state['evidence_inputs'])
    query, masks, pad, mass = state['indicators'][index]
    normalized = state['normalized'][index]
    probabilities = masks.sigmoid() * (~pad).unsqueeze(1).to(masks.dtype)
    weighted = probabilities * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection).clamp_min(1e-6)
    diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0.)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    context = torch.einsum('bij,bjd->bid', competition, normalized)
    reference = torch.cat((normalized, context, normalized - context, overlap.max(-1).values.unsqueeze(-1)), -1)
    assert module is head.evidence_norm and torch.equal(args[0], reference)
    assert args[0].requires_grad and value.requires_grad
    args[0].retain_grad()
    value.retain_grad()
    state['evidence_inputs'].append(args[0])
    state['evidence_outputs'].append(value)


def capture_first_linear(module, args, value):
    index = len(state['first_linear'])
    assert module is head.head[0] and args[0] is state['evidence_outputs'][index]
    state['first_linear'].append(value)


def capture_first_hidden(module, args, value):
    index = len(state['first_hidden'])
    assert module is head.head[1] and args[0] is state['first_linear'][index] and value.requires_grad
    value.retain_grad()
    state['first_hidden'].append(value)


def capture_second_linear(module, args, value):
    index = len(state['second_linear'])
    assert module is head.head[2] and args[0] is state['first_hidden'][index]
    state['second_linear'].append(value)


def capture_second_hidden(module, args, value):
    index = len(state['second_hidden'])
    assert module is head.head[3] and args[0] is state['second_linear'][index] and value.requires_grad
    value.retain_grad()
    state['second_hidden'].append(value)


def capture_gate(module, args, value):
    index = len(state['gated'])
    assert module is gate and len(args) == 2
    first, second = args
    assert first is state['first_hidden'][index] and second is state['second_hidden'][index]
    assert first.requires_grad and second.requires_grad and value.requires_grad
    scale = first.shape[-1] ** .5
    reference_gate = 1 + torch.tanh((first.double() @ module.weight.double()) / scale)
    reference = second.double() * reference_gate
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    measured_gate = 1 + torch.tanh((first @ module.weight) / scale)
    assert torch.equal(value, second * measured_gate)
    assert torch.isfinite(measured_gate).all() and measured_gate.min() >= 0 and measured_gate.max() <= 2
    nonzero = bool(torch.count_nonzero(module.weight))
    assert nonzero == bool(len(step_rows))
    gate_delta = float((measured_gate - 1).abs().max())
    hidden_delta = float((value - second).abs().max())
    assert gate_delta > 0 and hidden_delta > 0 if nonzero else gate_delta == hidden_delta == 0
    value.retain_grad()
    state['gated'].append(value)
    path_rows.append(dict(step=len(step_rows), stage=index, weight_nonzero=nonzero,
        gate_delta_max=gate_delta, hidden_delta_max=hidden_delta,
        original_query_identity=True, original_mask_identity=True, original_mass_identity=True,
        original_evidence_exact=True, first_and_second_hidden_identity=True,
        both_inputs_undetached=True, float64_gate_reference=True,
        final_linear_consumes_gated_hidden=False))


def before_final_linear(module, args):
    index = len(state['final_consumers'])
    assert module is head.head[4] and args[0] is state['gated'][index] and args[0].requires_grad
    state['final_consumers'].append(index)
    assert path_rows[-1]['stage'] == index and path_rows[-1]['step'] == len(step_rows)
    path_rows[-1]['final_linear_consumes_gated_hidden'] = True


def before_coarse(module, args):
    assert module is model.dec.qia_layers[state['coarse']] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def gradient_record():
    assert all(len(state[key]) == 7 for key in
        ('queries', 'masks', 'gates', 'indicators', 'normalized', 'evidence_inputs', 'evidence_outputs',
         'first_linear', 'first_hidden', 'second_linear', 'second_hidden', 'gated', 'final_consumers'))
    assert state['final_consumers'] == list(range(7)) and state['coarse'] == 6
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    original = list(head.head.parameters()) + list(head.query_norm.parameters()) + list(head.evidence_norm.parameters())
    assert len(original) == 10
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in original)
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for key in ('evidence_inputs', 'evidence_outputs', 'first_hidden', 'second_hidden', 'gated')
               for value in state[key])
    gradient = gate.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_calls=7,
        shared_query_norm_calls=7, original_evidence_norm_calls=7, original_MLP_layer_calls=35,
        first_hidden_consumers=7, second_hidden_consumers=7, hidden_gate_calls=7,
        final_linear_consumers=7, original_overlap_consumers=7, coarse_gate_consumers=6,
        original_head_parameter_tensors=10, input_gradients_finite=True,
        new_gradient_max=float(gradient.abs().max()))


def install_hooks():
    hooks = [
        head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_query_norm),
        head.evidence_norm.register_forward_hook(capture_evidence),
        head.head[0].register_forward_hook(capture_first_linear),
        head.head[1].register_forward_hook(capture_first_hidden),
        head.head[2].register_forward_hook(capture_second_linear),
        head.head[3].register_forward_hook(capture_second_hidden),
        gate.register_forward_hook(capture_gate),
        head.head[4].register_forward_pre_hook(before_final_linear),
    ]
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks

"""Run the unchanged native TRAIN loop for two real task updates."""
class FirstTwoBatches:
    sampler = loader.sampler

    def __len__(self):
        return 2

    def __iter__(self):
        for batch in itertools.islice(loader, 2):
            assert len(batch['scan_ids']) == 2
            batches.append(dict(scan_ids=[str(value) for value in batch['scan_ids']],
                expressions=len(batch['ann_ids']), points=int(batch['feats'].shape[0])))
            yield batch


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def before_model(module, args):
    norms = [m for block in (module.input_conv, module.unet, module.output_layer)
             for m in block.modules() if isinstance(m, nn.BatchNorm1d)]
    assert module.training and norms and all(m.training for m in norms)
    assert module.text_encoder.training and all(not p.requires_grad for p in module.text_encoder.parameters())
    modes.append(True)


def after_model(module, args, value):
    loss = value[0]
    assert loss.requires_grad and torch.isfinite(loss).all()
    losses.append(float(loss.detach().cpu()))


def before_decoder(module, args):
    assert len(args) == 8 and len(gradient_checks) == len(step_rows) and len(step_rows) < 2
    reset_state()


step = optimizer.step


def checked_step(*args, **kwargs):
    index = len(step_rows)
    assert index < 2 and len(losses) == len(modes) == index + 1
    record = gradient_record()
    before = gate.weight.detach().clone()
    value = step(*args, **kwargs)
    update = float((gate.weight - before).abs().max())
    assert update > 0 and torch.isfinite(gate.weight).all()
    record.update(step=index, parameter_update_max=update)
    gradient_checks.append(record)
    step_rows.append(dict(record))
    return value


hooks = install_hooks() + [model.register_forward_pre_hook(before_model),
    model.register_forward_hook(after_model), model.dec.register_forward_pre_hook(before_decoder)]
model.dec.prediction_head = checked_prediction
optimizer.step = checked_step
torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1, model, FirstTwoBatches(), optimizer, scheduler, cfg, logger, Writer())
    torch.cuda.synchronize()
    assert len(losses) == len(gradient_checks) == len(batches) == len(modes) == 2
    assert len({scene for batch in batches for scene in batch['scan_ids']}) == 4
    check_response_rows(path_rows)
    peak = int(torch.cuda.max_memory_reserved())
    assert 0 < peak < 24 * 2**30
finally:
    optimizer.step = step
    model.dec.prediction_head = original_prediction
    for hook in hooks:
        hook.remove()
    if getattr(loader, '_iterator', None) is not None:
        loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'],
    code_commit=FROZEN['code_commit'], goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'], native_entry='tools/train.py::train',
    split='train', steps=2, optimizer_steps=2, losses=losses, native_data_loaded=True,
    backward_finite=True, intended_path_observed=True, batches=batches, gradient_checks=gradient_checks,
    hidden_paths=path_rows, hidden_gate_calls=14, first_hidden_consumers=14, second_hidden_consumers=14,
    final_linear_consumers=14, original_MLP_layer_calls=70, shared_query_norm_calls=14,
    original_evidence_norm_calls=14, original_overlap_consumers=14, coarse_gate_consumers=12,
    static_mask_features_preserved=True, original_competition_and_evidence_preserved=True,
    original_hard_gate_algorithm_preserved=True, native_modes_verified=True, peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
                     config_data_root=cfg.data.train.data_root),
    validation_used=False, model_weights_saved=False, admission=False,
    limitation='Actual native TRAIN two-step task only; efficacy and full-dataset resource bounds unmeasured.')),
    encoding='utf8')
