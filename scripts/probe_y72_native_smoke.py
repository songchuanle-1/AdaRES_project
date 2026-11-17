"""Two original native TRAIN steps with current-stage mask-feature path observation."""
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
assert row['id'] == 'AdaRES-Y72'
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
losses = []; gradient_checks = []; batches = []; modes = []; states = []; paths = []
refinement = model.dec.visual_mask_refinement
head = model.dec.indi_embedding
assert len(list(head.parameters())) == 10
assert torch.equal(refinement.weight.detach(), torch.zeros_like(refinement.weight))


def check_path_calls(visual_stages, refinement_stages, prediction_stages, indicator_count, norm_calls):
    assert visual_stages == refinement_stages == list(range(6))
    assert prediction_stages == list(range(7))
    assert all(type(i) is int for i in visual_stages + refinement_stages + prediction_stages)
    assert type(indicator_count) is int and indicator_count == 7
    assert norm_calls == [True] * 7 and all(type(v) is bool for v in norm_calls)


def check_native_inputs(features, original_features, visual, current_visual, padded, original_padding):
    assert features is original_features
    assert visual is current_visual
    assert padded is original_padding


def check_response_rows(rows):
    from math import isfinite
    assert len(rows) == 12
    for index, record in enumerate(rows):
        assert type(record['stage']) is int and record['stage'] == index % 6
        assert type(record['weight_nonzero']) is bool
        for key in ('native_current_visual_identity_verified', 'static_mask_identity_verified',
                    'padding_identity_verified', 'prediction_consumption_verified', 'padding_rows_unchanged'):
            assert record[key] is True
        count = record['expressions']
        assert type(count) is int and count >= 2
        assert record['reference_expression_rows'] == [0, count - 1]
        lengths = record['reference_valid_feature_counts']
        assert len(lengths) == 2 and all(type(n) is int and n >= 3 for n in lengths)
        assert record['reference_feature_indices'] == [[0, n // 2, n - 1] for n in lengths]
        assert record['reference_channels'] == [0, 127, 255]
        for key in ('reference_error_max', 'feature_delta_max', 'visual_abs_max'):
            assert isfinite(record[key]) and record[key] >= 0
        assert type(record['padded_rows']) is int and record['padded_rows'] >= 0
        if index < 6:
            assert record['weight_nonzero'] is False and record['feature_delta_max'] == 0
        else:
            assert record['weight_nonzero'] is True and record['feature_delta_max'] > 0
            assert record['visual_abs_max'] > 0


class FirstTwoBatches:
    sampler = loader.sampler

    def __len__(self):
        return 2

    def __iter__(self):
        for batch in itertools.islice(loader, 2):
            assert len(batch['scan_ids']) == 2
            batches.append(dict(scan_ids=[str(s) for s in batch['scan_ids']],
                expressions=len(batch['ann_ids']), points=int(batch['feats'].shape[0])))
            yield batch


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def before_model(module, args):
    bn = [m for block in (module.input_conv, module.unet, module.output_layer)
          for m in block.modules() if isinstance(m, nn.BatchNorm1d)]
    assert module.training and bn and all(m.training for m in bn)
    assert module.text_encoder.training and all(not p.requires_grad for p in module.text_encoder.parameters())
    modes.append(True)


def after_model(module, args, value):
    loss = value[0]
    assert loss.requires_grad and torch.isfinite(loss).all()
    losses.append(float(loss.detach().cpu()))


def before_decoder(module, args):
    assert len(args) == 8 and len(states) == len(gradient_checks)
    states.append(dict(visual_stages=[], visuals=[], refinement_stages=[], outputs=[],
                       predictions=[], indicators=[], norms=[], records=[]))


def capture_visual(index, module, args, value):
    state = states[-1]
    assert index == len(state['visuals'])
    assert value.requires_grad and torch.isfinite(value).all()
    value.retain_grad()
    state['visual_stages'].append(index)
    state['visuals'].append(value)


def capture_refinement(module, args, value):
    state = states[-1]
    index = len(state['outputs'])
    assert index < 6 and len(state['visuals']) == index + 1
    features, visual, padded = args
    check_native_inputs(features, state['base'], visual, state['visuals'][index], padded, state['padding'])
    assert features.shape == visual.shape == value.shape and features.shape[-1] == 256
    assert features.ndim == 3 and padded.shape == features.shape[:2]
    assert all(t.requires_grad and torch.isfinite(t).all() for t in (features, visual, value))
    value.retain_grad()
    with torch.no_grad():
        expressions = features.shape[0]
        expression_rows = [0, expressions - 1]
        channels = [0, 127, 255]
        lengths = []; selected = []; reference = []; actual = []
        for b in expression_rows:
            valid = torch.nonzero(~padded[b], as_tuple=False).flatten()
            # Native get_batches places each expression's valid features before its padding.
            assert torch.equal(valid, torch.arange(len(valid), device=valid.device))
            assert len(valid) >= 3
            indices = [0, len(valid) // 2, len(valid) - 1]
            lengths.append(len(valid)); selected.append(indices)
            for position in indices:
                for d in channels:
                    residual = (visual[b, position].double() * module.weight[:, d].double()).sum()
                    reference.append(features[b, position, d].double() + residual)
                    actual.append(value[b, position, d].double())
        expected = torch.stack(reference); observed = torch.stack(actual)
        torch.testing.assert_close(observed, expected, rtol=4e-5, atol=4e-6)
        assert torch.equal(value[padded], features[padded])
        assert torch.equal(features[padded], torch.zeros_like(features[padded]))
        nonzero = bool(module.weight.detach().abs().max() > 0)
        if not nonzero:
            assert torch.equal(value, features)
        record = dict(stage=index, weight_nonzero=nonzero, expressions=expressions,
            reference_expression_rows=expression_rows, reference_valid_feature_counts=lengths,
            reference_feature_indices=selected, reference_channels=channels,
            reference_error_max=float((observed - expected).abs().max()),
            feature_delta_max=float((value - features).abs().max()),
            visual_abs_max=float(visual.abs().max()), padded_rows=int(padded.sum()),
            native_current_visual_identity_verified=True, static_mask_identity_verified=True,
            padding_identity_verified=True, padding_rows_unchanged=True,
            prediction_consumption_verified=False)
    state['refinement_stages'].append(index)
    state['outputs'].append(value)
    state['records'].append(record)
    paths.append(record)


prediction = model.dec.prediction_head


def checked_prediction(query, features, padded):
    state = states[-1]
    index = len(state['predictions'])
    assert index < 7
    if index == 0:
        assert not state['outputs'] and not state['visuals']
        assert features.requires_grad
        features.retain_grad()
        state['base'] = features
        state['padding'] = padded
    else:
        assert features is state['outputs'][index - 1]
        assert padded is state['padding']
        state['records'][index - 1]['prediction_consumption_verified'] = True
    state['predictions'].append(index)
    return prediction(query, features, padded)


def capture_indicator(module, args, value):
    assert args[0].requires_grad and args[1].requires_grad and torch.isfinite(value).all()
    states[-1]['indicators'].append(True)


def capture_norm(module, args, value):
    states[-1]['norms'].append(bool(args[0].requires_grad))


step = optimizer.step


def checked_step(*args, **kwargs):
    index = len(gradient_checks)
    assert index < 2 and len(states) == index + 1
    state = states[index]
    check_path_calls(state['visual_stages'], state['refinement_stages'], state['predictions'],
                     len(state['indicators']), state['norms'])
    active = [p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    headgrads = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(headgrads) == 10 and all(torch.isfinite(g).all() for g in headgrads)
    grad = refinement.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0
    tensors = [state['base']] + state['visuals'] + state['outputs']
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in tensors)
    before = refinement.weight.detach().clone()
    gradmax = float(grad.abs().max())
    result = step(*args, **kwargs)
    update = float((refinement.weight.detach() - before).abs().max())
    assert update > 0 and torch.isfinite(refinement.weight).all()
    gradient_checks.append(dict(all_finite=True, tensors=len(active), indicator_tensors=len(headgrads),
        indicator_calls=7, refinement_calls=6, prediction_consumption_calls=6,
        current_visual_and_static_mask_gradients_finite=True,
        refinement_weight_gradient_max=gradmax, parameter_update_max=update))
    return result


hooks = [model.register_forward_pre_hook(before_model), model.register_forward_hook(after_model),
         model.dec.register_forward_pre_hook(before_decoder),
         refinement.register_forward_hook(capture_refinement),
         head.register_forward_hook(capture_indicator), head.query_norm.register_forward_hook(capture_norm)]
for index, layer in enumerate(model.dec.ila_ffn_layers):
    hooks.append(layer.register_forward_hook(partial(capture_visual, index)))
model.dec.prediction_head = checked_prediction
optimizer.step = checked_step
torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1, model, FirstTwoBatches(), optimizer, scheduler, cfg, logger, Writer())
    torch.cuda.synchronize()
    assert len(losses) == len(gradient_checks) == len(batches) == len(modes) == len(states) == 2
    assert len({s for b in batches for s in b['scan_ids']}) == 4
    check_response_rows(paths)
    peak = int(torch.cuda.max_memory_reserved())
    assert 0 < peak < 24 * 2**30
finally:
    optimizer.step = step
    model.dec.prediction_head = prediction
    for hook in hooks:
        hook.remove()
    if getattr(loader, '_iterator', None) is not None:
        loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'],
    code_commit=FROZEN['code_commit'], goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'], native_entry='tools/train.py::train',
    split='train', steps=2, optimizer_steps=2, losses=losses, native_data_loaded=True,
    backward_finite=True, intended_path_observed=True, batches=batches, gradient_checks=gradient_checks,
    refinement_paths=paths, refinement_calls=sum(len(s['outputs']) for s in states),
    prediction_consumption_calls=sum(len(s['outputs']) for s in states),
    shared_head_calls=sum(len(s['indicators']) for s in states), native_modes_verified=True,
    peak_reserved_bytes=peak, data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
    config_data_root=cfg.data.train.data_root), validation_used=False, model_weights_saved=False,
    admission=False, limitation='Actual native two-step TRAIN execution only; not efficacy or a full-dataset memory bound.')),
    encoding='utf-8')
