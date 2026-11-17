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
assert row['id'] == 'AdaRES-Y97'
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
assert sum(p.numel() for p in model.parameters()) == 164826245
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)

dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)






losses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}
head=model.dec.indi_embedding;residual=model.dec.language_quadratic;original_prediction=model.dec.prediction_head
"""Same raw-token/projection/norm/consumer callbacks for CPU and native task."""
def reset_state():
    state.clear()
    state.update(predictions=[], masks=[], gates=[], indicators=[], norms=[], language_self=[],
                 coarse=0, features=None, padding=None, mass=None)


def capture_projection(module, args, value):
    assert module is model.dec.lang_proj and 'projected' not in state
    assert args[0] is state['raw_expected'] and args[0].shape[-1] == 768
    assert value.shape[-1] == 256 and value.requires_grad
    state['projected'], state['raw'] = value, args[0]


def capture_adapter(module, args, value):
    assert module is residual and len(args) == 2 and 'adapted' not in state
    projected, raw = args
    assert projected is state['projected'] and raw is state['raw'] and projected.requires_grad
    assert raw is state['raw_expected'] and value.requires_grad
    reference = projected.double() + raw.double().square() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    assert torch.equal(value, projected + raw.square() @ module.weight)
    projected.retain_grad()
    value.retain_grad()
    if raw.requires_grad:
        raw.retain_grad()
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - projected).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    state['adapted'] = value
    path_rows.append(dict(step=len(step_rows), weight_nonzero=nonzero, language_delta_max=delta,
        raw_requires_grad=bool(raw.requires_grad), original_raw_token_identity=True,
        original_projection_identity=True, full768_channels_retained=True,
        float64_quadratic_projection_reference=True, original_language_norm_consumes_adapter=False,
        original_sampling_consumes_normalized_language=False, original_first_language_self_consumes_norm=False))


def capture_language_norm(module, args, value):
    assert module is model.dec.lang_norm and args[0] is state['adapted'] and 'language' not in state
    assert module.eps == 1e-5 and value.requires_grad
    reference = torch.nn.functional.layer_norm(args[0].double(), module.normalized_shape,
        module.weight.double(), module.bias.double(), module.eps)
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    value.retain_grad()
    state['language'] = value
    path_rows[-1]['original_language_norm_consumes_adapter'] = True


def before_sampling(module, args):
    assert module is model.dec.sampling_module and args[1] is state['language']
    assert torch.equal(args[3], ~state['mask_expected'].bool()) and args[3].dtype == torch.bool
    assert not args[3].requires_grad and 'sampling' not in state
    state['sampling'] = True
    path_rows[-1]['original_sampling_consumes_normalized_language'] = True


def before_language_self(module, args):
    index = len(state['language_self'])
    assert module is model.dec.lla_layers[index]
    assert torch.equal(args[1], ~state['mask_expected'].bool())
    if index == 0:
        assert args[0] is state['language']
        path_rows[-1]['original_first_language_self_consumes_norm'] = True
    state['language_self'].append(index)


def checked_prediction(query, features, padding):
    if not state['predictions']:
        state['features'], state['padding'] = features, padding
    assert features is state['features'] and padding is state['padding']
    state['predictions'].append(query)
    value = original_prediction(query, features, padding)
    state['masks'].append(value[1])
    expected = (value[1].sigmoid() < .5) | padding.unsqueeze(1)
    expected[expected.all(-1)] = False
    expected = expected | padding.unsqueeze(1)
    assert torch.equal(value[2], expected) and not value[2].requires_grad
    state['gates'].append(value[2])
    return value


def before_coarse(module, args):
    index = state['coarse']
    assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def before_indicator(module, args):
    assert module is head and args[0] is state['predictions'][-1] and args[1] is state['masks'][-1]
    assert args[2] is state['padding']
    if not state['indicators']:
        state['mass'] = args[3]
    assert args[3] is state['mass'] and not args[3].requires_grad
    state['indicators'].append(len(state['predictions']) - 1)


def capture_query_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['predictions'][-1]
    assert args[0].requires_grad and value.requires_grad
    state['norms'].append(True)


def gradient_record():
    assert len(state['predictions']) == len(state['masks']) == len(state['gates']) == 7
    assert state['indicators'] == list(range(7)) and state['norms'] == [True] * 7
    assert state['language_self'] == list(range(6)) and state['coarse'] == 6 and state['sampling']
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    head_gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert len(head_gradients) == 10 and all(torch.isfinite(gradient).all() for gradient in head_gradients)
    original = list(model.dec.lang_proj.parameters()) + list(model.dec.lang_norm.parameters())
    assert len(original) == 4 and all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                      for parameter in original)
    for key in ('projected', 'adapted', 'language'):
        assert state[key].grad is not None and torch.isfinite(state[key].grad).all()
    if state['raw'].requires_grad:
        assert state['raw'].grad is not None and torch.isfinite(state['raw'].grad).all()
    gradient = residual.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10, indicator_calls=7, prediction_calls=7,
        query_norm_consumption_calls=7, language_adapter_calls=1, original_projection_calls=1,
        original_language_norm_calls=1, original_sampling_consumers=1, original_language_self_calls=6,
        coarse_gate_consumers=6, original_language_parameter_tensors=4,
        projected_adapter_normalized_gradients_finite=True, frozen_text_encoder_preserved=True,
        raw_requires_grad=bool(state['raw'].requires_grad), new_gradient_max=float(gradient.abs().max()))


def check_response_rows(rows):
    import math
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index)
        assert type(row['raw_requires_grad']) is bool
        for key in ('original_raw_token_identity', 'original_projection_identity', 'full768_channels_retained',
                    'float64_quadratic_projection_reference', 'original_language_norm_consumes_adapter',
                    'original_sampling_consumes_normalized_language', 'original_first_language_self_consumes_norm'):
            assert row[key] is True, key
        delta = row['language_delta_max']
        assert math.isfinite(delta) and (delta > 0 if index else delta == 0)


def install_hooks():
    hooks = [model.dec.lang_proj.register_forward_hook(capture_projection),
        residual.register_forward_hook(capture_adapter), model.dec.lang_norm.register_forward_hook(capture_language_norm),
        model.dec.sampling_module.register_forward_pre_hook(before_sampling),
        head.register_forward_pre_hook(before_indicator), head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(module.register_forward_pre_hook(before_language_self) for module in model.dec.lla_layers)
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks

"""Execute only the first two original TRAIN batches after an authorized idle slot."""
class FirstTwoBatches:
    sampler = loader.sampler

    def __len__(self):
        return 2

    def __iter__(self):
        for batch in itertools.islice(loader, 2):
            assert len(batch['scan_ids']) == 2
            batches.append(dict(scan_ids=[str(scene) for scene in batch['scan_ids']],
                expressions=len(batch['ann_ids']), points=int(batch['feats'].shape[0])))
            yield batch


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def before_model(module, args):
    norms = [layer for block in (module.input_conv, module.unet, module.output_layer)
             for layer in block.modules() if isinstance(layer, nn.BatchNorm1d)]
    assert module.training and norms and all(layer.training for layer in norms)
    assert module.text_encoder.training and all(not parameter.requires_grad for parameter in module.text_encoder.parameters())
    modes.append(True)


def after_model(module, args, value):
    loss = value[0]
    assert loss.requires_grad and torch.isfinite(loss).all()
    losses.append(float(loss.detach().cpu()))


def before_decoder(module, args):
    assert len(args) == 8 and len(gradient_checks) == len(step_rows) and len(step_rows) < 2
    reset_state()
    state['raw_expected'] = args[3]
    state['mask_expected'] = args[4]


original_step = optimizer.step


def checked_step(*args, **kwargs):
    index = len(step_rows)
    assert index < 2 and len(losses) == len(modes) == index + 1
    record = gradient_record()
    before = residual.weight.detach().clone()
    value = original_step(*args, **kwargs)
    update = float((residual.weight.detach() - before).abs().max())
    assert update > 0 and torch.isfinite(residual.weight).all()
    record.update(step=index, parameter_update_max=update)
    gradient_checks.append(record)
    step_rows.append(dict(record))
    return value


hooks = install_hooks() + [model.register_forward_pre_hook(before_model), model.register_forward_hook(after_model),
                          model.dec.register_forward_pre_hook(before_decoder)]
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
    optimizer.step = original_step
    model.dec.prediction_head = original_prediction
    for hook in hooks:
        hook.remove()
    if getattr(loader, '_iterator', None) is not None:
        loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(
    complete=True, id=row['id'], config=row['config'], code_commit=FROZEN['code_commit'],
    goal_sha256=FROZEN['goal_sha256'], native_YAML_sha256=row['config_data']['sha256'],
    native_entry='tools/train.py::train', split='train', steps=2, optimizer_steps=2, losses=losses,
    native_data_loaded=True, backward_finite=True, intended_path_observed=True, batches=batches,
    gradient_checks=gradient_checks, language_paths=path_rows, language_adapter_calls=2,
    original_projection_calls=2, original_language_norm_calls=2, coarse_gate_consumers=12, query_norm_consumption_calls=14, shared_head_calls=14,
    static_mask_features_preserved=True, original_hard_gate_algorithm_preserved=True, native_modes_verified=True,
    peak_reserved_bytes=peak, data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
        config_data_root=cfg.data.train.data_root),
    validation_used=False, model_weights_saved=False, admission=False,
    limitation='Actual original native two-step TRAIN only; not trained efficacy or a full-dataset memory guarantee.')),
    encoding='utf8')
