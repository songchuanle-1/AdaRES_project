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
assert row['id'] == 'AdaRES-Y96'
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
head=model.dec.indi_embedding;residual=model.dec.seed_query_skip;original_prediction=model.dec.prediction_head
"""Callbacks shared verbatim by actual CPU and pending native sparse TRAIN probes."""
def reset_state():
    state.clear()
    state.update(predictions=[], masks=[], gates=[], indicators=[], norms=[], generator_layers=[],
                 coarse=0, features=None, padding=None, mass=None)


def capture_scg(module, args, value):
    assert module is model.dec.scg_head and 'scg' not in state
    state['scg'] = value


def capture_sampling(module, args, value):
    assert module is model.dec.sampling_module and args[0] is state['scg']
    assert len(value) == 2 and 'indices' not in state
    state['indices'] = value[0].long()


def capture_generator_layer(module, args, value):
    index = len(state['generator_layers'])
    assert module is model.dec.query_generator[index] and value.requires_grad
    if index == 0:
        assert 'seed' not in state
        indices = state['indices']
        expected = state['scg'].gather(1, indices.unsqueeze(-1).expand(-1, -1, 256))
        assert torch.equal(args[0], expected)
        assert args[0].shape[1:] == (128, 256) and args[0].requires_grad
        state['seed'] = args[0]
    else:
        assert args[0] is state['generator_layers'][-1]
    state['generator_layers'].append(value)


def capture_generated(module, args, value):
    assert module is model.dec.query_generator and len(state['generator_layers']) == 5
    assert args[0] is state['seed'] and value is state['generator_layers'][-1]
    assert 'generated' not in state
    state['generated'] = value


def capture_skip(module, args, value):
    assert module is residual and len(args) == 2 and 'initialized' not in state
    generated, seed = args
    assert generated is state['generated'] and seed is state['seed']
    assert generated.requires_grad and seed.requires_grad and value.requires_grad
    for tensor in args + (value,):
        tensor.retain_grad()
    reference = generated.double() + seed.double() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    assert torch.equal(value, generated + seed @ module.weight)
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - generated).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    state['initialized'] = value
    path_rows.append(dict(step=len(step_rows), weight_nonzero=nonzero, query_delta_max=delta,
        original_sampled_SCG_seed_exact=True, original_generator_input_identity=True,
        original_five_generator_layers_consumed=True, original_generated_output_identity=True,
        both_inputs_undetached=True, float64_seed_projection_reference=True,
        original_initial_prediction_consumes_skip=False, original_first_coarse_consumes_skip=False))


def checked_prediction(query, features, padding):
    if not state['predictions']:
        assert query is state['initialized']
        state['features'], state['padding'] = features, padding
        path_rows[-1]['original_initial_prediction_consumes_skip'] = True
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
    if index == 0:
        assert args[1] is state['initialized']
        path_rows[-1]['original_first_coarse_consumes_skip'] = True
    state['coarse'] += 1


def before_indicator(module, args):
    assert module is head and len(args) == 4
    assert args[0] is state['predictions'][-1] and args[1] is state['masks'][-1]
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
    assert len(state['generator_layers']) == 5 and state['coarse'] == 6
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    head_gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert len(head_gradients) == 10 and all(torch.isfinite(gradient).all() for gradient in head_gradients)
    for key in ('seed', 'generated', 'initialized'):
        assert state[key].grad is not None and torch.isfinite(state[key].grad).all()
    gradient = residual.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10,
        indicator_calls=7, prediction_calls=7, query_norm_consumption_calls=7, generator_layer_calls=5,
        initialization_calls=1, coarse_gate_consumers=6, both_input_gradients_finite=True,
        new_gradient_max=float(gradient.abs().max()))


def check_response_rows(rows):
    import math
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index)
        for key in ('original_sampled_SCG_seed_exact', 'original_generator_input_identity',
                    'original_five_generator_layers_consumed', 'original_generated_output_identity',
                    'both_inputs_undetached', 'float64_seed_projection_reference',
                    'original_initial_prediction_consumes_skip', 'original_first_coarse_consumes_skip'):
            assert row[key] is True, key
        delta = row['query_delta_max']
        assert math.isfinite(delta) and (delta > 0 if index else delta == 0)


def install_hooks():
    hooks = [model.dec.scg_head.register_forward_hook(capture_scg),
        model.dec.sampling_module.register_forward_hook(capture_sampling),
        model.dec.query_generator.register_forward_hook(capture_generated),
        residual.register_forward_hook(capture_skip), head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(module.register_forward_hook(capture_generator_layer) for module in model.dec.query_generator)
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
    gradient_checks=gradient_checks, initialization_paths=path_rows, initialization_calls=2,
    generator_layer_calls=10, coarse_gate_consumers=12, query_norm_consumption_calls=14, shared_head_calls=14,
    static_mask_features_preserved=True, original_hard_gate_algorithm_preserved=True, native_modes_verified=True,
    peak_reserved_bytes=peak, data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
        config_data_root=cfg.data.train.data_root),
    validation_used=False, model_weights_saved=False, admission=False,
    limitation='Actual original native two-step TRAIN only; not trained efficacy or a full-dataset memory guarantee.')),
    encoding='utf8')
