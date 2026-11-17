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
assert row['id'] == 'AdaRES-Y88'
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
head=model.dec.indi_embedding;projection=model.dec.mask_query_quadratic;original_prediction=model.dec.prediction_head
"""Observe the exact normalized-query, mask, indicator and attention consumers."""
def check_response_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index // 7)
        for key in ('original_normalized_query_identity', 'input_undetached',
                    'float64_quadratic_projection_reference', 'original_mask_consumer'):
            assert row[key] is True
        delta = row['delta_max']
        assert math.isfinite(delta) and (delta > 0 if index // 7 else delta == 0)


def reset_state():
    state.clear()
    state.update(queries=[], normalized=[], projected=[], gates=[], indicators=[],
                 indicator_norms=[], scores=[], coarse=0, features=None, padding=None)


def checked_prediction(query, features, pad):
    if not state['queries']:
        state['features'], state['padding'] = features, pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(query)
    value = original_prediction(query, features, pad)
    index = len(state['queries']) - 1
    assert len(state['projected']) == index + 1
    expected_mask = torch.einsum('bnd,bmd->bnm', state['projected'][index], features)
    assert torch.equal(value[1], expected_mask)
    state['mask'] = value[1]
    expected_gate = (value[1].sigmoid() < .5) | pad.unsqueeze(1)
    expected_gate[expected_gate.all(-1)] = False
    expected_gate = expected_gate | pad.unsqueeze(1)
    assert torch.equal(value[2], expected_gate) and not value[2].requires_grad
    state['gates'].append(value[2])
    return value


def capture_out_norm(module, args, value):
    assert args[0] is state['queries'][-1] and value.requires_grad
    state['normalized'].append(value)


def before_score(module, args):
    index = len(state['scores'])
    assert args[0] is state['normalized'][index]
    state['scores'].append(index)


def capture_projection(module, args, value):
    index = len(state['projected'])
    assert module is projection and len(args) == 1
    assert args[0] is state['normalized'][index] and args[0].requires_grad and value.requires_grad
    args[0].retain_grad()
    value.retain_grad()
    reference = args[0].double() + args[0].double().square() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - args[0]).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    state['projected'].append(value)
    path_rows.append(dict(step=len(step_rows), stage=index, weight_nonzero=nonzero,
        delta_max=delta, original_normalized_query_identity=True, input_undetached=True,
        float64_quadratic_projection_reference=True, original_mask_consumer=True))


def before_indicator(module, args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries']) - 1)


def capture_indicator_norm(module, args, value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad
    state['indicator_norms'].append(True)


def before_coarse(module, args):
    assert module is model.dec.qia_layers[state['coarse']] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def gradient_record():
    assert all(len(state[key]) == 7 for key in
        ('queries', 'normalized', 'projected', 'gates', 'indicators', 'indicator_norms', 'scores'))
    assert state['scores'] == state['indicators'] == list(range(7)) and state['coarse'] == 6
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in state['normalized'] + state['projected'])
    gradient = projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10,
        projection_calls=7, original_mask_consumers=7, original_score_consumers=7,
        indicator_calls=7, query_norm_consumption_calls=7, coarse_gate_consumers=6,
        input_gradients_finite=True, new_gradient_max=float(gradient.abs().max()))


def install_hooks():
    hooks = [projection.register_forward_hook(capture_projection),
        model.dec.out_norm.register_forward_hook(capture_out_norm),
        model.dec.out_score.register_forward_pre_hook(before_score),
        head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_indicator_norm)]
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks

"""Run two original TRAIN steps without changing targets, losses or modes."""
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
    before = projection.weight.detach().clone()
    value = step(*args, **kwargs)
    update = float((projection.weight - before).abs().max())
    assert update > 0 and torch.isfinite(projection.weight).all()
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
    projection_paths=path_rows, projection_calls=14, original_mask_consumers=14,
    original_score_consumers=14, shared_head_calls=14, query_norm_consumption_calls=14,
    coarse_gate_consumers=12, static_mask_features_preserved=True,
    original_hard_gate_algorithm_preserved=True, native_modes_verified=True, peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
                     config_data_root=cfg.data.train.data_root),
    validation_used=False, model_weights_saved=False, admission=False,
    limitation='Actual native TRAIN two-step task only; efficacy and full-dataset resource bounds unmeasured.')),
    encoding='utf8')
