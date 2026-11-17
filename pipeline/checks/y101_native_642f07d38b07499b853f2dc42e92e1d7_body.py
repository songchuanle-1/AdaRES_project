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
assert row['id'] == 'AdaRES-Y101'
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
head=model.dec.indi_embedding;residual=model.dec.query_ffn_skip;original_prediction=model.dec.prediction_head
"""The same six original-FFN/parallel-input callbacks for CPU and native TRAIN."""
def reset_state():
    state.clear()
    state.update(predictions=[], masks=[], gates=[], indicators=[], norms=[], coarse=0,
                 fusion=[], ffn_inputs=[], ffn_outputs=[], adapted=[], features=None, padding=None, mass=None)


def capture_initial(module, args, value):
    assert module is model.dec.query_generator and 'initial' not in state and value.requires_grad
    state['initial'] = value


def capture_mask(module, args, value):
    assert module is model.dec.x_mask and 'mask_base' not in state and value.requires_grad
    value.retain_grad()
    state['mask_base'] = value


def capture_fusion(module, args, value):
    assert module is model.dec.norm_qqa and len(state['fusion']) == len(state['adapted']) < 6
    assert value.requires_grad and value.shape[-1] == 256
    value.retain_grad()
    state['fusion'].append(value)


def before_ffn(module, args):
    index = len(state['ffn_inputs'])
    assert index == len(state['adapted']) < 6 and module is model.dec.qqa_ffn_layers[index]
    assert args[0] is state['fusion'][index] and args[0].requires_grad
    state['ffn_inputs'].append(args[0])


def after_ffn(module, args, value):
    index = len(state['ffn_outputs'])
    assert module is model.dec.qqa_ffn_layers[index] and args[0] is state['ffn_inputs'][index]
    assert value.requires_grad
    value.retain_grad()
    state['ffn_outputs'].append(value)


def capture_skip(module, args, value):
    index = len(state['adapted'])
    assert module is residual and len(args) == 2 and index < 6
    base, current = args
    assert base is state['ffn_outputs'][index] and current is state['ffn_inputs'][index]
    assert base.requires_grad and current.requires_grad and value.requires_grad
    torch.testing.assert_allclose(value.double(), base.double() + current.double() @ module.weight.double(), rtol=1e-5, atol=1e-6)
    assert torch.equal(value, base + current @ module.weight)
    value.retain_grad()
    state['adapted'].append(value)
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - base).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    path_rows.append(dict(step=len(step_rows), stage=index, weight_nonzero=nonzero, query_delta_max=delta,
        original_full256_FFN_input_identity=True, complete_original_FFN_output_identity=True,
        original_shared_fusion_norm_consumed=True, both_inputs_undetached=True,
        float64_full_query_projection_reference=True, prediction_consumes_adapted_query=False,
        original_static_mask_preserved=False))


def checked_prediction(query, features, padding):
    index = len(state['predictions'])
    if index == 0:
        assert query is state['initial'] and not state['adapted']
        expected, expected_padding = model.dec.get_batches(state['mask_base'], state['batch_offsets_expected'])
        assert torch.equal(features, expected) and torch.equal(padding, expected_padding)
        state['features'], state['padding'] = features, padding
    else:
        assert query is state['adapted'][index - 1] and len(state['adapted']) == index
        path_rows[-1]['prediction_consumes_adapted_query'] = True
        path_rows[-1]['original_static_mask_preserved'] = True
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
    assert state['indicators'] == list(range(7)) and state['norms'] == [True] * 7 and state['coarse'] == 6
    assert all(len(state[k]) == 6 for k in ('fusion', 'ffn_inputs', 'ffn_outputs', 'adapted'))
    active = [p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    hg = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(hg) == 10 and all(torch.isfinite(g).all() for g in hg)
    original = [p for m in model.dec.qqa_ffn_layers for p in m.parameters()]
    assert len(original) == 36 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in original)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.dec.norm_qqa.parameters())
    assert state['mask_base'].grad is not None and torch.isfinite(state['mask_base'].grad).all()
    for k in ('fusion', 'ffn_inputs', 'ffn_outputs', 'adapted'):
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in state[k])
    gradient = residual.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10, indicator_calls=7, prediction_calls=7,
        query_norm_consumption_calls=7, coarse_gate_consumers=6, query_FFN_skip_calls=6, original_query_FFN_calls=6,
        original_query_FFN_parameter_tensors=36, full_input_base_adapted_and_original_FFN_gradients_finite=True,
        original_shared_fusion_norm_gradients_finite=True, static_mask_gradients_finite=True,
        frozen_text_encoder_preserved=True, new_gradient_max=float(gradient.abs().max()))


def check_response_rows(rows):
    import math
    assert len(rows) == 12
    for i, row in enumerate(rows):
        step, stage = divmod(i, 6)
        assert type(row['step']) is int and row['step'] == step and type(row['stage']) is int and row['stage'] == stage
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(step)
        for k in ('original_full256_FFN_input_identity', 'complete_original_FFN_output_identity',
                  'original_shared_fusion_norm_consumed', 'both_inputs_undetached', 'float64_full_query_projection_reference',
                  'prediction_consumes_adapted_query', 'original_static_mask_preserved'):
            assert row[k] is True, k
        assert math.isfinite(row['query_delta_max']) and (row['query_delta_max'] > 0 if step else row['query_delta_max'] == 0)


def install_hooks():
    hooks = [model.dec.query_generator.register_forward_hook(capture_initial), model.dec.x_mask.register_forward_hook(capture_mask),
        model.dec.norm_qqa.register_forward_hook(capture_fusion), residual.register_forward_hook(capture_skip),
        head.register_forward_pre_hook(before_indicator), head.query_norm.register_forward_hook(capture_query_norm)]
    for m in model.dec.qqa_ffn_layers:
        hooks += [m.register_forward_pre_hook(before_ffn), m.register_forward_hook(after_ffn)]
    hooks.extend(m.register_forward_pre_hook(before_coarse) for m in model.dec.qia_layers)
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
    state['batch_offsets_expected'] = args[2]


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
    gradient_checks=gradient_checks, query_FFN_paths=path_rows, query_FFN_skip_calls=12,
    original_query_FFN_calls=12, coarse_gate_consumers=12, query_norm_consumption_calls=14, shared_head_calls=14,
    static_mask_features_preserved=True, original_hard_gate_algorithm_preserved=True, native_modes_verified=True,
    peak_reserved_bytes=peak, data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
        config_data_root=cfg.data.train.data_root),
    validation_used=False, model_weights_saved=False, admission=False,
    limitation='Actual original native two-step TRAIN only; not trained efficacy or a full-dataset memory guarantee.')),
    encoding='utf8')
