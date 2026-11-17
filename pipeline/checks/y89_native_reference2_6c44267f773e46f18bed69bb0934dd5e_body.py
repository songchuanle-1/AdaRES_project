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
assert row['id'] == 'AdaRES-Y89'
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
assert sum(p.numel() for p in model.parameters()) == 164638865
assert not hasattr(model.dec, 'refinement_indicators')
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)

dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
loader = native.build_dataloader(dataset, dist=False, **cfg.dataloader.train)






losses,gradient_checks,batches,modes,step_rows,path_rows,state=[],[],[],[],[],[],{}
head=model.dec.indi_embedding;norms=[head.evidence_norm]+list(head.refinement_evidence_norms);original_prediction=model.dec.prediction_head
"""Exact stage-private evidence normalization and shared-head consumer checks."""
def check_norm_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        for key in ('original_query_identity', 'original_mask_identity', 'original_mass_identity',
                    'private_norm_identity', 'undetached_evidence', 'independent_competition_reference',
                    'original_epsilon', 'shared_MLP_consumer'):
            assert row[key] is True
        assert row['affine_nondefault'] is bool(index // 7)
        delta = row['affine_delta_max']
        assert math.isfinite(delta) and (delta > 0 if index // 7 else delta == 0)


def reset_state():
    state.clear()
    state.update(queries=[], masks=[], gates=[], query_values=[], evidence_inputs=[],
        norm_values=[], norm_consumers=[], heads=[], coarse=0, features=None, padding=None,
        mass=None, indicator_args=[])


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
    index = len(state['indicator_args'])
    assert module is head and len(args) == 5 and args[4] == index
    assert args[0] is state['queries'][index] and args[1] is state['masks'][index]
    assert args[2] is state['padding'] and args[0].requires_grad and args[1].requires_grad
    if index == 0:
        state['mass'] = args[3]
    assert args[3] is state['mass'] and not args[3].requires_grad
    state['indicator_args'].append(args)


def capture_query_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['queries'][-1] and value.requires_grad
    state['query_values'].append(value)


def capture_evidence_norm(module, args, value):
    index = len(state['norm_values'])
    assert module is norms[index] and len(args) == 1
    assert args[0].requires_grad and value.requires_grad and module.eps == 1e-5
    assert module.normalized_shape == (769,)
    query, masks, pad, mass, stage_index = state['indicator_args'][index]
    assert stage_index == index
    normalized = state['query_values'][index]
    probabilities = masks.sigmoid() * (~pad).unsqueeze(1).to(masks.dtype)
    weighted = probabilities * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection).clamp_min(1e-6)
    diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0.)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    peer = torch.einsum('bij,bjd->bid', competition, normalized)
    reference_evidence = torch.cat((normalized, peer, normalized - peer,
                                   overlap.max(-1).values.unsqueeze(-1)), -1).double()
    torch.testing.assert_allclose(args[0].double(), reference_evidence, rtol=1e-5, atol=1e-6)
    reference = torch.nn.functional.layer_norm(args[0].double(), module.normalized_shape,
        module.weight.double(), module.bias.double(), module.eps)
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    default = torch.nn.functional.layer_norm(args[0], module.normalized_shape, None, None, module.eps)
    nondefault = bool(torch.count_nonzero(module.weight - 1) or torch.count_nonzero(module.bias))
    delta = float((value - default).abs().max())
    assert nondefault == bool(len(step_rows)) and (delta > 0 if nondefault else delta == 0)
    args[0].retain_grad()
    value.retain_grad()
    state['evidence_inputs'].append(args[0])
    state['norm_values'].append(value)
    state['norm_consumers'].append(index)
    path_rows.append(dict(step=len(step_rows), stage=index, original_query_identity=True,
        original_mask_identity=True, original_mass_identity=True, private_norm_identity=True,
        undetached_evidence=True, independent_competition_reference=True, original_epsilon=True,
        shared_MLP_consumer=True, affine_nondefault=nondefault, affine_delta_max=delta))


def before_shared_MLP(module, args):
    index = len(state['heads'])
    assert module is head.head and args[0] is state['norm_values'][index]
    state['heads'].append(index)


def before_coarse(module, args):
    assert module is model.dec.qia_layers[state['coarse']] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def gradient_record():
    assert all(len(state[key]) == 7 for key in
        ('queries', 'masks', 'gates', 'query_values', 'evidence_inputs', 'norm_values', 'indicator_args'))
    assert state['heads'] == state['norm_consumers'] == list(range(7)) and state['coarse'] == 6
    parameters = [parameter for norm in norms for parameter in norm.parameters()]
    assert len(parameters) == 14 and len({parameter.data_ptr() for parameter in parameters}) == 14
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in list(head.head.parameters()) + list(head.query_norm.parameters()))
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().max() > 0 for parameter in parameters)
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in state['evidence_inputs'] + state['norm_values'])
    return dict(all_finite=True, tensors=len(active), indicator_calls=7, shared_MLP_calls=7,
        shared_query_norm_calls=7, private_evidence_norm_calls=7, coarse_gate_consumers=6,
        shared_MLP_parameter_tensors=6, shared_query_norm_parameter_tensors=2,
        evidence_norm_gradient_tensors=14, input_gradients_finite=True,
        norm_gradient_maxima=[float(parameter.grad.abs().max()) for parameter in parameters])


def install_hooks():
    hooks = [head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_query_norm),
        head.head.register_forward_pre_hook(before_shared_MLP)]
    hooks.extend(norm.register_forward_hook(capture_evidence_norm) for norm in norms)
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks

"""Use the unchanged original native TRAIN entry for two real optimizer steps."""
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
    bn = [value for block in (module.input_conv, module.unet, module.output_layer)
          for value in block.modules() if isinstance(value, nn.BatchNorm1d)]
    assert module.training and bn and all(value.training for value in bn)
    assert module.text_encoder.training and all(not parameter.requires_grad for parameter in module.text_encoder.parameters())
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
    parameters_ = [parameter for norm in norms for parameter in norm.parameters()]
    before = [parameter.detach().clone() for parameter in parameters_]
    value = step(*args, **kwargs)
    updates = [float((parameter - saved).abs().max()) for parameter, saved in zip(parameters_, before)]
    assert all(update > 0 for update in updates) and all(torch.isfinite(parameter).all() for parameter in parameters_)
    record.update(step=index, norm_parameter_updates=updates)
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
    check_norm_rows(path_rows)
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
    norm_paths=path_rows, private_evidence_norm_calls=14, shared_MLP_calls=14,
    shared_query_norm_calls=14, shared_indicator_calls=14, coarse_gate_consumers=12,
    original_competition_evidence_preserved=True, static_mask_features_preserved=True,
    original_hard_gate_algorithm_preserved=True, native_modes_verified=True, peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
                     config_data_root=cfg.data.train.data_root),
    validation_used=False, model_weights_saved=False, admission=False,
    limitation='Actual original native two-step TRAIN only; trained efficacy and full-data memory limits unmeasured.')),
    encoding='utf8')
