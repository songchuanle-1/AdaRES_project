"""Check the full native YAML model and real TRAIN loading without CUDA."""

import base64
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import gorilla
import torch
from torch import nn
import yaml

assert P['physical_gpu'] is None
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
root = Path(P['root'])
stage = Path(P['output']).with_suffix('')
assert not stage.exists()
stage.mkdir()
row = PAYLOAD['variant']
package = stage / row['package']
package.mkdir()
for name, value in row['python_files'].items():
    data = base64.b64decode(value['base64'])
    assert hashlib.sha256(data).hexdigest() == value['sha256']
    (package / name).write_bytes(data)
config_path = stage / Path(row['config']).name
data = base64.b64decode(row['config_data']['base64'])
assert hashlib.sha256(data).hexdigest() == row['config_data']['sha256']
config_path.write_bytes(data)
assert yaml.safe_load(config_path.read_text()) == yaml.safe_load(
    (root / PAYLOAD['parent_config']).read_text())
native = importlib.import_module('train')
assert hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['train_source_lf_sha256']
cfg = gorilla.Config.fromfile(str(config_path))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30
assert cfg.dataloader.train.batch_size == 2 and cfg.model.sampling_module.num_proposal == 128
assert cfg.model.dec.num_layer == 6 and cfg.model.dec.iter_pred
assert cfg.data.train.training and cfg.data.train.prefix == 'train'
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
model = Candidate(**cfg.model)
assert all(p.device.type == 'cpu' for p in model.parameters())
logger = importlib.import_module('ipdn.utils').get_root_logger()
native.load_shape_compatible_pretrain(model, cfg.train.pretrain, logger)
dataset = native.resolve_train_dataset(str(config_path), cfg.data.train.copy(), logger)
gorilla.set_random_seed(cfg.train.seed)
batch = dataset.collate_fn([dataset[0], dataset[1]])
assert len(batch['scan_ids']) == 2 and len(batch['ann_ids']) >= 2
assert torch.isfinite(batch['feats']).all()
assert batch['lang_tokenss'].device.type == 'cpu'
model.train(True)
assert model.training and model.text_encoder.training
assert all(not p.requires_grad for p in model.text_encoder.parameters())
assert all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm1d))
heads = [model.dec.indi_embedding] + list(model.dec.refinement_indicators)
assert len(heads) == 7
assert len({p.data_ptr() for h in heads for p in h.parameters()}) == sum(
    len(list(h.parameters())) for h in heads)
reference = heads[0].state_dict()
assert all(all(torch.equal(h.state_dict()[key], value) for key, value in reference.items())
           for h in heads[1:])

controls = []
for dtype in (torch.float32, torch.float64):
    for head in heads:
        head.to(dtype=dtype)
        head.zero_grad(set_to_none=True)
    query = torch.randn(2, 8, 256, dtype=dtype)
    masks = torch.randn(2, 8, 13, dtype=dtype)
    padded = torch.zeros(2, 13, dtype=torch.bool)
    padded[0, -3:] = True
    masses = torch.arange(1, 14, dtype=dtype).repeat(2, 1)
    values = [h(query, masks, padded, masses) for h in heads]
    assert all(torch.equal(value, values[0]) for value in values[1:])
    original_bias = heads[-1].head[-1].bias.detach().clone()
    with torch.no_grad():
        heads[-1].head[-1].bias.add_(torch.tensor([0.25, -0.25], dtype=dtype))
        changed = [h(query, masks, padded, masses) for h in heads]
        assert all(torch.equal(changed[i], values[i]) for i in range(6))
        assert not torch.equal(changed[-1], values[-1])
        heads[-1].head[-1].bias.copy_(original_bias)
    target = torch.arange(16).reshape(2, 8).remainder(2).long()
    private_loss = torch.nn.functional.cross_entropy(values[-1].reshape(-1, 2), target.reshape(-1))
    private_loss.backward()
    assert all(p.grad is None for h in heads[:-1] for p in h.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads[-1].parameters())
    assert heads[-1].head[-1].weight.grad.abs().max() > 0
    for head in heads:
        head.zero_grad(set_to_none=True)
    losses = []
    for index, head in enumerate(heads):
        value = head(query + index / 7.0, masks + index / 9.0, padded, masses)
        losses.append(torch.nn.functional.cross_entropy(value.reshape(-1, 2), target.reshape(-1)))
    sum(losses).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for h in heads for p in h.parameters())
    assert all(h.head[-1].weight.grad.abs().max() > 0 for h in heads)
    controls.append(dict(dtype=str(dtype), copied_initial_logits_exactly_equal=True,
        disjoint_parameter_storage=True, one_head_intervention_isolated=True,
        private_loss_no_sibling_gradient=True, all_seven_heads_finite_backward=True,
        component_loss=float(sum(losses).detach())))
for head in heads:
    head.float()
model.train(False)
assert all(not m.training for m in model.modules())
model.train(True)
assert model.training and all(h.training for h in heads)
assert model.text_encoder.training
assert all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm1d))
extra = sum(p.numel() for h in heads[1:] for p in h.parameters())
parameters = sum(p.numel() for p in model.parameters())
assert parameters - extra == 164629637
assert not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'],
    implementation_test_passed=True, full_model_CPU_constructed=True,
    configured_TRAIN_data_collated=True, released_backbone_loaded=True,
    native_YAML_semantically_unchanged=True, train_eval_train_roundtrip=True,
    dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
    dataset_source_sha256=hashlib.sha256(Path(inspect.getfile(type(dataset))).read_bytes()).hexdigest(),
    points=int(batch['feats'].shape[0]), expressions=len(batch['ann_ids']),
    indicators=7, parameters=parameters, additional_parameters=extra,
    component_controls=controls, CUDA_initialized=False, native_smoke_passed=False,
    native_task_backward=False, native_sparse_forward=False, validation_used=False,
    admission=False, limitation='Synthetic component controls plus actual native YAML constructor, '
    'released pretrain and real TRAIN collate only. Full sparse task backward, actual decoder-stage '
    'execution and GPU resource checks still require an exclusive GPU; no task efficacy measured.')),
    encoding='utf-8')
