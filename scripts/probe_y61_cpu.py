"""Validate the native CPU model, real TRAIN loading and exact gradient branch."""

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
assert yaml.safe_load(config_path.read_text()) == yaml.safe_load((root / PAYLOAD['parent_config']).read_text())
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
model.train(True)
assert model.training and model.text_encoder.training
assert all(not p.requires_grad for p in model.text_encoder.parameters())
assert all(m.training for m in model.modules() if isinstance(m, nn.BatchNorm1d))
parent_dec = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
controls = []

for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    candidate = copy.deepcopy(model.dec.indi_embedding).to(dtype=dtype)
    parent = parent_dec.QueryCompetitionIndicator(256).to(dtype=dtype)
    parent.load_state_dict(candidate.state_dict(), strict=True)
    query0 = torch.randn(2, 8, 256, dtype=dtype)
    masks0 = torch.randn(2, 8, 13, dtype=dtype)
    padded = torch.zeros(2, 13, dtype=torch.bool)
    padded[0, -3:] = True
    masses = torch.arange(1, 14, dtype=dtype).repeat(2, 1)

    def evaluate(head):
        head.zero_grad(set_to_none=True)
        query = query0.detach().clone().requires_grad_(True)
        masks = masks0.detach().clone().requires_grad_(True)
        logits = head(query, masks, padded, masses)
        loss = torch.nn.functional.cross_entropy(logits[:, 0], torch.tensor([0, 1]))
        loss.backward()
        return dict(logits=logits.detach(), query=query.grad.detach().clone(),
                    masks=masks.grad.detach().clone(),
                    parameters={n: p.grad.detach().clone() for n, p in head.named_parameters()})

    baseline = evaluate(parent)
    isolated = evaluate(candidate)
    assert torch.equal(baseline['logits'], isolated['logits'])
    assert torch.equal(baseline['masks'], isolated['masks'])
    assert torch.equal(baseline['query'][:, 0], isolated['query'][:, 0])
    assert baseline['query'][:, 1:].abs().max() > 0
    assert isolated['query'][:, 1:].count_nonzero() == 0
    assert isolated['query'][:, 0].abs().max() > 0
    assert isolated['masks'][:, 1:].abs().max() > 0
    tolerance = 64 * torch.finfo(dtype).eps
    for key, grad in baseline['parameters'].items():
        assert torch.isfinite(isolated['parameters'][key]).all()
        torch.testing.assert_allclose(grad, isolated['parameters'][key], rtol=tolerance, atol=tolerance)
    assert baseline['masks'][0, :, -3:].count_nonzero() == 0
    permutation = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4])
    with torch.no_grad():
        permuted = candidate(query0[:, permutation], masks0[:, permutation], padded, masses)
        torch.testing.assert_allclose(permuted, isolated['logits'][:, permutation], rtol=tolerance, atol=tolerance)
        repeat = candidate(query0, masks0, padded, masses)
        assert torch.equal(repeat, isolated['logits'])
    controls.append(dict(dtype=str(dtype), forward_bitwise_equal=True,
        focal_query_gradient_bitwise_equal=True, mask_gradient_bitwise_equal=True,
        parent_peer_value_gradient_max=float(baseline['query'][:, 1:].abs().max()),
        candidate_peer_value_gradient_max=float(isolated['query'][:, 1:].abs().max()),
        mask_peer_gradient_max=float(isolated['masks'][:, 1:].abs().max()),
        trainable_affine_and_head_gradients_preserved=True, query_permutation_equivariant=True,
        repeated_forward_exact=True, padding_gradient_zero=True,
        additional_parameters=0, tolerance=tolerance))
model.eval()
assert all(not m.training for m in model.modules())
model.train(True)
assert model.training and model.text_encoder.training
parameters = sum(p.numel() for p in model.parameters())
assert parameters == 164629637
assert not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'],
    implementation_test_passed=True, full_model_CPU_constructed=True,
    configured_TRAIN_data_collated=True, released_backbone_loaded=True,
    native_YAML_semantically_unchanged=True, train_eval_train_roundtrip=True,
    dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
    dataset_source_sha256=hashlib.sha256(Path(inspect.getfile(type(dataset))).read_bytes()).hexdigest(),
    points=int(batch['feats'].shape[0]), expressions=len(batch['ann_ids']), parameters=parameters,
    component_controls=controls, CUDA_initialized=False, native_smoke_passed=False,
    native_task_backward=False, native_sparse_forward=False, validation_used=False,
    limitation='Real TRAIN collate and full native YAML CPU construction with synthetic indicator '
        'controls only. Peer query gradients here treat masks as independent leaves; indirect paths '
        'through native mask generation and other decoder attention remain present. This is not '
        'native sparse task backward, GPU resource proof or evidence of trained efficacy.')),
    encoding='utf-8')
