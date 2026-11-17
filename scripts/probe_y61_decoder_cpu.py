"""Run actual W2/Y61 decoder graphs on synthetic CPU inputs, not task data."""

import base64
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import gorilla
import torch
import yaml

assert P['physical_gpu'] is None and not torch.cuda.is_initialized()
torch.set_num_threads(2)
root = Path(P['root'])
stage = Path(P['output']).with_suffix('')
assert not stage.exists()
stage.mkdir()


def unpack(row, deployed=False):
    folder = stage / row['package']
    folder.mkdir()
    for name, entry in row['python_files'].items():
        raw = base64.b64decode(entry['base64'])
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        (folder / name).write_bytes(raw)
        if deployed:
            actual = (root / 'ipdn/model_paper5' / row['package'] / name).read_bytes()
            assert actual.replace(b'\r\n', b'\n') == raw.replace(b'\r\n', b'\n')
    config = stage / Path(row['config']).name
    raw = base64.b64decode(row['config_data']['base64'])
    assert hashlib.sha256(raw).hexdigest() == row['config_data']['sha256']
    config.write_bytes(raw)
    namespace = 'ipdn.model_paper5.' + row['package']
    importlib.import_module('ipdn.model_paper5')
    assert namespace not in sys.modules
    spec = importlib.util.spec_from_file_location(namespace, str(folder / '__init__.py'),
        submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = module
    spec.loader.exec_module(module)
    return gorilla.Config.fromfile(str(config)), importlib.import_module(namespace + '.dec').DEC


parent_cfg, Parent = unpack(PACKET['parent'], deployed=True)
cfg, Candidate = unpack(PACKET['variant'])
assert yaml.safe_load(base64.b64decode(PACKET['parent']['config_data']['base64'])) == yaml.safe_load(
    base64.b64decode(PACKET['variant']['config_data']['base64']))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30 and cfg.model.dec.num_layer == 6
assert cfg.model.sampling_module.num_proposal == 128 and cfg.model.dec.iter_pred


def construct(cls):
    gorilla.set_random_seed(cfg.train.seed)
    return cls(**cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)


parent = construct(Parent)
candidate = construct(Candidate)
assert set(parent.state_dict()) == set(candidate.state_dict())
assert all(torch.equal(v, candidate.state_dict()[key]) for key, v in parent.state_dict().items())
assert sum(p.numel() for p in parent.parameters()) == sum(p.numel() for p in candidate.parameters())
gorilla.set_random_seed(cfg.train.seed)
lengths = (512, 513)
total = sum(lengths)
inputs = dict(x=torch.randn(total, cfg.model.media),
    fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2, 1),
    batch_offsets=torch.tensor([0, lengths[0], total], dtype=torch.int32),
    lang_feats=torch.randn(2, 9, candidate.lang_proj.in_features),
    lang_masks=torch.ones(2, 9, dtype=torch.long),
    sp_pos=torch.randn(total, 3), feats_2d=torch.randn(total, candidate.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float() + 1)
inputs['lang_masks'][0, -2:] = 0
inputs['lang_feats'][0, -2:] = 0


def parts(output):
    return list(output['aux_outputs']) + [output]


parent.eval()
candidate.eval()
with torch.no_grad():
    baseline = parent(**inputs)
    isolated = candidate(**inputs)
assert len(parts(baseline)) == len(parts(isolated)) == 7
assert all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(isolated))
           for k in ('masks', 'scores', 'indis'))
del baseline, isolated

records = dict(parent=[], candidate=[])
norm_inputs = []


def observe_head(name):
    def hook(module, args, output):
        query, masks, valid_mask, masses = args
        assert query.requires_grad and masks.requires_grad
        assert output.shape == (2, 128, 2) and torch.isfinite(output).all()
        query.retain_grad()
        masks.retain_grad()
        records[name].append((query, masks))
    return hook


def observe_norm(module, args, output):
    norm_inputs.append(bool(args[0].requires_grad))


handles = [parent.indi_embedding.register_forward_hook(observe_head('parent')),
           candidate.indi_embedding.register_forward_hook(observe_head('candidate')),
           candidate.indi_embedding.query_norm.register_forward_hook(observe_norm)]
try:
    parent.train(True)
    candidate.train(True)
    labels = torch.arange(256).reshape(2, 128).remainder(2).long()
    def losses(output):
        return [torch.nn.functional.cross_entropy(p['indis'].reshape(-1, 2), labels.reshape(-1)) +
                p['masks'].square().mean() for p in parts(output)]

    gorilla.set_random_seed(cfg.train.seed)
    baseline = parent(**inputs)
    gorilla.set_random_seed(cfg.train.seed)
    isolated = candidate(**inputs)
    assert len(records['parent']) == len(records['candidate']) == 7
    assert norm_inputs == [True, False] * 7
    assert all(torch.equal(a[k], b[k]) for a, b in zip(parts(baseline), parts(isolated))
               for k in ('masks', 'scores', 'indis'))
    baseline_losses, isolated_losses = losses(baseline), losses(isolated)
    assert all(torch.equal(a, b) and torch.isfinite(a) for a, b in zip(baseline_losses, isolated_losses))
    sum(baseline_losses).backward()
    sum(isolated_losses).backward()
    for model in (parent, candidate):
        active = [p.grad for p in model.parameters() if p.grad is not None]
        assert active and all(torch.isfinite(g).all() for g in active)
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.indi_embedding.parameters())
        assert model.indi_embedding.head[-1].weight.grad.abs().max() > 0
    query_differences, mask_differences = [], []
    tolerance = 64 * torch.finfo(torch.float32).eps
    for (pq, pm), (cq, cm) in zip(records['parent'], records['candidate']):
        assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (pq, pm, cq, cm))
        torch.testing.assert_allclose(pm.grad, cm.grad, rtol=tolerance, atol=tolerance)
        query_differences.append(float((pq.grad - cq.grad).abs().max()))
        mask_differences.append(float((pm.grad - cm.grad).abs().max()))
    assert query_differences[-1] > 0
    for name, parameter in parent.indi_embedding.named_parameters():
        other = dict(candidate.indi_embedding.named_parameters())[name]
        torch.testing.assert_allclose(parameter.grad, other.grad, rtol=tolerance, atol=tolerance)
    component_losses = [float(value.detach()) for value in isolated_losses]
finally:
    for handle in handles:
        handle.remove()
assert not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=PACKET['variant']['id'],
    config=PACKET['variant']['config'], base_commit=PACKET['base_commit'],
    goal_sha256=PACKET['goal_sha256'], actual_decoder_class_executed=True,
    common_initial_state_bitwise_equal=True, parent_deployed_source_LF_match=True,
    all_stage_forward_bitwise_equal=True, seven_confidence_calls=len(records['candidate']),
    query_norm_input_requires_grad=norm_inputs, all_stage_backward_finite=True,
    direct_mask_gradients_preserved=True, final_query_gradient_changed=query_differences[-1] > 0,
    stage_query_gradient_max_differences=query_differences,
    stage_mask_gradient_max_differences=mask_differences,
    head_parameter_gradients_preserved=True, additional_parameters=0,
    component_losses=component_losses, CUDA_initialized=False, native_data_loaded=False,
    native_task_backward=False, native_smoke_passed=False, admission=False,
    limitation='Actual seven-stage DEC on synthetic CPU inputs and surrogate CE/mask-square losses, '
        'not native sparse-backbone TRAIN data or task criterion/optimizer steps. The mask-output '
        'gradients at indicator inputs are preserved; gradients into earlier decoder/backbone '
        'parameters can differ as intended. No efficacy, GPU cost or hardware stability measured.')),
    encoding='utf-8')
