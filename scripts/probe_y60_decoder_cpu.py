"""Exercise actual W2/Y60 decoder graphs on CPU fixtures, not sparse task data."""

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


def unpack(row):
    folder = stage / row['package']
    folder.mkdir()
    for name, entry in row['python_files'].items():
        raw = base64.b64decode(entry['base64'])
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        (folder / name).write_bytes(raw)
        if row['package'] == PACKET['parent']['package']:
            deployed = root / 'ipdn/model_paper5' / row['package'] / name
            assert deployed.read_bytes().replace(b'\r\n', b'\n') == raw.replace(b'\r\n', b'\n')
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


parent_cfg, Parent = unpack(PACKET['parent'])
cfg, Candidate = unpack(PACKET['frozen']['variant'])
assert yaml.safe_load(base64.b64decode(PACKET['parent']['config_data']['base64'])) == yaml.safe_load(
    base64.b64decode(PACKET['frozen']['variant']['config_data']['base64']))
assert cfg.train.seed == 1999 and cfg.train.epochs == 30 and cfg.model.dec.num_layer == 6
assert cfg.model.sampling_module.num_proposal == 128 and cfg.model.dec.iter_pred


def construct(cls):
    gorilla.set_random_seed(cfg.train.seed)
    return cls(**cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)


parent = construct(Parent)
candidate = construct(Candidate)
parent_state = parent.state_dict()
candidate_state = candidate.state_dict()
shared_keys = set(parent_state)
assert shared_keys < set(candidate_state)
assert all(torch.equal(parent_state[key], candidate_state[key]) for key in shared_keys)
extra_keys = set(candidate_state) - shared_keys
assert all(key.startswith('refinement_indicators.') for key in extra_keys)
heads = [candidate.indi_embedding] + list(candidate.refinement_indicators)
assert len(heads) == 7
assert len({p.data_ptr() for h in heads for p in h.parameters()}) == sum(
    len(list(h.parameters())) for h in heads)
gorilla.set_random_seed(cfg.train.seed)
lengths = (512, 513)
total = sum(lengths)
offsets = torch.tensor([0, lengths[0], total], dtype=torch.int32)
inputs = dict(x=torch.randn(total, cfg.model.media),
    fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2, 1), batch_offsets=offsets,
    lang_feats=torch.randn(2, 9, candidate.lang_proj.in_features), lang_masks=torch.ones(2, 9, dtype=torch.long),
    sp_pos=torch.randn(total, 3), feats_2d=torch.randn(total, candidate.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float() + 1)
inputs['lang_masks'][0, -2:] = 0
inputs['lang_feats'][0, -2:] = 0
calls = [0] * len(heads)


def hook_for(index):
    def observed(module, args, output):
        assert output.shape == (2, 128, 2) and torch.isfinite(output).all()
        calls[index] += 1
    return observed


hooks = [head.register_forward_hook(hook_for(i)) for i, head in enumerate(heads)]
try:
    parent.eval()
    candidate.eval()
    with torch.no_grad():
        baseline = parent(**inputs)
        cloned = candidate(**inputs)
    assert calls == [1] * 7
    for key in ('masks', 'indis', 'scores'):
        assert torch.equal(baseline[key], cloned[key])
    assert len(baseline['aux_outputs']) == len(cloned['aux_outputs']) == 6
    for left, right in zip(baseline['aux_outputs'], cloned['aux_outputs']):
        for key in ('masks', 'indis', 'scores'):
            assert torch.equal(left[key], right[key])
    del baseline, cloned
    calls[:] = [0] * 7
    candidate.train(True)
    parent.train(True)
    labels = torch.arange(256).reshape(2, 128).remainder(2).long()

    def stage_losses(output):
        outputs = list(output['aux_outputs']) + [output]
        return [torch.nn.functional.cross_entropy(part['indis'].reshape(-1, 2), labels.reshape(-1))
                for part in outputs]

    output = candidate(**inputs)
    assert calls == [1] * 7
    losses = stage_losses(output)
    assert len(losses) == 7 and all(torch.isfinite(v) for v in losses)
    sum(losses).backward()
    head_norms = []
    for head in heads:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert head.head[-1].weight.grad.abs().max() > 0
        head_norms.append(float(head.head[-1].weight.grad.norm()))
    active = [p.grad for p in candidate.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    private_sum = sum(head.head[-1].weight.grad.detach().clone() for head in heads)
    parent_output = parent(**inputs)
    parent_losses = stage_losses(parent_output)
    sum(parent_losses).backward()
    shared = parent.indi_embedding.head[-1].weight.grad
    assert torch.isfinite(shared).all()
    accumulation_error = float((shared - private_sum).abs().max())
    final_vs_shared = float((shared - heads[-1].head[-1].weight.grad).norm())
    component_losses = [float(value.detach()) for value in losses]
    del output, parent_output, losses, parent_losses
    candidate.zero_grad(set_to_none=True)
    calls[:] = [0] * 7
    final_output = candidate(**inputs)
    final_loss = stage_losses(final_output)[-1]
    final_loss.backward()
    assert calls == [1] * 7
    assert all(p.grad is None for head in heads[:-1] for p in head.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads[-1].parameters())
    assert heads[-1].head[-1].weight.grad.abs().max() > 0
finally:
    for handle in hooks:
        handle.remove()
assert not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y60',
    code_commit=PACKET['frozen']['code_commit'], config=PACKET['frozen']['variant']['config'],
    goal_sha256=PACKET['frozen']['goal_sha256'], actual_decoder_class_executed=True,
    parent_deployed_source_LF_match=True, common_decoder_initial_state_bitwise_equal=True,
    initial_all_stage_masks_scores_confidence_bitwise_equal=True,
    actual_seven_stage_head_calls=calls, all_stage_component_backward_finite=True,
    final_only_loss_no_sibling_head_gradient=True, stage_final_weight_gradient_norms=head_norms,
    parent_shared_vs_summed_private_gradient_max_error=accumulation_error,
    parent_shared_vs_final_private_gradient_norm_difference=final_vs_shared,
    component_losses=component_losses, CUDA_initialized=False, native_data_loaded=False,
    native_task_backward=False, native_smoke_passed=False, admission=False,
    limitation='Full native decoder graph on synthetic CPU tensors using unchanged native YAML dimensions; '
    'not sparse-backbone TRAIN data, task criterion, optimizer steps, GPU memory or trained efficacy. '
    'Gradient differences describe head sharing on this fixture and are not evidence of harmful interference.')),
    encoding='utf-8')
