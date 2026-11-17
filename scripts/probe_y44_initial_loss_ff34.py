from pathlib import Path
import hashlib
assert P['physical_gpu'] == 1
assert Path('.codex_source_commit').read_text().strip() == '6fc7d4e46b6c26a11fede7f12d4e79b3471cc1cc'
_terminal = json.loads(Path('/tmp/adares_y42_terminal_ff34_' + P['token'] + '.json').read_text())
_y43_command = Path('/proc/13155/cmdline').read_bytes()
assert b'configs_song/model_paper5/adares_y43_point_asymmetry_res.yaml' in _y43_command
assert _y43_command.split(b'\x00')[_y43_command.split(b'\x00').index(b'--gpu_ids') + 1] == b'0'
_protected = dict(_terminal['source_hashes'], **_terminal['protected_source_hashes'])

def check_originals():
    assert Path('/proc/13155/cmdline').read_bytes() == _y43_command
    assert all((hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in _protected.items()))
check_originals()
'Investigate adverse initial loss without selecting settings or deploying weights.'
import copy
import gc
from pathlib import Path
import statistics
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain
torch.set_num_threads(4)
config = 'configs_song/model_paper5/adares_y44_coordinate_feature_res.yaml'
cfg = gorilla.Config.fromfile(config)
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), get_root_logger())
indices, seen = ([], set())
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]['scene_id'] not in seen:
        seen.add(chunk[0]['scene_id'])
        indices.append(index)
    if len(indices) == 8:
        break
mode, cache = ('enabled', {})

def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)((clone(v) for v in value))
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    return value

def capture_dec(module, args):
    cache['dec'] = clone(args)

def capture_criterion(module, args):
    cache['criterion'] = clone(args[1:])

def shape_input(module, args):
    q, masks, pad, mass, third = args
    if mode == 'rolled':
        return (q, masks.roll(1, 1), pad, mass, third)
    if mode == 'wrong_third':
        changed = third.clone()
        for b in range(third.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = third[b, :count].roll(1, 0)
        return (q, masks, pad, mass, changed)
results = {}
for phase in ('trained', 'released_init'):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    if phase == 'trained':
        ck = torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', map_location='cpu')
        loaded = model.load_state_dict(ck['model'], strict=False)
        assert set(loaded.missing_keys) == {k for k in model.state_dict() if '.point_readout.' in k}
        del ck
    else:
        load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
    model.train()
    captured = []
    hooks = [model.dec.register_forward_pre_hook(capture_dec), model.criterion.register_forward_pre_hook(capture_criterion)]
    with torch.no_grad():
        for index in indices[:8 if phase == 'trained' else 4]:
            cache.clear()
            model(dataset.collate_fn([dataset[index]]), mode='loss')
            captured.append((cache['dec'], cache['criterion']))
    for hook in hooks:
        hook.remove()
    new = {k: p for k, p in model.named_parameters() if '.point_readout.' in k}
    assert len(new) == 8
    for k, p in model.named_parameters():
        p.requires_grad_(k in new)
    initial = copy.deepcopy(model.dec.point_readout.state_dict())
    handle = model.dec.point_readout.register_forward_pre_hook(shape_input)
    outcomes = {}

    def losses():
        return [model.criterion(model.dec(*d), *c)[0] for d, c in captured]
    for mode in ('enabled', 'rolled', 'wrong_third'):
        model.dec.point_readout.load_state_dict(initial)
        optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
        readings, gradients = ([], [])
        with torch.no_grad():
            readings.append([float(v) for v in losses()])
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            mean_loss = torch.stack(losses()).mean()
            mean_loss.backward()
            norms = {k: float(p.grad.norm()) for k, p in new.items()}
            assert all((v > 0 for v in norms.values()))
            assert all((torch.isfinite(p.grad).all() for p in new.values()))
            gradients.append(norms)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            with torch.no_grad():
                readings.append([float(v) for v in losses()])
            del mean_loss
        outcomes[mode] = dict(per_scene_losses=readings, mean_losses=[statistics.mean(r) for r in readings], gradients=gradients, mean_change=statistics.mean(readings[-1]) - statistics.mean(readings[0]))
        del optimizer
    results[phase] = outcomes
    print(json.dumps(dict(phase=phase, changes={k: v['mean_change'] for k, v in outcomes.items()})), flush=True)
    handle.remove()
    del captured, initial, model, new
    cache.clear()
    gc.collect()
    torch.cuda.empty_cache()
result = dict(complete=True, id='AdaRES-Y44', summary=results, frozen_native_optimizer=dict(type='AdamW', lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, grad_clip=cfg.train.grad_clip), temporary_new_path_steps=2, source_weights_or_production_checkpoint_changed=False, gpu_precheck=dict(memory_mib=used, utilization=util), limitations=['Fixed native TRAIN8/init4 encoded inputs; only new8 tensors optimized temporarily.', 'All correct/rolled/wrong_third shape outcomes retained, no setting selected.', 'Learnability is not geometry necessity, generalization or removal of original adverse losses.', 'Production starts from unchanged released backbone and new initialization; diagnostic weights not deployed.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
check_originals()
