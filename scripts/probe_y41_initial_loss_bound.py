import hashlib
from pathlib import Path
manifest=[{'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/__init__.py', 'sha256': '8014a039f2388c8ca57692438985474724a17785186036e5208771b856072722'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/backbone.py', 'sha256': '1eb4ac3483961e708c23b0abad396c181dbf38c56fcca559d88b9622af46bafd'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/dec.py', 'sha256': '33d27cdfd93c39f2b9889d6bf7f3abad3acd4c403dec5a14c8da0f16f6852bd9'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/ipdn.py', 'sha256': '63c8f30bda20300d98d3d9a8a6478fc1a841cbaa3b334ee71688711e06375d97'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/loss.py', 'sha256': '4aa2818948ebfd9fe416b3b310a053384508ef92c3b67831833fece560523084'}, {'path': 'AdaRES/ipdn/model_paper5/adares_y41_boundary_probability/sample_model.py', 'sha256': '03fd683acc6d49d9b53e634b42bd150a3259a1a15c7dc8fe2badf115870a345b'}, {'path': 'AdaRES/configs_song/model_paper5/adares_y41_boundary_probability_res.yaml', 'sha256': '1a170434f63da98e10f59c5b17473cde5cf3ff6be0e4381e2771f322853ffa03'}]
root=Path('/bak_home/songcl/Project/AdaRES_project')
assert all(hashlib.sha256((root / row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest)
preserved_cmd=Path('/proc/102267/cmdline').read_bytes().split(b'\0')
assert b'configs_song/model_paper5/adares_y39_point_distribution_res.yaml' in preserved_cmd
assert preserved_cmd[preserved_cmd.index(b'--gpu_ids')+1]==b'1'
assert P['physical_gpu']==0
"""Investigate adverse initial loss without selecting settings or deploying weights."""
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
config = 'configs_song/model_paper5/adares_y41_boundary_probability_res.yaml'
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
    q, masks, pad, mass, tokens, coords = args
    if mode == 'rolled':
        return (q, masks.roll(1, 1), pad, mass, tokens, coords)
    if mode == 'wrong_pairs':
        changed = coords.clone()
        for b in range(coords.shape[0]):
            count = int((~pad[b]).sum())
            changed[b, :count] = coords[b, :count].roll(1, 0)
        return (q, masks, pad, mass, tokens, changed)
results = {}
for phase in ('trained', 'released_init'):
    gorilla.set_random_seed(1999)
    model = resolve_model_class(config)(**cfg.model).cuda()
    if phase == 'trained':
        ck = torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', map_location='cpu')
        loaded = model.load_state_dict(ck['model'], strict=False)
        assert set(loaded.missing_keys) == {k for k in model.state_dict() if '.boundary_readout.' in k}
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
    new = {k: p for k, p in model.named_parameters() if '.boundary_readout.' in k}
    assert len(new) == 4
    for k, p in model.named_parameters():
        p.requires_grad_(k in new)
    initial = copy.deepcopy(model.dec.boundary_readout.state_dict())
    handle = model.dec.boundary_readout.register_forward_pre_hook(shape_input)
    outcomes = {}

    def losses():
        return [model.criterion(model.dec(*d), *c)[0] for d, c in captured]
    for mode in ('enabled', 'rolled', 'wrong_pairs'):
        model.dec.boundary_readout.load_state_dict(initial)
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
result = dict(complete=True, id='AdaRES-Y41', summary=results, frozen_native_optimizer=dict(type='AdamW', lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, grad_clip=cfg.train.grad_clip), temporary_new_path_steps=2, source_weights_or_production_checkpoint_changed=False, gpu_precheck=dict(memory_mib=used, utilization=util), limitations=['Fixed native TRAIN8/init4 encoded inputs; only new4 tensors optimized temporarily.', 'All correct/rolled/wrong_pairs shape outcomes retained, no setting selected.', 'Learnability is not geometry necessity, generalization or removal of original adverse losses.', 'Production starts from unchanged released backbone and new initialization; diagnostic weights not deployed.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')

assert Path('/proc/102267/cmdline').read_bytes().split(b'\0')==preserved_cmd
assert all(hashlib.sha256((root / row['path']).read_bytes()).hexdigest()==row['sha256'] for row in manifest)
