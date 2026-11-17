"""Physical Y41 native two-step entry, checkpoint resume and cached controls."""
import copy
import gc
from pathlib import Path
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain, train
from probe_adares_y15_resume import rng, restore, differences, cpu_state
torch.set_num_threads(4)
gorilla.set_random_seed(1999)
config = 'configs_song/model_paper5/adares_y41_boundary_probability_res.yaml'
parent = 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
cfg = gorilla.Config.fromfile(config)
logger = get_root_logger()
dataset = build_dataset(cfg.data.train.copy(), logger)
indices, seen = ([], set())
for index, chunk in enumerate(dataset.scene_inputs):
    if chunk[0]['scene_id'] not in seen:
        seen.add(chunk[0]['scene_id'])
        indices.append(index)
    if len(indices) == 4:
        break
batches = [dataset.collate_fn([dataset[i] for i in indices[j:j + 2]]) for j in (0, 2)]
model = resolve_model_class(config)(**cfg.model).cuda()
load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
assert len(list(model.dec.boundary_readout.parameters())) == 4
assert all((not hasattr(layer, 'postnorm') for layer in model.dec.qla_layers))
opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
parameters = sum((p.numel() for p in model.parameters()))
records = []
original_step = opt.step

def audit_step(*args, **kwargs):
    norms = {name: p for name, p in model.dec.named_parameters() if '.boundary_readout.' in '.' + name}
    assert len(norms) == 4
    assert all((p.grad is not None and torch.isfinite(p.grad).all() and (p.grad.abs().max() > 0) for p in norms.values()))
    decoder = [p.grad for p in model.dec.parameters() if p.grad is not None]
    assert all((torch.isfinite(g).all() for g in decoder))
    records.append(dict(norm_gradients={name: float(p.grad.norm()) for name, p in norms.items()}, decoder_gradient_tensors=len(decoder)))
    return original_step(*args, **kwargs)

class Batches:
    sampler = None

    def __len__(self):
        return len(batches)

    def __iter__(self):
        return iter(batches)

class Writer:

    def add_scalar(self, *args, **kwargs):
        pass
opt.step = audit_step
cfg.dist = False
before = cpu_state(model)
train(1, model, Batches(), opt, sched, cfg, logger, Writer())
assert len(records) == 2
changed = differences(before, cpu_state(model))
assert not changed['equal']
affine_changes = {name: float((value - before[name]).abs().max()) for name, value in cpu_state(model).items() if '.boundary_readout.' in name}
assert len(affine_changes) == 4 and all((v > 0 for v in affine_changes.values()))
del before
opt.step = original_step
checkpoint = str(Path(P['output']).with_suffix('.pth'))
assert not Path(checkpoint).exists()
gorilla.save_checkpoint(model, checkpoint, opt, sched, dict(epoch=1, best_miou=0.0))
saved = torch.load(checkpoint, map_location='cpu')
state = rng()

def step():
    model.train()
    opt.zero_grad(set_to_none=True)
    loss, _ = model(batches[0], mode='loss')
    assert torch.isfinite(loss)
    loss.backward()
    assert all((torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    opt.step()
    return float(loss)
continued = step()
final = cpu_state(model)
final_opt = copy.deepcopy(opt.state_dict())
del model, opt, sched, original_step
gc.collect()
torch.cuda.empty_cache()
model = resolve_model_class(config)(**cfg.model).cuda()
opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
meta = gorilla.resume(model=model, filename=checkpoint, optimizer=opt, scheduler=sched, strict=False, map_location='cpu')
allowed = {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'}
resume = dict(model=differences({k: v for k, v in saved['model'].items() if k not in allowed}, cpu_state(model)), optimizer=differences(saved['optimizer'], opt.state_dict()), scheduler=differences(saved['scheduler'], sched.state_dict()), epoch=int(meta['epoch']))
assert all((resume[k]['equal'] for k in ('model', 'optimizer', 'scheduler')))
restore(state)
resumed = step()
continuation = dict(loss_difference=resumed - continued, model=differences(final, cpu_state(model)), optimizer=differences(final_opt, opt.state_dict()))
del final, final_opt, saved, model, opt, sched, batches
gc.collect()
torch.cuda.empty_cache()
dc = cfg.data.train.copy()
dc.training = False
dc.aug = False
dc.with_elastic = False
evaluation = build_dataset(dc, logger)
model = resolve_model_class(config)(**cfg.model).cuda()
ck = torch.load('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth', map_location='cpu')
loaded = model.load_state_dict(ck['model'], strict=False)
assert set(loaded.missing_keys) == {k for k in model.state_dict() if '.boundary_readout.' in k}
assert not set(loaded.unexpected_keys) - allowed
model.eval()
capture, cached, inference = ({}, [], [])

def prehook(module, args):
    capture['args'] = tuple((v.detach().clone() if torch.is_tensor(v) else v for v in args))
hook = model.dec.register_forward_pre_hook(prehook)
with torch.no_grad():
    for index in indices:
        batch = evaluation.collate_fn([evaluation[index]])
        capture.clear()
        gorilla.set_random_seed(1999)
        native = model(batch, mode='predict')
        assert all((v.shape[0] == 128 for v in native['indi']))
        assert all((v.numel() == len(batch['superpoints']) for v in native['pred_pmask']))
        args = capture['args']
        hook.remove()
        connected = model.dec(*args)
        repeated = model.dec(*args)
        h = model.dec.boundary_readout.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
        disabled = model.dec(*args)
        h.remove()
        cached.append((args, {k: disabled[k].detach().clone() for k in ('masks', 'indis', 'sample_inds')}))
        inference.append(dict(scan_id=batch['scan_ids'][0], enabled_repeat_mask_max=float((connected['masks'] - repeated['masks']).abs().max()), enabled_disabled_mask_max=float((connected['masks'] - disabled['masks']).abs().max()), enabled_finite=bool(torch.isfinite(connected['masks']).all())))
        hook = model.dec.register_forward_pre_hook(prehook)
hook.remove()
del model, native, connected, repeated, disabled
gc.collect()
torch.cuda.empty_cache()
model = resolve_model_class(parent)(**cfg.model).cuda()
loaded = model.load_state_dict(ck['model'], strict=False)
assert not loaded.missing_keys and (not set(loaded.unexpected_keys) - allowed)
model.eval()
with torch.no_grad():
    for row, (args, disabled) in zip(inference, cached):
        original = model.dec(*args)
        row.update(disabled_parent_mask_max=float((disabled['masks'] - original['masks']).abs().max()), disabled_parent_indicator_max=float((disabled['indis'] - original['indis']).abs().max()), disabled_parent_sample_mismatches=int((disabled['sample_inds'] != original['sample_inds']).sum()))
assert all((r['disabled_parent_mask_max'] == r['disabled_parent_indicator_max'] == r['disabled_parent_sample_mismatches'] == r['enabled_repeat_mask_max'] == 0 for r in inference))
result = dict(complete=True, id='AdaRES-Y41', parameters=parameters, added_parameters=parameters - 164629637, native_optimizer_steps=2, native_gradient_records=records, peer_parameter_changes=affine_changes, model_change=changed, resume_before_continuation=resume, continuation=continuation, cached_inference_rows=inference, inference_queries=128, inference_selected=1, gpu_precheck=dict(memory_mib=used, utilization=util), limitations=['Native entry two steps/resume and cached four-scan inference, not production training.', 'Y41 changes TRAIN/EVAL confidence only; parent identity with peer path disabled.', 'Exact model/AdamW/PolyLR restoration before continuation, next-step numerical differences retained.', 'No whole-loader/RNG trajectory, global deterministic inference or efficacy claim.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
