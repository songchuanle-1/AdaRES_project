"""Independent Y48 source audit, released-init native training and resources."""
import ast
import copy
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time
import gorilla
import torch
from ipdn.dataset import build_dataset
from ipdn.utils import get_root_logger
from train import resolve_model_class, load_shape_compatible_pretrain, train
from probe_adares_y15_resume import cpu_state, differences
assert P['physical_gpu'] == 1
torch.set_num_threads(4)
root = Path(P['root'])
base = root / 'ipdn/model_paper5'
parent_path = base / 'adares_w2_pointmass_competition'
candidate_path = base / 'adares_y48_mask_equivalence'
protected = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in parent_path.glob('*.py')}
sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in candidate_path.glob('*.py')}
assert sources['ipdn/model_paper5/adares_y48_mask_equivalence/loss.py'] == 'db6509d329002674b0d9cbaf84f43151948f3afca12be11f45d0cf148611efed'
for name in ('__init__.py', 'backbone.py', 'dec.py', 'ipdn.py', 'sample_model.py'):
    assert (candidate_path / name).read_bytes() == (parent_path / name).read_bytes()
tree = ast.parse((candidate_path / 'loss.py').read_text())
tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name == 'close_nonempty_mask_indices')]
criterion = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Criterion'))
for method in criterion.body:
    if not isinstance(method, ast.FunctionDef):
        continue
    if method.name == 'loss_indi':
        assert [arg.arg for arg in method.args.args[-2:]] == ['mask_pred', 'mask_padding']
        method.args.args = method.args.args[:-2]
        method.args.defaults = method.args.defaults[:-2]
        assert isinstance(method.body[0], ast.If)
        method.body = method.body[1:]
    if method.name == 'forward':
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and (node.func.attr == 'loss_indi')]
        assert len(calls) == 1 and len(calls[0].args) == 6
        calls[0].args = calls[0].args[:-2]
assert ast.dump(tree) == ast.dump(ast.parse((parent_path / 'loss.py').read_text()))
config = 'configs_song/model_paper5/adares_y48_mask_equivalence_res.yaml'
parent_config = 'configs_song/model_paper5/adares_w2_pointmass_competition_res.yaml'
assert (root / config).read_bytes() == (root / parent_config).read_bytes()
cfg = gorilla.Config.fromfile(config)
logger = get_root_logger()
module = importlib.import_module('ipdn.model_paper5.adares_y48_mask_equivalence.loss')
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.loss')
actual = module.Criterion(**cfg.model.criterion)
reference = parent_module.Criterion(**cfg.model.criterion)
assert actual.eos_coef == reference.eos_coef == 0.1
cpu_rows = []
for seed in range(32):
    torch.manual_seed(1999 + seed)
    masks = torch.full((1, 128, 16), -1.0)
    masks[:, :32, :8] = 1
    masks[:, 32:64, 8:] = 1
    masks[:, 96:] = 1
    padding = torch.zeros(1, 16, dtype=torch.bool)
    matched = [(torch.tensor([0, 40, 70]), torch.zeros(3, dtype=torch.long))]
    logits = torch.randn(1, 128, 2, requires_grad=True)
    normalizer = torch.tensor(1.0)
    closed = module.close_nonempty_mask_indices(masks, padding, matched)
    a = actual.loss_indi(logits, [], matched, normalizer, masks, padding)
    b = reference.loss_indi(logits, [], closed, normalizer)
    assert torch.equal(a, b)
    ga = torch.autograd.grad(a, logits, retain_graph=True)[0]
    gb = torch.autograd.grad(b, logits)[0]
    assert torch.equal(ga, gb)
    assert torch.equal(actual.loss_indi(logits, [], matched, normalizer), reference.loss_indi(logits, [], matched, normalizer))
    cpu_rows.append(dict(seed=seed, actual_production_loss_and_gradient_exact=True, omitted_masks_parent_loss_exact=True))
del actual, reference
gorilla.set_random_seed(1999)
dataset = build_dataset(cfg.data.train.copy(), logger)
indices = [0, 44, 60, 101]
assert len({dataset.scene_inputs[i][0]['scene_id'] for i in indices}) == 4
batches = [dataset.collate_fn([dataset[i] for i in indices[j:j + 2]]) for j in (0, 2)]
model = resolve_model_class(config)(**cfg.model).cuda()
load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
parameters = sum((p.numel() for p in model.parameters()))
assert parameters == 164629637
optimizer = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
scheduler = gorilla.build_lr_scheduler(optimizer, copy.deepcopy(cfg.lr_scheduler))
original_closure = module.close_nonempty_mask_indices
closure_records = []

def audit_closure(masks, padding, matched):
    result = original_closure(masks, padding, matched)
    before = torch.zeros(masks.shape[:2], device=masks.device, dtype=torch.bool)
    after = before.clone()
    for b, ((left, _), (new_left, _)) in enumerate(zip(matched, result)):
        before[b, left] = True
        after[b, new_left] = True
        binary = masks[b, :, ~padding[b]].detach().gt(0)
        assert torch.equal(before[b][~binary.any(-1)], after[b][~binary.any(-1)])
    assert (before | after).equal(after)
    closure_records.append(dict(original_positive_queries=int(before.sum()), new_positive_queries=int((after & ~before).sum()), empty_targets_preserved=True))
    return result
module.close_nonempty_mask_indices = audit_closure
steps = []
original_step = optimizer.step

def audited_step(*args, **kwargs):
    gradients = {name: float(value.grad.norm()) for name, value in model.dec.indi_embedding.named_parameters() if value.grad is not None}
    assert gradients and any((value > 0 for value in gradients.values()))
    assert all((torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None))
    steps.append(dict(indicator_gradients=gradients))
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
cfg.dist = False
before = {name: value.detach().cpu().clone() for name, value in model.dec.indi_embedding.named_parameters()}
optimizer.step = audited_step
train(1, model, Batches(), optimizer, scheduler, cfg, logger, Writer())
assert len(steps) == 2 and len(closure_records) == 2
assert sum((row['new_positive_queries'] for row in closure_records)) > 0
updates = {name: float((value.detach().cpu() - before[name]).abs().max()) for name, value in model.dec.indi_embedding.named_parameters()}
assert any((value > 0 for value in updates.values()))
optimizer.step = original_step
module.close_nonempty_mask_indices = original_closure
checkpoint_path = str(Path(P['output']).with_suffix('.pth'))
assert not Path(checkpoint_path).exists()
gorilla.save_checkpoint(model, checkpoint_path, optimizer, scheduler, dict(epoch=1, best_miou=0.0))
saved = torch.load(checkpoint_path, map_location='cpu')
assert all((key in saved for key in ('model', 'optimizer', 'scheduler', 'meta')))
saved_state = cpu_state(model)
model.eval()
captured = {}

def clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        return type(value)((clone(item) for item in value))
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    return value
prediction_cfg = cfg.data.train.copy()
prediction_cfg.training = prediction_cfg.aug = prediction_cfg.with_elastic = False
prediction_dataset = build_dataset(prediction_cfg, logger)
prediction_batches = []
for index in indices:
    gorilla.set_random_seed(1999)
    prediction_batches.append(prediction_dataset.collate_fn([prediction_dataset[index]]))
handle = model.dec.register_forward_pre_hook(lambda mod, inputs: captured.update(inputs=clone(inputs)))
physical = []
inference_readings = []
with torch.no_grad():
    for batch_index, batch in enumerate(prediction_batches):
        for _ in range(2):
            gorilla.set_random_seed(1999)
            model(batch, mode='predict')
        readings = []
        for _ in range(3):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            gorilla.set_random_seed(1999)
            started = time.monotonic()
            model(batch, mode='predict')
            torch.cuda.synchronize()
            readings.append(dict(seconds=time.monotonic() - started, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30))
        inference_readings.append(dict(manifest_indices=indices[batch_index:batch_index + 1], scene_batch_size=1, readings=readings))
        gorilla.set_random_seed(1999)
        prediction = model(batch, mode='predict')
        assert all((value.shape[0] == 128 for value in prediction['indi']))
        args = clone(captured['inputs'])
        output = model.dec(*args)
        physical.append((args, {key: output[key].detach().cpu().clone() for key in ('masks', 'indis', 'sample_inds')}))
handle.remove()
del optimizer, scheduler, original_step, model, batches, captured, prediction
gc.collect()
torch.cuda.empty_cache()
restored = resolve_model_class(config)(**cfg.model).cuda()
restored.load_state_dict(saved['model'], strict=True)
opt = gorilla.build_optimizer(restored, copy.deepcopy(cfg.optimizer))
sched = gorilla.build_lr_scheduler(opt, copy.deepcopy(cfg.lr_scheduler))
opt.load_state_dict(saved['optimizer'])
sched.load_state_dict(saved['scheduler'])
assert differences(saved_state, cpu_state(restored))['equal']
assert differences(saved['optimizer'], opt.state_dict())['equal']
assert differences(saved['scheduler'], sched.state_dict())['equal']
del restored, opt, sched, saved
gc.collect()
torch.cuda.empty_cache()
parent = resolve_model_class(parent_config)(**cfg.model).cuda()
parent.load_state_dict(saved_state, strict=True)
parent.eval()
with torch.no_grad():
    for args, expected in physical:
        output = parent.dec(*args)
        assert all((torch.equal(value, output[key].cpu()) for key, value in expected.items()))
del parent, physical, saved_state
gc.collect()
torch.cuda.empty_cache()
gorilla.set_random_seed(1999)
model = resolve_model_class(config)(**cfg.model).cuda()
load_shape_compatible_pretrain(model, cfg.train.pretrain, None)
model.train()
opt = gorilla.build_optimizer(model, copy.deepcopy(cfg.optimizer))
batch = dataset.collate_fn([dataset[728], dataset[6780]])
assert batch['scan_ids'] == ['scene0054_00', 'scene0497_00']
measurements = []
for iteration in range(5):
    opt.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    loss, _ = model(batch, mode='loss')
    assert torch.isfinite(loss)
    loss.backward()
    assert all((torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None))
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    opt.step()
    torch.cuda.synchronize()
    if iteration >= 2:
        measurements.append(dict(seconds=time.monotonic() - started, loss=float(loss), peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30))
    del loss
assert max((row['peak_reserved_gib'] for row in measurements)) < 24
assert all((hashlib.sha256(path.read_bytes()).hexdigest() == sha for path, sha in protected.items()))
assert all((hashlib.sha256((root / path).read_bytes()).hexdigest() == sha for path, sha in sources.items()))
result = dict(complete=True, id='AdaRES-Y48', source_hashes=sources, stripped_AST_exact_parent=True, independent_imports=True, configuration_byte_identical_parent=True, CPU_cases=32, production_loss_gradient_controls=cpu_rows, released_backbone=cfg.train.pretrain, actual_tools_train_steps=2, released_init_closure_records=closure_records, step_records=steps, indicator_parameter_updates=updates, parameters=parameters, added_parameters=0, model_optimizer_scheduler_restore_exact=True, cached_physical_parent_decoder_equal=True, inference_queries=128, selected_queries=1, released_init_inference_readings=inference_readings, released_init_native_stress=measurements, scientific_status='production_executability_controls_not_efficacy', limitations=['Diagnostic optimizer updates remain temporary; never deploy these weights.', 'Four TRAIN scenes for actual tools.train and cached decoder equality, not population benefit.', 'Restored state exact; no whole-loader bitwise next-step trajectory claim.', 'Native largest-pair stress is released-init executability, not trained Pareto evidence.'])
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
