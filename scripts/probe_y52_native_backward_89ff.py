"""Append to the frozen native setup: SAME-forward paired gradients and restore."""
import gc
import random
import numpy as np

from ipdn.model_paper5.adares_w2_pointmass_competition.loss import Criterion as ParentCriterion
parent_criterion = ParentCriterion(**cfg.model.criterion).cuda()
original_forward = model.criterion.forward
original_masks = model.criterion.loss_masks
parent_masks = parent_criterion.loss_masks
initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
phase = {}
queries = []


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(value):
    random.setstate(value[0])
    np.random.set_state(value[1])
    torch.set_rng_state(value[2])
    torch.cuda.set_rng_state_all(value[3])


def containers(value):
    if isinstance(value, dict):
        return {k: containers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [containers(v) for v in value]
    if isinstance(value, tuple):
        return tuple(containers(v) for v in value)
    return value


def query_hook(module, args):
    queries.append(args[0])


def masks_call(function, point_target, pred_masks, pred_scores, targets, indices, num_boxes):
    role = phase['role']
    records = phase['records'][role]
    for mask, target, match in zip(pred_masks, targets, indices):
        qids, tids = match
        selected = mask[qids].detach()
        with torch.no_grad():
            values = (quality.get_point_iou(selected, *target['point_counts']) if point_target
                      else quality.get_iou(selected, target['masks'][tids]))
        phase['active'][role] += int((values > .5).sum().item())
        records.append([qids.detach().cpu().tolist(), tids.detach().cpu().tolist()])
    result = function(pred_masks, pred_scores, targets, indices, num_boxes)
    phase['scores'][role].append(result[2])
    return result


def corrected_masks(*args):
    return masks_call(original_masks, True, *args)


def control_masks(*args):
    return masks_call(parent_masks, False, *args)


def paired_forward(*args, **kwargs):
    before = rng_state()
    phase['role'] = 'corrected'
    corrected = original_forward(*containers(args), **containers(kwargs))
    after = rng_state()
    restore_rng(before)
    phase['role'] = 'parent'
    parent_kwargs = containers(kwargs)
    parent_kwargs.pop('point_targets', None)
    parent = parent_criterion(*containers(args), **parent_kwargs)
    restore_rng(before)
    phase['role'] = 'repeat'
    repeated = original_forward(*containers(args), **containers(kwargs))
    restore_rng(after)
    assert phase['records']['corrected'] == phase['records']['parent'] == phase['records']['repeat']
    phase['losses'] = [corrected[0], parent[0], repeated[0]]
    return corrected


def gradients(loss):
    if not loss.requires_grad:
        return [None for _ in queries]
    return list(torch.autograd.grad(loss, queries, retain_graph=True, allow_unused=True))


def grad_delta(left, right):
    maximum = 0.
    squared = 0.
    norm = 0.
    for query, a, b in zip(queries, left, right):
        a = torch.zeros_like(query) if a is None else a
        b = torch.zeros_like(query) if b is None else b
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = a - b
        maximum = max(maximum, float(delta.abs().max().item()))
        squared += float(delta.square().sum().item())
        norm += float(a.square().sum().item())
    return dict(max_abs=maximum, l2=squared ** .5, corrected_l2=norm ** .5)


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def equal_state(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


items = []
for scene in frozen:
    gorilla.set_random_seed(1999)
    item = dataset[scene['dataset_index']]
    single = dataset.collate_fn([copy.deepcopy(item)])
    assert single['scan_ids'][0] == scene['scan_id']
    assert [str(a) for a in single['ann_ids']] == [e['ann_id'] for e in scene['expressions']]
    assert single['object_idss'] == [e['object_ids'] for e in scene['expressions']]
    items.append(item)

hook = model.dec.out_norm.register_forward_pre_hook(query_hook)
model.criterion.forward = paired_forward
model.criterion.loss_masks = corrected_masks
parent_criterion.loss_masks = control_masks
paired_rows = []
torch.cuda.reset_peak_memory_stats()
trained_parent = Path('/bak_home/songcl/Project/AdaRES_project/AdaRES_checkpoints/adares_w2_pointmass_competition_res/20260926_145041/best.pth')
assert hashlib.sha256(trained_parent.read_bytes()).hexdigest() == 'e3407283384b4899524d055ac146c7bd6cf39cedb10e58a44ad74eb522009f52'
try:
    for state_name in ('released_init', 'trained_parent_diagnostic_only'):
        if state_name == 'released_init':
            model.load_state_dict(initial_state, strict=True)
        else:
            checkpoint = torch.load(str(trained_parent), map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            gc.collect()
        model.train()
        for pair in range(0, 8, 2):
            queries.clear()
            phase.clear()
            phase.update(records={k: [] for k in ('corrected', 'parent', 'repeat')},
                active={k: 0 for k in ('corrected', 'parent', 'repeat')},
                scores={k: [] for k in ('corrected', 'parent', 'repeat')})
            gorilla.set_random_seed(1999)
            batch = dataset.collate_fn(copy.deepcopy(items[pair:pair + 2]))
            model.zero_grad(set_to_none=True)
            loss, logs = model(batch, mode='loss')
            assert len(queries) == 7 and all(q.requires_grad for q in queries)
            corrected, parent, repeated = phase['losses']
            assert all(torch.isfinite(value).all() for value in phase['losses'])
            cg, pg, rg = [gradients(value) for value in phase['losses']]
            full_signal, full_noise = grad_delta(cg, pg), grad_delta(cg, rg)
            del cg, pg, rg
            scores = {k: sum(phase['scores'][k]) for k in phase['scores']}
            cs, ps, rs = [gradients(scores[k]) for k in ('corrected', 'parent', 'repeat')]
            score_signal, score_noise = grad_delta(cs, ps), grad_delta(cs, rs)
            del cs, ps, rs
            corrected.backward()
            live_gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert live_gradients and all(torch.isfinite(g).all() for g in live_gradients)
            head = {name: dict(present=p.grad is not None,
                finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None,
                norm=float(p.grad.norm().item()) if p.grad is not None else 0.)
                for name, p in model.dec.out_score.named_parameters()}
            paired_rows.append(dict(state=state_name, batch=pair // 2, expressions=sum(batch['scenes_len']),
                identical_actual_match_indices=True, actual_query_tensors=len(queries),
                losses=[float(v.item()) for v in phase['losses']],
                unweighted_score_losses={k: float(v.item()) for k, v in scores.items()},
                active_matched_queries=dict(phase['active']), full_query_gradient_signal=full_signal,
                full_query_repeat_noise=full_noise, score_query_gradient_signal=score_signal,
                score_query_repeat_noise=score_noise, score_head_gradients=head,
                native_backward_finite=True, parameter_gradient_tensors=len(live_gradients)))
            del loss, logs, corrected, parent, repeated, scores, live_gradients, batch
            phase.clear()
            queries.clear()
            model.zero_grad(set_to_none=True)
            gc.collect()
    paired_peak = int(torch.cuda.max_memory_reserved())
finally:
    hook.remove()
    model.criterion.forward = original_forward
    model.criterion.loss_masks = original_masks
    parent_criterion.loss_masks = parent_masks
    phase.clear()
    queries.clear()

# Native optimizer settings; RAM-only snapshots include nonempty optimizer state.
model.load_state_dict(initial_state, strict=True)
gorilla.set_random_seed(1999)
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
steps = []
snapshot = None
for step in range(2):
    batch = dataset.collate_fn(copy.deepcopy(items[2 * step:2 * step + 2]))
    optimizer.zero_grad(set_to_none=True)
    loss, logs = model(batch, mode='loss')
    assert torch.isfinite(loss).all()
    loss.backward()
    values = [p.grad for p in model.parameters() if p.grad is not None]
    assert values and all(torch.isfinite(g).all() for g in values)
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip).item())
    optimizer.step()
    scheduler.step()
    steps.append(dict(step=step + 1, native_loss=float(loss.item()), gradient_norm_before_clip=norm,
                      optimizer_state_entries=len(optimizer.state), scheduler_epoch=scheduler.last_epoch))
    del batch, loss, logs, values
    if step == 0:
        snapshot = (cpu_copy(model.state_dict()), cpu_copy(optimizer.state_dict()),
                    cpu_copy(scheduler.state_dict()), rng_state())
    gc.collect()
assert snapshot is not None and steps[0]['optimizer_state_entries'] > 0
changed_after_second_step = not equal_state(snapshot[0], model.state_dict())
assert changed_after_second_step
model.load_state_dict(snapshot[0], strict=True)
optimizer.load_state_dict(snapshot[1])
scheduler.load_state_dict(snapshot[2])
restore_rng(snapshot[3])
restored = dict(model=equal_state(snapshot[0], model.state_dict()),
                optimizer=equal_state(snapshot[1], optimizer.state_dict()),
                scheduler=equal_state(snapshot[2], scheduler.state_dict()),
                python_rng=random.getstate() == snapshot[3][0],
                numpy_rng=all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                              for a, b in zip(np.random.get_state(), snapshot[3][1])),
                torch_rng=torch.equal(torch.get_rng_state(), snapshot[3][2]),
                cuda_rng=all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), snapshot[3][3])))
assert all(restored.values())
result = dict(complete=True, id='AdaRES-Y52', actual_native_sparse_forward_and_backward=True,
    frozen_TRAIN_scenes=8, expressions_per_state=32, fixed_seed=1999,
    paired_same_forward=True, parent_criterion_byte_bound=True, candidate_model_revisions=0,
    independent_corrected_repeat_noise=True, rows=paired_rows, optimizer_steps=steps,
    nonempty_optimizer_restore=restored, state_changed_before_restore=changed_after_second_step,
    paired_peak_reserved_bytes=paired_peak, total_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    model_or_checkpoint_weights_saved=False, admission=False,
    limitation='Released-init and a fixed trained W2 state are separate diagnostic conditions, not trained Y52 efficacy. No new parameters. Source-binding repair only; zero initial activity is reported, not promoted to PASS. Paired score-gradient activity is not selector improvement or a lightweight Pareto result.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
