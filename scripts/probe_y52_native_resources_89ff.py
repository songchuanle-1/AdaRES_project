"""Append to frozen native setup: paired inference and frozen-largest-pair stress."""
import gc
import statistics
from ipdn.model_paper5.adares_w2_pointmass_competition.ipdn import IPDN as Parent

candidate = model.cpu()
initial = {k: v.detach().cpu().clone() for k, v in candidate.state_dict().items()}
parent = Parent(**cfg.model)
parent.load_state_dict(initial, strict=True)
assert all(torch.equal(initial[k], v.cpu()) for k, v in parent.state_dict().items())
counts = {role: sum(p.numel() for p in value.parameters())
          for role, value in (('parent', parent), ('candidate', candidate))}
assert counts['parent'] == counts['candidate']
items = []
for scene in frozen:
    gorilla.set_random_seed(1999)
    item = dataset[scene['dataset_index']]
    single = dataset.collate_fn([copy.deepcopy(item)])
    assert single['scan_ids'][0] == scene['scan_id']
    assert [str(a) for a in single['ann_ids']] == [e['ann_id'] for e in scene['expressions']]
    items.append(item)
batches = [dataset.collate_fn(copy.deepcopy(items[i:i + 2])) for i in range(0, 8, 2)]
point_counts = [int(batch['coords_float'].shape[0]) for batch in batches]
largest = max(range(4), key=lambda i: point_counts[i])
observations = {'parent': {}, 'candidate': {}}
profile = []


def snapshot(output):
    return dict(identities=[(str(scan), str(ann)) for scan, ann in zip(output['scan_id'], output['ann_id'])],
        masks=[torch.as_tensor(mask).detach().cpu().clone() for mask in output['pred_pmask']],
        pious=[float(v) for v in output['piou']],
        queries=[int(torch.as_tensor(v).numel()) for v in output['indi']])


def difference(left, right):
    assert left['identities'] == right['identities']
    assert all(a.shape == b.shape for a, b in zip(left['masks'], right['masks']))
    return dict(mask_points_changed=sum(int((a != b).sum().item()) for a, b in zip(left['masks'], right['masks'])),
        maximum_expression_iou_difference=max(abs(a - b) for a, b in zip(left['pious'], right['pious'])))


for role, tested in (('parent', parent), ('candidate', candidate)):
    torch.cuda.empty_cache()
    tested.cuda().eval()
    for index, batch in enumerate(batches):
        with torch.no_grad():
            for _ in range(2):
                gorilla.set_random_seed(1999)
                tested(copy.deepcopy(batch), mode='predict')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            outputs = []
            for _ in range(3):
                gorilla.set_random_seed(1999)
                current = copy.deepcopy(batch)
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = tested(current, mode='predict')
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                outputs.append(snapshot(prediction))
                del prediction, current
        assert all(q == 128 for q in outputs[0]['queries'])
        observations[role][index] = outputs
        profile.append(dict(role=role, batch=index, expressions=len(outputs[0]['identities']),
            points=point_counts[index], two_warmups=True, measured_seconds=times,
            median_seconds=float(statistics.median(times)),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved())))
    tested.cpu()
    gc.collect()
    torch.cuda.empty_cache()

parity = []
for index in range(4):
    a, b = observations['parent'][index], observations['candidate'][index]
    parent_noise = [difference(a[0], value) for value in a[1:]]
    candidate_noise = [difference(b[0], value) for value in b[1:]]
    parity.append(dict(batch=index, matched=difference(a[0], b[0]),
                       parent_repeat=parent_noise, candidate_repeat=candidate_noise))
del observations
gc.collect()

candidate.cuda().train()
stress = []
for iteration in range(5):
    candidate.load_state_dict(initial, strict=True)
    gorilla.set_random_seed(1999)
    candidate.zero_grad(set_to_none=True)
    batch = copy.deepcopy(batches[largest])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss, logs = candidate(batch, mode='loss')
    assert torch.isfinite(loss).all()
    loss.backward()
    grads = [p.grad for p in candidate.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    reserved = int(torch.cuda.max_memory_reserved())
    assert reserved < 24 * 2**30
    stress.append(dict(iteration=iteration, warmup=iteration < 2,
        seconds=elapsed, native_loss=float(loss.item()), finite_backward=True,
        gradient_tensors=len(grads), peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=reserved))
    del loss, logs, grads, batch
    candidate.zero_grad(set_to_none=True)
    gc.collect()
result = dict(complete=True, id='AdaRES-Y52', exact_initial_state_equal=True,
    parameter_counts=counts, parameter_delta=0, frozen_TRAIN_scenes=8, expressions=32,
    actual_parent_and_candidate_predict=True, actual_single_argmax_queries128=True,
    inference_profile=profile, inference_parity=parity,
    stress_input=dict(batch=largest, points=point_counts[largest],
                      selection='largest input point count among the same four frozen production batchsize2 pairs'),
    stress=stress, model_or_checkpoint_weights_saved=False, optimizer_steps=0, admission=False,
    limitation='Measured paired inference and finite native backward resource scope on frozen TRAIN inputs only. Not full-dataset worst case, trained Y52 accuracy, a guaranteed Pareto gain, commit authorization or training admission.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
