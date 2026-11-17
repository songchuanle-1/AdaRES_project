"""Actual matched native loss/backward and inference costs for fixed TRAIN inputs."""

import statistics
from ipdn.model_paper5.adares_w2_pointmass_competition.ipdn import IPDN as Parent

model = initialize()
pretrain_sha = digest(cfg.train.pretrain)
initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
dataset, items, native_batches = batches()
points = [int(batch['coords_float'].shape[0]) for batch in native_batches]
largest = max(range(len(points)), key=lambda i: points[i])
model.cpu()
torch.cuda.empty_cache()
gorilla.set_random_seed(1999)
parent = Parent(**cfg.model)
parent.load_state_dict(initial, strict=True)
counts = dict(parent=sum(p.numel() for p in parent.parameters()),
              candidate=sum(p.numel() for p in model.parameters()))
assert counts['parent'] == counts['candidate']
stress, profile = [], []
for role, tested in (('parent', parent), ('candidate', model)):
    tested.cuda()
    for iteration in range(5):
        tested.load_state_dict(initial, strict=True)
        tested.train()
        tested.zero_grad(set_to_none=True)
        gorilla.set_random_seed(1999)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss, logs = tested(copy.deepcopy(native_batches[largest]), mode='loss')
        assert torch.isfinite(loss).all()
        loss.backward()
        gradients = [p.grad for p in tested.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        torch.cuda.synchronize()
        reserved = int(torch.cuda.max_memory_reserved())
        assert reserved < 24 * 2**30
        stress.append(dict(role=role, iteration=iteration, warmup=iteration < 2,
                           seconds=time.perf_counter()-started, loss=float(loss.item()),
                           finite_backward=True, peak_reserved_bytes=reserved,
                           peak_allocated_bytes=int(torch.cuda.max_memory_allocated())))
        del loss, logs, gradients
        tested.zero_grad(set_to_none=True)
        gc.collect()
    tested.load_state_dict(initial, strict=True)
    tested.eval()
    for index, item in enumerate(items):
        batch = dataset.collate_fn([copy.deepcopy(item)])
        with torch.no_grad():
            for _ in range(2):
                gorilla.set_random_seed(1999)
                tested(copy.deepcopy(batch), mode='predict')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(3):
                gorilla.set_random_seed(1999)
                current = copy.deepcopy(batch)
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = tested(current, mode='predict')
                torch.cuda.synchronize()
                times.append(time.perf_counter()-started)
                assert all(int(torch.as_tensor(v).numel()) == 128 for v in output['indi'])
                del current, output
        profile.append(dict(role=role, scene=PAYLOAD['scenes'][index]['scan_id'],
                            expressions=len(batch['ann_ids']), measured_seconds=times,
                            median_seconds=statistics.median(times),
                            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                            peak_reserved_bytes=int(torch.cuda.max_memory_reserved())))
    tested.cpu()
    gc.collect()
    torch.cuda.empty_cache()
assert digest(cfg.train.pretrain) == pretrain_sha and preserved()
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y56', parameter_counts=counts,
    parameter_delta=0, strict_same_weights=True,
    stress_input=dict(batch=largest, points=points[largest], selection='Largest point count among fixed5TRAIN pairs'),
    stress=stress, paired_native_predict_profile=profile, active_queries=128, selected_queries=1,
    fixed_scenes=10, optimizer_steps=0, weights_saved=False, admission=False,
    limitation='Bounded same-weight cost on fixed TRAIN inputs, not global Pareto/worst case or trained efficacy.')),
    encoding='utf-8')
