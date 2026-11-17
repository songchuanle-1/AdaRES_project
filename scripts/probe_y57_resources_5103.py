"""Bounded actual native stress and paired parent/candidate inference cost."""

import statistics
from ipdn.model_paper5.adares_w2_pointmass_competition.ipdn import IPDN as Parent

model = initialize()
pretrain_sha = digest(cfg.train.pretrain)
initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
dataset, items, native_batches = batches()
points = [int(batch["coords_float"].shape[0]) for batch in native_batches]
largest = max(range(4), key=lambda i: points[i])
stress = []
for iteration in range(5):
    model.load_state_dict(initial, strict=True)
    model.train().zero_grad(set_to_none=True)
    gorilla.set_random_seed(1999)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss, logs = model(copy.deepcopy(native_batches[largest]), mode="loss")
    assert torch.isfinite(loss).all()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    torch.cuda.synchronize()
    reserved = int(torch.cuda.max_memory_reserved())
    assert reserved < 24 * 2**30
    stress.append(dict(iteration=iteration, warmup=iteration < 2,
                       seconds=time.perf_counter() - started, native_loss=float(loss.item()),
                       finite_backward=True, peak_reserved_bytes=reserved,
                       peak_allocated_bytes=int(torch.cuda.max_memory_allocated())))
    del loss, logs, gradients
    model.zero_grad(set_to_none=True)
    gc.collect()
model.load_state_dict(initial, strict=True)
model.cpu()
torch.cuda.empty_cache()
gorilla.set_random_seed(1999)
parent = Parent(**cfg.model)
parent_state = parent.state_dict()
common = {k: v for k, v in initial.items() if k in parent_state and v.shape == parent_state[k].shape}
loaded = parent.load_state_dict(common, strict=False)
changed_shapes = sorted(k for k in initial if k in parent_state and initial[k].shape != parent_state[k].shape)
assert changed_shapes == ["dec.indi_embedding.evidence_norm.bias", "dec.indi_embedding.evidence_norm.weight",
                           "dec.indi_embedding.head.0.weight"]
assert sorted(loaded.missing_keys) == changed_shapes and not loaded.unexpected_keys
assert all(torch.equal(parent.state_dict()[k], value) for k, value in common.items())
counts = dict(parent=sum(p.numel() for p in parent.parameters()),
              candidate=sum(p.numel() for p in model.parameters()))
assert counts["candidate"] - counts["parent"] == 66048
profile = []
for role, tested in (("parent", parent), ("candidate", model)):
    tested.cuda().eval()
    for index, item in enumerate(items):
        batch = dataset.collate_fn([copy.deepcopy(item)])
        with torch.no_grad():
            for _ in range(2):
                gorilla.set_random_seed(1999)
                tested(copy.deepcopy(batch), mode="predict")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(3):
                gorilla.set_random_seed(1999)
                current = copy.deepcopy(batch)
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = tested(current, mode="predict")
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                assert all(int(v.numel()) == 128 for v in output["indi"])
                del current, output
        profile.append(dict(role=role, scene=PAYLOAD["scenes"][index]["scan_id"],
                            expressions=len(batch["ann_ids"]), measured_seconds=times,
                            median_seconds=statistics.median(times),
                            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                            peak_reserved_bytes=int(torch.cuda.max_memory_reserved())))
        del batch
    tested.cpu()
    gc.collect()
    torch.cuda.empty_cache()
assert digest(cfg.train.pretrain) == pretrain_sha and preserved()
result = dict(complete=True, id="AdaRES-Y57", parameter_counts=counts, parameter_delta=66048,
              same_common_parameter_state=True, changed_shape_keys=changed_shapes,
              stress_input=dict(batch=largest, points=points[largest],
                                selection="Largest point count among the fixed four TRAIN batch2 inputs only"),
              stress=stress, paired_native_predict_profile=profile,
              active_queries=128, selected_queries=1, fixed_scenes=8,
              optimizer_steps=0, weights_saved=False, admission=False,
              limitation="Bounded TRAIN inference/native-backward feasibility and cost, not global worst case/Pareto or trained benefit.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
