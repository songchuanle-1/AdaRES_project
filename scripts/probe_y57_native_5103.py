"""Physical candidate, released initialization and actual native task backward."""

model = initialize()
pretrain_sha = digest(cfg.train.pretrain)
dataset, items, native_batches = batches()
initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
all_rows, controls = [], []
peak = 0
for index, batch in enumerate(native_batches):
    model.load_state_dict(initial, strict=True)
    model.train().zero_grad(set_to_none=True)
    gorilla.set_random_seed(1999)
    evidence, inputs = [], []
    def evidence_hook(module, args):
        assert args[0].shape[-1] == 1025 and args[0].requires_grad
        args[0].retain_grad()
        evidence.append(args[0])
    def input_hook(module, args):
        if index == 0:
            inputs.append(tuple(v.detach().clone() for v in args))
    hooks = [model.dec.indi_embedding.evidence_norm.register_forward_pre_hook(evidence_hook),
             model.dec.indi_embedding.register_forward_pre_hook(input_hook)]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        loss, logs = model(copy.deepcopy(batch), mode="loss")
        assert torch.isfinite(loss).all()
        loss.backward()
        assert len(evidence) == 7
        gradients = {}
        for name, parameter in model.dec.indi_embedding.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            maximum = float(parameter.grad.abs().max())
            assert maximum > 0, name
            gradients[name] = maximum
        column_signal = float(model.dec.indi_embedding.head[0].weight.grad[:, -256:].abs().max())
        assert column_signal > 0
        layers = []
        for layer, value in enumerate(evidence):
            assert value.grad is not None and torch.isfinite(value.grad).all()
            variance = value.detach()[..., -256:]
            gradient = value.grad[..., -256:]
            assert float(variance.max()) > 0
            assert float(gradient.abs().max()) > 0
            layers.append(dict(layer=layer, variance_max=float(variance.max()),
                               raw_variance_gradient_max=float(gradient.abs().max())))
        all_gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert all_gradients and all(torch.isfinite(g).all() for g in all_gradients)
        torch.cuda.synchronize()
        reserved = int(torch.cuda.max_memory_reserved())
        assert reserved < 24 * 2**30
        peak = max(peak, reserved)
        all_rows.append(dict(batch=index, expressions=8, loss=float(loss.item()),
                             indicator_gradients=gradients, added_columns_gradient=column_signal,
                             layers=layers, native_gradient_tensors=len(all_gradients),
                             seconds=time.perf_counter() - started, peak_reserved_bytes=reserved))
    finally:
        for handle in hooks:
            handle.remove()
    if index == 0:
        assert len(inputs) == 7
        head = model.dec.indi_embedding
        for layer, args in enumerate(inputs):
            q, m, padding, mass = args
            with torch.no_grad():
                baseline = head(*args)
                repeat = head(*args)
                perm = torch.roll(torch.arange(128, device=q.device), 1)
                permuted = head(q[:, perm], m[:, perm], padding, mass)[:, torch.argsort(perm)]
                def shifted(module, values):
                    altered = values[0].clone()
                    altered[..., -256:] = torch.roll(altered[..., -256:], 1, dims=1)
                    return (altered,)
                handle = head.evidence_norm.register_forward_pre_hook(shifted)
                changed = head(*args)
                handle.remove()
            bound = 128 * torch.finfo(q.dtype).eps * max(1.0, float(baseline.abs().max()))
            repeat_error = float((baseline - repeat).abs().max())
            perm_error = float((baseline - permuted).abs().max())
            assert repeat_error == 0 and perm_error <= bound
            controls.append(dict(layer=layer, primary=layer == 6, repeat_error=repeat_error,
                                 permutation_error=perm_error, arithmetic_bound=bound,
                                 same_width_shift_logit_change=float((baseline - changed).abs().max())))
        assert controls[6]["same_width_shift_logit_change"] > controls[6]["arithmetic_bound"]
    evidence.clear()
    inputs.clear()
    del loss, logs, all_gradients, variance, gradient, value
    model.zero_grad(set_to_none=True)
    gc.collect()
model.load_state_dict(initial, strict=True)
assert all(torch.equal(value.detach().cpu(), initial[name]) for name, value in model.state_dict().items())
assert digest(cfg.train.pretrain) == pretrain_sha and preserved()
result = dict(complete=True, id="AdaRES-Y57", physical_independent_candidate=True,
              initialization="Actual unchanged native released-SPFormer loader; not trained W2 weights",
              released_pretrain_sha256=pretrain_sha, fixed_scenes=8, expressions=32,
              actual_sparse_FPS_batch2=True, native_backward=True, rows=all_rows,
              matched_native_controls=controls, peak_reserved_bytes=peak,
              parameters_and_buffers_restored=True, every_indicator_tensor_finite_nonzero=True,
              added_input_columns_reached=True, all7variance_paths_reached=True,
              optimizer_steps=0, weights_saved=False, admission=False,
              limitation="Native reachability and finite backward only, not trained efficacy/causality/Pareto; resource stress still separate.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
