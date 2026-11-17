"""Investigate the failed fixed control without changing its numerical bound."""

model = initialize()
dataset, items, native_batches = batches()
released = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
assert digest(PAYLOAD['checkpoint']) == PAYLOAD['checkpoint_sha256']
checkpoint = torch.load(PAYLOAD['checkpoint'], map_location='cpu')['model']
diagnostic = {k: v.clone() for k, v in released.items()}
mapped = []
for name, value in checkpoint.items():
    if name not in diagnostic:
        assert name in {'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'}
        continue
    if value.shape == diagnostic[name].shape:
        diagnostic[name].copy_(value)
    elif name in ('dec.indi_embedding.evidence_norm.weight', 'dec.indi_embedding.evidence_norm.bias'):
        assert value.shape == (769,) and diagnostic[name].shape == (1025,)
        diagnostic[name][:769].copy_(value)
        mapped.append(name)
    else:
        assert name == 'dec.indi_embedding.head.0.weight'
        assert value.shape == (256, 769) and diagnostic[name].shape == (256, 1025)
        diagnostic[name][:, :769].copy_(value)
        mapped.append(name)
assert len(mapped) == 3
del checkpoint
rows = []
for state_name, state in (('released_initialization', released), ('trained_W2_diagnostic_ONLY', diagnostic)):
    for index, batch in enumerate(native_batches):
        model.load_state_dict(state, strict=True)
        model.train()
        model.zero_grad(set_to_none=True)
        gorilla.set_random_seed(1999)
        inputs, evidence = [], []
        def capture_inputs(module, args):
            inputs.append(tuple(v.detach().clone() for v in args))
        def capture_evidence(module, args):
            args[0].retain_grad()
            evidence.append(args[0])
        handles = [model.dec.indi_embedding.register_forward_pre_hook(capture_inputs),
                   model.dec.indi_embedding.evidence_norm.register_forward_pre_hook(capture_evidence)]
        torch.cuda.reset_peak_memory_stats()
        try:
            loss, logs = model(copy.deepcopy(batch), mode='loss')
            assert torch.isfinite(loss).all()
            loss.backward()
            assert len(evidence) == len(inputs) == 7
            grads = {}
            for name, parameter in model.dec.indi_embedding.named_parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                grads[name] = float(parameter.grad.abs().max())
            all_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert all_finite
            layers = [dict(layer=i, variance_max=float(e.detach()[..., -256:].max()),
                           variance_grad_max=float(e.grad[..., -256:].abs().max()))
                      for i, e in enumerate(evidence)]
            added_grad = float(model.dec.indi_embedding.head[0].weight.grad[:, -256:].abs().max())
            loss_value = float(loss.item())
            peak = int(torch.cuda.max_memory_reserved())
        finally:
            for handle in handles:
                handle.remove()
        controls = []
        head = model.dec.indi_embedding
        for layer, args in enumerate(inputs):
            q, m, padding, mass = args
            with torch.no_grad():
                baseline = head(*args)
                repeated = head(*args)
                perm = torch.roll(torch.arange(128, device=q.device), 1)
                permuted = head(q[:, perm], m[:, perm], padding, mass)[:, torch.argsort(perm)]
                def shift(module, values):
                    value = values[0].clone()
                    value[..., -256:] = torch.roll(value[..., -256:], 1, dims=1)
                    return (value,)
                handle = head.evidence_norm.register_forward_pre_hook(shift)
                shifted = head(*args)
                handle.remove()
            bound = 128 * torch.finfo(q.dtype).eps * max(1., float(baseline.abs().max()))
            signal = float((baseline - shifted).abs().max())
            repeat = float((baseline - repeated).abs().max())
            permutation = float((baseline - permuted).abs().max())
            controls.append(dict(layer=layer, primary=layer == 6, repeat_error=repeat,
                                 permutation_error=permutation, arithmetic_bound=bound,
                                 same_width_shift_signal=signal,
                                 original_control_passed=repeat == 0 and permutation <= bound and signal > bound))
        rows.append(dict(state=state_name, batch=index, expressions=8, loss=loss_value,
                         finite_backward=True, gradients=grads,
                         every_indicator_tensor_nonzero=all(v > 0 for v in grads.values()),
                         added_columns_gradient=added_grad, layers=layers, controls=controls,
                         peak_reserved_bytes=peak))
        evidence.clear()
        inputs.clear()
        del loss, logs
        model.zero_grad(set_to_none=True)
        gc.collect()
model.load_state_dict(released, strict=True)
assert all(torch.equal(v.detach().cpu(), released[k]) for k, v in model.state_dict().items())
assert preserved()
result = dict(complete=True, id='AdaRES-Y57', rows=rows, shape_mapped_diagnostic_keys=mapped,
              formal_model_bundle_unchanged=True, all_original_control_bounds_retained=True,
              failed_controls_retained_not_relabeled=True, restored_released_initialization=True,
              optimizer_steps=0, weights_saved=False, admission=False,
              limitation='Trained W2 is a separate diagnostic state, never formal training initialization. '
                         'All released/control failures and gradients retained; review before any gate decision.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
