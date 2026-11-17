"""Frozen component controls; no CUDA, task efficacy or production weights."""

assert P["physical_gpu"] is None
rows = []
for dtype in (torch.float32, torch.float64):
    gorilla.set_random_seed(1999)
    head = Indicator(256).to(dtype=dtype)
    assert head.evidence_norm.normalized_shape == (1025,)
    q = torch.randn(2, 128, 256, dtype=dtype)
    masks = torch.randn(2, 128, 17, dtype=dtype)
    padding = torch.zeros((2, 17), dtype=torch.bool)
    padding[1, -3:] = True
    mass = torch.arange(1, 18, dtype=dtype).repeat(2, 1)
    evidence = []
    handle = head.evidence_norm.register_forward_pre_hook(
        lambda module, args: evidence.append(args[0].detach().clone()))
    original = head(q, masks, padding, mass)
    handle.remove()
    assert len(evidence) == 1 and evidence[0].shape == (2, 128, 1025)
    assert evidence[0][..., -256:].min() >= 0 and evidence[0][..., -256:].max() > 0
    repeat = head(q, masks, padding, mass)
    permutation = torch.roll(torch.arange(128), 1)
    permuted = head(q[:, permutation], masks[:, permutation], padding, mass)[:, torch.argsort(permutation)]
    bound = 128 * torch.finfo(dtype).eps * max(1.0, float(original.detach().abs().max()))
    assert float((original - repeat).abs().max()) == 0
    permutation_error = float((original - permuted).abs().max())
    assert permutation_error <= bound
    def shift(module, args):
        changed = args[0].clone()
        changed[..., -256:] = torch.roll(changed[..., -256:], 1, dims=1)
        return (changed,)
    handle = head.evidence_norm.register_forward_pre_hook(shift)
    shifted = head(q, masks, padding, mass)
    handle.remove()
    signal = float((original - shifted).abs().max())
    assert signal > bound
    changed_masks, changed_mass = masks.clone(), mass.clone()
    changed_masks[1, :, -3:] = 1000
    changed_mass[1, -3:] = 999
    padding_error = float((original - head(q, changed_masks, padding, changed_mass)).abs().max())
    assert padding_error == 0
    same = q[:, :1].repeat(1, 128, 1)
    captured = []
    handle = head.evidence_norm.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    same_output = head(same, masks, padding, mass)
    handle.remove()
    variance_bound = 128 * torch.finfo(dtype).eps * max(1.0, float(head.query_norm(same).square().max()))
    assert float(captured[0][..., -256:].abs().max()) <= variance_bound
    empty = head(q, masks, torch.ones_like(padding), mass)
    assert torch.isfinite(empty).all()
    loss = original.square().mean()
    loss.backward()
    gradients = {}
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        maximum = float(parameter.grad.abs().max())
        assert maximum > 0, name
        gradients[name] = maximum
    assert float(head.head[0].weight.grad[:, -256:].abs().max()) > 0
    rows.append(dict(dtype=str(dtype), repeat_error=0, permutation_error=permutation_error,
                     arithmetic_bound=bound, matched_shift_signal=signal,
                     padding_error=padding_error, equal_peer_variance_max=float(
                         captured[0][..., -256:].abs().max()), equal_peer_bound=variance_bound,
                     finite_component_gradients=gradients))
assert not torch.cuda.is_initialized()
result = dict(complete=True, id="AdaRES-Y57", rows=rows, physical_package_import=True,
              independent1025input_head=True, CUDA_initialized=False, preserved=preserved(),
              native_backward=False, optimizer_steps=0, weights_saved=False, admission=False,
              limitation="Component CPU controls only; not native task reachability, trained efficacy or resource PASS.")
Path(P["output"]).write_text(json.dumps(result), encoding="utf-8")
