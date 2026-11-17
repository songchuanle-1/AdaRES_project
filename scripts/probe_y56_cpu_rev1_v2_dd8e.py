"""Exact-class controls on physical package, including duplicated focal."""

assert P['physical_gpu'] is None
scope = dict(torch=torch, nn=nn)
exec(PAYLOAD['parent_class'], scope)
ParentIndicator = scope['QueryCompetitionIndicator']
initial_scope = dict(torch=torch, nn=nn)
exec(PAYLOAD['initial_class'], initial_scope)
InitialIndicator = initial_scope['QueryCompetitionIndicator']
rows = []
for dtype in (torch.float32, torch.float64):
    gorilla.set_random_seed(1999)
    head = Indicator(256).to(dtype=dtype)
    parent = ParentIndicator(256).to(dtype=dtype)
    parent.load_state_dict(head.state_dict(), strict=True)
    rng_state = torch.get_rng_state()
    initial_head = InitialIndicator(256).to(dtype=dtype)
    initial_head.load_state_dict(head.state_dict(), strict=True)
    torch.set_rng_state(rng_state)
    padding = torch.zeros((1, 17), dtype=torch.bool)
    padding[:, -3:] = True
    mass = torch.arange(1, 18, dtype=dtype).unsqueeze(0)
    q = torch.randn(1, 3, 256, dtype=dtype)
    masks = torch.randn(1, 3, 17, dtype=dtype)
    reference = parent(q, masks, padding, mass)
    cases = []
    parent_focals = []
    for counts in ([1, 63, 64], [1, 64, 63], [2, 62, 64]):
        labels = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
        queries, logits = q[:, labels], masks[:, labels]
        actual = head(queries, logits, padding, mass)
        assert torch.equal(actual, initial_head(queries, logits, padding, mass))
        expected = reference[:, labels]
        bound = 128 * torch.finfo(dtype).eps * max(1.0, float(expected.detach().abs().max()))
        error = float((actual - expected).detach().abs().max())
        assert error <= bound, (counts, error, bound)
        repeat = head(queries, logits, padding, mass)
        permutation = torch.roll(torch.arange(128), 17)
        permuted = head(queries[:, permutation], logits[:, permutation], padding, mass)[:, torch.argsort(permutation)]
        assert torch.equal(actual, repeat)
        permutation_error = float((actual - permuted).detach().abs().max())
        assert permutation_error <= bound
        parent_focals.append(parent(queries, logits, padding, mass)[:, 0].detach())
        cases.append(dict(counts=counts, unique_reference_error=error, arithmetic_bound=bound,
                          repeat_error=0, permutation_error=permutation_error))
    parent_signal = float((parent_focals[0] - parent_focals[1]).abs().max())
    assert parent_signal > cases[0]['arithmetic_bound']
    queries = torch.randn(2, 128, 256, dtype=dtype, requires_grad=True)
    logits = torch.randn(2, 128, 17, dtype=dtype, requires_grad=True)
    pads, masses = padding.repeat(2, 1), mass.repeat(2, 1)
    exact = head(queries, logits, pads, masses)
    native = parent(queries, logits, pads, masses)
    assert torch.equal(exact, native)
    assert torch.equal(exact, initial_head(queries, logits, pads, masses))
    candidate_gradients = torch.autograd.grad(exact.sum(), (queries, logits), retain_graph=True)
    initial_gradients = torch.autograd.grad(initial_head(queries, logits, pads, masses).sum(),
                                          (queries, logits), retain_graph=True)
    assert all(torch.equal(a, b) for a, b in zip(candidate_gradients, initial_gradients))
    changed_logits, changed_mass = logits.detach().clone(), masses.clone()
    changed_logits[:, :, -3:] = 1000
    changed_mass[:, -3:] = 999
    assert torch.equal(exact, head(queries, changed_logits, pads, changed_mass))
    # Equality is raw and joint, not query-only, mask-only, or approximate.
    negatives = []
    for negative_q, negative_m in ((queries[:, :1].repeat(1, 128, 1), logits),
                                   (queries, logits[:, :1].repeat(1, 128, 1))):
        assert torch.equal(head(negative_q, negative_m, pads, masses),
                           parent(negative_q, negative_m, pads, masses))
        negatives.append(True)
    tiny_q = q[:, :1].repeat(1, 2, 1).clone()
    tiny_m = masks[:, :1].repeat(1, 2, 1).clone()
    tiny_q[0, 1, 0] = torch.nextafter(tiny_q[0, 1, 0], torch.tensor(float('inf'), dtype=dtype))
    assert torch.equal(head(tiny_q, tiny_m, padding, mass), parent(tiny_q, tiny_m, padding, mass))
    single_q, single_m = q[:, :1], masks[:, :1]
    repeated_q, repeated_m = single_q.repeat(1, 128, 1), single_m.repeat(1, 128, 1)
    singleton = parent(single_q, single_m, padding, mass)
    all_equal = head(repeated_q, repeated_m, padding, mass)
    equal_error = float((all_equal - singleton).detach().abs().max())
    assert equal_error <= cases[0]['arithmetic_bound']
    for pads2, masses2 in ((torch.ones_like(padding), mass), (padding, torch.zeros_like(mass))):
        assert torch.isfinite(head(repeated_q, repeated_m, pads2, masses2)).all()
    exact.square().mean().backward()
    gradients = {}
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max()) > 0
        gradients[name] = float(p.grad.abs().max())
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # Duplicate graph also backpropagates to original per-copy rows.
    dq, dm = repeated_q.clone().requires_grad_(), repeated_m.clone().requires_grad_()
    mixed_q = torch.cat((dq[:, :64], queries.detach()[:1, :64]), dim=1)
    mixed_m = torch.cat((dm[:, :64], logits.detach()[:1, :64]), dim=1)
    head(mixed_q, mixed_m, padding, mass).square().mean().backward()
    assert dq.grad is not None and torch.isfinite(dq.grad).all() and float(dq.grad.abs().max()) > 0
    assert dm.grad is not None and torch.isfinite(dm.grad).all() and float(dm.grad.abs().max()) > 0
    cq, cm = mixed_q.detach().requires_grad_(), mixed_m.detach().requires_grad_()
    cg = torch.autograd.grad(head(cq, cm, padding, mass).sum(), (cq, cm))
    ig = torch.autograd.grad(initial_head(cq, cm, padding, mass).sum(), (cq, cm))
    assert all(torch.equal(a, b) for a, b in zip(cg, ig))
    rows.append(dict(dtype=str(dtype), cases=cases, parent_multiplicity_signal=parent_signal,
                     no_copy_bitwise_equal=True, joint_equality_negatives=negatives,
                     one_bit_not_merged=True, singleton_class_error=equal_error,
                     padding_invariant=True, empty_and_zero_support_finite=True,
                     finite_nonzero_component_gradients=gradients, duplicate_input_backward=True,
                     initial_revision_output_and_input_gradients_bitwise_equal=True))
assert not torch.cuda.is_initialized() and preserved()
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y56', rows=rows,
    physical_candidate=True, native_backward=False, optimizer_steps=0, weights_saved=False,
    limitation='CPU mechanism controls only, not native reachability or trained efficacy.')), encoding='utf-8')
