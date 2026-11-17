"""Additional exact-reference properties, not another production revision."""

assert P['physical_gpu'] is None
scope = dict(torch=torch, nn=nn)
exec(PAYLOAD['initial_class'], scope)
InitialIndicator = scope['QueryCompetitionIndicator']
rows = []
for dtype in (torch.float32, torch.float64):
    for case in ('singleton', 'three_classes', 'query_equal_mask_distinct',
                 'query_distinct_mask_equal', 'partial_joint_copies',
                 'padding_only_difference', 'empty_support', 'zero_mass', 'mixed_batch'):
        gorilla.set_random_seed(1999)
        head = Indicator(256).to(dtype=dtype)
        old = InitialIndicator(256).to(dtype=dtype)
        old.load_state_dict(head.state_dict(), strict=True)
        size = 1 if case == 'singleton' else 128
        q = torch.randn(2, size, 256, dtype=dtype)
        m = torch.randn(2, size, 13, dtype=dtype)
        padding = torch.zeros(2, 13, dtype=torch.bool)
        padding[:, -3:] = True
        mass = torch.arange(1, 14, dtype=dtype).repeat(2, 1)
        if case in ('three_classes', 'padding_only_difference', 'empty_support', 'zero_mass', 'mixed_batch'):
            labels = torch.arange(size) % 3
            q[0] = q[0, labels].clone()
            m[0] = m[0, labels].clone()
            if case != 'mixed_batch':
                q[1] = q[1, labels].clone()
                m[1] = m[1, labels].clone()
        if case == 'query_equal_mask_distinct':
            q = q[:, :1].repeat(1, size, 1)
        if case == 'query_distinct_mask_equal':
            m = m[:, :1].repeat(1, size, 1)
        if case == 'partial_joint_copies':
            q[:, -3:] = q[:, :1].clone()
            m[:, -2:] = m[:, :1].clone()
        if case == 'padding_only_difference':
            m[:, :, -3:] = torch.arange(size, dtype=dtype).reshape(1, size, 1)
        if case == 'empty_support':
            padding[:] = True
        if case == 'zero_mass':
            mass.zero_()
        q.requires_grad_()
        m.requires_grad_()
        actual = head(q, m, padding, mass)
        reference = old(q, m, padding, mass)
        assert torch.equal(actual, reference), (dtype, case, 'output')
        actual_gradients = torch.autograd.grad(actual.square().mean(), [q, m] + list(head.parameters()),
                                             allow_unused=True)
        old_gradients = torch.autograd.grad(reference.square().mean(), [q, m] + list(old.parameters()),
                                          allow_unused=True)
        finite = 0
        for index, (a, b) in enumerate(zip(actual_gradients, old_gradients)):
            assert (a is None) == (b is None)
            if a is not None:
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                assert torch.equal(a, b), (dtype, case, 'gradient', index)
                finite += 1
        rows.append(dict(dtype=str(dtype), case=case, queries=size, batch=2,
                         original_output_bitwise_equal=True,
                         input_and_every_parameter_gradients_bitwise_equal=True,
                         finite_gradient_tensors=finite))
assert preserved() and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y56', revision=2,
    rows=rows, cases=18, physical_candidate=True, initial_reference=True,
    optimizer_steps=0, weights_saved=False, native_backward=False,
    limitation='CPU exact-reference edge cases/gradients, not native efficacy or global equivalence proof.')),
    encoding='utf-8')
