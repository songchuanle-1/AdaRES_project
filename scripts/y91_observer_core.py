"""Verify original evidence and all consumers of the internal confidence gate."""
def check_response_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        assert row['weight_nonzero'] is bool(index // 7)
        for key in ('original_query_identity', 'original_mask_identity', 'original_mass_identity',
                    'original_evidence_exact', 'first_and_second_hidden_identity',
                    'both_inputs_undetached', 'float64_gate_reference', 'final_linear_consumes_gated_hidden'):
            assert row[key] is True, key
        for key in ('gate_delta_max', 'hidden_delta_max'):
            assert math.isfinite(row[key])
            assert row[key] > 0 if index // 7 else row[key] == 0


def reset_state():
    state.clear()
    state.update(queries=[], masks=[], gates=[], indicators=[], normalized=[], evidence_inputs=[],
        evidence_outputs=[], first_linear=[], first_hidden=[], second_linear=[], second_hidden=[],
        gated=[], final_consumers=[], coarse=0, features=None, padding=None, mass=None)


def checked_prediction(query, features, pad):
    if not state['queries']:
        state['features'], state['padding'] = features, pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(query)
    value = original_prediction(query, features, pad)
    state['masks'].append(value[1])
    expected = (value[1].sigmoid() < .5) | pad.unsqueeze(1)
    expected[expected.all(-1)] = False
    expected = expected | pad.unsqueeze(1)
    assert torch.equal(value[2], expected) and not value[2].requires_grad
    state['gates'].append(value[2])
    return value


def before_indicator(module, args):
    index = len(state['indicators'])
    assert module is head and len(args) == 4
    assert args[0] is state['queries'][index] and args[1] is state['masks'][index]
    assert args[2] is state['padding'] and args[0].requires_grad and args[1].requires_grad
    if index == 0:
        state['mass'] = args[3]
    assert args[3] is state['mass'] and not args[3].requires_grad
    state['indicators'].append(args)


def capture_query_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['queries'][-1]
    assert args[0].requires_grad and value.requires_grad
    state['normalized'].append(value)


def capture_evidence(module, args, value):
    index = len(state['evidence_inputs'])
    query, masks, pad, mass = state['indicators'][index]
    normalized = state['normalized'][index]
    probabilities = masks.sigmoid() * (~pad).unsqueeze(1).to(masks.dtype)
    weighted = probabilities * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection).clamp_min(1e-6)
    diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0.)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    context = torch.einsum('bij,bjd->bid', competition, normalized)
    reference = torch.cat((normalized, context, normalized - context, overlap.max(-1).values.unsqueeze(-1)), -1)
    assert module is head.evidence_norm and torch.equal(args[0], reference)
    assert args[0].requires_grad and value.requires_grad
    args[0].retain_grad()
    value.retain_grad()
    state['evidence_inputs'].append(args[0])
    state['evidence_outputs'].append(value)


def capture_first_linear(module, args, value):
    index = len(state['first_linear'])
    assert module is head.head[0] and args[0] is state['evidence_outputs'][index]
    state['first_linear'].append(value)


def capture_first_hidden(module, args, value):
    index = len(state['first_hidden'])
    assert module is head.head[1] and args[0] is state['first_linear'][index] and value.requires_grad
    value.retain_grad()
    state['first_hidden'].append(value)


def capture_second_linear(module, args, value):
    index = len(state['second_linear'])
    assert module is head.head[2] and args[0] is state['first_hidden'][index]
    state['second_linear'].append(value)


def capture_second_hidden(module, args, value):
    index = len(state['second_hidden'])
    assert module is head.head[3] and args[0] is state['second_linear'][index] and value.requires_grad
    value.retain_grad()
    state['second_hidden'].append(value)


def capture_gate(module, args, value):
    index = len(state['gated'])
    assert module is gate and len(args) == 2
    first, second = args
    assert first is state['first_hidden'][index] and second is state['second_hidden'][index]
    assert first.requires_grad and second.requires_grad and value.requires_grad
    scale = first.shape[-1] ** .5
    reference_gate = 1 + torch.tanh((first.double() @ module.weight.double()) / scale)
    reference = second.double() * reference_gate
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    measured_gate = 1 + torch.tanh((first @ module.weight) / scale)
    assert torch.equal(value, second * measured_gate)
    assert torch.isfinite(measured_gate).all() and measured_gate.min() >= 0 and measured_gate.max() <= 2
    nonzero = bool(torch.count_nonzero(module.weight))
    assert nonzero == bool(len(step_rows))
    gate_delta = float((measured_gate - 1).abs().max())
    hidden_delta = float((value - second).abs().max())
    assert gate_delta > 0 and hidden_delta > 0 if nonzero else gate_delta == hidden_delta == 0
    value.retain_grad()
    state['gated'].append(value)
    path_rows.append(dict(step=len(step_rows), stage=index, weight_nonzero=nonzero,
        gate_delta_max=gate_delta, hidden_delta_max=hidden_delta,
        original_query_identity=True, original_mask_identity=True, original_mass_identity=True,
        original_evidence_exact=True, first_and_second_hidden_identity=True,
        both_inputs_undetached=True, float64_gate_reference=True,
        final_linear_consumes_gated_hidden=False))


def before_final_linear(module, args):
    index = len(state['final_consumers'])
    assert module is head.head[4] and args[0] is state['gated'][index] and args[0].requires_grad
    state['final_consumers'].append(index)
    assert path_rows[-1]['stage'] == index and path_rows[-1]['step'] == len(step_rows)
    path_rows[-1]['final_linear_consumes_gated_hidden'] = True


def before_coarse(module, args):
    assert module is model.dec.qia_layers[state['coarse']] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def gradient_record():
    assert all(len(state[key]) == 7 for key in
        ('queries', 'masks', 'gates', 'indicators', 'normalized', 'evidence_inputs', 'evidence_outputs',
         'first_linear', 'first_hidden', 'second_linear', 'second_hidden', 'gated', 'final_consumers'))
    assert state['final_consumers'] == list(range(7)) and state['coarse'] == 6
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    original = list(head.head.parameters()) + list(head.query_norm.parameters()) + list(head.evidence_norm.parameters())
    assert len(original) == 10
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in original)
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for key in ('evidence_inputs', 'evidence_outputs', 'first_hidden', 'second_hidden', 'gated')
               for value in state[key])
    gradient = gate.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_calls=7,
        shared_query_norm_calls=7, original_evidence_norm_calls=7, original_MLP_layer_calls=35,
        first_hidden_consumers=7, second_hidden_consumers=7, hidden_gate_calls=7,
        final_linear_consumers=7, original_overlap_consumers=7, coarse_gate_consumers=6,
        original_head_parameter_tensors=10, input_gradients_finite=True,
        new_gradient_max=float(gradient.abs().max()))


def install_hooks():
    hooks = [
        head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_query_norm),
        head.evidence_norm.register_forward_hook(capture_evidence),
        head.head[0].register_forward_hook(capture_first_linear),
        head.head[1].register_forward_hook(capture_first_hidden),
        head.head[2].register_forward_hook(capture_second_linear),
        head.head[3].register_forward_hook(capture_second_hidden),
        gate.register_forward_hook(capture_gate),
        head.head[4].register_forward_pre_hook(before_final_linear),
    ]
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks
