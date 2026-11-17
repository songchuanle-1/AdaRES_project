"""Observe all original competition inputs and learned peer-weight consumers."""
def check_response_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        assert row['weight_nonzero'] is bool(index // 7)
        for key in ('original_query_identity', 'original_mask_identity', 'original_mass_identity',
                    'normalized_query_input_undetached', 'float64_bilinear_gate_reference',
                    'original_overlap_and_max_preserved', 'weighted_peer_context_consumed'):
            assert row[key] is True
        for key in ('gate_delta_max', 'context_delta_max'):
            assert math.isfinite(row[key]) and row[key] >= 0
        assert row['gate_delta_max'] > 0 if index // 7 else row['gate_delta_max'] == 0
        if not index // 7:
            assert row['context_delta_max'] == 0
    # A constant gate can cancel under row normalization at an individual stage.
    assert max(row['context_delta_max'] for row in rows[7:]) > 0


def reset_state():
    state.clear()
    state.update(queries=[], masks=[], gates=[], indicator_args=[], normalized=[], peer_gates=[],
        gate_inputs=[], contexts=[], evidence_inputs=[], heads=[], coarse=0, features=None, padding=None, mass=None)


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
    index = len(state['indicator_args'])
    assert module is head and len(args) == 4
    assert args[0] is state['queries'][index] and args[1] is state['masks'][index]
    assert args[2] is state['padding'] and args[0].requires_grad and args[1].requires_grad
    if index == 0:
        state['mass'] = args[3]
    assert args[3] is state['mass'] and not args[3].requires_grad
    state['indicator_args'].append(args)


def capture_query_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['queries'][-1] and value.requires_grad
    state['normalized'].append(value)


def capture_peer_gate(module, args, value):
    index = len(state['peer_gates'])
    assert module is affinity and len(args) == 1 and args[0] is state['normalized'][index]
    assert args[0].requires_grad and value.requires_grad
    reference = 1 + torch.tanh(torch.einsum('bid,de,bje->bij',
        args[0].double(), module.weight.double(), args[0].double()) / (args[0].shape[-1] ** .5))
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(value).all() and value.min() >= 0 and value.max() <= 2
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - 1).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    args[0].retain_grad()
    value.retain_grad()
    state['gate_inputs'].append(args[0])
    state['peer_gates'].append(value)


def capture_evidence_input(module, args):
    index = len(state['evidence_inputs'])
    query, masks, pad, mass = state['indicator_args'][index]
    normalized = state['normalized'][index]
    probabilities = masks.sigmoid() * (~pad).unsqueeze(1).to(masks.dtype)
    weighted = probabilities * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection).clamp_min(1e-6)
    diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0.)
    weighted_overlap = overlap * state['peer_gates'][index]
    competition = weighted_overlap / weighted_overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    context = torch.einsum('bij,bjd->bid', competition, normalized)
    original_competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    original_context = torch.einsum('bij,bjd->bid', original_competition, normalized)
    reference = torch.cat((normalized, context, normalized-context, overlap.max(-1).values.unsqueeze(-1)), -1)
    assert module is head.evidence_norm and torch.equal(args[0], reference)
    assert args[0].requires_grad and torch.count_nonzero(weighted_overlap.masked_select(diagonal)) == 0
    args[0].retain_grad()
    state['contexts'].append(context)
    state['evidence_inputs'].append(args[0])
    path_rows.append(dict(step=len(step_rows), stage=index,
        weight_nonzero=bool(torch.count_nonzero(affinity.weight)),
        gate_delta_max=float((state['peer_gates'][index]-1).abs().max()),
        context_delta_max=float((context-original_context).abs().max()),
        original_query_identity=True, original_mask_identity=True, original_mass_identity=True,
        normalized_query_input_undetached=True, float64_bilinear_gate_reference=True,
        original_overlap_and_max_preserved=True, weighted_peer_context_consumed=True))


def before_shared_MLP(module, args):
    assert module is head.head and args[0].requires_grad
    state['heads'].append(len(state['heads']))


def before_coarse(module, args):
    assert module is model.dec.qia_layers[state['coarse']] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def gradient_record():
    assert all(len(state[key]) == 7 for key in
        ('queries', 'masks', 'gates', 'indicator_args', 'normalized', 'peer_gates',
         'gate_inputs', 'contexts', 'evidence_inputs', 'heads'))
    assert state['heads'] == list(range(7)) and state['coarse'] == 6
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in list(head.head.parameters()) + list(head.query_norm.parameters()) + list(head.evidence_norm.parameters()))
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in state['gate_inputs'] + state['peer_gates'] + state['evidence_inputs'])
    gradient = affinity.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True,tensors=len(active),indicator_calls=7,shared_MLP_calls=7,
        shared_query_norm_calls=7,original_evidence_norm_calls=7,peer_gate_calls=7,
        original_overlap_consumers=7,weighted_context_consumers=7,coarse_gate_consumers=6,
        original_head_parameter_tensors=10,input_gradients_finite=True,
        new_gradient_max=float(gradient.abs().max()))


def install_hooks():
    hooks = [head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_query_norm),
        affinity.register_forward_hook(capture_peer_gate),
        head.evidence_norm.register_forward_pre_hook(capture_evidence_input),
        head.head.register_forward_pre_hook(before_shared_MLP)]
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks
