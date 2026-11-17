"""Exact stage-private evidence normalization and shared-head consumer checks."""
def check_norm_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        for key in ('original_query_identity', 'original_mask_identity', 'original_mass_identity',
                    'private_norm_identity', 'undetached_evidence', 'independent_competition_reference',
                    'original_epsilon', 'shared_MLP_consumer'):
            assert row[key] is True
        assert row['affine_nondefault'] is bool(index // 7)
        delta = row['affine_delta_max']
        assert math.isfinite(delta) and (delta > 0 if index // 7 else delta == 0)


def reset_state():
    state.clear()
    state.update(queries=[], masks=[], gates=[], query_values=[], evidence_inputs=[],
        norm_values=[], norm_consumers=[], heads=[], coarse=0, features=None, padding=None,
        mass=None, indicator_args=[])


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
    assert module is head and len(args) == 5 and args[4] == index
    assert args[0] is state['queries'][index] and args[1] is state['masks'][index]
    assert args[2] is state['padding'] and args[0].requires_grad and args[1].requires_grad
    if index == 0:
        state['mass'] = args[3]
    assert args[3] is state['mass'] and not args[3].requires_grad
    state['indicator_args'].append(args)


def capture_query_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['queries'][-1] and value.requires_grad
    state['query_values'].append(value)


def capture_evidence_norm(module, args, value):
    index = len(state['norm_values'])
    assert module is norms[index] and len(args) == 1
    assert args[0].requires_grad and value.requires_grad and module.eps == 1e-5
    assert module.normalized_shape == (769,)
    query, masks, pad, mass, stage_index = state['indicator_args'][index]
    assert stage_index == index
    normalized = state['query_values'][index]
    probabilities = masks.sigmoid() * (~pad).unsqueeze(1).to(masks.dtype)
    weighted = probabilities * mass.unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    overlap = intersection / (areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection).clamp_min(1e-6)
    diagonal = torch.eye(query.shape[1], device=query.device, dtype=torch.bool).unsqueeze(0)
    overlap = overlap.masked_fill(diagonal, 0.)
    competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
    peer = torch.einsum('bij,bjd->bid', competition, normalized)
    reference_evidence = torch.cat((normalized, peer, normalized - peer,
                                   overlap.max(-1).values.unsqueeze(-1)), -1).double()
    torch.testing.assert_allclose(args[0].double(), reference_evidence, rtol=1e-5, atol=1e-6)
    reference = torch.nn.functional.layer_norm(args[0].double(), module.normalized_shape,
        module.weight.double(), module.bias.double(), module.eps)
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    default = torch.nn.functional.layer_norm(args[0], module.normalized_shape, None, None, module.eps)
    nondefault = bool(torch.count_nonzero(module.weight - 1) or torch.count_nonzero(module.bias))
    delta = float((value - default).abs().max())
    assert nondefault == bool(len(step_rows)) and (delta > 0 if nondefault else delta == 0)
    args[0].retain_grad()
    value.retain_grad()
    state['evidence_inputs'].append(args[0])
    state['norm_values'].append(value)
    state['norm_consumers'].append(index)
    path_rows.append(dict(step=len(step_rows), stage=index, original_query_identity=True,
        original_mask_identity=True, original_mass_identity=True, private_norm_identity=True,
        undetached_evidence=True, independent_competition_reference=True, original_epsilon=True,
        shared_MLP_consumer=True, affine_nondefault=nondefault, affine_delta_max=delta))


def before_shared_MLP(module, args):
    index = len(state['heads'])
    assert module is head.head and args[0] is state['norm_values'][index]
    state['heads'].append(index)


def before_coarse(module, args):
    assert module is model.dec.qia_layers[state['coarse']] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def gradient_record():
    assert all(len(state[key]) == 7 for key in
        ('queries', 'masks', 'gates', 'query_values', 'evidence_inputs', 'norm_values', 'indicator_args'))
    assert state['heads'] == state['norm_consumers'] == list(range(7)) and state['coarse'] == 6
    parameters = [parameter for norm in norms for parameter in norm.parameters()]
    assert len(parameters) == 14 and len({parameter.data_ptr() for parameter in parameters}) == 14
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in list(head.head.parameters()) + list(head.query_norm.parameters()))
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().max() > 0 for parameter in parameters)
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in state['evidence_inputs'] + state['norm_values'])
    return dict(all_finite=True, tensors=len(active), indicator_calls=7, shared_MLP_calls=7,
        shared_query_norm_calls=7, private_evidence_norm_calls=7, coarse_gate_consumers=6,
        shared_MLP_parameter_tensors=6, shared_query_norm_parameter_tensors=2,
        evidence_norm_gradient_tensors=14, input_gradients_finite=True,
        norm_gradient_maxima=[float(parameter.grad.abs().max()) for parameter in parameters])


def install_hooks():
    hooks = [head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_query_norm),
        head.head.register_forward_pre_hook(before_shared_MLP)]
    hooks.extend(norm.register_forward_hook(capture_evidence_norm) for norm in norms)
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks
