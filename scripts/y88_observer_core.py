"""Observe the exact normalized-query, mask, indicator and attention consumers."""
def check_response_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index // 7)
        for key in ('original_normalized_query_identity', 'input_undetached',
                    'float64_quadratic_projection_reference', 'original_mask_consumer'):
            assert row[key] is True
        delta = row['delta_max']
        assert math.isfinite(delta) and (delta > 0 if index // 7 else delta == 0)


def reset_state():
    state.clear()
    state.update(queries=[], normalized=[], projected=[], gates=[], indicators=[],
                 indicator_norms=[], scores=[], coarse=0, features=None, padding=None)


def checked_prediction(query, features, pad):
    if not state['queries']:
        state['features'], state['padding'] = features, pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(query)
    value = original_prediction(query, features, pad)
    index = len(state['queries']) - 1
    assert len(state['projected']) == index + 1
    expected_mask = torch.einsum('bnd,bmd->bnm', state['projected'][index], features)
    assert torch.equal(value[1], expected_mask)
    state['mask'] = value[1]
    expected_gate = (value[1].sigmoid() < .5) | pad.unsqueeze(1)
    expected_gate[expected_gate.all(-1)] = False
    expected_gate = expected_gate | pad.unsqueeze(1)
    assert torch.equal(value[2], expected_gate) and not value[2].requires_grad
    state['gates'].append(value[2])
    return value


def capture_out_norm(module, args, value):
    assert args[0] is state['queries'][-1] and value.requires_grad
    state['normalized'].append(value)


def before_score(module, args):
    index = len(state['scores'])
    assert args[0] is state['normalized'][index]
    state['scores'].append(index)


def capture_projection(module, args, value):
    index = len(state['projected'])
    assert module is projection and len(args) == 1
    assert args[0] is state['normalized'][index] and args[0].requires_grad and value.requires_grad
    args[0].retain_grad()
    value.retain_grad()
    reference = args[0].double() + args[0].double().square() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - args[0]).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    state['projected'].append(value)
    path_rows.append(dict(step=len(step_rows), stage=index, weight_nonzero=nonzero,
        delta_max=delta, original_normalized_query_identity=True, input_undetached=True,
        float64_quadratic_projection_reference=True, original_mask_consumer=True))


def before_indicator(module, args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries']) - 1)


def capture_indicator_norm(module, args, value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad
    state['indicator_norms'].append(True)


def before_coarse(module, args):
    assert module is model.dec.qia_layers[state['coarse']] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def gradient_record():
    assert all(len(state[key]) == 7 for key in
        ('queries', 'normalized', 'projected', 'gates', 'indicators', 'indicator_norms', 'scores'))
    assert state['scores'] == state['indicators'] == list(range(7)) and state['coarse'] == 6
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in state['normalized'] + state['projected'])
    gradient = projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10,
        projection_calls=7, original_mask_consumers=7, original_score_consumers=7,
        indicator_calls=7, query_norm_consumption_calls=7, coarse_gate_consumers=6,
        input_gradients_finite=True, new_gradient_max=float(gradient.abs().max()))


def install_hooks():
    hooks = [projection.register_forward_hook(capture_projection),
        model.dec.out_norm.register_forward_hook(capture_out_norm),
        model.dec.out_score.register_forward_pre_hook(before_score),
        head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_indicator_norm)]
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks
