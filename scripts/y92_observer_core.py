"""Observe the original visual feature, one projection and all seven mask readers."""
def check_response_rows(rows):
    import math
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index)
        for key in ('original_x_mask_output_identity', 'input_undetached',
                    'float64_quadratic_projection_reference', 'shared_module_identity',
                    'all_seven_original_mask_consumers'):
            assert row[key] is True
        delta = row['delta_max']
        assert math.isfinite(delta) and (delta > 0 if index else delta == 0)


def reset_state():
    state.clear()
    state.update(queries=[], normalized=[], gates=[], indicators=[], indicator_norms=[],
                 scores=[], batches=[], coarse=0)


def capture_mask_raw(module, args, value):
    assert module is model.dec.x_mask and 'raw' not in state and value.requires_grad
    value.retain_grad()
    state['raw'] = value


def capture_projection(module, args, value):
    assert module is projection and len(args) == 1 and 'projected' not in state
    assert args[0] is state['raw'] and args[0].requires_grad and value.requires_grad
    reference = args[0].double() + args[0].double().square() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    value.retain_grad()
    state['projected'] = value
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - args[0]).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    path_rows.append(dict(step=len(step_rows), weight_nonzero=nonzero, delta_max=delta,
        original_x_mask_output_identity=True, input_undetached=True,
        float64_quadratic_projection_reference=True, shared_module_identity=True,
        all_seven_original_mask_consumers=False))


def tracked_batch(*args, **kwargs):
    if len(state['batches']) == 1:
        assert args[0] is state['projected']
    result = original_batch(*args, **kwargs)
    state['batches'].append(result)
    return result


def checked_prediction(query, features, pad):
    assert len(state['batches']) == 4
    assert features is state['batches'][1][0] and pad is state['batches'][0][1]
    assert torch.count_nonzero(features[pad]) == 0
    state['queries'].append(query)
    value = original_prediction(query, features, pad)
    index = len(state['queries']) - 1
    assert len(state['normalized']) == index + 1
    assert torch.equal(value[1], torch.einsum('bnd,bmd->bnm', state['normalized'][index], features))
    state['mask'], state['padding'] = value[1], pad
    expected = (value[1].sigmoid() < .5) | pad.unsqueeze(1)
    expected[expected.all(-1)] = False
    expected = expected | pad.unsqueeze(1)
    assert torch.equal(value[2], expected) and not value[2].requires_grad
    state['gates'].append(value[2])
    return value


def capture_out_norm(module, args, value):
    assert module is model.dec.out_norm and args[0] is state['queries'][-1] and value.requires_grad
    value.retain_grad()
    state['normalized'].append(value)


def before_score(module, args):
    index = len(state['scores'])
    assert module is model.dec.out_score and args[0] is state['normalized'][index]
    state['scores'].append(index)


def before_indicator(module, args):
    assert module is head and args[0] is state['queries'][-1]
    assert args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries']) - 1)


def capture_indicator_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['queries'][-1] and args[0].requires_grad
    state['indicator_norms'].append(True)


def before_coarse(module, args):
    assert module is model.dec.qia_layers[state['coarse']] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def gradient_record():
    assert all(len(state[key]) == 7 for key in
        ('queries', 'normalized', 'gates', 'indicators', 'indicator_norms', 'scores'))
    assert len(state['batches']) == 4 and state['coarse'] == 6
    assert state['scores'] == state['indicators'] == list(range(7))
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in [state['raw'], state['projected']] + state['normalized'])
    gradient = projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    assert path_rows[-1]['step'] == len(step_rows)
    path_rows[-1]['all_seven_original_mask_consumers'] = True
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10,
        projection_calls=1, original_mask_consumers=7, original_score_consumers=7,
        indicator_calls=7, query_norm_consumption_calls=7, coarse_gate_consumers=6,
        input_gradients_finite=True, new_gradient_max=float(gradient.abs().max()))


def install_hooks():
    hooks = [model.dec.x_mask.register_forward_hook(capture_mask_raw),
        projection.register_forward_hook(capture_projection),
        model.dec.out_norm.register_forward_hook(capture_out_norm),
        model.dec.out_score.register_forward_pre_hook(before_score),
        head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_indicator_norm)]
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    model.dec.get_batches = tracked_batch
    return hooks
