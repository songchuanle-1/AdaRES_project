"""Same raw-token/projection/norm/consumer callbacks for CPU and native task."""
def reset_state():
    state.clear()
    state.update(predictions=[], masks=[], gates=[], indicators=[], norms=[], language_self=[],
                 coarse=0, features=None, padding=None, mass=None)


def capture_projection(module, args, value):
    assert module is model.dec.lang_proj and 'projected' not in state
    assert args[0] is state['raw_expected'] and args[0].shape[-1] == 768
    assert value.shape[-1] == 256 and value.requires_grad
    state['projected'], state['raw'] = value, args[0]


def capture_adapter(module, args, value):
    assert module is residual and len(args) == 2 and 'adapted' not in state
    projected, raw = args
    assert projected is state['projected'] and raw is state['raw'] and projected.requires_grad
    assert raw is state['raw_expected'] and value.requires_grad
    reference = projected.double() + raw.double().square() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    assert torch.equal(value, projected + raw.square() @ module.weight)
    projected.retain_grad()
    value.retain_grad()
    if raw.requires_grad:
        raw.retain_grad()
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - projected).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    state['adapted'] = value
    path_rows.append(dict(step=len(step_rows), weight_nonzero=nonzero, language_delta_max=delta,
        raw_requires_grad=bool(raw.requires_grad), original_raw_token_identity=True,
        original_projection_identity=True, full768_channels_retained=True,
        float64_quadratic_projection_reference=True, original_language_norm_consumes_adapter=False,
        original_sampling_consumes_normalized_language=False, original_first_language_self_consumes_norm=False))


def capture_language_norm(module, args, value):
    assert module is model.dec.lang_norm and args[0] is state['adapted'] and 'language' not in state
    assert module.eps == 1e-5 and value.requires_grad
    reference = torch.nn.functional.layer_norm(args[0].double(), module.normalized_shape,
        module.weight.double(), module.bias.double(), module.eps)
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    value.retain_grad()
    state['language'] = value
    path_rows[-1]['original_language_norm_consumes_adapter'] = True


def before_sampling(module, args):
    assert module is model.dec.sampling_module and args[1] is state['language']
    assert torch.equal(args[3], ~state['mask_expected'].bool()) and args[3].dtype == torch.bool
    assert not args[3].requires_grad and 'sampling' not in state
    state['sampling'] = True
    path_rows[-1]['original_sampling_consumes_normalized_language'] = True


def before_language_self(module, args):
    index = len(state['language_self'])
    assert module is model.dec.lla_layers[index]
    assert torch.equal(args[1], ~state['mask_expected'].bool())
    if index == 0:
        assert args[0] is state['language']
        path_rows[-1]['original_first_language_self_consumes_norm'] = True
    state['language_self'].append(index)


def checked_prediction(query, features, padding):
    if not state['predictions']:
        state['features'], state['padding'] = features, padding
    assert features is state['features'] and padding is state['padding']
    state['predictions'].append(query)
    value = original_prediction(query, features, padding)
    state['masks'].append(value[1])
    expected = (value[1].sigmoid() < .5) | padding.unsqueeze(1)
    expected[expected.all(-1)] = False
    expected = expected | padding.unsqueeze(1)
    assert torch.equal(value[2], expected) and not value[2].requires_grad
    state['gates'].append(value[2])
    return value


def before_coarse(module, args):
    index = state['coarse']
    assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1]
    state['coarse'] += 1


def before_indicator(module, args):
    assert module is head and args[0] is state['predictions'][-1] and args[1] is state['masks'][-1]
    assert args[2] is state['padding']
    if not state['indicators']:
        state['mass'] = args[3]
    assert args[3] is state['mass'] and not args[3].requires_grad
    state['indicators'].append(len(state['predictions']) - 1)


def capture_query_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['predictions'][-1]
    assert args[0].requires_grad and value.requires_grad
    state['norms'].append(True)


def gradient_record():
    assert len(state['predictions']) == len(state['masks']) == len(state['gates']) == 7
    assert state['indicators'] == list(range(7)) and state['norms'] == [True] * 7
    assert state['language_self'] == list(range(6)) and state['coarse'] == 6 and state['sampling']
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    head_gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert len(head_gradients) == 10 and all(torch.isfinite(gradient).all() for gradient in head_gradients)
    original = list(model.dec.lang_proj.parameters()) + list(model.dec.lang_norm.parameters())
    assert len(original) == 4 and all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                      for parameter in original)
    for key in ('projected', 'adapted', 'language'):
        assert state[key].grad is not None and torch.isfinite(state[key].grad).all()
    if state['raw'].requires_grad:
        assert state['raw'].grad is not None and torch.isfinite(state['raw'].grad).all()
    gradient = residual.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10, indicator_calls=7, prediction_calls=7,
        query_norm_consumption_calls=7, language_adapter_calls=1, original_projection_calls=1,
        original_language_norm_calls=1, original_sampling_consumers=1, original_language_self_calls=6,
        coarse_gate_consumers=6, original_language_parameter_tensors=4,
        projected_adapter_normalized_gradients_finite=True, frozen_text_encoder_preserved=True,
        raw_requires_grad=bool(state['raw'].requires_grad), new_gradient_max=float(gradient.abs().max()))


def check_response_rows(rows):
    import math
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index)
        assert type(row['raw_requires_grad']) is bool
        for key in ('original_raw_token_identity', 'original_projection_identity', 'full768_channels_retained',
                    'float64_quadratic_projection_reference', 'original_language_norm_consumes_adapter',
                    'original_sampling_consumes_normalized_language', 'original_first_language_self_consumes_norm'):
            assert row[key] is True, key
        delta = row['language_delta_max']
        assert math.isfinite(delta) and (delta > 0 if index else delta == 0)


def install_hooks():
    hooks = [model.dec.lang_proj.register_forward_hook(capture_projection),
        residual.register_forward_hook(capture_adapter), model.dec.lang_norm.register_forward_hook(capture_language_norm),
        model.dec.sampling_module.register_forward_pre_hook(before_sampling),
        head.register_forward_pre_hook(before_indicator), head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(module.register_forward_pre_hook(before_language_self) for module in model.dec.lla_layers)
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks
