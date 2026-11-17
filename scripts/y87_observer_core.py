"""Original self/language tensors and their existing norm/FFN consumers."""
def check_path_calls(state):
    for key in ('language', 'self', 'fused', 'self_inputs', 'language_inputs', 'normalized'):
        assert len(state[key]) == 6
    assert len(state['queries']) == len(state['gates']) == 7
    for key in ('norm_consumers', 'ffn_consumers'):
        assert state[key] == list(range(6))
    assert state['indicators'] == list(range(7)) and state['norms'] == [True] * 7


def check_response_rows(rows):
    import math
    assert len(rows) == 12
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 6
        assert type(row['stage']) is int and row['stage'] == index % 6
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index // 6)
        for key in ('original_self_identity', 'original_language_identity',
                    'both_inputs_undetached', 'float64_sum_product_projection_reference'):
            assert row[key] is True
        delta = row['fusion_delta_max']
        assert math.isfinite(delta) and (delta > 0 if index // 6 else delta == 0)


def checked_prediction(query, features, pad):
    if not state['queries']:
        state['features'], state['padding'] = features, pad
    assert features is state['features'] and pad is state['padding']
    state['queries'].append(query)
    value = original_prediction(query, features, pad)
    state['mask'] = value[1]
    expected = (value[1].sigmoid() < .5) | pad.unsqueeze(1)
    expected[expected.all(-1)] = False
    expected = expected | pad.unsqueeze(1)
    assert torch.equal(value[2], expected) and not value[2].requires_grad
    state['gates'].append(value[2])
    return value


def capture_language(module, args, value):
    index = len(state['language'])
    assert module is model.dec.qla_layers[index]
    state['language'].append(value[0])


def capture_self(module, args, value):
    index = len(state['self'])
    assert module is model.dec.qqa_layers[index] and args[0] is state['language'][index]
    state['self'].append(value)


def capture_fusion(module, args, value):
    index = len(state['fused'])
    assert len(args) == 2 and args[0] is state['self'][index] and args[1] is state['language'][index]
    assert all(tensor.requires_grad for tensor in args)
    for tensor in args + (value,):
        tensor.retain_grad()
    reference = args[0].double() + args[1].double() + (args[0].double() * args[1].double()) @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - (args[0] + args[1])).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    state['self_inputs'].append(args[0])
    state['language_inputs'].append(args[1])
    state['fused'].append(value)
    path_rows.append(dict(step=len(step_rows), stage=index, weight_nonzero=nonzero,
        fusion_delta_max=delta, original_self_identity=True, original_language_identity=True,
        both_inputs_undetached=True, float64_sum_product_projection_reference=True))


def before_original_norm(module, args):
    index = len(state['norm_consumers'])
    assert module is model.dec.norm_qqa and module.eps == 1e-5 and args[0] is state['fused'][index]
    state['norm_consumers'].append(index)


def capture_original_norm(module, args, value):
    assert module is model.dec.norm_qqa and args[0] is state['fused'][len(state['normalized'])]
    reference = torch.nn.functional.layer_norm(args[0].double(), module.normalized_shape,
                                               module.weight.double(), module.bias.double(), module.eps)
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    state['normalized'].append(value)


def before_ffn(module, args):
    index = len(state['ffn_consumers'])
    assert module is model.dec.qqa_ffn_layers[index] and args[0] is state['normalized'][index]
    state['ffn_consumers'].append(index)


def before_indicator(module, args):
    assert args[0] is state['queries'][-1] and args[1] is state['mask'] and args[2] is state['padding']
    state['indicators'].append(len(state['queries']) - 1)


def capture_query_norm(module, args, value):
    assert args[0] is state['queries'][-1] and args[0].requires_grad
    state['norms'].append(True)


def reset_state():
    state.clear()
    state.update(features=None, padding=None, queries=[], gates=[], language=[], self=[], fused=[],
        self_inputs=[], language_inputs=[], normalized=[], norm_consumers=[], ffn_consumers=[],
        indicators=[], norms=[])


def gradient_record():
    check_path_calls(state)
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    head_gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert len(head_gradients) == 10 and all(torch.isfinite(gradient).all() for gradient in head_gradients)
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all()
               for tensor in state['self_inputs'] + state['language_inputs'] + state['fused'])
    gradient = fusion.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10, indicator_calls=7,
        fusion_calls=6, original_norm_consumption_calls=6, original_ffn_consumption_calls=6,
        query_norm_consumption_calls=7, both_input_gradients_finite=True, new_gradient_max=float(gradient.abs().max()))


def install_hooks():
    hooks = [fusion.register_forward_hook(capture_fusion), head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_query_norm),
        model.dec.norm_qqa.register_forward_pre_hook(before_original_norm),
        model.dec.norm_qqa.register_forward_hook(capture_original_norm)]
    hooks.extend(module.register_forward_hook(capture_language) for module in model.dec.qla_layers)
    hooks.extend(module.register_forward_hook(capture_self) for module in model.dec.qqa_layers)
    hooks.extend(module.register_forward_pre_hook(before_ffn) for module in model.dec.qqa_ffn_layers)
    return hooks
