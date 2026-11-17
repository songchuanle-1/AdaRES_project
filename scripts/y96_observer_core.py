"""Callbacks shared verbatim by actual CPU and pending native sparse TRAIN probes."""
def reset_state():
    state.clear()
    state.update(predictions=[], masks=[], gates=[], indicators=[], norms=[], generator_layers=[],
                 coarse=0, features=None, padding=None, mass=None)


def capture_scg(module, args, value):
    assert module is model.dec.scg_head and 'scg' not in state
    state['scg'] = value


def capture_sampling(module, args, value):
    assert module is model.dec.sampling_module and args[0] is state['scg']
    assert len(value) == 2 and 'indices' not in state
    state['indices'] = value[0].long()


def capture_generator_layer(module, args, value):
    index = len(state['generator_layers'])
    assert module is model.dec.query_generator[index] and value.requires_grad
    if index == 0:
        assert 'seed' not in state
        indices = state['indices']
        expected = state['scg'].gather(1, indices.unsqueeze(-1).expand(-1, -1, 256))
        assert torch.equal(args[0], expected)
        assert args[0].shape[1:] == (128, 256) and args[0].requires_grad
        state['seed'] = args[0]
    else:
        assert args[0] is state['generator_layers'][-1]
    state['generator_layers'].append(value)


def capture_generated(module, args, value):
    assert module is model.dec.query_generator and len(state['generator_layers']) == 5
    assert args[0] is state['seed'] and value is state['generator_layers'][-1]
    assert 'generated' not in state
    state['generated'] = value


def capture_skip(module, args, value):
    assert module is residual and len(args) == 2 and 'initialized' not in state
    generated, seed = args
    assert generated is state['generated'] and seed is state['seed']
    assert generated.requires_grad and seed.requires_grad and value.requires_grad
    for tensor in args + (value,):
        tensor.retain_grad()
    reference = generated.double() + seed.double() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    assert torch.equal(value, generated + seed @ module.weight)
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - generated).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    state['initialized'] = value
    path_rows.append(dict(step=len(step_rows), weight_nonzero=nonzero, query_delta_max=delta,
        original_sampled_SCG_seed_exact=True, original_generator_input_identity=True,
        original_five_generator_layers_consumed=True, original_generated_output_identity=True,
        both_inputs_undetached=True, float64_seed_projection_reference=True,
        original_initial_prediction_consumes_skip=False, original_first_coarse_consumes_skip=False))


def checked_prediction(query, features, padding):
    if not state['predictions']:
        assert query is state['initialized']
        state['features'], state['padding'] = features, padding
        path_rows[-1]['original_initial_prediction_consumes_skip'] = True
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
    if index == 0:
        assert args[1] is state['initialized']
        path_rows[-1]['original_first_coarse_consumes_skip'] = True
    state['coarse'] += 1


def before_indicator(module, args):
    assert module is head and len(args) == 4
    assert args[0] is state['predictions'][-1] and args[1] is state['masks'][-1]
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
    assert len(state['generator_layers']) == 5 and state['coarse'] == 6
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    head_gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert len(head_gradients) == 10 and all(torch.isfinite(gradient).all() for gradient in head_gradients)
    for key in ('seed', 'generated', 'initialized'):
        assert state[key].grad is not None and torch.isfinite(state[key].grad).all()
    gradient = residual.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10,
        indicator_calls=7, prediction_calls=7, query_norm_consumption_calls=7, generator_layer_calls=5,
        initialization_calls=1, coarse_gate_consumers=6, both_input_gradients_finite=True,
        new_gradient_max=float(gradient.abs().max()))


def check_response_rows(rows):
    import math
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index)
        for key in ('original_sampled_SCG_seed_exact', 'original_generator_input_identity',
                    'original_five_generator_layers_consumed', 'original_generated_output_identity',
                    'both_inputs_undetached', 'float64_seed_projection_reference',
                    'original_initial_prediction_consumes_skip', 'original_first_coarse_consumes_skip'):
            assert row[key] is True, key
        delta = row['query_delta_max']
        assert math.isfinite(delta) and (delta > 0 if index else delta == 0)


def install_hooks():
    hooks = [model.dec.scg_head.register_forward_hook(capture_scg),
        model.dec.sampling_module.register_forward_hook(capture_sampling),
        model.dec.query_generator.register_forward_hook(capture_generated),
        residual.register_forward_hook(capture_skip), head.register_forward_pre_hook(before_indicator),
        head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend(module.register_forward_hook(capture_generator_layer) for module in model.dec.query_generator)
    hooks.extend(module.register_forward_pre_hook(before_coarse) for module in model.dec.qia_layers)
    return hooks
