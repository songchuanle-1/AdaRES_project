"""The same six original-FFN/parallel-input callbacks for CPU and native TRAIN."""
def reset_state():
    state.clear()
    state.update(predictions=[], masks=[], gates=[], indicators=[], norms=[], coarse=0,
                 fusion=[], ffn_inputs=[], ffn_outputs=[], adapted=[], features=None, padding=None, mass=None)


def capture_initial(module, args, value):
    assert module is model.dec.query_generator and 'initial' not in state and value.requires_grad
    state['initial'] = value


def capture_mask(module, args, value):
    assert module is model.dec.x_mask and 'mask_base' not in state and value.requires_grad
    value.retain_grad()
    state['mask_base'] = value


def capture_fusion(module, args, value):
    assert module is model.dec.norm_qqa and len(state['fusion']) == len(state['adapted']) < 6
    assert value.requires_grad and value.shape[-1] == 256
    value.retain_grad()
    state['fusion'].append(value)


def before_ffn(module, args):
    index = len(state['ffn_inputs'])
    assert index == len(state['adapted']) < 6 and module is model.dec.qqa_ffn_layers[index]
    assert args[0] is state['fusion'][index] and args[0].requires_grad
    state['ffn_inputs'].append(args[0])


def after_ffn(module, args, value):
    index = len(state['ffn_outputs'])
    assert module is model.dec.qqa_ffn_layers[index] and args[0] is state['ffn_inputs'][index]
    assert value.requires_grad
    value.retain_grad()
    state['ffn_outputs'].append(value)


def capture_skip(module, args, value):
    index = len(state['adapted'])
    assert module is residual and len(args) == 2 and index < 6
    base, current = args
    assert base is state['ffn_outputs'][index] and current is state['ffn_inputs'][index]
    assert base.requires_grad and current.requires_grad and value.requires_grad
    torch.testing.assert_allclose(value.double(), base.double() + current.double() @ module.weight.double(), rtol=1e-5, atol=1e-6)
    assert torch.equal(value, base + current @ module.weight)
    value.retain_grad()
    state['adapted'].append(value)
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - base).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    path_rows.append(dict(step=len(step_rows), stage=index, weight_nonzero=nonzero, query_delta_max=delta,
        original_full256_FFN_input_identity=True, complete_original_FFN_output_identity=True,
        original_shared_fusion_norm_consumed=True, both_inputs_undetached=True,
        float64_full_query_projection_reference=True, prediction_consumes_adapted_query=False,
        original_static_mask_preserved=False))


def checked_prediction(query, features, padding):
    index = len(state['predictions'])
    if index == 0:
        assert query is state['initial'] and not state['adapted']
        expected, expected_padding = model.dec.get_batches(state['mask_base'], state['batch_offsets_expected'])
        assert torch.equal(features, expected) and torch.equal(padding, expected_padding)
        state['features'], state['padding'] = features, padding
    else:
        assert query is state['adapted'][index - 1] and len(state['adapted']) == index
        path_rows[-1]['prediction_consumes_adapted_query'] = True
        path_rows[-1]['original_static_mask_preserved'] = True
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
    assert state['indicators'] == list(range(7)) and state['norms'] == [True] * 7 and state['coarse'] == 6
    assert all(len(state[k]) == 6 for k in ('fusion', 'ffn_inputs', 'ffn_outputs', 'adapted'))
    active = [p.grad for p in model.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(g).all() for g in active)
    assert all(p.grad is None for p in model.text_encoder.parameters())
    hg = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(hg) == 10 and all(torch.isfinite(g).all() for g in hg)
    original = [p for m in model.dec.qqa_ffn_layers for p in m.parameters()]
    assert len(original) == 36 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in original)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.dec.norm_qqa.parameters())
    assert state['mask_base'].grad is not None and torch.isfinite(state['mask_base'].grad).all()
    for k in ('fusion', 'ffn_inputs', 'ffn_outputs', 'adapted'):
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in state[k])
    gradient = residual.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10, indicator_calls=7, prediction_calls=7,
        query_norm_consumption_calls=7, coarse_gate_consumers=6, query_FFN_skip_calls=6, original_query_FFN_calls=6,
        original_query_FFN_parameter_tensors=36, full_input_base_adapted_and_original_FFN_gradients_finite=True,
        original_shared_fusion_norm_gradients_finite=True, static_mask_gradients_finite=True,
        frozen_text_encoder_preserved=True, new_gradient_max=float(gradient.abs().max()))


def check_response_rows(rows):
    import math
    assert len(rows) == 12
    for i, row in enumerate(rows):
        step, stage = divmod(i, 6)
        assert type(row['step']) is int and row['step'] == step and type(row['stage']) is int and row['stage'] == stage
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(step)
        for k in ('original_full256_FFN_input_identity', 'complete_original_FFN_output_identity',
                  'original_shared_fusion_norm_consumed', 'both_inputs_undetached', 'float64_full_query_projection_reference',
                  'prediction_consumes_adapted_query', 'original_static_mask_preserved'):
            assert row[k] is True, k
        assert math.isfinite(row['query_delta_max']) and (row['query_delta_max'] > 0 if step else row['query_delta_max'] == 0)


def install_hooks():
    hooks = [model.dec.query_generator.register_forward_hook(capture_initial), model.dec.x_mask.register_forward_hook(capture_mask),
        model.dec.norm_qqa.register_forward_hook(capture_fusion), residual.register_forward_hook(capture_skip),
        head.register_forward_pre_hook(before_indicator), head.query_norm.register_forward_hook(capture_query_norm)]
    for m in model.dec.qqa_ffn_layers:
        hooks += [m.register_forward_pre_hook(before_ffn), m.register_forward_hook(after_ffn)]
    hooks.extend(m.register_forward_pre_hook(before_coarse) for m in model.dec.qia_layers)
    return hooks
