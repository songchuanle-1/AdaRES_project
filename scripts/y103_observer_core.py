"""Shared CPU/native callbacks for the full original spatial-memory bypass."""

def reset_state():
    state.clear()
    state.update(predictions=[], masks=[], gates=[], indicators=[], norms=[], coarse=0, memory_consumers=0, features=None, padding=None, mass=None)

def capture_original_spatial(module, args, value):
    assert module is model.dec.input_proj and 'spatial_projected' not in state
    assert args[0] is state['raw_spatial_expected'] and args[0].shape[-1] == 32
    assert value.shape[-1] == 256 and value.requires_grad
    value.retain_grad()
    state['spatial_projected'] = value

def capture_appearance(module, args, value):
    assert module is model.dec.input_proj_2d and args[0] is state['raw_appearance_expected']
    assert value.requires_grad and value.shape[-1] == 256
    value.retain_grad()
    state['appearance_projected'] = value

def capture_base_mask(module, args, value):
    assert module is model.dec.x_mask and args[0] is state['raw_spatial_expected']
    assert 'mask_base' not in state and value.requires_grad and (value.shape[-1] == 256)
    value.retain_grad()
    state['mask_base'] = value

def capture_skip(module, args, value):
    assert module is residual and len(args) == 2 and ('spatial_adapted' not in state)
    base, spatial = args
    assert base is state['spatial_projected'] and spatial is state['raw_spatial_expected']
    assert spatial.shape[-1] == 32 and base.requires_grad and value.requires_grad
    reference = base.double() + spatial.double() @ module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-05, atol=1e-06)
    assert torch.equal(value, base + spatial @ module.weight)
    value.retain_grad()
    if spatial.requires_grad:
        spatial.retain_grad()
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - base).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    state['spatial_adapted'] = value
    path_rows.append(dict(step=len(step_rows), weight_nonzero=nonzero, spatial_feature_delta_max=delta, raw_spatial_requires_grad=bool(spatial.requires_grad), original_full32_spatial_identity=True, original_full1024_appearance_identity=True, original_spatial_projection_identity=True, both_inputs_undetached=True, float64_full_spatial_projection_reference=True, original_sum_norm_consumes_skip=False, original_language_memory_consumes_skip=False, all_seven_original_static_masks_preserved=False))

def capture_fusion(module, args, value):
    assert module is model.dec.sum_norm and 'fused' not in state
    assert torch.equal(args[0], state['spatial_adapted'] + state['appearance_projected'])
    assert value.requires_grad
    value.retain_grad()
    state['fused'] = value
    path_rows[-1]['original_sum_norm_consumes_skip'] = True

def before_memory(module, args):
    assert module is model.dec.lia_layers[0] and state['memory_consumers'] == 0
    expected, padding = model.dec.get_batches(state['fused'], state['batch_offsets_expected'])
    assert torch.equal(args[0], expected) and torch.equal(args[2], padding)
    path_rows[-1]['original_language_memory_consumes_skip'] = True
    state['memory_consumers'] += 1

def checked_prediction(query, features, padding):
    if not state['predictions']:
        expected, expected_padding = model.dec.get_batches(state['mask_base'], state['batch_offsets_expected'])
        assert torch.equal(features, expected) and torch.equal(padding, expected_padding)
        state['features'], state['padding'] = (features, padding)
    assert features is state['features'] and padding is state['padding']
    state['predictions'].append(query)
    value = original_prediction(query, features, padding)
    state['masks'].append(value[1])
    expected = (value[1].sigmoid() < 0.5) | padding.unsqueeze(1)
    expected[expected.all(-1)] = False
    expected = expected | padding.unsqueeze(1)
    assert torch.equal(value[2], expected) and (not value[2].requires_grad)
    state['gates'].append(value[2])
    if len(state['predictions']) == 7:
        path_rows[-1]['all_seven_original_static_masks_preserved'] = True
    return value

def before_coarse(module, args):
    index = state['coarse']
    assert module is model.dec.qia_layers[index] and args[3] is state['gates'][-1]
    state['coarse'] += 1

def before_indicator(module, args):
    assert module is head and args[0] is state['predictions'][-1] and (args[1] is state['masks'][-1])
    assert args[2] is state['padding']
    if not state['indicators']:
        state['mass'] = args[3]
    assert args[3] is state['mass'] and (not args[3].requires_grad)
    state['indicators'].append(len(state['predictions']) - 1)

def capture_query_norm(module, args, value):
    assert module is head.query_norm and args[0] is state['predictions'][-1]
    assert args[0].requires_grad and value.requires_grad
    state['norms'].append(True)

def gradient_record():
    assert len(state['predictions']) == len(state['masks']) == len(state['gates']) == 7
    assert state['indicators'] == list(range(7)) and state['norms'] == [True] * 7
    assert state['coarse'] == 6 and state['memory_consumers'] == 1
    active = [p.grad for p in model.parameters() if p.grad is not None]
    assert active and all((torch.isfinite(g).all() for g in active))
    assert all((p.grad is None for p in model.text_encoder.parameters()))
    head_gradients = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(head_gradients) == 10 and all((torch.isfinite(g).all() for g in head_gradients))
    original = list(model.dec.x_mask.parameters()) + list(model.dec.input_proj.parameters()) + list(model.dec.input_proj_2d.parameters())
    assert len(original) == 14 and all((p.grad is not None and torch.isfinite(p.grad).all() for p in original))
    for key in ('mask_base', 'spatial_projected', 'spatial_adapted', 'appearance_projected', 'fused'):
        assert state[key].grad is not None and torch.isfinite(state[key].grad).all()
    raw = state['raw_spatial_expected']
    assert raw.requires_grad and raw.grad is not None and torch.isfinite(raw.grad).all()
    gradient = residual.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and (gradient.abs().max() > 0)
    return dict(all_finite=True, tensors=len(active), indicator_tensors=10, indicator_calls=7, prediction_calls=7, query_norm_consumption_calls=7, spatial_input_calls=1, original_spatial_projection_calls=1, original_appearance_projection_calls=1, original_base_mask_calls=1, original_sum_norm_calls=1, first_language_memory_consumers=1, coarse_gate_consumers=6, original_visual_parameter_tensors=14, base_adapted_fusion_and_mask_gradients_finite=True, raw_spatial_requires_grad=bool(raw.requires_grad), frozen_text_encoder_preserved=True, new_gradient_max=float(gradient.abs().max()))

def check_response_rows(rows):
    import math
    assert len(rows) == 2
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index)
        assert type(row['raw_spatial_requires_grad']) is bool
        for key in ('original_full32_spatial_identity', 'original_full1024_appearance_identity', 'original_spatial_projection_identity', 'both_inputs_undetached', 'float64_full_spatial_projection_reference', 'original_sum_norm_consumes_skip', 'original_language_memory_consumes_skip', 'all_seven_original_static_masks_preserved'):
            assert row[key] is True, key
        delta = row['spatial_feature_delta_max']
        assert math.isfinite(delta) and (delta > 0 if index else delta == 0)

def install_hooks():
    hooks = [model.dec.input_proj.register_forward_hook(capture_original_spatial), model.dec.input_proj_2d.register_forward_hook(capture_appearance), model.dec.x_mask.register_forward_hook(capture_base_mask), residual.register_forward_hook(capture_skip), model.dec.sum_norm.register_forward_hook(capture_fusion), model.dec.lia_layers[0].register_forward_pre_hook(before_memory), head.register_forward_pre_hook(before_indicator), head.query_norm.register_forward_hook(capture_query_norm)]
    hooks.extend((m.register_forward_pre_hook(before_coarse) for m in model.dec.qia_layers))
    return hooks
