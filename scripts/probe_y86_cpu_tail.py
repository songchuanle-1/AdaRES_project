"""Synthetic CPU optimizer exercise; this is never a native sparse TRAIN task."""
hooks = [head.register_forward_pre_hook(before_indicator),
    head.head.register_forward_hook(capture_original_head), shortcut.register_forward_hook(capture_shortcut),
    head.query_norm.register_forward_hook(capture_norm)]
decoder.get_batches = tracked_batch
decoder.prediction_head = checked_prediction
try:
    model.train(True)
    for index in range(2):
        state.clear()
        state.update(batches=[], predictions=0, indicators=0, norms=0, shortcuts=0, retained=[])
        optimizer.zero_grad()
        torch.manual_seed(1999 + index)
        value = decoder(**inputs)
        assert state['predictions'] == state['indicators'] == state['norms'] == state['shortcuts'] == 7
        labels = torch.arange(256).reshape(2, 128).remainder(2).long()
        loss = sum(torch.nn.functional.cross_entropy(stage['indis'].reshape(-1, 2), labels.reshape(-1))
                   + stage['masks'].square().mean() for stage in parts(value))
        assert torch.isfinite(loss)
        loss.backward()
        active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert active and all(torch.isfinite(gradient).all() for gradient in active)
        gradient = shortcut.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.all(gradient.abs() > 0)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in original_head_parameters)
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in state['retained'])
        before = shortcut.weight.detach().clone()
        optimizer.step()
        updates = (shortcut.weight - before).abs()
        assert torch.all(updates > 0)
        step_rows.append(dict(step=index, synthetic_CPU_decoder_loss=float(loss),
            new_gradients=gradient.detach().abs().tolist(), parameter_updates=updates.detach().tolist(),
            prediction_calls=7, indicator_calls=7, query_norm_calls=7, shortcut_calls=7,
            original_head_gradient_tensors=10, original_query_and_mask_gradients_finite=True))
    check_shortcut_rows(path_rows)
finally:
    decoder.get_batches = original_batch
    decoder.prediction_head = original_prediction
    for hook in hooks:
        hook.remove()
model.eval()
assert all(not module.training for module in model.modules())
model.train(True)
assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'],
    goal_sha256=PAYLOAD['goal_sha256'], implementation_test_passed=True,
    full_model_CPU_constructed=True, configured_TRAIN_data_collated=True, released_backbone_loaded=True,
    native_YAML_semantically_unchanged=True, train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]), expressions=len(batch['ann_ids']), parameters=parameters,
    additional_parameters=2, component_controls=controls,
    decoder_initial_RNG_and_parent_weights_exact=True, decoder_all_stages_anchor_exact=True,
    decoder_input_gradients_anchor_close=True, nonzero_confidence_only_intervention_restored=True,
    original_shared_head_query_norm_consumption_verified=True, original_static_masks_and_hard_gates_verified=True,
    synthetic_CPU_decoder_optimizer_steps=2, CPU_decoder_steps=step_rows, CPU_shortcut_paths=path_rows,
    CUDA_initialized=False, native_smoke_passed=False, native_task_backward=False, native_sparse_forward=False,
    validation_used=False, admission=False,
    limitation='Full CPU model/backbone/TRAIN collation and synthetic decoder only; no native GPU task, efficacy or full-dataset memory bound.')), encoding='utf8')
