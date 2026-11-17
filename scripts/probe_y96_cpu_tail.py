"""Causal bypass and two synthetic updates with the same pending native callbacks."""
parent.zero_grad(set_to_none=True)
decoder.zero_grad(set_to_none=True)
with torch.no_grad():
    decoder.seed_query_skip.weight[0, 1] = .25
    changed = decoder(**inputs)
    assert not torch.allclose(parts(changed)[0]['masks'], parts(baseline)[0]['masks'])
    assert not torch.allclose(changed['masks'], baseline['masks'])
    active = decoder.seed_query_skip.forward
    decoder.seed_query_skip.forward = lambda generated, seed: generated
    bypassed = decoder(**inputs)
    assert all(torch.equal(a[key], b[key]) for a, b in zip(parts(baseline), parts(bypassed))
               for key in ('masks', 'scores', 'indis'))
    decoder.seed_query_skip.forward = active
    decoder.seed_query_skip.weight.zero_()
model.dec = decoder
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
head, residual = decoder.indi_embedding, decoder.seed_query_skip
step_rows, path_rows, state = [], [], {}
original_prediction = decoder.prediction_head
# SHARED_OBSERVER_CORE
hooks = install_hooks()
decoder.prediction_head = checked_prediction
try:
    model.train(True)
    for index in range(2):
        reset_state()
        optimizer.zero_grad()
        torch.manual_seed(1999 + index)
        feed = dict(inputs)
        for key in ('x', 'lang_feats', 'feats_2d'):
            feed[key] = inputs[key].clone().requires_grad_()
        value = decoder(**feed)
        labels = torch.arange(256).reshape(2, 128).remainder(2).long()
        loss = sum(torch.nn.functional.cross_entropy(part['indis'].reshape(-1, 2), labels.reshape(-1))
            + part['masks'].square().mean() for part in parts(value))
        assert torch.isfinite(loss)
        loss.backward()
        record = gradient_record()
        assert all(feed[key].grad is not None and torch.isfinite(feed[key].grad).all()
                   for key in ('x', 'lang_feats', 'feats_2d'))
        before = residual.weight.detach().clone()
        optimizer.step()
        update = float((residual.weight.detach() - before).abs().max())
        assert update > 0 and torch.isfinite(residual.weight).all()
        record.update(step=index, synthetic_CPU_decoder_loss=float(loss), parameter_update_max=update,
                      decoder_input_gradients_finite=True)
        step_rows.append(record)
    check_response_rows(path_rows)
finally:
    decoder.prediction_head = original_prediction
    for hook in hooks:
        hook.remove()
model.eval()
assert all(not module.training for module in model.modules())
model.train(True)
assert model.training and model.text_encoder.training and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(
    complete=True, id=row['id'], config=row['config'], goal_sha256=PAYLOAD['goal_sha256'],
    implementation_test_passed=True, full_model_CPU_constructed=True, configured_TRAIN_data_collated=True,
    released_backbone_loaded=True, native_YAML_semantically_unchanged=True, train_eval_train_roundtrip=True,
    points=int(batch['feats'].shape[0]), expressions=len(batch['ann_ids']), parameters=parameters,
    additional_parameters=65536, component_controls=controls, decoder_initial_RNG_and_parent_weights_exact=True,
    decoder_all_stages_anchor_exact=True, decoder_input_gradients_anchor_close=True,
    original_sampled_seed_and_generator_consumers_verified=True, initial_and_first_coarse_consumers_verified=True,
    static_mask_and_original_hard_gate_preserved=True, original_shared_head_query_norm_consumption_verified=True,
    nonzero_seed_skip_changes_initial_and_final_masks=True, exact_bypass_restoration=True,
    synthetic_CPU_decoder_optimizer_steps=2, CPU_decoder_steps=step_rows, CPU_initialization_paths=path_rows,
    CUDA_initialized=False, native_smoke_passed=False, native_task_backward=False, native_sparse_forward=False,
    validation_used=False, admission=False,
    limitation='Original CPU YAML/model/pretrain/TRAIN collation plus synthetic decoder checks only; '
        'not native sparse GPU task, admission, trained efficacy or full-data memory proof.')), encoding='utf8')
