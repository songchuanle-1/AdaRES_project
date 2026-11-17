"""Two synthetic decoder updates with the identical future native observers."""
hooks = install_hooks()
model.dec.prediction_head = checked_prediction
try:
    model.train(True)
    for index in range(2):
        reset_state()
        optimizer.zero_grad()
        torch.manual_seed(1999 + index)
        value = model.dec(**inputs)
        labels = torch.arange(256).reshape(2, 128).remainder(2).long()
        loss = sum(torch.nn.functional.cross_entropy(part['indis'].reshape(-1, 2), labels.reshape(-1))
                   + part['masks'].square().mean() for part in parts(value))
        assert torch.isfinite(loss)
        loss.backward()
        record = gradient_record()
        parameters_ = [parameter for norm in norms for parameter in norm.parameters()]
        before = [parameter.detach().clone() for parameter in parameters_]
        optimizer.step()
        updates = [float((parameter - saved).abs().max()) for parameter, saved in zip(parameters_, before)]
        assert all(update > 0 for update in updates) and all(torch.isfinite(parameter).all() for parameter in parameters_)
        record.update(step=index, synthetic_CPU_decoder_loss=float(loss), norm_parameter_updates=updates)
        step_rows.append(record)
    check_norm_rows(path_rows)
finally:
    model.dec.prediction_head = original_prediction
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
    additional_parameters=9228, component_controls=controls, decoder_initial_RNG_and_parent_weights_exact=True,
    decoder_all_stages_anchor_exact=True, decoder_input_gradients_anchor_close=True,
    aggregate_shared_evidence_norm_gradient_anchor_close=True,
    private_final_intervention_changes_only_final_confidence=True, restoration_exact=True,
    original_queries_masks_scores_and_hard_gate_preserved=True,
    original_competition_evidence_and_shared_MLP_consumers_verified=True,
    synthetic_CPU_decoder_optimizer_steps=2, CPU_decoder_steps=step_rows, CPU_norm_paths=path_rows,
    CUDA_initialized=False, native_smoke_passed=False, native_task_backward=False, native_sparse_forward=False,
    validation_used=False, admission=False,
    limitation='Original CPU model/backbone/TRAIN loading and synthetic decoder only; native task backward and trained efficacy untested.')),
    encoding='utf8')
