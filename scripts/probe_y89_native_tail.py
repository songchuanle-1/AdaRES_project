"""Use the unchanged original native TRAIN entry for two real optimizer steps."""
class FirstTwoBatches:
    sampler = loader.sampler
    def __len__(self):
        return 2
    def __iter__(self):
        for batch in itertools.islice(loader, 2):
            assert len(batch['scan_ids']) == 2
            batches.append(dict(scan_ids=[str(value) for value in batch['scan_ids']],
                expressions=len(batch['ann_ids']), points=int(batch['feats'].shape[0])))
            yield batch


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def before_model(module, args):
    bn = [value for block in (module.input_conv, module.unet, module.output_layer)
          for value in block.modules() if isinstance(value, nn.BatchNorm1d)]
    assert module.training and bn and all(value.training for value in bn)
    assert module.text_encoder.training and all(not parameter.requires_grad for parameter in module.text_encoder.parameters())
    modes.append(True)


def after_model(module, args, value):
    loss = value[0]
    assert loss.requires_grad and torch.isfinite(loss).all()
    losses.append(float(loss.detach().cpu()))


def before_decoder(module, args):
    assert len(args) == 8 and len(gradient_checks) == len(step_rows) and len(step_rows) < 2
    reset_state()


step = optimizer.step


def checked_step(*args, **kwargs):
    index = len(step_rows)
    assert index < 2 and len(losses) == len(modes) == index + 1
    record = gradient_record()
    parameters_ = [parameter for norm in norms for parameter in norm.parameters()]
    before = [parameter.detach().clone() for parameter in parameters_]
    value = step(*args, **kwargs)
    updates = [float((parameter - saved).abs().max()) for parameter, saved in zip(parameters_, before)]
    assert all(update > 0 for update in updates) and all(torch.isfinite(parameter).all() for parameter in parameters_)
    record.update(step=index, norm_parameter_updates=updates)
    gradient_checks.append(record)
    step_rows.append(dict(record))
    return value


hooks = install_hooks() + [model.register_forward_pre_hook(before_model),
    model.register_forward_hook(after_model), model.dec.register_forward_pre_hook(before_decoder)]
model.dec.prediction_head = checked_prediction
optimizer.step = checked_step
torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1, model, FirstTwoBatches(), optimizer, scheduler, cfg, logger, Writer())
    torch.cuda.synchronize()
    assert len(losses) == len(gradient_checks) == len(batches) == len(modes) == 2
    assert len({scene for batch in batches for scene in batch['scan_ids']}) == 4
    check_norm_rows(path_rows)
    peak = int(torch.cuda.max_memory_reserved())
    assert 0 < peak < 24 * 2**30
finally:
    optimizer.step = step
    model.dec.prediction_head = original_prediction
    for hook in hooks:
        hook.remove()
    if getattr(loader, '_iterator', None) is not None:
        loader._iterator._shutdown_workers()
Path(P['output']).write_text(json.dumps(dict(complete=True, id=row['id'], config=row['config'],
    code_commit=FROZEN['code_commit'], goal_sha256=FROZEN['goal_sha256'],
    native_YAML_sha256=row['config_data']['sha256'], native_entry='tools/train.py::train',
    split='train', steps=2, optimizer_steps=2, losses=losses, native_data_loaded=True,
    backward_finite=True, intended_path_observed=True, batches=batches, gradient_checks=gradient_checks,
    norm_paths=path_rows, private_evidence_norm_calls=14, shared_MLP_calls=14,
    shared_query_norm_calls=14, shared_indicator_calls=14, coarse_gate_consumers=12,
    original_competition_evidence_preserved=True, static_mask_features_preserved=True,
    original_hard_gate_algorithm_preserved=True, native_modes_verified=True, peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
                     config_data_root=cfg.data.train.data_root),
    validation_used=False, model_weights_saved=False, admission=False,
    limitation='Actual original native two-step TRAIN only; trained efficacy and full-data memory limits unmeasured.')),
    encoding='utf8')
