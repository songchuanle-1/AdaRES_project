"""Original native two-step harness for the frozen overlap-shortcut model."""
losses, gradient_checks, batches, modes, step_rows, path_rows, state = [], [], [], [], [], [], {}
head = model.dec.indi_embedding
shortcut = head.overlap_readout
original_prediction, original_batch = model.dec.prediction_head, model.dec.get_batches
original_head_parameters = [parameter for name, parameter in head.named_parameters()
                            if not name.startswith('overlap_readout.')]
assert len(original_head_parameters) == 10 and not torch.count_nonzero(shortcut.weight)


class FirstTwoBatches:
    sampler = loader.sampler

    def __len__(self):
        return 2

    def __iter__(self):
        for batch in itertools.islice(loader, 2):
            assert len(batch['scan_ids']) == 2
            batches.append(dict(scan_ids=[str(scan) for scan in batch['scan_ids']],
                expressions=len(batch['ann_ids']), points=int(batch['feats'].shape[0])))
            yield batch


class Writer:
    def add_scalar(self, *args, **kwargs):
        pass


def before_model(module, args):
    norms = [norm for block in (module.input_conv, module.unet, module.output_layer)
             for norm in block.modules() if isinstance(norm, nn.BatchNorm1d)]
    assert module.training and norms and all(norm.training for norm in norms)
    assert module.text_encoder.training and all(not parameter.requires_grad
                                                for parameter in module.text_encoder.parameters())
    modes.append(True)


def after_model(module, args, value):
    loss = value[0]
    assert loss.requires_grad and torch.isfinite(loss).all()
    losses.append(float(loss.detach().cpu()))


def before_decoder(module, args):
    assert len(args) == 8 and len(gradient_checks) == len(step_rows) < 2
    state.clear()
    state.update(batches=[], predictions=0, indicators=0, norms=0, shortcuts=0, retained=[])


step = optimizer.step


def checked_step(*args, **kwargs):
    index = len(step_rows)
    assert index < 2 and len(losses) == len(modes) == index + 1
    assert state['predictions'] == state['indicators'] == state['norms'] == state['shortcuts'] == 7
    active = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(gradient).all() for gradient in active)
    assert all(parameter.grad is None for parameter in model.text_encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in original_head_parameters)
    gradient = shortcut.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and torch.all(gradient.abs() > 0)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in state['retained'])
    before = shortcut.weight.detach().clone()
    value = step(*args, **kwargs)
    updates = (shortcut.weight - before).abs()
    assert torch.all(updates > 0)
    row = dict(step=index, all_finite=True, tensors=len(active), original_head_gradient_tensors=10,
        new_gradients=gradient.detach().abs().tolist(), parameter_updates=updates.detach().tolist(),
        prediction_calls=7, indicator_calls=7, query_norm_calls=7, shortcut_calls=7,
        original_query_and_mask_gradients_finite=True)
    gradient_checks.append(row)
    step_rows.append(dict(row))
    return value


hooks = [head.register_forward_pre_hook(before_indicator), head.head.register_forward_hook(capture_original_head),
    shortcut.register_forward_hook(capture_shortcut), head.query_norm.register_forward_hook(capture_norm),
    model.register_forward_pre_hook(before_model), model.register_forward_hook(after_model),
    model.dec.register_forward_pre_hook(before_decoder)]
model.dec.get_batches = tracked_batch
model.dec.prediction_head = checked_prediction
optimizer.step = checked_step
torch.cuda.reset_peak_memory_stats()
try:
    gorilla.set_random_seed(cfg.train.seed)
    native.train(1, model, FirstTwoBatches(), optimizer, scheduler, cfg, logger, Writer())
    torch.cuda.synchronize()
    assert len(losses) == len(gradient_checks) == len(batches) == len(modes) == 2
    assert len({scan for batch in batches for scan in batch['scan_ids']}) == 4
    check_shortcut_rows(path_rows)
    peak = int(torch.cuda.max_memory_reserved())
    assert 0 < peak < 24 * 2**30
finally:
    optimizer.step = step
    model.dec.get_batches = original_batch
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
    shortcut_paths=path_rows, shortcut_calls=14, shared_head_calls=14, query_norm_consumption_calls=14,
    static_mask_features_preserved=True, original_hard_gate_algorithm_preserved=True,
    original_max_overlap_formula_preserved=True, native_modes_verified=True, peak_reserved_bytes=peak,
    data_source=dict(dataset_class=type(dataset).__module__ + '.' + type(dataset).__name__,
        config_data_root=cfg.data.train.data_root), validation_used=False, model_weights_saved=False,
    admission=False, limitation='Actual native two-step TRAIN only; no efficacy or full-data memory guarantee.')),
    encoding='utf8')
