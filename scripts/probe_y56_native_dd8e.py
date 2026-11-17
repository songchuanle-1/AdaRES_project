"""Physical Y56 loss/backward and separate trained-parent case diagnostics."""

scope = dict(torch=torch, nn=nn)
exec(PAYLOAD['parent_class'], scope)
ParentIndicator = scope['QueryCompetitionIndicator']
model = initialize()
pretrain_sha = digest(cfg.train.pretrain)
dataset, items, native_batches = batches()
initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
rows, controls = [], []
peak = 0


def run_batch(index, batch, regime, role):
    model.zero_grad(set_to_none=True)
    gorilla.set_random_seed(1999)
    head = model.dec.indi_embedding
    original_forward = head.forward
    if role == 'parent':
        head.forward = lambda *args: ParentIndicator.forward(head, *args)
    outputs, observations = [], []
    def observe(module, args, value):
        query, masks, padding, mass = args
        value.retain_grad()
        outputs.append(value)
        if role == 'parent':
            return
        with torch.no_grad():
            native = ParentIndicator.forward(head, *args)
            layer = len(observations)
            stats = []
            for expression in range(query.shape[0]):
                valid = ~padding[expression]
                joint = torch.cat((query[expression].detach(), masks[expression, :, valid].detach()), dim=-1)
                unique, inverse, count = torch.unique(joint, dim=0, return_inverse=True, return_counts=True)
                excess = int(query.shape[1] - unique.shape[0])
                delta = float((value[expression] - native[expression]).abs().max())
                if excess == 0:
                    assert torch.equal(value[expression], native[expression])
                reference_error = None
                if excess:
                    representatives = torch.stack([torch.nonzero(inverse == k, as_tuple=False)[0, 0]
                                                   for k in range(len(unique))])
                    reference = ParentIndicator.forward(head, query[expression:expression+1, representatives],
                        masks[expression:expression+1, representatives], padding[expression:expression+1],
                        mass[expression:expression+1])[0, inverse]
                    reference_error = float((value[expression] - reference).abs().max())
                    bound = 128 * torch.finfo(query.dtype).eps * max(1.0, float(reference.abs().max()))
                    assert reference_error <= bound, (regime, index, layer, expression, reference_error, bound)
                stats.append(dict(expression=expression, duplicate_excess=excess,
                                  parent_logit_change=delta, unique_reference_error=reference_error,
                                  no_copy_exact=excess == 0 and delta == 0))
            observations.append(dict(layer=layer, expressions=stats))
    hook = head.register_forward_hook(observe)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        loss, logs = model(copy.deepcopy(batch), mode='loss')
        assert torch.isfinite(loss).all() and len(outputs) == 7
        loss.backward()
        gradients = {}
        for name, p in head.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            gradients[name] = float(p.grad.abs().max())
            assert gradients[name] > 0, name
        layer_gradients = []
        for layer, output in enumerate(outputs):
            assert output.grad is not None and torch.isfinite(output.grad).all()
            maximum = float(output.grad.abs().max())
            assert maximum > 0
            layer_gradients.append(dict(layer=layer, native_loss_logit_gradient_max=maximum))
        all_gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert all_gradients and all(torch.isfinite(g).all() for g in all_gradients)
        torch.cuda.synchronize()
        row = dict(regime=regime, pair=index, role=role, expressions=len(batch['ann_ids']),
                   loss=float(loss.item()), indicator_gradients=gradients,
                   layer_task_gradients=layer_gradients, activity=observations,
                   seconds=time.perf_counter()-started,
                   peak_reserved_bytes=int(torch.cuda.max_memory_reserved()), finite_backward=True)
        assert row['peak_reserved_bytes'] < 24 * 2**30
        return row
    finally:
        hook.remove()
        head.forward = original_forward
        outputs.clear()
        model.zero_grad(set_to_none=True)
        gc.collect()


for index, batch in enumerate(native_batches):
    model.load_state_dict(initial, strict=True)
    model.train()
    row = run_batch(index, batch, 'released_SPFormer_formal_init', 'candidate')
    rows.append(row)
    peak = max(peak, row['peak_reserved_bytes'])

for index in (0, 4):
    for role in ('parent', 'repeat'):
        model.load_state_dict(initial, strict=True)
        model.train()
        rows.append(run_batch(index, native_batches[index], 'released_SPFormer_formal_init', role))
model.load_state_dict(initial, strict=True)
assert all(torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items())

# Strict same-shape trained state is diagnostic only, never formal initialization.
assert digest(PAYLOAD['checkpoint']) == PAYLOAD['checkpoint_sha256']
checkpoint = torch.load(PAYLOAD['checkpoint'], map_location='cpu')
weights = dict(checkpoint['model'])
extras = set(weights) - set(model.state_dict())
assert extras.issubset({'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'})
for key in extras:
    assert torch.equal(weights[key].cpu(), dict(model.named_buffers())[key].cpu())
    del weights[key]
model.load_state_dict(weights, strict=True)
trained = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
del weights, checkpoint
for p in model.parameters():
    p.requires_grad_(False)
for p in model.dec.indi_embedding.parameters():
    p.requires_grad_(True)
for index in (0, 4):
    for role in ('candidate', 'parent', 'repeat'):
        model.load_state_dict(trained, strict=True)
        model.train()
        rows.append(run_batch(index, native_batches[index], 'trained_W2_diagnostic_only', role))
model.load_state_dict(initial, strict=True)
assert all(torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items())
assert digest(cfg.train.pretrain) == pretrain_sha and digest(PAYLOAD['checkpoint']) == PAYLOAD['checkpoint_sha256']
assert preserved()
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y56', rows=rows,
    formal_initialization='Released SPFormer, unchanged native loader; strict trained-W2 diagnostic is separate.',
    actual_native_batch2=True, fixed_scenes=10, formal_expressions=40, all7_sites=True,
    native_backward=True, physical_independent_package=True, parameter_and_buffer_state_restored=True,
    peak_reserved_bytes=peak, optimizer_steps=0, weights_saved=False, admission=False,
    limitation='Native reachability and case-conditioned mechanism controls, not trained benefit or prevalence.')),
    encoding='utf-8')
