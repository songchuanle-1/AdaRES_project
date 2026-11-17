"""Append frozen W2 setup: actual-copy correction through native task loss."""

capture_path = Path('/tmp/adares_y56_broader_unique_6debf88fed634a13b772f27bdc244bbd.json')
assert digest(capture_path) == 'c02ac3d48780e9b5557e9edacaecdcc8a98b9c27d0b35074fb4ee7e14a607cee'
capture = json.loads(capture_path.read_text())
head = model.dec.indi_embedding
for parameter in head.parameters():
    parameter.requires_grad_(True)
initial = {name: value.detach().clone() for name, value in head.state_dict().items()}
original = head.forward
rows = []
for pair in (0, 8):
    items = []
    for scene in capture['cohort'][pair * 2:pair * 2 + 2]:
        gorilla.set_random_seed(1999)
        item = dataset[scene['dataset_index']]
        single = dataset.collate_fn([copy.deepcopy(item)])
        assert str(single['scan_ids'][0]) == scene['scene']
        assert [str(v) for v in single['ann_ids']] == scene['ann_ids']
        items.append(item)
    batch = dataset.collate_fn(copy.deepcopy(items))
    for mode in ('original', 'repeat', 'inverse_multiplicity'):
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
        head.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        gorilla.set_random_seed(1999)
        activity, outputs = [], []
        def intervention(query, masks, padding, mass):
            native = original(query, masks, padding, mass)
            counts_per_expression = []
            changed = []
            for index in range(query.shape[0]):
                valid = ~padding[index]
                joint = torch.cat((query[index].detach(), masks[index, :, valid].detach()), dim=-1)
                unique, inverse, counts = torch.unique(joint, dim=0, return_inverse=True, return_counts=True)
                excess = 128 - len(unique)
                counts_per_expression.append(excess)
                if mode != 'inverse_multiplicity' or excess == 0:
                    changed.append(native[index])
                    continue
                probability = masks[index, :, valid].sigmoid()
                weighted = probability * mass[index, valid].unsqueeze(0)
                area = weighted.sum(-1)
                intersection = weighted @ probability.T
                overlap = intersection / (area[:, None] + area[None, :] - intersection).clamp_min(1e-6)
                overlap.fill_diagonal_(0)
                corrected = overlap / counts[inverse].to(query.dtype).unsqueeze(0)
                corrected = corrected / corrected.sum(-1, keepdim=True).clamp_min(1e-6)
                normalized = head.query_norm(query[index])
                context = corrected @ normalized
                evidence = torch.cat((normalized, context, normalized - context,
                                      overlap.max(-1).values.unsqueeze(-1)), dim=-1)
                changed.append(head.head(head.evidence_norm(evidence)))
            value = torch.stack(changed)
            value.retain_grad()
            outputs.append(value)
            activity.append(dict(layer=len(activity), duplicate_excess=counts_per_expression,
                                 logit_change=float((value.detach() - native.detach()).abs().max())))
            return value
        head.forward = intervention
        try:
            loss, logs = model(copy.deepcopy(batch), mode='loss')
            assert torch.isfinite(loss).all() and len(activity) == 7
            loss.backward()
            gradients = {name: float(p.grad.abs().max()) if p.grad is not None else None
                         for name, p in head.named_parameters()}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            layers = []
            for layer, value in enumerate(outputs):
                assert value.grad is not None and torch.isfinite(value.grad).all()
                layers.append(dict(layer=layer, native_loss_logit_gradient_max=float(value.grad.abs().max())))
            rows.append(dict(pair=pair, mode=mode, native_loss=float(loss.item()),
                             layer_activity=activity, layer_task_gradients=layers,
                             indicator_gradients=gradients, finite_backward=True))
        finally:
            head.forward = original
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        outputs.clear()
        del loss, logs
        model.zero_grad(set_to_none=True)
        gc.collect()
head.load_state_dict(initial, strict=True)
assert all(value._version == parameter_versions[name] for name, value in model.named_parameters()
           if not name.startswith('dec.indi_embedding.'))
assert all(torch.equal(value, initial[name]) for name, value in head.state_dict().items())
assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
assert digest(checkpoint_path) == FROZEN['checkpoint_sha256']
result = dict(complete=True, id='AdaRES-Y56', rows=rows, pairs=[0,8], expressions=16,
              native_TRAIN_batch2=True, actual_task_backward=True,
              input_selection='Fixed discovery pair8 and metadata first disjoint no-copy pair0; case-conditioned, not prevalence.',
              optimizer_steps=0, weights_saved=False, restored=True, admission=False,
              limitation='Task-loss reachability/gradient witness only, not trained benefit or causal population harm. '
                         'Model weights unchanged; load_state_dict increments head version counters without optimizing.')
Path(P['output']).write_text(json.dumps(result), encoding='utf-8')
