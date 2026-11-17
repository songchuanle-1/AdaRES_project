"""Class-margin/task readout and order-balanced native inference timing."""

import statistics
from ipdn.model_paper5.adares_w2_pointmass_competition.ipdn import IPDN as Parent

scope = dict(torch=torch, nn=nn)
exec(PAYLOAD['parent_class'], scope)
ParentIndicator = scope['QueryCompetitionIndicator']
model = initialize()
pretrain_sha = digest(cfg.train.pretrain)
initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
dataset, items, native_batches = batches()
audit = []
for regime in ('released_SPFormer_formal_init', 'trained_W2_diagnostic_only'):
    if regime == 'trained_W2_diagnostic_only':
        assert digest(PAYLOAD['checkpoint']) == PAYLOAD['checkpoint_sha256']
        weights = dict(torch.load(PAYLOAD['checkpoint'], map_location='cpu')['model'])
        extras = set(weights)-set(model.state_dict())
        assert extras.issubset({'text_encoder.embeddings.position_ids', 'text_encoder.embeddings.token_type_ids'})
        for key in extras:
            assert torch.equal(weights[key].cpu(), dict(model.named_buffers())[key].cpu())
            del weights[key]
        model.load_state_dict(weights, strict=True)
        del weights
    else:
        model.load_state_dict(initial, strict=True)
    model.train()
    model.zero_grad(set_to_none=True)
    captured = []
    def observe(head, args, output):
        output.retain_grad()
        with torch.no_grad():
            parent_output = ParentIndicator.forward(head, *args)
            repeat = Indicator.forward(head, *args)
            assert torch.equal(output, repeat)
        captured.append((output, parent_output))
    hook = model.dec.indi_embedding.register_forward_hook(observe)
    gorilla.set_random_seed(1999)
    try:
        loss, logs = model(copy.deepcopy(native_batches[4]), mode='loss')
        loss.backward()
        assert len(captured) == 7 and torch.isfinite(loss).all()
        layers = []
        for layer, (output, native) in enumerate(captured):
            assert output.grad is not None and torch.isfinite(output.grad).all()
            delta = output.detach()-native
            margin_delta = delta[..., 1]-delta[..., 0]
            probability_delta = output.detach().softmax(-1)[..., 1]-native.softmax(-1)[..., 1]
            layers.append(dict(layer=layer, maximum_logit_delta=float(delta.abs().max()),
                               maximum_class_margin_delta=float(margin_delta.abs().max()),
                               maximum_confidence_delta=float(probability_delta.abs().max()),
                               native_task_first_order_delta=float((output.grad*delta).sum()),
                               native_loss_logit_gradient_max=float(output.grad.abs().max()),
                               logit_arithmetic_bound=128*torch.finfo(output.dtype).eps*max(1.0, float(native.abs().max())),
                               probability_arithmetic_bound=128*torch.finfo(output.dtype).eps,
                               repeat_error=0))
        audit.append(dict(regime=regime, pair=4, loss=float(loss.item()), layers=layers))
    finally:
        hook.remove()
    captured.clear()
    model.zero_grad(set_to_none=True)
    del loss, logs
    gc.collect()

model.load_state_dict(initial, strict=True)
gorilla.set_random_seed(1999)
parent = Parent(**cfg.model).cuda()
parent.load_state_dict(initial, strict=True)
parent.eval()
model.eval()
models = dict(parent=parent, candidate=model)
profile = []
for index, item in enumerate(items):
    batch = dataset.collate_fn([copy.deepcopy(item)])
    for iteration in range(7):
        roles = ('parent', 'candidate') if (index+iteration) % 2 == 0 else ('candidate', 'parent')
        for position, role in enumerate(roles):
            with torch.no_grad():
                gorilla.set_random_seed(1999)
                current = copy.deepcopy(batch)
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = models[role](current, mode='predict')
                torch.cuda.synchronize()
                seconds = time.perf_counter()-started
                assert all(int(torch.as_tensor(v).numel()) == 128 for v in output['indi'])
            profile.append(dict(scene=PAYLOAD['scenes'][index]['scan_id'], iteration=iteration,
                                warmup=iteration < 2, role=role, order_position=position,
                                seconds=seconds))
            del current, output
assert digest(cfg.train.pretrain) == pretrain_sha and preserved()
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y56', revision=2,
    rows=audit, cases=profile, fixed_TRAIN_scenes=10, timing_warmups=2, timing_measured=5,
    same_strict_initial_weights=True, two_models_resident_in_one_diagnostic_process=True,
    memory_scope='Dual residency is for paired timing only; production memory uses prior single-model stress.',
    optimizer_steps=0, weights_saved=False, admission=False,
    limitation='Class-margin/native-gradient case readout and order-balanced bounded costs; not trained efficacy or global Pareto.')),
    encoding='utf-8')
