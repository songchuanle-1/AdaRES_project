"""Check a full normalized-evidence residual into the second hidden state."""
parameters = sum(parameter.numel() for parameter in model.parameters())
assert parameters == 164826501
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    rng = torch.get_rng_state().clone()
    layer = dec_module.EvidenceHiddenSkip(11, 7).to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    evidence = torch.randn(2, 5, 11, dtype=dtype, requires_grad=True)
    second = torch.randn(2, 5, 7, dtype=dtype, requires_grad=True)
    initial = layer(evidence, second)
    assert torch.equal(initial, second)
    with torch.no_grad():
        layer.weight.copy_((torch.arange(77, dtype=dtype).reshape(11, 7) - 38) / 100)
    value = layer(evidence, second)
    reference = torch.empty_like(value, dtype=torch.float64)
    for b in range(2):
        for i in range(5):
            for j in range(7):
                reference[b, i, j] = second[b, i, j].double() + sum(
                    evidence[b, i, k].double() * layer.weight[k, j].double() for k in range(11))
    tolerance = 128 * torch.finfo(dtype).eps
    torch.testing.assert_allclose(value.double(), reference, rtol=tolerance, atol=tolerance)
    order = torch.tensor([3, 1, 4, 0, 2])
    torch.testing.assert_allclose(layer(evidence[:, order], second[:, order]), value[:, order],
                                 rtol=tolerance, atol=tolerance)
    eperm = torch.tensor([3, 1, 6, 0, 2, 5, 4, 10, 8, 7, 9])
    hperm = torch.tensor([3, 1, 6, 0, 2, 5, 4])
    other = dec_module.EvidenceHiddenSkip(11, 7).to(dtype=dtype)
    with torch.no_grad():
        other.weight.copy_(layer.weight[eperm][:, hperm])
    torch.testing.assert_allclose(other(evidence[:, :, eperm], second[:, :, hperm]), value[:, :, hperm],
                                 rtol=tolerance, atol=tolerance)
    assert torch.equal(layer(torch.zeros_like(evidence), second), second)
    assert not torch.equal(layer(evidence, torch.zeros_like(second)), torch.zeros_like(second))
    assert not torch.equal(layer(evidence.roll(1, 1), second), value)
    assert torch.isfinite(value).all()
    value.square().mean().backward()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().max() > 0
               for tensor in (evidence, second, layer.weight))
    with torch.no_grad():
        layer.weight.zero_()
        assert torch.equal(layer(evidence, second), initial)
    evidence_zero, second_identity = torch.autograd.grad(layer(evidence, second).sum(), (evidence, second))
    assert torch.equal(evidence_zero, torch.zeros_like(evidence_zero))
    assert torch.equal(second_identity, torch.ones_like(second_identity))
    controls.append(dict(dtype=str(dtype), no_RNG=True, zero_weight_exact_identity=True,
        independent_scalar_residual_reference=True, query_and_both_channel_permutations=True,
        zero_evidence_preserves_hidden=True, zero_hidden_receives_evidence=True,
        evidence_causal_intervention=True, both_input_and_parameter_gradients_finite_nonzero=True,
        zero_restoration_exact=True, zero_evidence_input_gradient_and_identity_hidden_gradient=True))
kwargs = dict(cfg.model.dec, sampling_module=cfg.model.sampling_module, in_channel=cfg.model.media)
torch.manual_seed(1999)
parent = parent_module.DEC(**kwargs)
rng = torch.get_rng_state().clone()
torch.manual_seed(1999)
decoder = dec_module.DEC(**kwargs)
assert torch.equal(rng, torch.get_rng_state())
assert all(torch.equal(value, decoder.state_dict()[key]) for key, value in parent.state_dict().items())
decoder.load_state_dict(model.dec.state_dict())
parent.load_state_dict({key: value for key, value in decoder.state_dict().items()
                       if not key.startswith('indi_embedding.evidence_skip.')})
torch.manual_seed(1999)
lengths = (512, 513)
total = sum(lengths)
inputs = dict(x=torch.randn(total, cfg.model.media), fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2, 1),
    batch_offsets=torch.tensor([0, lengths[0], total], dtype=torch.int32),
    lang_feats=torch.randn(2, 9, decoder.lang_proj.in_features), lang_masks=torch.ones(2, 9, dtype=torch.long),
    sp_pos=torch.randn(total, 3), feats_2d=torch.randn(total, decoder.input_proj_2d[0].in_features),
    point_mass=torch.arange(total).remainder(13).float() + 1)
inputs['lang_masks'][0, -2:] = 0
inputs['lang_feats'][0, -2:] = 0
parts = lambda value: list(value['aux_outputs']) + [value]
parent.eval()
decoder.eval()
with torch.no_grad():
    baseline = parent(**inputs)
    output = decoder(**inputs)
assert len(parts(output)) == 7
assert all(torch.equal(a[key], b[key]) for a, b in zip(parts(baseline), parts(output))
           for key in ('masks', 'scores', 'indis'))


def input_gradients(net):
    feed = dict(inputs)
    names = ('x', 'lang_feats', 'feats_2d')
    for key in names:
        feed[key] = inputs[key].clone().requires_grad_()
    value = net(**feed)
    loss = sum(part[key].square().mean() for part in parts(value) for key in ('masks', 'scores', 'indis'))
    return torch.autograd.grad(loss, tuple(feed[key] for key in names))


for a, b in zip(input_gradients(parent), input_gradients(decoder)):
    torch.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
parent.zero_grad(set_to_none=True)
decoder.zero_grad(set_to_none=True)
with torch.no_grad():
    decoder.indi_embedding.evidence_skip.weight[0, 1] = .25
    changed = decoder(**inputs)
    assert all(torch.equal(a[key], b[key]) for a, b in zip(parts(baseline), parts(changed))
               for key in ('masks', 'scores'))
    assert any(not torch.equal(a['indis'], b['indis']) for a, b in zip(parts(baseline), parts(changed)))
    active = decoder.indi_embedding.evidence_skip.forward
    decoder.indi_embedding.evidence_skip.forward = lambda evidence, second: second
    bypass = decoder(**inputs)
    assert all(torch.equal(a[key], b[key]) for a, b in zip(parts(baseline), parts(bypass))
               for key in ('masks', 'scores', 'indis'))
    decoder.indi_embedding.evidence_skip.forward = active
    decoder.indi_embedding.evidence_skip.weight.zero_()
model.dec = decoder
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
head = decoder.indi_embedding
residual = head.evidence_skip
step_rows, path_rows, state = [], [], {}
original_prediction = decoder.prediction_head
