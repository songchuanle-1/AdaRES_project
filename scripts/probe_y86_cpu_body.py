"""CPU controls for an all-stage direct readout of already present maximum overlap."""
parameters = sum(parameter.numel() for parameter in model.parameters())
assert parameters == 164629639
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    rng = torch.get_rng_state().clone()
    module = dec_module.OverlapReadout().to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    assert module.weight.shape == (2,) and not torch.count_nonzero(module.weight)
    logits = torch.randn(2, 5, 2, dtype=dtype, requires_grad=True)
    overlap = torch.rand(2, 5, 1, dtype=dtype, requires_grad=True)
    assert torch.equal(module(logits, overlap), logits)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([.3, -.2], dtype=dtype))
    output = module(logits, overlap)
    reference = torch.stack([torch.stack([torch.stack([
        logits[b, q, channel].double() + overlap[b, q, 0].double() * module.weight[channel].double()
        for channel in range(2)]) for q in range(5)]) for b in range(2)])
    tolerance = 128 * torch.finfo(dtype).eps
    torch.testing.assert_allclose(output.double(), reference, rtol=tolerance, atol=tolerance)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    torch.testing.assert_allclose(module(logits[:, permutation], overlap[:, permutation]),
                                 output[:, permutation], rtol=tolerance, atol=tolerance)
    assert torch.equal(module(logits, torch.zeros_like(overlap)), logits)
    output.square().mean().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               and value.grad.abs().max() > 0 for value in (logits, overlap, module.weight))
    with torch.no_grad():
        module.weight.zero_()
        assert torch.equal(module(logits, overlap), logits)
    controls.append(dict(dtype=str(dtype), no_RNG_zero_identity=True,
        independent_scalar_reference=True, joint_query_permutation=True, zero_overlap_negative_control=True,
        input_and_weight_gradients_finite_nonzero=True, intervention_and_restoration=True))

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
                       if not key.startswith('indi_embedding.overlap_readout.')})
torch.manual_seed(1999)
lengths = (512, 513)
total = sum(lengths)
inputs = dict(x=torch.randn(total, cfg.model.media), fps_seed_sp=torch.arange(cfg.model.fps_num).repeat(2, 1),
    batch_offsets=torch.tensor([0, lengths[0], total], dtype=torch.int32),
    lang_feats=torch.randn(2, 9, decoder.lang_proj.in_features),
    lang_masks=torch.ones(2, 9, dtype=torch.long), sp_pos=torch.randn(total, 3),
    feats_2d=torch.randn(total, decoder.input_proj_2d[0].in_features),
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
assert all(torch.equal(left[key], right[key]) for left, right in zip(parts(baseline), parts(output))
           for key in ('masks', 'scores', 'indis'))


def input_gradients(network):
    feed = dict(inputs)
    names = ('x', 'lang_feats', 'feats_2d')
    for name in names:
        feed[name] = inputs[name].clone().requires_grad_()
    value = network(**feed)
    loss = sum(stage[key].square().mean() for stage in parts(value) for key in ('masks', 'scores', 'indis'))
    return torch.autograd.grad(loss, tuple(feed[name] for name in names))


for left, right in zip(input_gradients(parent), input_gradients(decoder)):
    torch.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
parent.zero_grad(set_to_none=True)
decoder.zero_grad(set_to_none=True)
with torch.no_grad():
    decoder.indi_embedding.overlap_readout.weight.copy_(torch.tensor([.3, -.2]))
    changed = decoder(**inputs)
    assert all(torch.equal(left[key], right[key]) for left, right in zip(parts(baseline), parts(changed))
               for key in ('masks', 'scores'))
    assert all(not torch.equal(left['indis'], right['indis']) for left, right in zip(parts(baseline), parts(changed)))
    decoder.indi_embedding.overlap_readout.weight.zero_()
    restored = decoder(**inputs)
    assert all(torch.equal(left[key], right[key]) for left, right in zip(parts(baseline), parts(restored))
               for key in ('masks', 'scores', 'indis'))
model.dec = decoder
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
head = decoder.indi_embedding
shortcut = head.overlap_readout
step_rows, path_rows, state = [], [], {}
original_prediction, original_batch = decoder.prediction_head, decoder.get_batches
original_head_parameters = [parameter for name, parameter in head.named_parameters()
                            if not name.startswith('overlap_readout.')]
assert len(original_head_parameters) == 10
