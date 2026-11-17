"""Controls following the established complete CPU model and TRAIN loading."""
import copy
parameters = sum(parameter.numel() for parameter in model.parameters())
assert parameters == 164638865
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    shared = nn.LayerNorm(7).to(dtype=dtype)
    rng = torch.get_rng_state().clone()
    private = [shared] + [copy.deepcopy(shared) for _ in range(6)]
    assert torch.equal(rng, torch.get_rng_state())
    assert len({parameter.data_ptr() for norm in private for parameter in norm.parameters()}) == 14
    x = torch.randn(2, 5, 7, dtype=dtype, requires_grad=True)
    values = [norm(x) for norm in private]
    assert all(torch.equal(value, values[0]) for value in values)
    xd = x.double()
    mean = xd.mean(-1, keepdim=True)
    var = ((xd - mean).square()).mean(-1, keepdim=True)
    reference = (xd - mean) / (var + shared.eps).sqrt() * shared.weight.double() + shared.bias.double()
    tolerance = 128 * torch.finfo(dtype).eps
    torch.testing.assert_allclose(values[0].double(), reference, rtol=tolerance, atol=tolerance)
    order = torch.tensor([3, 1, 4, 0, 2])
    torch.testing.assert_allclose(shared(x[:, order]), values[0][:, order], rtol=tolerance, atol=tolerance)
    with torch.no_grad():
        saved = private[4].bias.clone()
        private[4].bias[0] += .25
        changed = [norm(x) for norm in private]
        assert all(torch.equal(changed[index], values[index]) for index in range(7) if index != 4)
        assert not torch.equal(changed[4], values[4])
        private[4].bias.copy_(saved)
        assert torch.equal(private[4](x), values[4])
    (private[5](x).square() * torch.arange(1, 8, dtype=dtype)).mean().backward()
    assert all(parameter.grad is None for index, norm in enumerate(private) if index != 5 for parameter in norm.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0
               for parameter in private[5].parameters())
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().max() > 0
    controls.append(dict(dtype=str(dtype),no_RNG=True,seven_disjoint_affine_norms=True,
        identical_initial_outputs=True,independent_scalar_LayerNorm=True,query_permutation=True,
        private_intervention_restoration=True,no_sibling_parameter_gradients=True,
        input_and_private_gradients_finite_nonzero=True))
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
                       if not key.startswith('indi_embedding.refinement_evidence_norms.')})
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


def aggregate_norm_gradients(net):
    net.zero_grad(set_to_none=True)
    value = net(**inputs)
    sum(part['indis'].square().mean() for part in parts(value)).backward()
    head = net.indi_embedding
    ns = [head.evidence_norm] + list(getattr(head, 'refinement_evidence_norms', []))
    return [sum(norm.state_dict(keep_vars=True)[name].grad for norm in ns).detach().clone()
            for name in ('weight', 'bias')]


for a, b in zip(aggregate_norm_gradients(parent), aggregate_norm_gradients(decoder)):
    torch.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
parent.zero_grad(set_to_none=True)
decoder.zero_grad(set_to_none=True)
with torch.no_grad():
    norm = decoder.indi_embedding.refinement_evidence_norms[-1]
    saved = norm.bias.clone()
    norm.bias[0] += .25
    changed = decoder(**inputs)
    assert all(torch.equal(a[key], b[key]) for a, b in zip(parts(baseline), parts(changed))
               for key in ('masks', 'scores'))
    assert all(torch.equal(a['indis'], b['indis']) for a, b in zip(parts(baseline)[:-1], parts(changed)[:-1]))
    assert not torch.allclose(changed['indis'], baseline['indis'])
    norm.bias.copy_(saved)
    restored = decoder(**inputs)
    assert all(torch.equal(a[key], b[key]) for a, b in zip(parts(baseline), parts(restored))
               for key in ('masks', 'scores', 'indis'))
model.dec = decoder
optimizer = gorilla.build_optimizer(model, cfg.optimizer)
head = decoder.indi_embedding
norms = [head.evidence_norm] + list(head.refinement_evidence_norms)
assert len(norms) == 7
step_rows, path_rows, state = [], [], {}
original_prediction = decoder.prediction_head
