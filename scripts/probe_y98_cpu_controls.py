"""Independent full-appearance projection and row-local component controls."""
parameters = sum(parameter.numel() for parameter in model.parameters())
assert parameters == 164891781
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    rng = torch.get_rng_state().clone()
    layer = dec_module.AppearanceMaskSkip(11, 7).to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    base = torch.randn(10, 7, dtype=dtype, requires_grad=True)
    appearance = torch.randn(10, 11, dtype=dtype, requires_grad=True)
    anchor = layer(base, appearance)
    assert torch.equal(anchor, base)
    gradients = torch.autograd.grad(anchor.sum(), (base, appearance, layer.weight))
    assert torch.equal(gradients[0], torch.ones_like(base))
    assert torch.equal(gradients[1], torch.zeros_like(appearance))
    assert torch.isfinite(gradients[2]).all() and gradients[2].abs().max() > 0
    with torch.no_grad():
        layer.weight.copy_((torch.arange(77, dtype=dtype).reshape(11, 7) - 38) / 77)
        value = layer(base, appearance)
        reference = torch.empty_like(value, dtype=torch.float64)
        for i in range(10):
            for d in range(7):
                reference[i, d] = base[i, d].double() + sum(
                    appearance[i, k].double() * layer.weight[k, d].double() for k in range(11))
        tolerance = 128 * torch.finfo(dtype).eps
        torch.testing.assert_allclose(value.double(), reference, rtol=tolerance, atol=tolerance)
        order = torch.tensor([3, 1, 4, 0, 2, 9, 8, 5, 7, 6])
        torch.testing.assert_allclose(layer(base[order], appearance[order]), value[order],
                                     rtol=tolerance, atol=tolerance)
        iperm = torch.tensor([3, 1, 6, 0, 2, 5, 4, 10, 8, 7, 9])
        operm = torch.tensor([3, 1, 6, 0, 2, 5, 4])
        other = dec_module.AppearanceMaskSkip(11, 7).to(dtype=dtype)
        other.weight.copy_(layer.weight[iperm][:, operm])
        torch.testing.assert_allclose(other(base[:, operm], appearance[:, iperm]), value[:, operm],
                                     rtol=tolerance, atol=tolerance)
        assert torch.equal(layer(base, torch.zeros_like(appearance)), base)
        assert not torch.equal(layer(torch.zeros_like(base), appearance), torch.zeros_like(base))
        dirty = appearance.clone()
        dirty[2] += 1
        changed = layer(base, dirty)
        keep = torch.tensor([0, 1, 3, 4, 5, 6, 7, 8, 9])
        assert torch.equal(changed[keep], value[keep])
        assert not torch.equal(changed[2], value[2])
        torch.testing.assert_allclose(layer(base, -appearance) - base, -(value - base),
                                     rtol=tolerance, atol=tolerance)
        assert torch.isfinite(layer(base * 1e4, appearance * 1e4)).all()
    gradients = torch.autograd.grad(layer(base, appearance).square().mean(), (base, appearance, layer.weight))
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in gradients)
    with torch.no_grad():
        layer.weight.zero_()
        assert torch.equal(layer(base, appearance), anchor)
    controls.append(dict(dtype=str(dtype), no_RNG=True, zero_identity_and_input_gradients_exact=True,
        independent_scalar_full_appearance_projection=True, row_input_output_channel_permutations=True,
        zero_appearance_preserves_base=True, zero_base_receives_appearance=True,
        appearance_intervention_is_row_local=True, appearance_sign_reversal=True,
        both_input_and_parameter_gradients_finite_nonzero=True, large_finite_inputs_finite=True,
        exact_zero_restoration=True))
