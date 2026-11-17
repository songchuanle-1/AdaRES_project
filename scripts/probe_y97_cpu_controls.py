"""Full raw-token quadratic projection, independent of decoder implementation."""
parameters = sum(parameter.numel() for parameter in model.parameters())
assert parameters == 164826245
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    rng = torch.get_rng_state().clone()
    layer = dec_module.LanguageQuadraticInput(11, 7).to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    raw = torch.randn(2, 5, 11, dtype=dtype, requires_grad=True)
    projected = torch.randn(2, 5, 7, dtype=dtype, requires_grad=True)
    anchor = layer(projected, raw)
    assert torch.equal(anchor, projected)
    gradients = torch.autograd.grad(anchor.sum(), (projected, raw, layer.weight))
    assert torch.equal(gradients[0], torch.ones_like(projected))
    assert torch.equal(gradients[1], torch.zeros_like(raw))
    assert torch.isfinite(gradients[2]).all() and gradients[2].abs().max() > 0
    with torch.no_grad():
        layer.weight.copy_((torch.arange(77, dtype=dtype).reshape(11, 7) - 38) / 77)
        value = layer(projected, raw)
        reference = torch.empty_like(value, dtype=torch.float64)
        for b in range(2):
            for i in range(5):
                for d in range(7):
                    reference[b, i, d] = projected[b, i, d].double() + sum(
                        raw[b, i, k].double()**2 * layer.weight[k, d].double() for k in range(11))
        tolerance = 128 * torch.finfo(dtype).eps
        torch.testing.assert_allclose(value.double(), reference, rtol=tolerance, atol=tolerance)
        order = torch.tensor([3, 1, 4, 0, 2])
        torch.testing.assert_allclose(layer(projected[:, order], raw[:, order]), value[:, order],
                                     rtol=tolerance, atol=tolerance)
        assert torch.equal(layer(projected.flip(0), raw.flip(0)), value.flip(0))
        iperm = torch.tensor([3, 1, 6, 0, 2, 5, 4, 10, 8, 7, 9])
        operm = torch.tensor([3, 1, 6, 0, 2, 5, 4])
        other = dec_module.LanguageQuadraticInput(11, 7).to(dtype=dtype)
        other.weight.copy_(layer.weight[iperm][:, operm])
        torch.testing.assert_allclose(other(projected[:, :, operm], raw[:, :, iperm]), value[:, :, operm],
                                     rtol=tolerance, atol=tolerance)
        assert torch.equal(layer(projected, -raw), value)
        assert torch.equal(layer(projected, torch.zeros_like(raw)), projected)
        assert not torch.equal(layer(torch.zeros_like(projected), raw), torch.zeros_like(projected))
        dirty = raw.clone()
        dirty[:, 2] *= 2
        changed = layer(projected, dirty)
        assert torch.equal(changed[:, [0, 1, 3, 4]], value[:, [0, 1, 3, 4]])
        assert not torch.equal(changed[:, 2], value[:, 2])
        assert torch.isfinite(layer(projected * 1e4, raw * 1e4)).all()
    gradients = torch.autograd.grad(layer(projected, raw).square().mean(), (projected, raw, layer.weight))
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in gradients)
    with torch.no_grad():
        layer.weight.zero_()
        assert torch.equal(layer(projected, raw), anchor)
    controls.append(dict(dtype=str(dtype), no_RNG=True, zero_identity_and_input_gradients_exact=True,
        independent_scalar_raw_token_quadratic_projection=True, token_batch_input_output_channel_permutations=True,
        quadratic_sign_symmetry=True, zero_raw_preserves_projected=True, zero_projected_receives_raw=True,
        raw_intervention_is_token_local=True, both_input_and_parameter_gradients_finite_nonzero=True,
        large_finite_inputs_finite=True, exact_zero_restoration=True))
