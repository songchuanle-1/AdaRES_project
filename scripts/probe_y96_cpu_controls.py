"""Independent sampled-seed residual controls, followed by native decoder anchors."""
parameters = sum(parameter.numel() for parameter in model.parameters())
assert parameters == 164695173
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    rng = torch.get_rng_state().clone()
    layer = dec_module.SeedQuerySkip(7).to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    generated, seeds = [torch.randn(2, 5, 7, dtype=dtype, requires_grad=True) for _ in range(2)]
    anchor = layer(generated, seeds)
    assert torch.equal(anchor, generated)
    gradients = torch.autograd.grad(anchor.sum(), (generated, seeds, layer.weight))
    assert torch.equal(gradients[0], torch.ones_like(generated))
    assert torch.equal(gradients[1], torch.zeros_like(seeds))
    assert torch.isfinite(gradients[2]).all() and gradients[2].abs().max() > 0
    with torch.no_grad():
        layer.weight.copy_((torch.arange(49, dtype=dtype).reshape(7, 7) - 24) / 49)
        value = layer(generated, seeds)
        reference = torch.empty_like(value, dtype=torch.float64)
        for b in range(2):
            for i in range(5):
                for d in range(7):
                    reference[b, i, d] = generated[b, i, d].double() + sum(
                        seeds[b, i, k].double() * layer.weight[k, d].double() for k in range(7))
        tolerance = 128 * torch.finfo(dtype).eps
        torch.testing.assert_allclose(value.double(), reference, rtol=tolerance, atol=tolerance)
        order = torch.tensor([3, 1, 4, 0, 2])
        torch.testing.assert_allclose(layer(generated[:, order], seeds[:, order]), value[:, order],
                                     rtol=tolerance, atol=tolerance)
        assert torch.equal(layer(generated.flip(0), seeds.flip(0)), value.flip(0))
        perm = torch.tensor([3, 1, 6, 0, 2, 5, 4])
        other = dec_module.SeedQuerySkip(7).to(dtype=dtype)
        other.weight.copy_(layer.weight[perm][:, perm])
        torch.testing.assert_allclose(other(generated[:, :, perm], seeds[:, :, perm]), value[:, :, perm],
                                     rtol=tolerance, atol=tolerance)
        assert torch.equal(layer(generated, torch.zeros_like(seeds)), generated)
        assert not torch.equal(layer(torch.zeros_like(generated), seeds), torch.zeros_like(generated))
        dirty = seeds.clone()
        dirty[:, 2] += 1
        changed = layer(generated, dirty)
        assert torch.equal(changed[:, [0, 1, 3, 4]], value[:, [0, 1, 3, 4]])
        assert not torch.equal(changed[:, 2], value[:, 2])
        assert torch.isfinite(layer(generated * 1e4, seeds * 1e4)).all()
    gradients = torch.autograd.grad(layer(generated, seeds).square().mean(), (generated, seeds, layer.weight))
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in gradients)
    with torch.no_grad():
        layer.weight.zero_()
        assert torch.equal(layer(generated, seeds), anchor)
    controls.append(dict(dtype=str(dtype), no_RNG=True, zero_identity_and_input_gradients_exact=True,
        independent_scalar_seed_projection=True, joint_slot_batch_channel_permutations=True,
        zero_seed_preserves_generated=True, zero_generated_receives_seed=True,
        seed_intervention_is_slot_local=True, both_input_and_parameter_gradients_finite_nonzero=True,
        large_finite_inputs_finite=True, exact_zero_restoration=True))
