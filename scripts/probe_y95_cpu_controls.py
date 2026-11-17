"""Independent scalar and set-context controls before full decoder anchors."""
parameters = sum(parameter.numel() for parameter in model.parameters())
assert parameters == 164695173
parent_path = root / 'ipdn/model_paper5/adares_w2_pointmass_competition/dec.py'
assert hashlib.sha256(parent_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == PAYLOAD['parent_dec_lf_sha256']
parent_module = importlib.import_module('ipdn.model_paper5.adares_w2_pointmass_competition.dec')
dec_module = importlib.import_module(namespace + '.dec')
Fusion = dec_module.QuerySetMeanSkip
assert torch.equal(model.dec.query_set_skip.weight, torch.zeros(256, 256))
controls = []
for dtype in (torch.float32, torch.float64):
    torch.manual_seed(1999)
    rng = torch.get_rng_state().clone()
    branch = Fusion(7).to(dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    self_query, language = [torch.randn(2, 5, 7, dtype=dtype, requires_grad=True) for _ in range(2)]
    anchor = branch(self_query, language)
    assert torch.equal(anchor, self_query + language)
    gradients = torch.autograd.grad(anchor.sum(), (self_query, language, branch.weight))
    assert torch.equal(gradients[0], torch.ones_like(self_query))
    assert torch.equal(gradients[1], torch.ones_like(language))
    assert torch.isfinite(gradients[2]).all() and gradients[2].abs().max() > 0
    with torch.no_grad():
        branch.weight.copy_((torch.arange(49, dtype=dtype).reshape(7, 7) - 24) / 49)
        value = branch(self_query, language)
        reference = torch.empty_like(value, dtype=torch.float64)
        for b in range(2):
            for i in range(5):
                for d in range(7):
                    reference[b, i, d] = self_query[b, i, d].double() + language[b, i, d].double() + sum(
                        sum(language[b, j, k].double() for j in range(5)) / 5 * branch.weight[k, d].double()
                        for k in range(7))
        tolerance = 128 * torch.finfo(dtype).eps
        torch.testing.assert_allclose(value.double(), reference, rtol=tolerance, atol=tolerance)
        assert not torch.allclose(value, self_query + language, rtol=tolerance, atol=tolerance)
        order = torch.tensor([3, 1, 4, 0, 2])
        torch.testing.assert_allclose(branch(self_query[:, order], language[:, order]), value[:, order],
                                     rtol=tolerance, atol=tolerance)
        assert torch.equal(branch(self_query.flip(0), language.flip(0)), value.flip(0))
        perm = torch.tensor([3, 1, 6, 0, 2, 5, 4])
        other = Fusion(7).to(dtype=dtype)
        other.weight.copy_(branch.weight[perm][:, perm])
        torch.testing.assert_allclose(other(self_query[:, :, perm], language[:, :, perm]), value[:, :, perm],
                                     rtol=tolerance, atol=tolerance)
        torch.testing.assert_allclose(branch(self_query.repeat_interleave(2, 1), language.repeat_interleave(2, 1)),
                                     value.repeat_interleave(2, 1), rtol=tolerance, atol=tolerance)
        witness = Fusion(7).to(dtype=dtype)
        witness.weight.copy_(torch.eye(7, dtype=dtype))
        zero = torch.zeros(1, 4, 7, dtype=dtype)
        changed = zero.clone()
        changed[:, 1, 0] = 4
        assert torch.equal(zero[:, 0], changed[:, 0])
        assert not torch.equal(witness(zero, changed)[:, 0], witness(zero, zero)[:, 0])
        balanced = zero.clone()
        balanced[:, 0, 0], balanced[:, 1, 0] = 1, -1
        assert torch.equal(witness(zero, balanced), balanced)
        assert torch.isfinite(branch(self_query * 1e4, language * 1e4)).all()
    gradients = torch.autograd.grad(branch(self_query, language).square().mean(),
                                    (self_query, language, branch.weight))
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in gradients)
    controls.append(dict(dtype=str(dtype), no_RNG=True, zero_sum_and_input_gradients_exact=True,
        independent_all_query_scalar_mean_projection=True, joint_query_batch_channel_permutations=True,
        query_replication_invariant=True, changed_peer_affects_unchanged_focal_query=True,
        zero_mean_set_has_no_skip=True, large_finite_inputs_finite=True,
        both_input_and_parameter_gradients_finite_nonzero=True))
