"""Exact confidence-shortcut callbacks for both CPU and planned native checks."""
def before_indicator(module, args):
    query, masks, pad, mass = args
    assert query is state['query'] and masks is state['mask']
    assert masks.requires_grad and query.requires_grad
    masks.retain_grad()
    query.retain_grad()
    state['inputs'] = args
    state['retained'].extend([query, masks])
    state['indicators'] += 1


def capture_original_head(module, args, value):
    assert module is head.head and value.requires_grad
    state['original_logits'] = value


def capture_shortcut(module, args, value):
    logits, overlap = args
    assert module is shortcut and logits is state['original_logits']
    assert logits.requires_grad and overlap.requires_grad and value.requires_grad
    query, masks, pad, mass = state['inputs']
    valid = (~pad).unsqueeze(1).double()
    probabilities = masks.double().sigmoid() * valid
    weighted = probabilities * mass.double().unsqueeze(1)
    areas = weighted.sum(-1)
    intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
    union = areas.unsqueeze(-1) + areas.unsqueeze(1) - intersection
    matrix = intersection / union.clamp_min(1e-6)
    diagonal = torch.eye(matrix.shape[-1], dtype=torch.bool, device=matrix.device).unsqueeze(0)
    reference_overlap = matrix.masked_fill(diagonal, 0.).max(-1).values.unsqueeze(-1)
    torch.testing.assert_allclose(overlap.double(), reference_overlap, rtol=1e-5, atol=1e-6)
    reference = logits.double() + overlap.double() * module.weight.double()
    torch.testing.assert_allclose(value.double(), reference, rtol=1e-5, atol=1e-6)
    nonzero = bool(torch.count_nonzero(module.weight))
    delta = float((value - logits).abs().max())
    assert nonzero == bool(len(step_rows)) and (delta > 0 if nonzero else delta == 0)
    path_rows.append(dict(step=len(step_rows), stage=state['shortcuts'],
        original_logits_identity=True, original_all_peer_max_overlap_reference=True,
        no_new_overlap_or_detach=True, both_inputs_undetached=True,
        float64_scalar_projection_reference=True, weight_nonzero=nonzero, confidence_delta_max=delta))
    state['shortcuts'] += 1


def tracked_batch(*args, **kwargs):
    result = original_batch(*args, **kwargs)
    state['batches'].append(result)
    return result


def checked_prediction(query, features, pad):
    assert len(state['batches']) == 4
    assert features is state['batches'][1][0] and pad is state['batches'][0][1]
    state['query'] = query
    state['predictions'] += 1
    value = original_prediction(query, features, pad)
    state['mask'] = value[1]
    expected = (value[1].sigmoid() < .5) | pad.unsqueeze(1)
    expected[expected.all(-1)] = False
    expected = expected | pad.unsqueeze(1)
    assert torch.equal(value[2], expected) and not value[2].requires_grad
    return value


def capture_norm(module, args, value):
    assert args[0] is state['query']
    state['norms'] += 1


def check_shortcut_rows(rows):
    import math
    assert len(rows) == 14
    for index, row in enumerate(rows):
        assert type(row['step']) is int and row['step'] == index // 7
        assert type(row['stage']) is int and row['stage'] == index % 7
        assert type(row['weight_nonzero']) is bool and row['weight_nonzero'] == bool(index // 7)
        for key in ('original_logits_identity', 'original_all_peer_max_overlap_reference',
                    'no_new_overlap_or_detach', 'both_inputs_undetached', 'float64_scalar_projection_reference'):
            assert row[key] is True
        delta = row['confidence_delta_max']
        assert math.isfinite(delta)
        assert delta > 0 if index // 7 else delta == 0
