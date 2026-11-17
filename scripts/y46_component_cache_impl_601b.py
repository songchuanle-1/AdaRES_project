"""Exact per-call memoization of duplicate component inputs."""


def component_inputs(graph, masks, masses, tokens, collapse=False):
    started = time.monotonic()
    rows, weights, groups, cache = [], [], [], {}
    for query, mask in enumerate(masks):
        for side, selected in enumerate((mask, ~mask)):
            key = np.packbits(selected).tobytes()
            if key not in cache:
                indices = np.flatnonzero(selected)
                if not len(indices):
                    cache[key] = None
                else:
                    if collapse:
                        count, labels = 1, np.zeros(len(indices), dtype=np.int32)
                    else:
                        count, labels = connected_components(
                            graph[indices][:, indices], directed=False)
                    component_mass = np.bincount(
                        labels, weights=masses[indices], minlength=count)
                    feature_sum = np.zeros((count, tokens.shape[1]), dtype=np.float64)
                    np.add.at(feature_sum, labels, tokens[indices] * masses[indices, None])
                    total_mass = masses[indices].sum()
                    assert component_mass.sum() == total_mass
                    assert (component_mass > 0).all()
                    fraction = component_mass / total_mass
                    cache[key] = (
                        np.concatenate((
                            feature_sum / component_mass[:, None], fraction[:, None]), 1),
                        fraction)
            value = cache[key]
            if value is None:
                continue
            feature_rows, fraction = value
            rows.append(feature_rows)
            weights.append(fraction)
            groups.append(np.full(len(fraction), 2 * query + side, dtype=np.int64))
    result = (
        torch.from_numpy(np.concatenate(rows)),
        torch.from_numpy(np.concatenate(weights)),
        torch.from_numpy(np.concatenate(groups)))
    cached_seconds = time.monotonic() - started
    if P["physical_gpu"] is None:
        reference_started = time.monotonic()
        original = reference_component_inputs(graph, masks, masses, tokens, collapse)
        assert all(torch.equal(left, right) for left, right in zip(result, original))
        component_cache_checks.append(dict(
            exact_rows_weights_groups=True, masks=len(masks),
            unique_selected_sets=len(cache), original_selected_sets=2 * len(masks),
            cached_seconds=cached_seconds,
            reference_seconds=time.monotonic() - reference_started))
    return result
