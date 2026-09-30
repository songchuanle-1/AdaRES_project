"""Final spatial JS with scene-balanced reduction and strict pair identity."""

import torch


def target_pair_js(out, pairs, object_ids, scenes_len, point_mass, enabled=True):
    masks, indicators = out["masks"], out["indis"]
    zero = (masks.sum() + indicators.sum()) * 0
    if not enabled:
        return zero, 0
    if pairs is None:
        raise ValueError("Y17 training requires explicit target_pairs")
    if len(pairs) != len(scenes_len):
        raise ValueError("Pair scene count mismatch")
    probabilities = masks.sigmoid()
    weights = (indicators[..., 1]-indicators[..., 0]).softmax(-1)
    mixture = (weights.unsqueeze(-1)*probabilities).sum(1)
    scene_values, offset, pair_count = [], 0, 0
    for scene, count in enumerate(scenes_len):
        count = int(count)
        terms, used = [], set()
        for left, right in pairs[scene]:
            if not offset <= left < offset+count or not offset <= right < offset+count:
                raise ValueError("Cross-scene pair")
            if left == right or left in used or right in used:
                raise ValueError("Repeated/self training pair")
            if tuple(sorted(object_ids[left])) != tuple(sorted(object_ids[right])):
                raise ValueError("Different-object positive pair")
            used.update((left, right))
            valid = ~out["batch_mask"][left]
            if not torch.equal(valid, ~out["batch_mask"][right]):
                raise ValueError("Pair padding differs")
            mass = point_mass[left, valid]
            if not torch.equal(mass, point_mass[right, valid]) or mass.sum() <= 0:
                raise ValueError("Pair point mass differs or is empty")
            p = mixture[left, valid].clamp(1e-6, 1-1e-6)
            q = mixture[right, valid].clamp(1e-6, 1-1e-6)
            m = (p+q)/2
            kl_p = p*(p/m).log()+(1-p)*((1-p)/(1-m)).log()
            kl_q = q*(q/m).log()+(1-q)*((1-q)/(1-m)).log()
            terms.append((((kl_p+kl_q)/2)*mass).sum()/mass.sum())
        if terms:
            scene_values.append(torch.stack(terms).mean())
            pair_count += len(terms)
        offset += count
    if offset != len(masks) or offset != len(object_ids):
        raise ValueError("Pair reduction expression count mismatch")
    return (torch.stack(scene_values).mean() if scene_values else zero), pair_count
