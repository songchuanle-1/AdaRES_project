"""Final-only label closure on exact nonempty hard-mask equivalence classes."""


@torch.no_grad()
def close_nonempty_mask_indices(masks, padding, indices):
    labels = torch.zeros(masks.shape[:2], device=masks.device, dtype=torch.bool)
    for b, (left, _) in enumerate(indices):
        labels[b, left] = True
    closed = labels.clone()
    for b in range(len(masks)):
        binary = masks[b, :, ~padding[b]].detach().gt(0)
        keys, inverse = torch.unique(binary, dim=0, return_inverse=True)
        positives = torch.zeros(len(keys), device=masks.device, dtype=torch.long)
        positives.scatter_add_(0, inverse, labels[b].long())
        closed[b] |= (positives[inverse] > 0) & binary.any(-1)
    replacement = []
    for value, (_, right) in zip(closed, indices):
        left = value.nonzero(as_tuple=True)[0]
        replacement.append((left, torch.zeros_like(left, dtype=right.dtype)))
    return replacement
