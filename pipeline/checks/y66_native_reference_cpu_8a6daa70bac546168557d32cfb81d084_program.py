import json
from pathlib import Path
import torch
from torch import nn
torch.set_num_threads(2)
assert not torch.cuda.is_initialized()
def reference_peer_rows(query, masks, padded, mass):
    """Three complete peer rows; compute union directly per point, in double."""
    import torch
    probability = torch.sigmoid(masks.detach().double())
    probability = probability.masked_fill(padded.unsqueeze(1), 0.)
    weights = mass.detach().double().unsqueeze(1)
    peers = query.detach().double()
    result = []
    for index in (0, 64, 127):
        own = probability[:, index:index + 1]
        product = own * probability
        intersection = (product * weights).sum(-1)
        union = ((own + probability - product) * weights).sum(-1)
        overlap = intersection / union.clamp_min(1e-6)
        overlap[:, index] = 0.
        normalized = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
        result.append((normalized.unsqueeze(-1) * peers).sum(1))
    return torch.stack(result, dim=1)

class PeerStateFeedback(nn.Module):
    """Feed overlap-weighted peer states into the next query refinement."""

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model, d_model))

    def forward(self, query, pred_masks, batch_mask, point_mass):
        valid = (~batch_mask).unsqueeze(1).to(pred_masks.dtype)
        probabilities = pred_masks.sigmoid() * valid
        weighted = probabilities * point_mass.unsqueeze(1)
        area = weighted.sum(-1)
        intersection = torch.einsum('bim,bjm->bij', weighted, probabilities)
        union = area.unsqueeze(-1) + area.unsqueeze(1) - intersection
        overlap = intersection / union.clamp_min(1e-6)
        diagonal = torch.eye(overlap.shape[-1], dtype=torch.bool,
                             device=overlap.device).unsqueeze(0)
        overlap = overlap.masked_fill(diagonal, 0.0)
        competition = overlap / overlap.sum(-1, keepdim=True).clamp_min(1e-6)
        context = torch.einsum('bij,bjd->bid', competition, query)
        return query + context @ self.weight


cases = []
for dtype in (torch.float32, torch.float64):
    generator = torch.Generator().manual_seed(1999)
    query = torch.randn(2, 128, 5, dtype=dtype, generator=generator)
    masks = torch.randn(2, 128, 7, dtype=dtype, generator=generator)
    padded = torch.tensor([[False] * 5 + [True] * 2, [False] * 7])
    mass = torch.tensor([[1., 3., 2., 4., 5., 777., 999.], [1., 2., 1., 3., 1., 2., 4.]], dtype=dtype)
    module = PeerStateFeedback(5).to(dtype)
    assert torch.equal(module.weight, torch.zeros_like(module.weight))
    with torch.no_grad():
        module.weight.copy_(torch.arange(25, dtype=dtype).reshape(5, 5) / 200.)
    for name in ('ordinary', 'zero_mass', 'all_padding', 'extreme_logits'):
        current_masks = masks * (100. if name == 'extreme_logits' else 1.)
        current_mass = torch.zeros_like(mass) if name == 'zero_mass' else mass
        current_padding = torch.ones_like(padded) if name == 'all_padding' else padded
        context = reference_peer_rows(query, current_masks, current_padding, current_mass)
        # Full scalar loops over every peer and point, avoiding vectorized overlap algebra.
        scalar = torch.zeros(2, 3, 5, dtype=torch.float64)
        probabilities = current_masks.double().sigmoid()
        for batch in range(2):
            for row, own in enumerate((0, 64, 127)):
                overlaps = []
                for peer in range(128):
                    intersection = own_area = peer_area = 0.
                    for point in range(7):
                        if bool(current_padding[batch, point]): continue
                        a = float(probabilities[batch, own, point])
                        b = float(probabilities[batch, peer, point])
                        weight = float(current_mass[batch, point])
                        intersection += a * b * weight
                        own_area += a * weight
                        peer_area += b * weight
                    overlap = 0. if peer == own else intersection / max(own_area + peer_area - intersection, 1e-6)
                    overlaps.append(overlap)
                denominator = max(sum(overlaps), 1e-6)
                for peer, overlap in enumerate(overlaps):
                    scalar[batch, row] += query[batch, peer].double() * (overlap / denominator)
        torch.testing.assert_close(context, scalar, rtol=1e-10, atol=1e-10)
        observed = module(query, current_masks, current_padding, current_mass)[:, [0, 64, 127]].double()
        expected = query[:, [0, 64, 127]].double() + context @ module.weight.detach().double()
        torch.testing.assert_close(observed, expected, rtol=3e-5, atol=2e-6)
        cases.append(dict(dtype=str(dtype), case=name, scalar_reference_passed=True,
                          actual_component_matches_native_reference=True))
assert len(cases) == 8 and not torch.cuda.is_initialized()
Path(P['output']).write_text(json.dumps(dict(complete=True, id='AdaRES-Y66',
    native_reference_CPU_checked=True, cases=cases, CUDA_initialized=False,
    native_smoke_passed=False, native_task_backward=False, admission=False,
    limitation='CPU reference/guard preparation only, no native sparse TRAIN steps or efficacy.')),
    encoding='utf-8')
