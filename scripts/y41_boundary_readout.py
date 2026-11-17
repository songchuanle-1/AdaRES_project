import torch
from torch import nn


class MaskBoundaryProbabilityReadout(nn.Module):
    """Retain boundary probability mass that normalized messages can discard."""

    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model + 3, d_model), nn.GELU(), nn.Linear(d_model, 2),
        )

    @staticmethod
    def descriptor(masks, padding, point_mass, coordinates):
        readings = []
        for b in range(masks.shape[0]):
            valid = ~padding[b]
            points, mass = coordinates[b, valid], point_mass[b, valid]
            p = masks[b, :, valid].sigmoid()
            if points.shape[0] < 2:
                readings.append(p.new_zeros((p.shape[0], 3)))
                continue
            distance = torch.cdist(points, points)
            distance.fill_diagonal_(float("inf"))
            adjacency = (distance == distance.min(-1, keepdim=True).values).to(p.dtype)
            adjacency = adjacency / adjacency.sum(-1, keepdim=True)
            neighbor = p @ adjacency.transpose(0, 1)
            total = mass.sum().clamp_min(1e-6)
            foreground = (p * mass).sum(-1).clamp_min(1e-6)
            background = ((1 - p) * mass).sum(-1).clamp_min(1e-6)
            fg_cut = (p * (1 - neighbor) * mass).sum(-1) / foreground
            bg_cut = ((1 - p) * neighbor * mass).sum(-1) / background
            source, target = adjacency.nonzero(as_tuple=True)
            edge_weight = adjacency[source, target] * mass[source]
            edge_square = (p[:, source] - p[:, target]).square()
            mean = (p * mass).sum(-1) / total
            variance = ((p - mean[:, None]).square() * mass).sum(-1) / total
            energy = (edge_square * edge_weight).sum(-1) / total
            readings.append(torch.stack([
                fg_cut, bg_cut, energy / variance.clamp_min(1e-6),
            ], -1))
        return torch.stack(readings)

    def forward(self, query, masks, padding, point_mass, tokens, coordinates):
        descriptor = self.descriptor(masks, padding, point_mass, coordinates)
        normalized = nn.functional.layer_norm(query, (query.shape[-1],))
        return self.head(torch.cat([normalized, descriptor], -1))
