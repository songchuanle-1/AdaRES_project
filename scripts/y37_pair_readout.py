"""Fixed scene-normalized global pair-distribution readout."""

import torch
from torch import nn


class MaskPairDistributionReadout(nn.Module):
    """Read weighted FG/FG, BG/BG and FG/BG distance-kernel expectations."""

    def __init__(self, d_model):
        super().__init__()
        self.distribution_encoder = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2),
        )

    @staticmethod
    def descriptor(masks, padding, point_mass, coordinates):
        results = []
        for b in range(masks.shape[0]):
            valid = ~padding[b]
            points = coordinates[b, valid]
            mass = point_mass[b, valid]
            probability = masks[b, :, valid].sigmoid()
            center = (mass[:, None] * points).sum(0) / mass.sum().clamp_min(1e-6)
            centered = points - center
            scale = (mass * centered.square().sum(-1)).sum() / mass.sum().clamp_min(1e-6)
            distance = torch.cdist(centered, centered).square() / scale.clamp_min(1e-6)
            foreground = probability * mass
            background = (1 - probability) * mass
            foreground = foreground / foreground.sum(-1, keepdim=True).clamp_min(1e-6)
            background = background / background.sum(-1, keepdim=True).clamp_min(1e-6)
            ff, bb, fb = [], [], []
            # The three Laplace orders are fixed by the three coordinate dimensions.
            for order in (1, 2, 3):
                kernel = torch.exp(-order * distance)
                ff.append(((foreground @ kernel) * foreground).sum(-1))
                bb.append(((background @ kernel) * background).sum(-1))
                fb.append(((foreground @ kernel) * background).sum(-1))
            results.append(torch.stack([
                torch.stack(ff, -1), torch.stack(bb, -1), torch.stack(fb, -1),
            ], -2))
        return torch.stack(results)

    def forward(self, query, masks, padding, point_mass, tokens, coordinates):
        descriptor = self.descriptor(masks, padding, point_mass, coordinates)
        encoded = self.distribution_encoder(descriptor)
        normalized_query = nn.functional.layer_norm(query, (query.shape[-1],))
        return self.head(torch.cat([
            normalized_query, encoded[..., 0, :] - encoded[..., 1, :],
            encoded[..., 2, :],
        ], -1))
