"""Fixed higher-order incidence information path, no inference selector change."""
import torch
from torch import nn


def joint_incidence_descriptor(pred_masks, batch_mask, point_mass):
    valid = (~batch_mask).unsqueeze(1).to(pred_masks.dtype)
    probability = pred_masks.sigmoid() * valid
    mass = point_mass.unsqueeze(1) * valid
    weighted = probability * mass
    area = weighted.sum(-1, keepdim=True)
    total = mass.sum(-1, keepdim=True)
    mean = area / total.clamp_min(1e-6)
    centered = (probability - mean) * valid
    coverage = (centered.sum(1, keepdim=True) - centered) / max(1, pred_masks.shape[1] - 1)
    second = coverage.square()
    foreground = (weighted * second).sum(-1, keepdim=True) / area.clamp_min(1e-6)
    background_weight = (1.0 - probability) * mass
    background = (background_weight * second).sum(-1, keepdim=True) / background_weight.sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.cat((foreground, background), dim=-1)


class JointIncidenceReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, query, descriptor):
        normalized = nn.functional.layer_norm(query, (query.shape[-1],))
        return self.head(torch.cat((normalized, descriptor), dim=-1)).squeeze(-1)
