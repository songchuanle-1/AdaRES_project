"""Query-conditioned FP/FN decomposition from detached mask evidence."""

import torch
from torch import nn


class MetricDecompositionReadout(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 3))
        self.enabled = True

    def forward(self, query, masks, padding, mass, features):
        weights = masks.detach().sigmoid() * (~padding).unsqueeze(1)
        foreground = weights * mass.unsqueeze(1)
        background = ((1 - weights) * (~padding).unsqueeze(1)) * mass.unsqueeze(1)
        fg = torch.bmm(foreground, features) / foreground.sum(-1, keepdim=True).clamp_min(1)
        bg = torch.bmm(background, features) / background.sum(-1, keepdim=True).clamp_min(1)
        return self.net(torch.cat((query, fg, bg), dim=-1))
