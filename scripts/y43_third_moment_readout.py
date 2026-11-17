import torch
from torch import nn


class MaskPointAsymmetryReadout(nn.Module):
    def __init__(self, d_model, in_channel):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, query, masks, padding, point_mass, third):
        valid = (~padding).unsqueeze(1).to(masks.dtype)
        probability = masks.detach().sigmoid() * valid
        weight = point_mass.unsqueeze(1) * valid
        # Signed cube root has feature units; the machine floor bounds its gradient.
        scale = third.abs().clamp_min(torch.finfo(third.dtype).eps).pow(2.0 / 3.0)
        features = self.point_encoder(third / scale)
        foreground = probability * weight
        background = (1 - probability) * weight
        positive = torch.einsum("bqm,bmd->bqd", foreground, features)
        positive = positive / foreground.sum(-1, keepdim=True).clamp_min(1e-6)
        negative = torch.einsum("bqm,bmd->bqd", background, features)
        negative = negative / background.sum(-1, keepdim=True).clamp_min(1e-6)
        return self.head(torch.cat(
            [query, positive - negative, (positive + negative) * 0.5], dim=-1))
