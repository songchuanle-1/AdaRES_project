"""Prospective fixed nearest-neighbor edge realization for Y36."""

import torch
from torch import nn


class MaskLocalAdjacencyReadout(nn.Module):
    """Read boundary/interior feature messages from nearest spatial edges."""

    def __init__(self, d_model):
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, query, masks, padding, point_mass, tokens, coordinates):
        results = []
        for b in range(query.shape[0]):
            valid = ~padding[b]
            features = tokens[b, valid]
            probability = masks[b, :, valid].sigmoid()
            mass = point_mass[b, valid]
            points = coordinates[b, valid]
            if points.shape[0] < 2:
                boundary = interior = query[b].new_zeros(query[b].shape)
            else:
                # Average exact nearest-distance ties to preserve equivariance.
                distance = torch.cdist(points, points)
                distance.fill_diagonal_(float("inf"))
                closest = distance.min(-1, keepdim=True).values
                adjacency = (distance == closest).to(features.dtype)
                adjacency = adjacency / adjacency.sum(-1, keepdim=True)
                neighbor_features = adjacency @ features
                edge = self.edge_encoder(torch.cat(
                    [features, neighbor_features, features - neighbor_features], -1))
                neighbor_probability = probability @ adjacency.transpose(0, 1)
                boundary_weight = probability * (1 - neighbor_probability) * mass
                interior_weight = probability * neighbor_probability * mass
                boundary = (boundary_weight @ edge) / boundary_weight.sum(
                    -1, keepdim=True).clamp_min(1e-6)
                interior = (interior_weight @ edge) / interior_weight.sum(
                    -1, keepdim=True).clamp_min(1e-6)
            normalized_query = nn.functional.layer_norm(
                query[b], (query.shape[-1],))
            results.append(self.head(torch.cat(
                [normalized_query, boundary, interior], -1)))
        return torch.stack(results)
