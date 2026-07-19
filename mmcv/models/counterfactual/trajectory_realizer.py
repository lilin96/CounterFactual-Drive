"""Rule-based physically constrained response trajectory realization."""

import torch
from torch import nn


class RuleBasedTrajectoryRealizer(nn.Module):
    """Adjust base futures according to predicted response meta-actions.

    Inputs and outputs are local trajectories with shape (B, N, T, 2).
    If start_positions is provided, base_futures are absolute ego-frame
    positions and deltas are computed from each agent box center. Otherwise
    base_futures are treated as cumulative displacements from the origin.
    """

    def __init__(self, dt=0.5, max_speed=15.0, max_lat_step=1.2):
        super().__init__()
        self.dt = dt
        self.max_step = max_speed * dt
        self.max_lat_step = max_lat_step

    def forward(self, base_futures, speed_labels, path_labels, relevance, threshold=0.5, start_positions=None):
        if base_futures.numel() == 0:
            return base_futures
        if start_positions is not None:
            start_positions = start_positions.to(device=base_futures.device, dtype=base_futures.dtype)
            deltas = base_futures.diff(dim=-2, prepend=start_positions[..., None, :])
        else:
            deltas = base_futures.diff(dim=-2, prepend=torch.zeros_like(base_futures[..., :1, :]))
        speed_scale = torch.ones_like(relevance)
        speed_scale = torch.where(speed_labels == 1, speed_scale * 0.05, speed_scale)
        speed_scale = torch.where(speed_labels == 2, speed_scale * 0.65, speed_scale)
        speed_scale = torch.where(speed_labels == 3, speed_scale * 1.25, speed_scale)
        speed_scale = torch.where(speed_labels == 4, speed_scale * 0.8, speed_scale)
        speed_scale = torch.where(speed_labels == 5, speed_scale * 1.1, speed_scale)
        speed_scale = torch.where(speed_labels == 6, speed_scale * 0.45, speed_scale)

        lat_bias = torch.zeros_like(relevance)
        lat_bias = torch.where(path_labels == 2, lat_bias + 0.18, lat_bias)
        lat_bias = torch.where(path_labels == 3, lat_bias + 0.35, lat_bias)
        lat_bias = torch.where(path_labels == 4, lat_bias - 0.18, lat_bias)
        lat_bias = torch.where(path_labels == 5, lat_bias - 0.35, lat_bias)

        adjusted = deltas * speed_scale[..., None, None]
        t = torch.linspace(0.0, 1.0, base_futures.shape[-2], device=base_futures.device, dtype=base_futures.dtype)
        adjusted[..., 0] = adjusted[..., 0] + lat_bias[..., None] * t.clamp(max=self.max_lat_step)
        step_norm = torch.linalg.norm(adjusted, dim=-1, keepdim=True).clamp_min(1e-4)
        adjusted = adjusted * (self.max_step / step_norm).clamp(max=1.0)
        if start_positions is not None:
            realized = start_positions[..., None, :] + adjusted.cumsum(dim=-2)
        else:
            realized = adjusted.cumsum(dim=-2)
        return torch.where((relevance >= threshold)[..., None, None], realized, base_futures)
