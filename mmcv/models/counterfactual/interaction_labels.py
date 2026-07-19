"""Factual interaction relevance pseudo-labels for logged trajectories."""

import torch


def _temporal_steps(positions):
    """Return per-step motion without treating absolute origin as velocity."""
    if positions.shape[-2] <= 1:
        return positions.new_zeros(*positions.shape[:-2], 0, positions.shape[-1])
    return positions[..., 1:, :] - positions[..., :-1, :]


def interaction_relevance_labels(
    ego_future,
    agent_futures,
    distance_threshold=4.0,
    ttc_threshold=3.0,
    path_overlap_threshold=2.0,
    dt=0.5,
):
    """Build relevance labels from observed ego/agent futures.

    Args:
        ego_future (Tensor): Observed ego future, shape (B, T, 2) or (T, 2).
        agent_futures (Tensor): Observed agent futures, shape (B, N, T, 2).

    Returns:
        Tensor: Binary labels, shape (B, N).
    """
    if ego_future.dim() == 2:
        ego_future = ego_future.unsqueeze(0)
    if agent_futures.numel() == 0:
        return torch.zeros(agent_futures.shape[:2], device=agent_futures.device)
    t = min(ego_future.shape[-2], agent_futures.shape[-2])
    ego = ego_future[:, None, :t, :]
    agents = agent_futures[:, :, :t, :]
    dist = torch.linalg.norm(agents - ego, dim=-1)
    min_dist, min_idx = dist.min(dim=-1)

    ego_steps = _temporal_steps(ego)
    agent_steps = _temporal_steps(agents)
    if ego_steps.shape[-2] == 0:
        rel_speed = torch.ones_like(min_dist)
    else:
        rel_speed = torch.linalg.norm(ego_steps - agent_steps, dim=-1).mean(dim=-1).clamp_min(1e-3)
    ttc = min_dist / rel_speed
    closest_time = min_idx.to(agent_futures.dtype) * dt

    overlap = min_dist < path_overlap_threshold
    near = min_dist < distance_threshold
    urgent = (ttc < ttc_threshold) & (closest_time < ttc_threshold + dt)
    return (near | overlap | urgent).to(agent_futures.dtype)
