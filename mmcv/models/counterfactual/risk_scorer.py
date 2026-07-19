"""Rule-based risk scorer for counterfactual candidate scenes."""

import torch
from torch import nn


def _temporal_steps(positions):
    """Return adjacent-step motion for absolute or origin-relative positions."""
    if positions.shape[-2] <= 1:
        return positions.new_zeros(*positions.shape[:-2], 0, positions.shape[-1])
    return positions[..., 1:, :] - positions[..., :-1, :]


class RuleBasedRiskScorer(nn.Module):
    """Score candidate scenes from collision, TTC, comfort, progress, deviation."""

    def __init__(self, collision_distance=2.0, near_distance=5.0, dt=0.5):
        super().__init__()
        self.collision_distance = collision_distance
        self.near_distance = near_distance
        self.dt = dt

    def forward(self, ego_future, agent_futures, base_ego_future=None, relevance=None, map_context=None):
        if agent_futures.numel() == 0:
            b = ego_future.shape[0]
            zeros = ego_future.new_zeros(b)
            if base_ego_future is not None:
                target_progress = base_ego_future[..., -1, 1].clamp(min=0)
            else:
                target_progress = ego_future[..., -1, 1].detach().clamp(min=0)
            progress = torch.relu(target_progress - ego_future[..., -1, 1].clamp(min=0))
            nominal = torch.zeros_like(progress)
            if base_ego_future is not None:
                nominal = torch.linalg.norm(ego_future - base_ego_future[:, : ego_future.shape[-2]], dim=-1).mean(dim=-1)
            lateral = ego_future[..., 0].abs()
            map_rule = torch.relu(lateral.max(dim=-1).values - 3.7) * 3.0
            total = progress * 2.0 + nominal * 1.0 + map_rule
            return dict(total=total, collision=zeros, ttc=zeros, interaction=zeros, map_rule=map_rule, comfort=zeros, progress=progress, nominal=nominal)

        t = min(ego_future.shape[-2], agent_futures.shape[-2])
        ego = ego_future[:, None, :t, :]
        agents = agent_futures[:, :, :t, :]
        dist = torch.linalg.norm(agents - ego, dim=-1)
        min_dist = dist.min(dim=-1).values
        collision = torch.relu(self.collision_distance - min_dist).sum(dim=-1)
        near = torch.relu(self.near_distance - min_dist).sum(dim=-1)

        ego_step = _temporal_steps(ego)
        agent_step = _temporal_steps(agents)
        if ego_step.shape[-2] == 0:
            rel_speed = torch.ones_like(min_dist)
        else:
            rel_speed = torch.linalg.norm(ego_step - agent_step, dim=-1).mean(dim=-1).clamp_min(1e-3)
        ttc = (min_dist / rel_speed).clamp(max=10.0)
        ttc_risk = torch.relu(3.0 - ttc).sum(dim=-1)

        ego_deltas = ego_future.diff(dim=-2, prepend=torch.zeros_like(ego_future[..., :1, :]))
        accel = ego_deltas.diff(dim=-2, prepend=torch.zeros_like(ego_deltas[..., :1, :]))
        comfort = torch.linalg.norm(accel, dim=-1).mean(dim=-1)
        if base_ego_future is not None:
            target_progress = base_ego_future[..., -1, 1].clamp(min=0)
        else:
            target_progress = ego_future[..., -1, 1].detach().clamp(min=0)
        progress = torch.relu(target_progress - ego_future[..., -1, 1].clamp(min=0))
        nominal = torch.zeros_like(progress)
        if base_ego_future is not None:
            nominal = torch.linalg.norm(ego_future - base_ego_future[:, : ego_future.shape[-2]], dim=-1).mean(dim=-1)
        interaction = near
        if relevance is not None and relevance.numel() > 0:
            interaction = interaction * (1.0 + relevance.mean(dim=-1))
        lateral = ego_future[..., 0].abs()
        max_lateral = lateral.max(dim=-1).values
        final_lateral = lateral[..., -1]
        low_progress = torch.relu(5.0 - ego_future[..., -1, 1].clamp(min=0))
        map_rule = torch.relu(max_lateral - 3.7) * 3.0 + final_lateral * 0.3 + low_progress * final_lateral * 0.2
        # This head is a reranker, not an aggressive planner. Keep risk terms
        # meaningful but penalize large deviation and lost progress enough that
        # MindDrive's original trajectory remains the default unless interaction
        # risk is materially lower for an alternative.
        total = 8.0 * collision + 1.0 * ttc_risk + 0.2 * interaction + 0.2 * comfort + 2.0 * progress + 1.0 * nominal + map_rule
        return dict(total=total, collision=collision, ttc=ttc_risk, interaction=interaction, map_rule=map_rule, comfort=comfort, progress=progress, nominal=nominal)
