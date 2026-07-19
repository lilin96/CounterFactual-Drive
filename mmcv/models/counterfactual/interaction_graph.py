"""Lightweight candidate-conditioned interaction graph encoder."""

import torch
from torch import nn


class MetaActionInteractionGraph(nn.Module):
    """Encode scene/candidate/agent tuples with compact MLP message passing.

    Shapes:
        scene_tokens: (B, M, C)
        ego_future: (B, T, 2)
        agent_futures: (B, N, T, 2)
        boxes: optional box centers or boxes, (B, N, >=2)
    """

    def __init__(self, embed_dims=896, hidden_dims=256, future_steps=6):
        super().__init__()
        self.future_steps = future_steps
        self.scene_proj = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, hidden_dims),
            nn.ReLU(inplace=True),
        )
        self.traj_proj = nn.Sequential(
            nn.Linear(future_steps * 4 + 8, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
        )

    def _fit_steps(self, traj):
        if traj.shape[-2] == self.future_steps:
            return traj
        if traj.shape[-2] > self.future_steps:
            return traj[..., : self.future_steps, :]
        pad = traj[..., -1:, :].expand(*traj.shape[:-2], self.future_steps - traj.shape[-2], 2)
        return torch.cat([traj, pad], dim=-2)

    def forward(self, scene_tokens, ego_future, agent_futures, boxes=None, labels=None, scores=None, map_context=None):
        if agent_futures.numel() == 0:
            b = scene_tokens.shape[0]
            return scene_tokens.new_zeros((b, 0, self.fuse[-2].out_features))
        ego_future = self._fit_steps(ego_future)
        agent_futures = self._fit_steps(agent_futures)
        b, n = agent_futures.shape[:2]
        scene = self.scene_proj(scene_tokens).mean(dim=1)
        scene = scene[:, None, :].expand(b, n, -1)

        ego_flat = ego_future[:, None].expand(b, n, -1, -1).reshape(b, n, -1)
        agent_flat = agent_futures.reshape(b, n, -1)
        rel = agent_futures[:, :, -1] - ego_future[:, None, -1]
        start_rel = agent_futures[:, :, 0] - ego_future[:, None, 0]
        if boxes is not None:
            box_xy = boxes[..., :2]
            box_feat = torch.zeros((b, n, 4), device=agent_futures.device, dtype=agent_futures.dtype)
            box_feat[..., :2] = box_xy
        else:
            box_feat = torch.zeros((b, n, 4), device=agent_futures.device, dtype=agent_futures.dtype)
        geom = torch.cat([ego_flat, agent_flat, rel, start_rel, box_feat], dim=-1)
        traj = self.traj_proj(geom)
        return self.fuse(torch.cat([scene, traj], dim=-1))

