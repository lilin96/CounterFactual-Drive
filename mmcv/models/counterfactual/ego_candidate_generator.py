"""Fallback ego candidate generation aligned with MindDrive meta-actions."""

import torch
from torch import nn

from .meta_action_labels import PATH_META_ACTIONS, SPEED_META_ACTIONS, default_meta_actions


class EgoCandidateGenerator(nn.Module):
    """Generate lightweight ego candidates when Action Expert candidates are absent."""

    def __init__(self, future_steps=6, dt=0.5):
        super().__init__()
        self.future_steps = future_steps
        self.dt = dt

    def forward(self, batch_size, device, dtype=torch.float32):
        actions = default_meta_actions(device=device)
        trajs = []
        t = torch.arange(1, self.future_steps + 1, device=device, dtype=dtype) * self.dt
        for action in actions:
            speed = action["speed_idx"]
            path = action["path_idx"]
            base_v = torch.tensor([5.0, 0.0, 2.0, 7.0, 3.5, 9.0, 1.5], device=device, dtype=dtype)[speed]
            # MindDrive ego future tensors use (lateral, longitudinal) order.
            # Keep fallback candidates in that same convention so CF reranking
            # can compare them directly with ego_fut_preds.
            lateral = torch.zeros_like(t)
            longitudinal = base_v * t
            if PATH_META_ACTIONS[path] == "turn left":
                lateral = 0.5 * t * t
            elif PATH_META_ACTIONS[path] == "turn right":
                lateral = -0.5 * t * t
            elif PATH_META_ACTIONS[path] == "change lane left":
                lateral = torch.linspace(0.0, 3.5, self.future_steps, device=device, dtype=dtype)
            elif PATH_META_ACTIONS[path] == "change lane right":
                lateral = torch.linspace(0.0, -3.5, self.future_steps, device=device, dtype=dtype)
            # Avoid physically invalid fallback pairs such as "stop + turn",
            # which otherwise create pure lateral jumps with no forward motion.
            lateral = lateral * (base_v / 5.0).clamp(max=1.0)
            trajs.append(torch.stack([lateral, longitudinal], dim=-1))
        candidates = torch.stack(trajs, dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        return candidates, actions
