"""Meta-action conditioned interaction graph encoder.

This module is a lightweight GNN, not a black-box tuple MLP.  For each ego
candidate it builds a graph with:

  node 0: candidate ego future tau_0^k
  node i: surrounding-agent base or realized future tau_i

Edges are directed into each agent from:
  - the ego node, so every relevance prediction is candidate-conditioned
  - k nearest surrounding agents, so local multi-agent context is available

The output is one feature vector per surrounding agent, shape (B, N, H), used by
the response predictor for relevance and meta-action classification.
"""

import torch
from torch import nn


class MetaActionInteractionGraph(nn.Module):
    """Encode candidate-conditioned agent interactions with GNN message passing.

    Shapes:
        scene_tokens: MindDrive scene tokens, (B, M, C)
        ego_future: candidate ego trajectory, (B, T, 2)
        agent_futures: surrounding-agent trajectories, (B, N, T, 2)
        boxes: optional boxes or centers, (B, N, >=2) or (N, >=2)

    Coordinate convention follows MindDrive planning tensors:
        position[..., 0] = lateral, position[..., 1] = longitudinal.
    """

    def __init__(
        self,
        embed_dims=896,
        hidden_dims=256,
        future_steps=6,
        num_message_passing=2,
        agent_knn=8,
        use_ego_meta_embedding=False,
        num_speed=7,
        num_path=6,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.hidden_dims = hidden_dims
        self.num_message_passing = num_message_passing
        self.agent_knn = agent_knn
        self.use_ego_meta_embedding = use_ego_meta_embedding

        self.scene_proj = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, hidden_dims),
            nn.ReLU(inplace=True),
        )
        ego_meta_dims = hidden_dims if use_ego_meta_embedding else 0
        if use_ego_meta_embedding:
            self.speed_embedding = nn.Embedding(num_speed, hidden_dims // 2)
            self.path_embedding = nn.Embedding(num_path, hidden_dims // 2)
            self.meta_proj = nn.Sequential(
                nn.Linear(hidden_dims, hidden_dims),
                nn.ReLU(inplace=True),
            )
        self.ego_node_proj = nn.Sequential(
            nn.Linear(future_steps * 2 + hidden_dims + ego_meta_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
        )
        self.agent_node_proj = nn.Sequential(
            nn.Linear(future_steps * 2 + 4 + hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
        )
        # Edge feature: rel_start(2), rel_end(2), rel_velocity(2),
        # dist_start(1), dist_end(1), is_ego_sender(1), optional ego meta emb.
        self.edge_proj = nn.Sequential(
            nn.Linear(9 + ego_meta_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dims * 3, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
        )
        self.update = nn.GRUCell(hidden_dims, hidden_dims)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dims),
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

    def _agent_box_features(self, boxes, batch_size, num_agents, device, dtype):
        if boxes is None or num_agents == 0:
            return torch.zeros(batch_size, num_agents, 4, device=device, dtype=dtype)
        if hasattr(boxes, "tensor"):
            boxes = boxes.tensor
        boxes = boxes.to(device=device, dtype=dtype)
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
        if boxes.shape[0] == 1 and batch_size > 1:
            boxes = boxes.expand(batch_size, -1, -1)
        elif boxes.shape[0] < batch_size:
            pad = boxes.new_zeros(batch_size - boxes.shape[0], boxes.shape[1], boxes.shape[-1])
            boxes = torch.cat([boxes, pad], dim=0)
        if boxes.shape[1] < num_agents:
            pad = boxes.new_zeros(batch_size, num_agents - boxes.shape[1], boxes.shape[-1])
            boxes = torch.cat([boxes, pad], dim=1)
        boxes = boxes[:, :num_agents]
        if boxes.shape[-1] < 4:
            pad = boxes.new_zeros(batch_size, num_agents, 4 - boxes.shape[-1])
            boxes = torch.cat([boxes, pad], dim=-1)
        return boxes[..., :4]

    def _edge_features(self, receiver_future, sender_future, is_ego_sender):
        rel_start = sender_future[..., 0, :] - receiver_future[..., 0, :]
        rel_end = sender_future[..., -1, :] - receiver_future[..., -1, :]
        sender_vel = sender_future[..., -1, :] - sender_future[..., 0, :]
        receiver_vel = receiver_future[..., -1, :] - receiver_future[..., 0, :]
        rel_vel = sender_vel - receiver_vel
        dist_start = torch.linalg.norm(rel_start, dim=-1, keepdim=True)
        dist_end = torch.linalg.norm(rel_end, dim=-1, keepdim=True)
        ego_flag = torch.full_like(dist_end, float(is_ego_sender))
        return torch.cat([rel_start, rel_end, rel_vel, dist_start, dist_end, ego_flag], dim=-1)

    def _knn_indices(self, agent_futures):
        b, n = agent_futures.shape[:2]
        if n <= 1 or self.agent_knn <= 0:
            return None
        k = min(self.agent_knn, n - 1)
        centers = agent_futures[..., 0, :]
        dist = torch.cdist(centers, centers)
        eye = torch.eye(n, device=agent_futures.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
        return dist.topk(k, largest=False, dim=-1).indices

    def _ego_meta_feature(self, ego_meta, batch_size, device, dtype):
        if not self.use_ego_meta_embedding:
            return None
        if ego_meta is None:
            speed = torch.zeros(batch_size, device=device, dtype=torch.long)
            path = torch.zeros(batch_size, device=device, dtype=torch.long)
        elif isinstance(ego_meta, dict):
            speed = ego_meta.get("speed_idx", ego_meta.get("speed", 0))
            path = ego_meta.get("path_idx", ego_meta.get("path", 0))
            speed = torch.as_tensor(speed, device=device, dtype=torch.long).reshape(-1)
            path = torch.as_tensor(path, device=device, dtype=torch.long).reshape(-1)
        else:
            meta = torch.as_tensor(ego_meta, device=device, dtype=torch.long)
            if meta.dim() == 1:
                speed = meta
                path = torch.zeros_like(speed)
            else:
                speed = meta[..., 0].reshape(-1)
                path = meta[..., 1].reshape(-1)
        if speed.numel() == 1 and batch_size > 1:
            speed = speed.expand(batch_size)
        if path.numel() == 1 and batch_size > 1:
            path = path.expand(batch_size)
        speed = speed[:batch_size].clamp(0, self.speed_embedding.num_embeddings - 1)
        path = path[:batch_size].clamp(0, self.path_embedding.num_embeddings - 1)
        meta_feat = torch.cat([self.speed_embedding(speed), self.path_embedding(path)], dim=-1)
        return self.meta_proj(meta_feat).to(dtype=dtype)

    def forward(self, scene_tokens, ego_future, agent_futures, boxes=None, labels=None, scores=None, map_context=None, ego_meta=None):
        if agent_futures.numel() == 0:
            b = scene_tokens.shape[0]
            return scene_tokens.new_zeros((b, 0, self.hidden_dims))

        ego_future = self._fit_steps(ego_future)
        agent_futures = self._fit_steps(agent_futures)
        b, n = agent_futures.shape[:2]
        dtype = agent_futures.dtype
        device = agent_futures.device

        scene = self.scene_proj(scene_tokens).mean(dim=1)
        scene_agent = scene[:, None, :].expand(b, n, -1)
        scene_ego = scene

        ego_flat = ego_future.reshape(b, -1)
        meta_feat = self._ego_meta_feature(ego_meta, b, device, dtype)
        ego_inputs = [ego_flat, scene_ego]
        if meta_feat is not None:
            ego_inputs.append(meta_feat)
        ego_node = self.ego_node_proj(torch.cat(ego_inputs, dim=-1))

        box_feat = self._agent_box_features(boxes, b, n, device, dtype)
        agent_flat = agent_futures.reshape(b, n, -1)
        agent_node = self.agent_node_proj(torch.cat([agent_flat, box_feat, scene_agent], dim=-1))

        knn_idx = self._knn_indices(agent_futures)
        for _ in range(self.num_message_passing):
            receiver_node = agent_node
            ego_sender = ego_node[:, None, :].expand(b, n, -1)
            ego_future_sender = ego_future[:, None, :, :].expand(b, n, -1, -1)
            ego_edge_feat = self._edge_features(agent_futures, ego_future_sender, True)
            if meta_feat is not None:
                ego_edge_feat = torch.cat([ego_edge_feat, meta_feat[:, None, :].expand(b, n, -1)], dim=-1)
            ego_edge = self.edge_proj(ego_edge_feat)
            ego_msg = self.message_mlp(torch.cat([receiver_node, ego_sender, ego_edge], dim=-1))
            agg = ego_msg
            edge_count = 1.0

            if knn_idx is not None:
                k = knn_idx.shape[-1]
                gather_idx = knn_idx[..., None].expand(b, n, k, self.hidden_dims)
                nbr_node = torch.gather(agent_node[:, None].expand(b, n, n, self.hidden_dims), 2, gather_idx)
                traj_idx = knn_idx[..., None, None].expand(b, n, k, self.future_steps, 2)
                nbr_future = torch.gather(
                    agent_futures[:, None].expand(b, n, n, self.future_steps, 2),
                    2,
                    traj_idx,
                )
                recv_future = agent_futures[:, :, None, :, :].expand(b, n, k, -1, -1)
                recv_node = receiver_node[:, :, None, :].expand(b, n, k, -1)
                nbr_edge_feat = self._edge_features(recv_future, nbr_future, False)
                if meta_feat is not None:
                    zeros = meta_feat.new_zeros(b, n, k, meta_feat.shape[-1])
                    nbr_edge_feat = torch.cat([nbr_edge_feat, zeros], dim=-1)
                nbr_edge = self.edge_proj(nbr_edge_feat)
                nbr_msg = self.message_mlp(torch.cat([recv_node, nbr_node, nbr_edge], dim=-1)).mean(dim=2)
                agg = agg + nbr_msg
                edge_count += 1.0

            updated = self.update((agg / edge_count).reshape(b * n, -1), agent_node.reshape(b * n, -1))
            agent_node = updated.reshape(b, n, -1)

        return self.out_proj(agent_node)
