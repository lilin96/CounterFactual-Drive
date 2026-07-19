"""MindDrive-integrated counterfactual reasoning head."""

import torch
import torch.nn.functional as F
from torch import nn

from mmcv.models.builder import HEADS

from .ego_candidate_generator import EgoCandidateGenerator
from .interaction_graph import MetaActionInteractionGraph
from .interaction_labels import interaction_relevance_labels
from .meta_action_labels import PATH_META_ACTIONS, SPEED_META_ACTIONS, path_pseudo_labels, speed_pseudo_labels
from .response_predictor import ResponsePredictor
from .risk_scorer import RuleBasedRiskScorer
from .trajectory_realizer import RuleBasedTrajectoryRealizer


@HEADS.register_module()
class CounterfactualReasoningHead(nn.Module):
    """Counterfactual response and risk head for MindDrive scene tokens.

    scene_tokens are MindDrive intermediate tokens:
        z_t^MD = concat(Q_obj, Q_map), shape (B, M_obj + M_map, C).

    Training uses factual logged futures to learn interaction regularities.
    Inference replaces the logged ego future with each candidate ego trajectory
    and predicts candidate-conditioned agent response hypotheses.
    """

    def __init__(
        self,
        embed_dims=896,
        hidden_dims=256,
        future_steps=6,
        relevance_threshold=0.5,
        rel_loss_weight=1.0,
        speed_loss_weight=1.0,
        path_loss_weight=1.0,
        real_loss_weight=0.0,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.relevance_threshold = relevance_threshold
        self.rel_loss_weight = rel_loss_weight
        self.speed_loss_weight = speed_loss_weight
        self.path_loss_weight = path_loss_weight
        self.real_loss_weight = real_loss_weight
        self.graph = MetaActionInteractionGraph(embed_dims, hidden_dims, future_steps)
        self.predictor = ResponsePredictor(hidden_dims)
        self.realizer = RuleBasedTrajectoryRealizer()
        self.risk_scorer = RuleBasedRiskScorer()
        self.ego_generator = EgoCandidateGenerator(future_steps)

    def _ensure_batched_futures(self, futures, batch_size=None, device=None):
        if futures is None:
            if batch_size is None:
                batch_size = 1
            return torch.zeros(batch_size, 0, self.future_steps, 2, device=device)
        if isinstance(futures, (list, tuple)):
            if len(futures) == 0:
                return torch.zeros(batch_size or 1, 0, self.future_steps, 2, device=device)
            futures = torch.stack([x.to(device=device) for x in futures], dim=0)
        if futures.dim() == 2:
            if futures.shape[-1] == 2:
                futures = futures.unsqueeze(0).unsqueeze(0)
            else:
                futures = futures.reshape(1, futures.shape[0], -1, 2)
        elif futures.dim() == 3:
            if futures.shape[-1] == 2:
                futures = futures.unsqueeze(0)
            elif batch_size is not None and futures.shape[0] == batch_size:
                futures = futures.reshape(futures.shape[0], futures.shape[1], -1, 2)
            else:
                # MindDrive motion decoding may return unbatched futures as
                # (num_agents, num_modes, T * 2). Use the first mode as the
                # base surrounding-agent future for counterfactual response.
                futures = futures[:, 0].reshape(1, futures.shape[0], -1, 2)
        if futures.shape[-1] != 2:
            futures = futures.reshape(*futures.shape[:-1], -1, 2)
        if futures.shape[-2] > self.future_steps:
            futures = futures[..., : self.future_steps, :]
        if futures.shape[-2] < self.future_steps and futures.shape[-2] > 0:
            pad = futures[..., -1:, :].expand(*futures.shape[:-2], self.future_steps - futures.shape[-2], 2)
            futures = torch.cat([futures, pad], dim=-2)
        if device is not None:
            futures = futures.to(device=device)
        return futures

    def _extract_bbox_tensor(self, boxes_3d, device, batch_size):
        if boxes_3d is None:
            return torch.zeros(batch_size, 0, 4, device=device)
        if hasattr(boxes_3d, "tensor"):
            box = boxes_3d.tensor
        elif hasattr(boxes_3d, "center"):
            box = boxes_3d.center
        else:
            box = boxes_3d
        if isinstance(box, (list, tuple)):
            box = torch.stack(box, dim=0)
        box = box.to(device=device)
        if box.dim() == 2:
            box = box.unsqueeze(0)
        return box

    def _box_centers_for_agents(self, boxes, num_agents, device, dtype):
        if boxes is None or num_agents == 0:
            return None
        centers = boxes[..., :2].to(device=device, dtype=dtype)
        if centers.dim() == 2:
            centers = centers.unsqueeze(0)
        if centers.shape[1] < num_agents:
            pad = centers.new_zeros((centers.shape[0], num_agents - centers.shape[1], 2))
            centers = torch.cat([centers, pad], dim=1)
        return centers[:, :num_agents]

    def _extract_base_agent_futures(self, outs_bbox=None, bbox_result=None, boxes=None, device=None, batch_size=1):
        if bbox_result is not None and "trajs_3d" in bbox_result:
            trajs = bbox_result["trajs_3d"]
            if hasattr(trajs, "to"):
                trajs = trajs.to(device=device)
            futures = self._ensure_batched_futures(trajs, batch_size=batch_size, device=device)
            # MindDrive motion predictions are per-step displacements. Match
            # the original motion metric path: cumsum, then add box centers.
            futures = futures.cumsum(dim=-2)
            centers = self._box_centers_for_agents(boxes, futures.shape[1], device, futures.dtype)
            if centers is not None:
                futures = futures + centers[..., None, :]
            return futures
        if outs_bbox is None:
            return torch.zeros(batch_size, 0, self.future_steps, 2, device=device)
        for key in ("all_traj_preds", "traj_preds", "all_motion_preds"):
            if isinstance(outs_bbox, dict) and key in outs_bbox:
                futures = self._ensure_batched_futures(outs_bbox[key][-1], batch_size=batch_size, device=device)
                return futures.cumsum(dim=-2)
        return torch.zeros(batch_size, 0, self.future_steps, 2, device=device)

    def _extract_gt_agent_futures(self, gt_attr_labels, device, batch_size):
        if gt_attr_labels is None:
            return torch.zeros(batch_size, 0, self.future_steps, 2, device=device)
        if isinstance(gt_attr_labels, torch.Tensor) and gt_attr_labels.dim() == 3:
            samples = [sample for sample in gt_attr_labels]
        else:
            samples = gt_attr_labels if isinstance(gt_attr_labels, (list, tuple)) else [gt_attr_labels]
        futures = []
        max_agents = 0
        for attrs in samples:
            attrs = attrs.to(device=device)
            if attrs.numel() == 0:
                fut = attrs.new_zeros((0, self.future_steps, 2))
            else:
                fut = attrs[:, : self.future_steps * 2].reshape(attrs.shape[0], self.future_steps, 2)
                fut = fut.cumsum(dim=-2)
            futures.append(fut)
            max_agents = max(max_agents, fut.shape[0])
        padded = []
        for fut in futures:
            if fut.shape[0] < max_agents:
                pad = fut.new_zeros((max_agents - fut.shape[0], self.future_steps, 2))
                fut = torch.cat([fut, pad], dim=0)
            padded.append(fut)
        return torch.stack(padded, dim=0) if padded else torch.zeros(batch_size, 0, self.future_steps, 2, device=device)

    def _extract_ego_future(self, ego_fut_trajs, device, batch_size):
        if ego_fut_trajs is None:
            return torch.zeros(batch_size, self.future_steps, 2, device=device)
        ego = ego_fut_trajs.to(device=device)
        while ego.dim() > 3:
            ego = ego[:, 0]
        if ego.dim() == 2:
            ego = ego.unsqueeze(0)
        if ego.shape[-1] != 2:
            ego = ego.reshape(ego.shape[0], -1, 2)
        ego = ego[:, : self.future_steps].cumsum(dim=-2)
        return ego

    def forward_train(
        self,
        scene_tokens,
        outs_bbox,
        outs_lane,
        gt_bboxes_3d,
        gt_labels_3d,
        gt_attr_labels,
        ego_fut_trajs,
        ego_fut_masks=None,
        img_metas=None,
        **kwargs
    ):
        device = scene_tokens.device
        batch_size = scene_tokens.shape[0]
        ego_gt = self._extract_ego_future(ego_fut_trajs, device, batch_size)
        dtype = scene_tokens.dtype
        ego_gt = ego_gt.to(dtype=dtype)
        agent_gt = self._extract_gt_agent_futures(gt_attr_labels, device, batch_size).to(dtype=dtype)
        boxes = None
        if gt_bboxes_3d is not None:
            box_samples = []
            max_agents = agent_gt.shape[1]
            for boxes_3d in gt_bboxes_3d:
                box = self._extract_bbox_tensor(boxes_3d, device, 1).squeeze(0).to(dtype=dtype)
                if box.shape[-1] < 4:
                    box = torch.cat([box, box.new_zeros((box.shape[0], 4 - box.shape[-1]))], dim=-1)
                box = box[:, :4]
                if box.shape[0] < max_agents:
                    box = torch.cat([box, box.new_zeros((max_agents - box.shape[0], 4))], dim=0)
                box_samples.append(box[:max_agents])
            boxes = torch.stack(box_samples, dim=0) if box_samples else None

        agent_origins = self._box_centers_for_agents(boxes, agent_gt.shape[1], device, dtype) if boxes is not None else None
        agent_gt_abs = agent_gt + agent_origins[..., None, :] if agent_origins is not None else agent_gt

        if agent_gt_abs.shape[1] == 0:
            zero = scene_tokens.sum() * 0.0
            return dict(loss_cf_rel=zero, loss_cf_speed=zero, loss_cf_path=zero)

        # Factual supervision only: labels come from observed ego and observed
        # agent futures. No counterfactual ground-truth futures are assumed.
        relevance_label = interaction_relevance_labels(ego_gt, agent_gt_abs)
        speed_label = speed_pseudo_labels(agent_gt)
        path_label = path_pseudo_labels(agent_gt)

        features = self.graph(scene_tokens, ego_gt, agent_gt_abs, boxes=boxes, labels=gt_labels_3d)
        pred = self.predictor(features)
        valid = torch.ones_like(relevance_label, dtype=torch.bool)
        loss_rel = F.binary_cross_entropy_with_logits(pred["relevance_logits"][valid], relevance_label[valid])
        loss_speed = F.cross_entropy(pred["speed_logits"][valid], speed_label[valid])
        loss_path = F.cross_entropy(pred["path_logits"][valid], path_label[valid])
        losses = dict(
            loss_cf_rel=loss_rel * self.rel_loss_weight,
            loss_cf_speed=loss_speed * self.speed_loss_weight,
            loss_cf_path=loss_path * self.path_loss_weight,
        )
        if self.real_loss_weight > 0:
            relevance = relevance_label
            realized = self.realizer(
                agent_gt_abs,
                speed_label,
                path_label,
                relevance,
                threshold=self.relevance_threshold,
                start_positions=agent_origins,
            )
            losses["loss_cf_real"] = F.l1_loss(realized[valid], agent_gt_abs[valid]) * self.real_loss_weight
        return losses

    @torch.no_grad()
    def forward_infer(
        self,
        scene_tokens,
        bbox_result,
        lane_results,
        ego_candidates=None,
        candidate_meta_actions=None,
        map_context=None,
        **kwargs
    ):
        device = scene_tokens.device
        dtype = scene_tokens.dtype
        batch_size = scene_tokens.shape[0]
        boxes = self._extract_bbox_tensor(bbox_result.get("boxes_3d", None), device, batch_size).to(dtype=dtype) if bbox_result is not None else None
        base_agents = self._extract_base_agent_futures(
            bbox_result=bbox_result, boxes=boxes, device=device, batch_size=batch_size
        ).to(dtype=dtype)
        agent_origins = self._box_centers_for_agents(boxes, base_agents.shape[1], device, dtype) if boxes is not None else None
        base_ego_anchor = None
        lane0 = lane_results[0] if isinstance(lane_results, (list, tuple)) and len(lane_results) > 0 else lane_results
        if isinstance(lane0, dict) and lane0.get("ego_fut_preds", None) is not None:
            base_ego_anchor = lane0["ego_fut_preds"].to(device=device, dtype=dtype)
            if base_ego_anchor.dim() == 2:
                base_ego_anchor = base_ego_anchor.unsqueeze(0)
            if base_ego_anchor.shape[-2] > self.future_steps:
                base_ego_anchor = base_ego_anchor[..., : self.future_steps, :]
            if base_ego_anchor.shape[-2] < self.future_steps:
                pad = base_ego_anchor[..., -1:, :].expand(
                    *base_ego_anchor.shape[:-2], self.future_steps - base_ego_anchor.shape[-2], 2
                )
                base_ego_anchor = torch.cat([base_ego_anchor, pad], dim=-2)

        if ego_candidates is None:
            ego_candidates, candidate_meta_actions = self.ego_generator(batch_size, device, dtype=scene_tokens.dtype)
            if base_ego_anchor is not None:
                ego_candidates = torch.cat([base_ego_anchor[:, None], ego_candidates], dim=1)
                candidate_meta_actions = [
                    dict(speed="minddrive selected", path="minddrive selected", speed_idx=-1, path_idx=-1)
                ] + list(candidate_meta_actions)
        else:
            ego_candidates = ego_candidates.to(device=device, dtype=scene_tokens.dtype)
            if ego_candidates.dim() == 3:
                ego_candidates = ego_candidates.unsqueeze(0)
            if candidate_meta_actions is None:
                candidate_meta_actions = [dict(speed=SPEED_META_ACTIONS[0], path=PATH_META_ACTIONS[1], speed_idx=0, path_idx=1) for _ in range(ego_candidates.shape[1])]
        if base_ego_anchor is None:
            base_ego_anchor = ego_candidates[:, 0]

        risk_terms = []
        scenes = []
        relevance_all = []
        response_actions = []
        for k in range(ego_candidates.shape[1]):
            ego_k = ego_candidates[:, k]
            features = self.graph(scene_tokens, ego_k, base_agents, boxes=boxes, map_context=map_context)
            pred = self.predictor(features)
            relevance = torch.sigmoid(pred["relevance_logits"])
            speed_labels = pred["speed_logits"].argmax(dim=-1)
            path_labels = pred["path_logits"].argmax(dim=-1)
            realized_agents = self.realizer(
                base_agents,
                speed_labels,
                path_labels,
                relevance,
                threshold=self.relevance_threshold,
                start_positions=agent_origins,
            )
            risk = self.risk_scorer(ego_k, realized_agents, base_ego_future=base_ego_anchor, relevance=relevance, map_context=map_context)
            risk_terms.append(risk)
            scenes.append(dict(ego_future=ego_k.detach().cpu(), agent_futures=realized_agents.detach().cpu()))
            relevance_all.append(relevance.detach().cpu())
            response_actions.append(dict(speed=speed_labels.detach().cpu(), path=path_labels.detach().cpu()))

        total = torch.stack([r["total"] for r in risk_terms], dim=1)
        selected_idx = total.argmin(dim=1)
        selected = ego_candidates[torch.arange(batch_size, device=device), selected_idx]
        risk_scores = {
            name: torch.stack([r[name] for r in risk_terms], dim=1).detach().cpu()
            for name in risk_terms[0].keys()
        }
        selected_meta = []
        for b, idx in enumerate(selected_idx.detach().cpu().tolist()):
            selected_meta.append(candidate_meta_actions[idx])
        return dict(
            selected_ego_future=selected.detach().cpu(),
            selected_meta_action=selected_meta[0] if len(selected_meta) == 1 else selected_meta,
            risk_scores=risk_scores,
            counterfactual_scenes=scenes,
            interaction_relevance=relevance_all,
            response_meta_actions=response_actions,
        )
