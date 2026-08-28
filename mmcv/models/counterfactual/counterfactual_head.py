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
        agent_score_threshold=0.25,
        valid_agent_labels=(0, 1, 2, 3, 7, 8),
        min_valid_future_steps=2,
        rel_loss_weight=1.0,
        speed_loss_weight=1.0,
        path_loss_weight=1.0,
        real_loss_weight=0.0,
        save_counterfactual_scenes=True,
        use_ego_meta_embedding=False,
        train_candidate_augmentation=False,
        aug_num_candidates=4,
        aug_loss_weight=0.5,
        aug_response_loss_weight=0.25,
        use_candidate_conditioned_agent_response=True,
        rule_candidate_mode="all",
        rule_fixed_path_idx=1,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.relevance_threshold = relevance_threshold
        self.agent_score_threshold = agent_score_threshold
        self.valid_agent_labels = tuple(int(x) for x in valid_agent_labels) if valid_agent_labels is not None else None
        self.min_valid_future_steps = int(min_valid_future_steps)
        self.rel_loss_weight = rel_loss_weight
        self.speed_loss_weight = speed_loss_weight
        self.path_loss_weight = path_loss_weight
        self.real_loss_weight = real_loss_weight
        self.save_counterfactual_scenes = save_counterfactual_scenes
        self.use_ego_meta_embedding = use_ego_meta_embedding
        self.train_candidate_augmentation = train_candidate_augmentation
        self.aug_num_candidates = aug_num_candidates
        self.aug_loss_weight = aug_loss_weight
        self.aug_response_loss_weight = aug_response_loss_weight
        self.use_candidate_conditioned_agent_response = bool(use_candidate_conditioned_agent_response)
        self.rule_candidate_mode = rule_candidate_mode
        self.graph = MetaActionInteractionGraph(
            embed_dims,
            hidden_dims,
            future_steps,
            use_ego_meta_embedding=use_ego_meta_embedding,
        )
        self.predictor = ResponsePredictor(hidden_dims)
        self.realizer = RuleBasedTrajectoryRealizer()
        self.risk_scorer = RuleBasedRiskScorer()
        self.ego_generator = EgoCandidateGenerator(
            future_steps,
            candidate_mode=rule_candidate_mode,
            fixed_path_idx=rule_fixed_path_idx,
        )

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

    def _extract_agent_valid_mask(self, bbox_result, num_agents, device, batch_size):
        """Return the detection-confidence mask aligned with agent futures."""
        if num_agents == 0:
            return torch.zeros(batch_size, 0, device=device, dtype=torch.bool)
        scores = bbox_result.get("scores_3d", None) if bbox_result is not None else None
        if scores is None:
            return torch.ones(batch_size, num_agents, device=device, dtype=torch.bool)
        if isinstance(scores, (list, tuple)):
            scores = torch.stack([torch.as_tensor(x, device=device) for x in scores], dim=0)
        else:
            scores = torch.as_tensor(scores, device=device)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        scores = scores.reshape(scores.shape[0], -1)
        if scores.shape[0] == 1 and batch_size > 1:
            scores = scores.expand(batch_size, -1)
        if scores.shape[1] < num_agents:
            pad = scores.new_full((scores.shape[0], num_agents - scores.shape[1]), float("-inf"))
            scores = torch.cat([scores, pad], dim=1)
        valid = scores[:batch_size, :num_agents] >= self.agent_score_threshold
        labels = bbox_result.get("labels_3d", None) if bbox_result is not None else None
        if labels is not None and self.valid_agent_labels is not None:
            labels = torch.as_tensor(labels, device=device)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            labels = labels.reshape(labels.shape[0], -1)
            if labels.shape[0] == 1 and batch_size > 1:
                labels = labels.expand(batch_size, -1)
            if labels.shape[1] < num_agents:
                pad = labels.new_full((labels.shape[0], num_agents - labels.shape[1]), -1)
                labels = torch.cat([labels, pad], dim=1)
            valid_label = torch.zeros_like(valid)
            for label in self.valid_agent_labels:
                valid_label |= labels[:batch_size, :num_agents] == label
            valid &= valid_label
        return valid

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

    def _meta_tensor_from_actions(self, candidate_meta_actions, num_candidates, batch_size, device):
        meta = torch.zeros(num_candidates, 2, device=device, dtype=torch.long)
        if candidate_meta_actions is not None:
            for idx, action in enumerate(candidate_meta_actions[:num_candidates]):
                if isinstance(action, dict):
                    meta[idx, 0] = int(action.get("speed_idx", 0))
                    meta[idx, 1] = int(action.get("path_idx", 0))
        return meta[:, None, :].expand(num_candidates, batch_size, 2).reshape(num_candidates * batch_size, 2)

    def _factual_meta_ids(self, ego_future):
        speed = speed_pseudo_labels(ego_future)
        path = path_pseudo_labels(ego_future)
        return torch.stack([speed, path], dim=-1).long()

    def _build_augmented_ego_candidates(self, ego_gt):
        """Create candidate ego futures for factual augmentation.

        These are not treated as counterfactual ground truth. They only provide
        alternate ego conditions so geometric interaction relevance can be
        recomputed from logged agent futures during training.
        """
        b, t, _ = ego_gt.shape
        device = ego_gt.device
        dtype = ego_gt.dtype
        speed_ids = torch.tensor([1, 2, 3, 4, 6, 5, 0], device=device, dtype=torch.long)[: self.aug_num_candidates]
        path_ids = torch.tensor([0, 1, 0, 0, 0, 1, 0], device=device, dtype=torch.long)[: self.aug_num_candidates]

        deltas = ego_gt.diff(dim=-2, prepend=torch.zeros_like(ego_gt[:, :1]))
        speed_scale_values = ego_gt.new_tensor([1.0, 0.05, 0.65, 1.25, 0.8, 1.1, 0.45])
        lat_bias_values = ego_gt.new_tensor([0.0, 0.0, 0.18, 0.35, -0.18, -0.35])
        time = torch.linspace(0.0, 1.0, t, device=device, dtype=dtype)

        candidates = []
        for speed_id, path_id in zip(speed_ids.tolist(), path_ids.tolist()):
            adjusted = deltas * speed_scale_values[speed_id]
            adjusted = adjusted.clone()
            adjusted[..., 0] = adjusted[..., 0] + lat_bias_values[path_id] * time
            candidates.append(adjusted.cumsum(dim=-2))
        ego_aug = torch.stack(candidates, dim=1) if candidates else ego_gt[:, None]
        meta = torch.stack([speed_ids, path_ids], dim=-1)
        return ego_aug, meta

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
        attr_samples = list(gt_attr_labels) if isinstance(gt_attr_labels, (list, tuple)) else list(gt_attr_labels)
        label_samples = list(gt_labels_3d) if isinstance(gt_labels_3d, (list, tuple)) else list(gt_labels_3d)
        box_samples = list(gt_bboxes_3d) if isinstance(gt_bboxes_3d, (list, tuple)) else list(gt_bboxes_3d)

        # A rank can receive a scene with no valid dynamic agents.  In that
        # case every reported loss must still be connected to the trainable
        # counterfactual parameters, otherwise distributed backward fails
        # with "does not require grad" on that rank.  Touch all trainable
        # parameters with a zero coefficient so DDP also gets zero gradients
        # for them and stays synchronized with ranks that do have valid agents.
        zero = scene_tokens.new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad:
                zero = zero + parameter.sum() * 0.0
        rel_sum = zero
        speed_sum = zero
        path_sum = zero
        real_sum = zero
        factual_count = 0
        real_value_count = 0
        aug_rel_sum = zero
        aug_speed_sum = zero
        aug_path_sum = zero
        aug_count = 0

        ego_aug_all, aug_meta = (None, None)
        if self.train_candidate_augmentation and self.aug_num_candidates > 0:
            ego_aug_all, aug_meta = self._build_augmented_ego_candidates(ego_gt)

        for b in range(batch_size):
            attrs = attr_samples[b].to(device=device)
            gt_labels = label_samples[b].to(device=device, dtype=torch.long)
            gt_boxes = self._extract_bbox_tensor(box_samples[b], device, 1).squeeze(0).to(dtype=dtype)
            n = min(attrs.shape[0], gt_labels.shape[0], gt_boxes.shape[0])
            if n == 0:
                continue
            attrs = attrs[:n]
            gt_labels = gt_labels[:n]
            gt_boxes = gt_boxes[:n]

            future_offsets = attrs[:, : self.future_steps * 2].reshape(n, self.future_steps, 2)
            future_mask = attrs[:, self.future_steps * 2 : self.future_steps * 3]
            valid = future_mask.sum(dim=-1) >= self.min_valid_future_steps
            if self.valid_agent_labels is not None:
                dynamic = torch.zeros_like(valid, dtype=torch.bool)
                for label in self.valid_agent_labels:
                    dynamic |= gt_labels == label
                valid &= dynamic
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue

            agent_rel = future_offsets[valid_idx].to(dtype=dtype).cumsum(dim=-2).unsqueeze(0)
            labels_compact = gt_labels[valid_idx].unsqueeze(0)
            boxes_compact = gt_boxes[valid_idx]
            if boxes_compact.shape[-1] < 4:
                boxes_compact = torch.cat(
                    [boxes_compact, boxes_compact.new_zeros((boxes_compact.shape[0], 4 - boxes_compact.shape[-1]))],
                    dim=-1,
                )
            boxes_compact = boxes_compact[:, :4].unsqueeze(0)
            origins = boxes_compact[..., :2]
            agent_abs = agent_rel + origins[..., None, :]
            ego_b = ego_gt[b : b + 1]
            scene_b = scene_tokens[b : b + 1]

            relevance_target = interaction_relevance_labels(ego_b, agent_abs)
            speed_target = speed_pseudo_labels(agent_rel)
            path_target = path_pseudo_labels(agent_rel)
            factual_meta = self._factual_meta_ids(ego_b) if self.use_ego_meta_embedding else None
            features = self.graph(
                scene_b,
                ego_b,
                agent_abs,
                boxes=boxes_compact,
                labels=labels_compact,
                ego_meta=factual_meta,
            )
            pred = self.predictor(features)
            rel_sum = rel_sum + F.binary_cross_entropy_with_logits(
                pred["relevance_logits"], relevance_target, reduction="sum"
            )
            speed_sum = speed_sum + F.cross_entropy(
                pred["speed_logits"].reshape(-1, pred["speed_logits"].shape[-1]),
                speed_target.reshape(-1),
                reduction="sum",
            )
            path_sum = path_sum + F.cross_entropy(
                pred["path_logits"].reshape(-1, pred["path_logits"].shape[-1]),
                path_target.reshape(-1),
                reduction="sum",
            )
            factual_count += int(valid_idx.numel())

            if self.real_loss_weight > 0:
                realized = self.realizer(
                    agent_abs,
                    speed_target,
                    path_target,
                    relevance_target,
                    threshold=self.relevance_threshold,
                    start_positions=origins,
                )
                real_sum = real_sum + F.l1_loss(realized, agent_abs, reduction="sum")
                real_value_count += agent_abs.numel()

            if ego_aug_all is not None:
                ego_aug_b = ego_aug_all[b]
                num_aug = ego_aug_b.shape[0]
                scene_rep = scene_b.expand(num_aug, *scene_b.shape[1:])
                agent_rep = agent_abs.expand(num_aug, *agent_abs.shape[1:])
                boxes_rep = boxes_compact.expand(num_aug, *boxes_compact.shape[1:])
                labels_rep = labels_compact.expand(num_aug, *labels_compact.shape[1:])
                meta_rep = aug_meta if self.use_ego_meta_embedding else None
                aug_rel = interaction_relevance_labels(ego_aug_b, agent_rep)
                aug_speed = speed_target.expand(num_aug, -1)
                aug_path = path_target.expand(num_aug, -1)
                aug_features = self.graph(
                    scene_rep,
                    ego_aug_b,
                    agent_rep,
                    boxes=boxes_rep,
                    labels=labels_rep,
                    ego_meta=meta_rep,
                )
                aug_pred = self.predictor(aug_features)
                aug_rel_sum = aug_rel_sum + F.binary_cross_entropy_with_logits(
                    aug_pred["relevance_logits"], aug_rel, reduction="sum"
                )
                if self.aug_response_loss_weight > 0:
                    aug_speed_sum = aug_speed_sum + F.cross_entropy(
                        aug_pred["speed_logits"].reshape(-1, aug_pred["speed_logits"].shape[-1]),
                        aug_speed.reshape(-1),
                        reduction="sum",
                    )
                    aug_path_sum = aug_path_sum + F.cross_entropy(
                        aug_pred["path_logits"].reshape(-1, aug_pred["path_logits"].shape[-1]),
                        aug_path.reshape(-1),
                        reduction="sum",
                    )
                aug_count += int(valid_idx.numel()) * num_aug

        denom = max(factual_count, 1)
        losses = dict(
            loss_cf_rel=(rel_sum / denom) * self.rel_loss_weight,
            loss_cf_speed=(speed_sum / denom) * self.speed_loss_weight,
            loss_cf_path=(path_sum / denom) * self.path_loss_weight,
        )
        if self.real_loss_weight > 0:
            losses["loss_cf_real"] = (real_sum / max(real_value_count, 1)) * self.real_loss_weight
        if ego_aug_all is not None:
            aug_denom = max(aug_count, 1)
            losses["loss_cf_aug_rel"] = (aug_rel_sum / aug_denom) * self.aug_loss_weight
            if self.aug_response_loss_weight > 0:
                response_weight = self.aug_loss_weight * self.aug_response_loss_weight
                losses["loss_cf_aug_speed"] = (aug_speed_sum / aug_denom) * response_weight
                losses["loss_cf_aug_path"] = (aug_path_sum / aug_denom) * response_weight
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
        agent_valid_mask = self._extract_agent_valid_mask(
            bbox_result, base_agents.shape[1], device, batch_size
        )
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
            # The legacy all-combinations fallback keeps MindDrive's base plan
            # as candidate 0.  The paper Rule-K=7 ablation already contains
            # exactly seven speed-conditioned candidates, so prepending the
            # base plan would make an unfair K=8 comparison.
            if base_ego_anchor is not None and self.rule_candidate_mode == "all":
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

        # MindDrive's simple_test_pts currently runs with B=1. Compacting the
        # decoded agents before graph construction avoids padding invalid
        # queries back into agent-agent message passing. A batched inference
        # implementation needs per-sample packing because valid counts differ.
        if batch_size != 1:
            raise NotImplementedError(
                "Compact counterfactual inference currently requires batch_size=1; got {}.".format(batch_size)
            )
        valid_idx = torch.nonzero(agent_valid_mask[0], as_tuple=False).squeeze(-1)
        valid_base_agents = base_agents[:, valid_idx]
        valid_boxes = boxes[:, valid_idx] if boxes is not None else None
        valid_agent_origins = agent_origins[:, valid_idx] if agent_origins is not None else None
        num_agents = base_agents.shape[1]

        risk_terms = []
        scenes = []
        relevance_all = []
        response_actions = []
        for k in range(ego_candidates.shape[1]):
            ego_k = ego_candidates[:, k]
            ego_meta = None
            if self.use_ego_meta_embedding:
                ego_meta = self._meta_tensor_from_actions(candidate_meta_actions, ego_candidates.shape[1], batch_size, device)[
                    k * batch_size : (k + 1) * batch_size
                ]
            if valid_idx.numel() > 0:
                features = self.graph(
                    scene_tokens,
                    ego_k,
                    valid_base_agents,
                    boxes=valid_boxes,
                    map_context=map_context,
                    ego_meta=ego_meta,
                )
                pred = self.predictor(features)
                valid_relevance = torch.sigmoid(pred["relevance_logits"])
                valid_speed_probs = pred["speed_logits"].softmax(dim=-1)
                valid_path_probs = pred["path_logits"].softmax(dim=-1)
                valid_speed_labels = pred["speed_logits"].argmax(dim=-1)
                valid_path_labels = pred["path_logits"].argmax(dim=-1)
                if self.use_candidate_conditioned_agent_response:
                    valid_realized_agents = self.realizer(
                        valid_base_agents,
                        valid_speed_labels,
                        valid_path_labels,
                        valid_relevance,
                        threshold=self.relevance_threshold,
                        start_positions=valid_agent_origins,
                    )
                else:
                    # A2 paper ablation: every candidate is evaluated against
                    # the same original MindDrive agent futures.
                    valid_realized_agents = valid_base_agents
            else:
                valid_relevance = base_agents.new_zeros((batch_size, 0))
                valid_speed_probs = base_agents.new_zeros((batch_size, 0, len(SPEED_META_ACTIONS)))
                valid_path_probs = base_agents.new_zeros((batch_size, 0, len(PATH_META_ACTIONS)))
                valid_speed_labels = torch.empty((batch_size, 0), device=device, dtype=torch.long)
                valid_path_labels = torch.empty((batch_size, 0), device=device, dtype=torch.long)
                valid_realized_agents = valid_base_agents

            risk = self.risk_scorer(
                ego_k,
                valid_realized_agents,
                base_ego_future=base_ego_anchor,
                relevance=valid_relevance,
                map_context=map_context,
            )

            # Preserve the decoded bbox ordering expected by result consumers.
            relevance = base_agents.new_zeros((batch_size, num_agents))
            speed_probs = base_agents.new_zeros((batch_size, num_agents, len(SPEED_META_ACTIONS)))
            path_probs = base_agents.new_zeros((batch_size, num_agents, len(PATH_META_ACTIONS)))
            speed_labels = torch.full((batch_size, num_agents), -1, device=device, dtype=torch.long)
            path_labels = torch.full((batch_size, num_agents), -1, device=device, dtype=torch.long)
            realized_agents = base_agents.clone()
            relevance[:, valid_idx] = valid_relevance
            speed_probs[:, valid_idx] = valid_speed_probs
            path_probs[:, valid_idx] = valid_path_probs
            speed_labels[:, valid_idx] = valid_speed_labels
            path_labels[:, valid_idx] = valid_path_labels
            realized_agents[:, valid_idx] = valid_realized_agents
            risk_terms.append(risk)
            scenes.append(dict(ego_future=ego_k.detach().cpu(), agent_futures=realized_agents.detach().cpu()))
            relevance_all.append(relevance.detach().cpu())
            response_actions.append(
                dict(
                    speed=speed_labels.detach().cpu(),
                    path=path_labels.detach().cpu(),
                    speed_probs=speed_probs.detach().cpu(),
                    path_probs=path_probs.detach().cpu(),
                )
            )

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
        output = dict(
            selected_ego_future=selected.detach().cpu(),
            selected_meta_action=selected_meta[0] if len(selected_meta) == 1 else selected_meta,
            candidate_meta_actions=candidate_meta_actions,
            risk_scores=risk_scores,
            interaction_relevance=relevance_all,
            response_meta_actions=response_actions,
            use_candidate_conditioned_agent_response=self.use_candidate_conditioned_agent_response,
            rule_candidate_mode=self.rule_candidate_mode,
            agent_valid_mask=agent_valid_mask.detach().cpu(),
            num_valid_agents=agent_valid_mask.sum(dim=-1).detach().cpu(),
        )
        if self.save_counterfactual_scenes:
            output["counterfactual_scenes"] = scenes
        return output
