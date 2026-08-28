#!/usr/bin/env python
"""Scan a contiguous index range for candidate-conditioned CF differences."""

import argparse
import json
import os

import numpy as np
import torch

from mmcv.datasets import build_dataset
from mmcv.parallel import collate

from tools.live_cf_candidate_scenes_only import (
    build_live_model,
    draw_live_candidate_panels,
    prepare_cfg,
)
from tools.visualize_counterfactual_case_study import (
    agent_futures_from_result,
    get_pts_bbox,
    load_infos,
    to_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True, help="Inclusive end index.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--relevance-thr", type=float, default=0.5)
    parser.add_argument("--change-thr", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize-indices", default="", help="Comma-separated indices saved during sequential inference.")
    parser.add_argument("--visualize-dir", default=None)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", default=None)
    parser.set_defaults(use_rule_fallback_candidates=False, use_action_expert_candidates=True)
    return parser.parse_args()


def max_ref_diff(arrays):
    if len(arrays) < 2:
        return 0.0
    ref = arrays[0]
    values = []
    for other in arrays[1:]:
        if ref.shape != other.shape or ref.size == 0:
            continue
        values.append(float(np.max(np.abs(ref - other))))
    return max(values, default=0.0)


def scan_one(pts, info, index, relevance_thr, change_thr):
    boxes, base, scores, _ = agent_futures_from_result(pts)
    scenes = pts.get("cf_counterfactual_scenes", [])
    rels_raw = pts.get("cf_interaction_relevance", [])
    responses = pts.get("cf_response_meta_actions", [])
    egos, agents, rels, speed_probs, path_probs = [], [], [], [], []
    affected_sets = []
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    valid_detection = scores >= 0.25

    for k, scene in enumerate(scenes):
        ego = np.squeeze(to_numpy(scene.get("ego_future"))).astype(np.float32)
        realized = np.squeeze(to_numpy(scene.get("agent_futures"))).astype(np.float32)
        rel = np.squeeze(to_numpy(rels_raw[k])).astype(np.float32)[: len(boxes)]
        egos.append(ego)
        agents.append(realized)
        rels.append(rel)
        if k < len(responses):
            speed_probs.append(np.squeeze(to_numpy(responses[k].get("speed_probs"))).astype(np.float32))
            path_probs.append(np.squeeze(to_numpy(responses[k].get("path_probs"))).astype(np.float32))
        change = np.zeros(len(boxes), dtype=np.float32)
        n = min(len(realized), len(base), len(boxes))
        if n:
            change[:n] = np.linalg.norm(realized[:n, :, :2] - base[:n, :, :2], axis=-1).max(axis=-1)
        affected_sets.append(set(np.flatnonzero(valid_detection & ((rel >= relevance_thr) | (change >= change_thr))).tolist()))

    union = set().union(*affected_sets) if affected_sets else set()
    unique_sets = len({tuple(sorted(x)) for x in affected_sets})
    agent_diff = max_ref_diff(agents)
    ego_diff = max_ref_diff(egos)
    rel_diff = max_ref_diff(rels)
    speed_diff = max_ref_diff(speed_probs)
    path_diff = max_ref_diff(path_probs)
    # Prefer actual trajectory/set changes; ego/probability changes are useful
    # secondary evidence but should not dominate the visualization ranking.
    visual_score = 10.0 * agent_diff + 2.0 * max(0, unique_sets - 1) + ego_diff
    return dict(
        index=index,
        folder=info.get("folder"),
        frame_idx=info.get("frame_idx"),
        num_valid_agents=int(np.squeeze(to_numpy(pts.get("cf_num_valid_agents", 0)))),
        selected_candidate=int(np.argmin(np.squeeze(to_numpy(pts["cf_risk_scores"]["total"])))),
        ego_max_diff=ego_diff,
        agent_max_diff=agent_diff,
        relevance_max_diff=rel_diff,
        speed_prob_max_diff=speed_diff,
        path_prob_max_diff=path_diff,
        affected_counts=[len(x) for x in affected_sets],
        affected_union=sorted(union),
        unique_affected_sets=unique_sets,
        visual_score=visual_score,
    )


def save_plot_data(pts, info, index, out_dir, relevance_thr=0.5, change_thr=0.15, score_thr=0.25):
    """Save all numeric arrays needed to redraw candidate-conditioned panels."""
    boxes, base_futures, scores, labels = agent_futures_from_result(pts)
    boxes = np.asarray(boxes, dtype=np.float32)
    base_futures = np.asarray(base_futures, dtype=np.float32)
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    labels = np.asarray(labels if labels is not None else np.zeros(len(boxes)), dtype=np.int64)
    scenes = pts.get("cf_counterfactual_scenes", [])
    rels_raw = pts.get("cf_interaction_relevance", [])
    responses = pts.get("cf_response_meta_actions", [])

    ego_candidates = []
    realized_futures = []
    relevance = []
    speed_labels = []
    path_labels = []
    speed_probs = []
    path_probs = []
    affected_masks = []
    trajectory_change = []
    affected_ids = []
    valid_detection = scores >= score_thr

    for k, scene in enumerate(scenes):
        ego_k = np.squeeze(to_numpy(scene.get("ego_future"))).astype(np.float32)
        realized_k = np.squeeze(to_numpy(scene.get("agent_futures"))).astype(np.float32)
        rel_k = np.squeeze(to_numpy(rels_raw[k])).astype(np.float32)[: len(boxes)]
        response = responses[k]
        speed_k = np.atleast_1d(np.squeeze(to_numpy(response.get("speed")))).astype(np.int64)[: len(boxes)]
        path_k = np.atleast_1d(np.squeeze(to_numpy(response.get("path")))).astype(np.int64)[: len(boxes)]
        speed_prob_k = np.squeeze(to_numpy(response.get("speed_probs"))).astype(np.float32)[: len(boxes)]
        path_prob_k = np.squeeze(to_numpy(response.get("path_probs"))).astype(np.float32)[: len(boxes)]
        change_k = np.zeros(len(boxes), dtype=np.float32)
        n = min(len(realized_k), len(base_futures), len(boxes))
        if n:
            change_k[:n] = np.linalg.norm(
                realized_k[:n, :, :2] - base_futures[:n, :, :2], axis=-1
            ).max(axis=-1)
        affected_k = valid_detection & ((rel_k >= relevance_thr) | (change_k >= change_thr))

        ego_candidates.append(ego_k)
        realized_futures.append(realized_k)
        relevance.append(rel_k)
        speed_labels.append(speed_k)
        path_labels.append(path_k)
        speed_probs.append(speed_prob_k)
        path_probs.append(path_prob_k)
        trajectory_change.append(change_k)
        affected_masks.append(affected_k)
        affected_ids.append(np.flatnonzero(affected_k).astype(int).tolist())

    risk_scores = {
        key: np.atleast_1d(np.squeeze(to_numpy(value))).astype(np.float32)
        for key, value in pts.get("cf_risk_scores", {}).items()
    }
    valid_mask_raw = to_numpy(pts.get("cf_agent_valid_mask"))
    valid_mask = (
        np.squeeze(valid_mask_raw).astype(bool)
        if valid_mask_raw is not None
        else np.zeros(len(boxes), dtype=bool)
    )
    map_pts_raw = to_numpy(pts.get("map_pts_3d"))
    map_points = np.asarray(map_pts_raw, dtype=np.float32) if map_pts_raw is not None else np.empty((0, 0, 2), dtype=np.float32)

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "index_{}_plot_data.npz".format(index))
    np.savez_compressed(
        npz_path,
        ego_candidates=np.stack(ego_candidates),
        base_agent_futures=base_futures,
        realized_agent_futures=np.stack(realized_futures),
        boxes=boxes,
        scores=scores,
        labels=labels,
        agent_valid_mask=valid_mask,
        relevance=np.stack(relevance),
        speed_labels=np.stack(speed_labels),
        path_labels=np.stack(path_labels),
        speed_probs=np.stack(speed_probs),
        path_probs=np.stack(path_probs),
        trajectory_change=np.stack(trajectory_change),
        affected_mask=np.stack(affected_masks),
        map_points=map_points,
        **{"risk_{}".format(key): value for key, value in risk_scores.items()}
    )
    meta_path = os.path.join(out_dir, "index_{}_plot_data.json".format(index))
    total_risk = risk_scores.get("total", np.asarray([], dtype=np.float32))
    metadata = dict(
        index=index,
        folder=info.get("folder"),
        frame_idx=info.get("frame_idx"),
        candidate_source=pts.get("cf_ego_candidate_source", "unknown"),
        selected_candidate=int(np.argmin(total_risk)) if total_risk.size else None,
        score_threshold=score_thr,
        relevance_threshold=relevance_thr,
        change_threshold=change_thr,
        num_decoded_agents=len(boxes),
        num_valid_agents=int(valid_mask.sum()),
        affected_agent_ids=affected_ids,
        npz=npz_path,
    )
    with open(meta_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    return npz_path, meta_path


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = prepare_cfg(args)
    infos = load_infos(args.infos)
    dataset = build_dataset(cfg.data.test)
    model = build_live_model(cfg, args.checkpoint, args.device_id)
    visualize_indices = {int(x) for x in args.visualize_indices.split(",") if x.strip()}
    visualize_dir = args.visualize_dir or os.path.splitext(args.out)[0] + "_figures"
    records, failures = [], []
    for index in range(args.start, args.end + 1):
        try:
            data = collate([dataset[index]], samples_per_gpu=1)
            with torch.no_grad():
                result = model(data, return_loss=False)
            bbox_result = result["bbox_results"][0] if isinstance(result, dict) and "bbox_results" in result else result[0]
            pts = get_pts_bbox(bbox_result)
            records.append(scan_one(pts, infos[index], index, args.relevance_thr, args.change_thr))
            if index in visualize_indices:
                npz_path, meta_path = save_plot_data(
                    pts,
                    infos[index],
                    index,
                    visualize_dir,
                    relevance_thr=args.relevance_thr,
                    change_thr=args.change_thr,
                )
                print("saved plot data", npz_path, meta_path, flush=True)
                draw_live_candidate_panels(
                    pts=pts,
                    info=infos[index],
                    out_path=os.path.join(visualize_dir, "index_{}".format(index)),
                    candidate_text="0,1,2,3,4,5,6",
                    score_thr=0.25,
                    max_agents=10,
                    show_text=args.show_text,
                    figsize="21,6.8",
                    dpi=220,
                    relevance_thr=args.relevance_thr,
                    change_thr=args.change_thr,
                    max_context=5,
                    show_low_confidence_cf=False,
                )
        except Exception as exc:
            failures.append(dict(index=index, error=repr(exc)))
        if (index - args.start + 1) % args.progress_every == 0:
            print("processed {}/{}; failures={}".format(index - args.start + 1, args.end - args.start + 1, len(failures)), flush=True)

    ranked = sorted(records, key=lambda x: x["visual_score"], reverse=True)
    payload = dict(start=args.start, end=args.end, records=records, ranked=ranked, failures=failures)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(dict(top=ranked[:20], failures=failures), indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
