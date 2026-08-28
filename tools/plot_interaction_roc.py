"""Plot ROC for counterfactual interaction relevance prediction.

This evaluates factual interaction regularities only. Ground-truth relevance is
derived from logged ego and agent futures; no counterfactual ground-truth label
is assumed. Predicted relevance is taken from the selected CF ego candidate by
default, and each predicted agent is matched to a GT agent by current box center.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mmcv.models.counterfactual.interaction_labels import interaction_relevance_labels
from tools.analyze_cf_predictions import (
    agent_futures_from_infos,
    ego_future_from_infos,
    extract_infos,
    extract_results,
    match_agents,
    selected_cf_index,
    to_numpy,
)


def get_pts(sample):
    return sample.get("pts_bbox", sample)


def choose_candidate_index(pts, mode):
    if mode == "base":
        return 0
    if mode == "selected":
        idx = selected_cf_index(pts)
        return 0 if idx is None else idx
    raise ValueError("Unsupported candidate mode: {}".format(mode))


def roc_curve_np(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float64)
    order = np.argsort(-scores)
    labels = labels[order]
    scores = scores[order]

    pos = max(int(labels.sum()), 1)
    neg = max(int((1 - labels).sum()), 1)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)

    tpr = np.concatenate([[0.0], tp / pos, [1.0]])
    fpr = np.concatenate([[0.0], fp / neg, [1.0]])
    thresholds = np.concatenate([[np.inf], scores, [-np.inf]])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, thresholds, auc


def collect_scores(results, infos, max_samples=None, score_thr=0.25, match_dist=4.0, candidate="selected"):
    labels = []
    scores = []
    skipped = 0
    matched = 0
    max_n = min(len(results), len(infos))
    if max_samples is not None:
        max_n = min(max_n, max_samples)

    for idx in range(max_n):
        pts = get_pts(results[idx])
        relevance = pts.get("cf_interaction_relevance")
        if not relevance:
            skipped += 1
            continue

        cand_idx = choose_candidate_index(pts, candidate)
        if cand_idx >= len(relevance):
            skipped += 1
            continue
        pred_rel = np.squeeze(to_numpy(relevance[cand_idx])).astype(np.float32)

        pred_boxes = to_numpy(pts.get("boxes_3d"))
        pred_scores = to_numpy(pts.get("scores_3d"))
        if pred_boxes is None or pred_scores is None:
            skipped += 1
            continue

        keep = pred_scores >= score_thr
        kept_indices = np.where(keep)[0]
        if len(kept_indices) == 0:
            skipped += 1
            continue

        ego_gt, ego_mask = ego_future_from_infos(infos, idx)
        if ego_mask.sum() == 0:
            skipped += 1
            continue
        gt_agent_futs, gt_agent_masks, gt_boxes, _ = agent_futures_from_infos(infos, idx)
        if len(gt_boxes) == 0:
            skipped += 1
            continue

        pairs = match_agents(pred_boxes[keep], gt_boxes, max_dist=match_dist)
        if not pairs:
            continue

        ego_t = torch.from_numpy(ego_gt[None]).float()
        agent_t = torch.from_numpy(gt_agent_futs[None]).float()
        gt_rel = interaction_relevance_labels(ego_t, agent_t).squeeze(0).cpu().numpy()

        for kept_pi, gi, _ in pairs:
            pred_i = kept_indices[kept_pi]
            if pred_i >= len(pred_rel) or gi >= len(gt_rel):
                continue
            scores.append(float(pred_rel[pred_i]))
            labels.append(int(gt_rel[gi] > 0.5))
            matched += 1

    return np.asarray(labels), np.asarray(scores), {"samples": max_n, "skipped": skipped, "matched_agents": matched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--match-dist", type=float, default=4.0)
    parser.add_argument("--candidate", choices=["selected", "base"], default="selected")
    args = parser.parse_args()

    results = extract_results(args.results)
    infos = extract_infos(args.infos)
    labels, scores, counts = collect_scores(
        results,
        infos,
        max_samples=args.max_samples,
        score_thr=args.score_thr,
        match_dist=args.match_dist,
        candidate=args.candidate,
    )
    if len(labels) == 0 or labels.min() == labels.max():
        raise RuntimeError("Need both positive and negative matched labels for ROC; got {} labels.".format(len(labels)))

    fpr, tpr, thresholds, auc = roc_curve_np(labels, scores)
    os.makedirs(args.out_dir, exist_ok=True)

    fig_path = os.path.join(args.out_dir, "interaction_relevance_roc.png")
    plt.figure(figsize=(5.5, 5.0))
    plt.plot(fpr, tpr, color="#d62728", lw=2.0, label="CF interaction relevance AUC={:.3f}".format(auc))
    plt.plot([0, 1], [0, 1], color="#777777", lw=1.0, linestyle="--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Interaction Relevance ROC ({})".format(args.candidate))
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    metrics = {
        "auc": auc,
        "num_labels": int(len(labels)),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "score_mean": float(scores.mean()),
        "score_pos_mean": float(scores[labels == 1].mean()),
        "score_neg_mean": float(scores[labels == 0].mean()),
        "counts": counts,
        "candidate": args.candidate,
        "score_thr": args.score_thr,
        "match_dist": args.match_dist,
    }
    json_path = os.path.join(args.out_dir, "interaction_relevance_roc.json")
    npz_path = os.path.join(args.out_dir, "interaction_relevance_roc_points.npz")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(npz_path, fpr=fpr, tpr=tpr, thresholds=thresholds, labels=labels, scores=scores)
    print("AUC: {:.4f}".format(auc))
    print("positives: {} negatives: {}".format(metrics["num_positive"], metrics["num_negative"]))
    print("wrote", fig_path)
    print("wrote", json_path)
    print("wrote", npz_path)


if __name__ == "__main__":
    main()
