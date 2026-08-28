"""Analyze MindDrive/CF planning, motion and interaction predictions.

The script compares saved MindDrive test outputs with Bench2Drive info files
without loading images. Ground-truth futures are reconstructed with the same
relative-frame logic used by B2D_minddrive_Dataset:
  - ego_fut_trajs are per-step ego displacements, then cumsum to positions
  - gt_attr_labels agent futures are per-step displacements, then cumsum and
    shifted by the current GT box center

Interaction labels are factual pseudo-labels from logged ego/agent futures.
They are used only for validation of learned interaction regularities; no
counterfactual ground-truth supervision is assumed.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from mmcv.fileio.io import load

from mmcv.models.counterfactual.interaction_labels import interaction_relevance_labels
from mmcv.models.counterfactual.meta_action_labels import path_pseudo_labels, speed_pseudo_labels


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "tensor"):
        return to_numpy(x.tensor)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return np.asarray(x)


def load_pickle(path):
    return load(path)


def extract_results(path):
    data = load_pickle(path)
    if isinstance(data, dict) and "bbox_results" in data:
        return data["bbox_results"]
    if isinstance(data, list):
        return data
    raise TypeError("Unsupported results format: {}".format(type(data)))


def extract_infos(path):
    infos = load_pickle(path)
    if isinstance(infos, dict):
        infos = infos.get("infos", infos.get("data_list", infos))
    if not isinstance(infos, list):
        raise TypeError("Unsupported infos format: {}".format(type(infos)))
    return infos


def invert_pose(pose):
    return np.linalg.inv(np.asarray(pose))


def ego_future_from_infos(infos, idx, sample_rate=5, past_frames=2, future_frames=6):
    cur = infos[idx]
    world2lidar_cur = np.asarray(cur["sensors"]["LIDAR_TOP"]["world2lidar"])
    full_track = np.zeros((past_frames + future_frames + 1, 2), dtype=np.float32)
    mask = np.zeros(past_frames + future_frames + 1, dtype=np.float32)
    adj_indices = range(idx - past_frames * sample_rate, idx + (future_frames + 1) * sample_rate, sample_rate)
    for j, adj_idx in enumerate(adj_indices):
        if adj_idx < 0 or adj_idx >= len(infos):
            break
        adj = infos[adj_idx]
        if adj["folder"] != cur["folder"]:
            break
        world2lidar_adj = np.asarray(adj["sensors"]["LIDAR_TOP"]["world2lidar"])
        adj2cur = world2lidar_cur @ np.linalg.inv(world2lidar_adj)
        full_track[j] = adj2cur[:2, 3]
        mask[j] = 1
    offsets = full_track[1:] - full_track[:-1]
    for j in range(past_frames, past_frames + future_frames):
        if mask[j + 1] == 0:
            offsets[j] = 0
    fut = offsets[past_frames : past_frames + future_frames]
    fut_mask = mask[-future_frames:]
    return np.cumsum(fut, axis=0), fut_mask


def agent_futures_from_infos(infos, idx, sample_rate=5, future_frames=6):
    cur = infos[idx]
    world2lidar_cur = np.asarray(cur["sensors"]["LIDAR_TOP"]["world2lidar"])
    gt_boxes = np.asarray(cur["gt_boxes"])
    gt_ids = np.asarray(cur["gt_ids"])
    gt_names = np.asarray(cur["gt_names"])
    num_points = np.asarray(cur.get("num_points", np.ones(len(gt_ids))))
    valid_now = num_points != 0

    tracks = np.zeros((len(gt_ids), future_frames + 1, 2), dtype=np.float32)
    masks = np.zeros((len(gt_ids), future_frames + 1), dtype=np.float32)
    adj_indices = range(idx, idx + (future_frames + 1) * sample_rate, sample_rate)
    for i, box_id in enumerate(gt_ids):
        for j, adj_idx in enumerate(adj_indices):
            if adj_idx < 0 or adj_idx >= len(infos):
                break
            adj = infos[adj_idx]
            if adj["folder"] != cur["folder"]:
                break
            adj_ids = np.asarray(adj["gt_ids"])
            matched = np.where(adj_ids == box_id)[0]
            if len(matched) != 1:
                continue
            adj_box2lidar = world2lidar_cur @ np.asarray(adj["npc2world"][matched[0]])
            tracks[i, j] = adj_box2lidar[:2, 3]
            masks[i, j] = 1
    offsets = tracks[:, 1:] - tracks[:, :-1]
    fut_masks = masks[:, 1:]
    offsets[fut_masks == 0] = 0
    futures_abs = np.cumsum(offsets, axis=1) + gt_boxes[:, None, :2]
    return futures_abs[valid_now], fut_masks[valid_now], gt_boxes[valid_now], gt_names[valid_now]


def traj_metrics(pred, gt, mask=None):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    t = min(pred.shape[-2], gt.shape[-2])
    pred = pred[..., :t, :]
    gt = gt[..., :t, :]
    err = np.linalg.norm(pred - gt, axis=-1)
    if mask is not None:
        mask = np.asarray(mask)[..., :t].astype(bool)
        if not mask.any():
            return None
        ade = float(err[mask].mean())
        valid_last = np.where(mask)[0][-1] if mask.ndim == 1 else t - 1
        fde = float(err[..., valid_last])
    else:
        ade = float(err.mean())
        fde = float(err[..., -1])
    final_vec = pred[..., t - 1, :] - gt[..., t - 1, :]
    return dict(ade=ade, fde=fde, final_lateral=float(final_vec[..., 0]), final_longitudinal=float(final_vec[..., 1]))


def decode_pred_agent_futures(pts):
    trajs = to_numpy(pts.get("trajs_3d"))
    boxes = to_numpy(pts.get("boxes_3d"))
    scores = to_numpy(pts.get("scores_3d"))
    labels = to_numpy(pts.get("labels_3d"))
    if trajs is None or boxes is None:
        return None, None, None, None
    if trajs.ndim == 3 and trajs.shape[-1] != 2:
        trajs = trajs[:, 0].reshape(trajs.shape[0], -1, 2)
    elif trajs.ndim == 2:
        trajs = trajs.reshape(trajs.shape[0], -1, 2)
    elif trajs.ndim == 4:
        trajs = trajs[:, 0]
    trajs_abs = np.cumsum(trajs[:, :6], axis=1) + boxes[:, None, :2]
    return trajs_abs, boxes, scores, labels


def match_agents(pred_boxes, gt_boxes, max_dist=4.0):
    if pred_boxes is None or gt_boxes is None or len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    dist = np.linalg.norm(pred_boxes[:, None, :2] - gt_boxes[None, :, :2], axis=-1)
    pairs = []
    used_pred = set()
    used_gt = set()
    for _ in range(min(len(pred_boxes), len(gt_boxes))):
        flat = np.argmin(dist)
        pi, gi = np.unravel_index(flat, dist.shape)
        if dist[pi, gi] > max_dist:
            break
        if pi not in used_pred and gi not in used_gt:
            pairs.append((pi, gi, float(dist[pi, gi])))
            used_pred.add(pi)
            used_gt.add(gi)
        dist[pi, :] = np.inf
        dist[:, gi] = np.inf
    return pairs


def summarize(values):
    values = [float(v) for v in values if np.isfinite(v)]
    if not values:
        return dict(count=0, mean=None, median=None, p90=None)
    arr = np.asarray(values, dtype=np.float64)
    return dict(count=int(arr.size), mean=float(arr.mean()), median=float(np.median(arr)), p90=float(np.percentile(arr, 90)))


def selected_cf_index(pts):
    risks = pts.get("cf_risk_scores")
    if not risks or "total" not in risks:
        return None
    total = to_numpy(risks["total"])
    if total is None:
        return None
    total = np.squeeze(total)
    return int(np.argmin(total))


def analyze(results, infos, max_samples=None, score_thr=0.25):
    out = defaultdict(list)
    counts = defaultdict(int)
    max_n = min(len(results), len(infos))
    if max_samples is not None:
        max_n = min(max_n, max_samples)

    for idx in range(max_n):
        sample = results[idx]
        pts = sample.get("pts_bbox", sample)
        ego_gt, ego_mask = ego_future_from_infos(infos, idx)
        base_ego = to_numpy(pts.get("ego_fut_preds"))
        if base_ego is not None:
            m = traj_metrics(np.squeeze(base_ego), ego_gt, ego_mask)
            if m:
                for k, v in m.items():
                    out["ego_base_" + k].append(v)
                counts["ego_base_samples"] += 1

        cf_ego = to_numpy(pts.get("cf_selected_ego_future"))
        if cf_ego is not None:
            m = traj_metrics(np.squeeze(cf_ego), ego_gt, ego_mask)
            if m:
                for k, v in m.items():
                    out["ego_cf_" + k].append(v)
                counts["ego_cf_samples"] += 1

        gt_agent_futs, gt_agent_masks, gt_boxes, _ = agent_futures_from_infos(infos, idx)
        pred_futs, pred_boxes, pred_scores, _ = decode_pred_agent_futures(pts)
        if pred_futs is not None:
            keep = np.ones(len(pred_futs), dtype=bool)
            if pred_scores is not None:
                keep = pred_scores >= score_thr
            pred_futs_keep = pred_futs[keep]
            pred_boxes_keep = pred_boxes[keep]
            pairs = match_agents(pred_boxes_keep, gt_boxes)
            counts["agent_matched"] += len(pairs)
            counts["agent_gt_total"] += len(gt_boxes)
            counts["agent_pred_total"] += len(pred_boxes_keep)
            for pi, gi, center_dist in pairs:
                mask = gt_agent_masks[gi].astype(bool)
                if not mask.any():
                    continue
                err = np.linalg.norm(pred_futs_keep[pi, :6] - gt_agent_futs[gi, :6], axis=-1)
                out["agent_ade"].append(float(err[mask[:6]].mean()))
                out["agent_fde"].append(float(err[np.where(mask[:6])[0][-1]]))
                out["agent_center_match_dist"].append(center_dist)

        sel = selected_cf_index(pts)
        relevance_list = pts.get("cf_interaction_relevance")
        response_list = pts.get("cf_response_meta_actions")
        if sel is not None and relevance_list is not None and response_list is not None:
            if sel < len(relevance_list) and sel < len(response_list):
                pred_rel = np.squeeze(to_numpy(relevance_list[sel]))
                pred_speed = np.squeeze(to_numpy(response_list[sel]["speed"]))
                pred_path = np.squeeze(to_numpy(response_list[sel]["path"]))
                n = min(len(pred_rel), len(gt_agent_futs))
                if n > 0:
                    ego_t = torch.from_numpy(ego_gt[None]).float()
                    agent_t = torch.from_numpy(gt_agent_futs[None, :n]).float()
                    rel_label = interaction_relevance_labels(ego_t, agent_t).squeeze(0).numpy()
                    gt_rel_agent = torch.from_numpy((gt_agent_futs[:n] - gt_boxes[:n, None, :2])).float()
                    speed_label = speed_pseudo_labels(gt_rel_agent).numpy()
                    path_label = path_pseudo_labels(gt_rel_agent).numpy()
                    pred_rel_bin = pred_rel[:n] >= 0.5
                    valid = np.ones(n, dtype=bool)
                    counts["interaction_agents"] += int(valid.sum())
                    out["interaction_rel_acc"].append(float((pred_rel_bin[valid] == rel_label[:n][valid].astype(bool)).mean()))
                    out["interaction_rel_pos_rate"].append(float(pred_rel_bin.mean()))
                    out["interaction_rel_label_pos_rate"].append(float(rel_label[:n].mean()))
                    out["response_speed_acc"].append(float((pred_speed[:n] == speed_label[:n]).mean()))
                    out["response_path_acc"].append(float((pred_path[:n] == path_label[:n]).mean()))

    summary = {"counts": dict(counts)}
    for key, values in sorted(out.items()):
        summary[key] = summarize(values)
    if counts["agent_gt_total"]:
        summary["agent_match_recall"] = counts["agent_matched"] / max(counts["agent_gt_total"], 1)
    if counts["agent_pred_total"]:
        summary["agent_match_precision"] = counts["agent_matched"] / max(counts["agent_pred_total"], 1)
    return summary


def write_report(summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "trajectory_interaction_metrics.json")
    md_path = os.path.join(out_dir, "trajectory_interaction_metrics.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(md_path, "w") as f:
        f.write("# Trajectory and Interaction Metrics\n\n")
        f.write("## Counts\n\n")
        for k, v in sorted(summary.get("counts", {}).items()):
            f.write("- {}: {}\n".format(k, v))
        for k in ("agent_match_recall", "agent_match_precision"):
            if k in summary:
                f.write("- {}: {:.4f}\n".format(k, summary[k]))
        f.write("\n## Metrics\n\n")
        for k, v in sorted(summary.items()):
            if k == "counts" or not isinstance(v, dict) or v.get("count", 0) == 0:
                continue
            f.write("- {}: mean={:.4f}, median={:.4f}, p90={:.4f}, n={}\n".format(
                k, v["mean"], v["median"], v["p90"], v["count"]))
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.25)
    args = parser.parse_args()

    results = extract_results(args.results)
    infos = extract_infos(args.infos)
    summary = analyze(results, infos, max_samples=args.max_samples, score_thr=args.score_thr)
    json_path, md_path = write_report(summary, args.out_dir)
    print("wrote", json_path)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
