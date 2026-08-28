#!/usr/bin/env python
"""Visualize one counterfactual MindDrive case study.

The script reads MindDrive test outputs and B2D infos, then produces:
  1. camera input montage when image paths are available;
  2. BEV scene with map vectors, boxes, base agent futures, base/CF ego futures;
  3. candidate meta-action risk comparison;
  4. interaction relevance and response meta-action diagnostics.

If ``cf_counterfactual_scenes`` exists in the result file, candidate-conditioned
realized agent futures are drawn for the selected/top candidates. Older result
files usually omit this field to keep pkl size small; in that case the script
falls back to base agent futures plus candidate-conditioned relevance/response
diagnostics.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mmcv.fileio.io import load

try:
    from PIL import Image
except ImportError:
    Image = None

from mmcv.models.counterfactual.meta_action_labels import PATH_META_ACTIONS, SPEED_META_ACTIONS


CAM_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray([to_numpy(v) for v in x], dtype=object)
    return np.asarray(x)


def load_pickle(path):
    return load(path)


def result_items(results):
    if isinstance(results, dict) and "bbox_results" in results:
        return results["bbox_results"]
    return results


def get_pts_bbox(item):
    if isinstance(item, dict) and "pts_bbox" in item:
        return item["pts_bbox"]
    return item


def load_infos(path):
    data = load_pickle(path)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data


def ego_traj(x):
    arr = to_numpy(x).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim == 3:
        arr = arr[0]
    return arr[..., :2]


def agent_futures_from_result(pts):
    boxes = to_numpy(pts.get("boxes_3d"))
    trajs = to_numpy(pts.get("trajs_3d"))
    scores = to_numpy(pts.get("scores_3d"))
    labels = to_numpy(pts.get("labels_3d"))
    if boxes is None or trajs is None:
        return None, None, scores, labels
    boxes = np.asarray(boxes, dtype=np.float32)
    trajs = np.asarray(trajs, dtype=np.float32)
    if trajs.ndim == 4:
        trajs = trajs[:, 0]
    elif trajs.ndim == 3 and trajs.shape[-1] != 2:
        trajs = trajs[:, 0]
    if trajs.ndim == 2:
        trajs = trajs.reshape(trajs.shape[0], -1, 2)
    base_xy = boxes[:, :2]
    abs_trajs = np.cumsum(trajs[..., :2], axis=1) + base_xy[:, None, :]
    return boxes, abs_trajs, scores, labels


def map_polylines(pts):
    map_pts = to_numpy(pts.get("map_pts_3d"))
    if map_pts is None:
        return []
    map_pts = np.asarray(map_pts, dtype=np.float32)
    if map_pts.ndim == 2 and map_pts.shape[-1] >= 2:
        return [map_pts[:, :2]]
    if map_pts.ndim == 3:
        return [p[:, :2] for p in map_pts]
    return []


def risk_arrays(pts):
    risks = {}
    for key, value in pts.get("cf_risk_scores", {}).items():
        arr = np.squeeze(to_numpy(value)).astype(np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        risks[key] = arr
    if "total" not in risks and risks:
        length = max(v.size for v in risks.values())
        total = np.zeros(length, dtype=np.float32)
        for key, value in risks.items():
            if value.size == length and key != "total":
                total += value
        risks["total"] = total
    return risks


def candidate_meta(idx):
    if idx == 0:
        return "base MindDrive"
    j = idx - 1
    n_path = len(PATH_META_ACTIONS)
    speed_idx = j // n_path
    path_idx = j % n_path
    if speed_idx < len(SPEED_META_ACTIONS):
        return "{} / {}".format(SPEED_META_ACTIONS[speed_idx], PATH_META_ACTIONS[path_idx])
    return "candidate {}".format(idx)


def stored_candidate_meta(pts, idx):
    metas = pts.get("cf_candidate_meta_actions")
    if isinstance(metas, (list, tuple)) and 0 <= int(idx) < len(metas):
        meta = metas[int(idx)]
        if isinstance(meta, dict):
            speed = meta.get("speed", "")
            path = meta.get("path", "")
            if speed or path:
                return "{} / {}".format(speed, path).strip(" /")
            token_speed = meta.get("speed_token", "")
            token_path = meta.get("path_token", "")
            if token_speed or token_path:
                return "{} / {}".format(token_speed, token_path).strip(" /")
        return str(meta)
    return candidate_meta(idx)


def label_name(label, names):
    label = int(label)
    if 0 <= label < len(names):
        return names[label]
    return "unknown"


def response_labels_for_candidate(responses, idx, num_agents):
    if not responses or idx >= len(responses):
        return None, None, None, None
    resp = responses[idx]
    speed = np.squeeze(to_numpy(resp.get("speed"))).astype(np.int64)
    path = np.squeeze(to_numpy(resp.get("path"))).astype(np.int64)
    speed_probs = to_numpy(resp.get("speed_probs"))
    path_probs = to_numpy(resp.get("path_probs"))
    if speed_probs is not None:
        speed_probs = np.squeeze(speed_probs).astype(np.float32)[:num_agents]
    if path_probs is not None:
        path_probs = np.squeeze(path_probs).astype(np.float32)[:num_agents]
    return speed[:num_agents], path[:num_agents], speed_probs, path_probs


def response_text(label, names, probs=None):
    name = label_name(label, names)
    aliases = {
        "maintain moderate speed": "mod",
        "stop": "stop",
        "maintain slow speed": "slow",
        "speed up": "up",
        "slow down": "down",
        "maintain fast speed": "fast",
        "slow down rapidly": "rapid",
        "lanefollow": "lane",
        "straight": "str",
        "turn left": "left",
        "change lane left": "cl-left",
        "turn right": "right",
        "change lane right": "cl-right",
    }
    short = aliases.get(name, name[:10])
    if probs is None or int(label) >= len(probs):
        return short
    return "{}:{:.2f}".format(short, float(probs[int(label)]))


def draw_response_table(ax, selected_agents, rel, speed_labels, path_labels, speed_probs=None, path_probs=None, max_rows=5):
    if speed_labels is None or path_labels is None or not selected_agents:
        return
    rows = ["agent  rel   speed-p  path-p"]
    for agent_idx in selected_agents[:max_rows]:
        if agent_idx >= len(speed_labels) or agent_idx >= len(path_labels):
            continue
        speed_agent_probs = speed_probs[agent_idx] if speed_probs is not None and agent_idx < len(speed_probs) else None
        path_agent_probs = path_probs[agent_idx] if path_probs is not None and agent_idx < len(path_probs) else None
        speed = response_text(speed_labels[agent_idx], SPEED_META_ACTIONS, speed_agent_probs)
        path = response_text(path_labels[agent_idx], PATH_META_ACTIONS, path_agent_probs)
        rows.append("{:>3d}  {:.2f}  {:<8s} {}".format(int(agent_idx), float(rel[agent_idx]), speed[:8], path[:8]))
    if len(rows) == 1:
        return
    ax.text(
        0.02,
        0.98,
        "\n".join(rows),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.6,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.82, boxstyle="round,pad=0.25"),
    )


def fallback_candidate_ego(idx, future_steps=6, dt=0.5):
    """Reconstruct the lightweight fallback ego candidate for visualization.

    This mirrors ``EgoCandidateGenerator``. It is only used for plotting when
    the result file does not store all ego candidates explicitly.
    """
    if idx <= 0:
        return None
    j = idx - 1
    path_count = len(PATH_META_ACTIONS)
    speed_idx = j // path_count
    path_idx = j % path_count
    if speed_idx >= len(SPEED_META_ACTIONS):
        return None
    t = np.arange(1, future_steps + 1, dtype=np.float32) * dt
    base_v = np.asarray([5.0, 0.0, 2.0, 7.0, 3.5, 9.0, 1.5], dtype=np.float32)[speed_idx]
    lateral = np.zeros_like(t)
    longitudinal = base_v * t
    path = PATH_META_ACTIONS[path_idx]
    if path == "turn left":
        lateral = 0.5 * t * t
    elif path == "turn right":
        lateral = -0.5 * t * t
    elif path == "change lane left":
        lateral = np.linspace(0.0, 3.5, future_steps, dtype=np.float32)
    elif path == "change lane right":
        lateral = np.linspace(0.0, -3.5, future_steps, dtype=np.float32)
    lateral = lateral * min(base_v / 5.0, 1.0)
    return np.stack([lateral, longitudinal], axis=-1)


def candidate_ego_for_plot(pts, idx, future_steps=6, base_ego=None):
    """Return the exact stored ego candidate when it exists.

    Result files produced with ``save_counterfactual_scenes=True`` contain the
    ego trajectory actually consumed by the counterfactual head. Older result
    files do not, so only those fall back to the rule candidate visualization.
    """
    idx = int(idx)
    scenes = pts.get("cf_counterfactual_scenes")
    if scenes and idx < len(scenes):
        cand = ego_traj(scenes[idx].get("ego_future"))
        if cand is not None and cand.size:
            return cand
    if idx == 0 and base_ego is not None:
        return base_ego
    return fallback_candidate_ego(idx, future_steps=future_steps)


def select_case(items, requested_idx=None):
    if requested_idx is not None:
        return requested_idx
    best_idx = 0
    best_gain = -np.inf
    for idx, item in enumerate(items):
        pts = get_pts_bbox(item)
        risks = risk_arrays(pts)
        total = risks.get("total")
        if total is None or total.size < 2:
            continue
        gain = float(total[0] - np.min(total))
        rels = pts.get("cf_interaction_relevance", [])
        if rels:
            selected = int(np.argmin(total))
            rel = np.squeeze(to_numpy(rels[selected])).astype(np.float32)
            gain += 0.1 * float(np.nanmax(rel)) if rel.size else 0.0
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    return best_idx


def resolve_case_index(infos, index=None, scene_folder=None, scenario_name=None, frame_idx=None):
    if index is not None:
        return index
    if scene_folder is None and scenario_name is None:
        return None

    matches = []
    for i, info in enumerate(infos):
        folder = info.get("folder", "")
        if scene_folder is not None and folder != scene_folder:
            continue
        if scenario_name is not None and scenario_name not in folder:
            continue
        if frame_idx is not None and info.get("frame_idx") != frame_idx:
            continue
        matches.append(i)

    if not matches:
        raise ValueError(
            "No case found for scene_folder={!r}, scenario_name={!r}, frame_idx={!r}".format(
                scene_folder, scenario_name, frame_idx
            )
        )
    if frame_idx is None and len(matches) > 1:
        examples = [(i, infos[i].get("folder"), infos[i].get("frame_idx")) for i in matches[:10]]
        raise ValueError(
            "Matched {} frames. Please pass --frame-idx. First matches: {}".format(len(matches), examples)
        )
    return matches[0]


def resolve_case_indices(infos, index=None, scene_folder=None, scenario_name=None, frame_idx=None, all_frames=False):
    if all_frames:
        if index is not None:
            raise ValueError("--all-frames cannot be used together with --index.")
        if scene_folder is None and scenario_name is None:
            raise ValueError("--all-frames requires --scene-folder or --scenario-name.")
        matches = []
        for i, info in enumerate(infos):
            folder = info.get("folder", "")
            if scene_folder is not None and folder != scene_folder:
                continue
            if scenario_name is not None and scenario_name not in folder:
                continue
            if frame_idx is not None and info.get("frame_idx") != frame_idx:
                continue
            matches.append(i)
        if not matches:
            raise ValueError(
                "No frames found for scene_folder={!r}, scenario_name={!r}, frame_idx={!r}".format(
                    scene_folder, scenario_name, frame_idx
                )
            )
        return matches
    resolved = resolve_case_index(infos, index, scene_folder, scenario_name, frame_idx)
    return [select_case([], resolved) if resolved is not None else None]


def resolve_image_path(path, data_root):
    if not path:
        return None
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    candidates = [
        os.path.join(data_root, path),
        os.path.join("data/bench2drive", path),
        path,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def save_camera_montage(info, out_path, data_root):
    if Image is None or "sensors" not in info:
        return False
    images = []
    titles = []
    for cam in CAM_ORDER:
        sensor = info["sensors"].get(cam, {})
        path = resolve_image_path(sensor.get("data_path"), data_root)
        if path is None:
            continue
        with Image.open(path) as img:
            images.append(np.asarray(img.convert("RGB")))
            titles.append(cam)
    if not images:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def draw_box(ax, box, color="tab:blue", alpha=0.35):
    x, y = box[:2]
    w = float(box[3]) if len(box) > 3 else 1.8
    l = float(box[4]) if len(box) > 4 else 4.2
    yaw = float(box[6]) if len(box) > 6 else 0.0
    corners = np.array(
        [[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2], [l / 2, w / 2]],
        dtype=np.float32,
    )
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
    pts = corners @ rot.T + np.array([x, y], dtype=np.float32)
    ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, linewidth=0.8)


def draw_agent_rect(ax, xy, length=4.2, width=1.8, yaw=0.0, color="tab:blue", alpha=0.8, linewidth=1.2, linestyle="-"):
    x, y = float(xy[0]), float(xy[1])
    corners = np.array(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2], [length / 2, width / 2]],
        dtype=np.float32,
    )
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
    pts = corners @ rot.T + np.array([x, y], dtype=np.float32)
    ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)


def draw_future_rectangles(ax, future, box, color, alpha=0.85, linestyle="-", linewidth=1.1, stride=1):
    length = float(box[3]) if len(box) > 3 else 4.2
    width = float(box[4]) if len(box) > 4 else 1.8
    yaw = float(box[6]) if len(box) > 6 else 0.0
    for t, xy in enumerate(future[::stride]):
        step_alpha = alpha * (0.35 + 0.65 * (t + 1) / max(1, len(future[::stride])))
        draw_agent_rect(ax, xy, length=length, width=width, yaw=yaw, color=color, alpha=step_alpha, linewidth=linewidth, linestyle=linestyle)


def response_summary(pts, selected_idx, rel, score_mask):
    responses = pts.get("cf_response_meta_actions", [])
    if not responses or selected_idx >= len(responses):
        return {}
    resp = responses[selected_idx]
    rel_mask = rel >= 0.5
    mask = rel_mask & score_mask
    summary = {"num_relevant": int(mask.sum())}
    for key, names in (("speed", SPEED_META_ACTIONS), ("path", PATH_META_ACTIONS)):
        labels = np.squeeze(to_numpy(resp.get(key))).astype(np.int64)
        labels = labels[: len(mask)]
        counts = {}
        for label in labels[mask]:
            if 0 <= int(label) < len(names):
                counts[names[int(label)]] = counts.get(names[int(label)], 0) + 1
        summary[key] = counts
    return summary


def realize_agent_futures_np(base_futures, boxes, speed_labels, path_labels, relevance, threshold=0.5, dt=0.5, max_speed=15.0):
    """Numpy mirror of RuleBasedTrajectoryRealizer for visualization."""
    base_futures = np.asarray(base_futures, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    speed_labels = np.asarray(speed_labels, dtype=np.int64)[: base_futures.shape[0]]
    path_labels = np.asarray(path_labels, dtype=np.int64)[: base_futures.shape[0]]
    relevance = np.asarray(relevance, dtype=np.float32)[: base_futures.shape[0]]
    start = boxes[: base_futures.shape[0], :2]
    deltas = np.diff(base_futures, axis=1, prepend=start[:, None, :])

    speed_scale = np.ones_like(relevance, dtype=np.float32)
    speed_scale = np.where(speed_labels == 1, speed_scale * 0.05, speed_scale)
    speed_scale = np.where(speed_labels == 2, speed_scale * 0.65, speed_scale)
    speed_scale = np.where(speed_labels == 3, speed_scale * 1.25, speed_scale)
    speed_scale = np.where(speed_labels == 4, speed_scale * 0.8, speed_scale)
    speed_scale = np.where(speed_labels == 5, speed_scale * 1.1, speed_scale)
    speed_scale = np.where(speed_labels == 6, speed_scale * 0.45, speed_scale)

    lat_bias = np.zeros_like(relevance, dtype=np.float32)
    lat_bias = np.where(path_labels == 2, lat_bias + 0.18, lat_bias)
    lat_bias = np.where(path_labels == 3, lat_bias + 0.35, lat_bias)
    lat_bias = np.where(path_labels == 4, lat_bias - 0.18, lat_bias)
    lat_bias = np.where(path_labels == 5, lat_bias - 0.35, lat_bias)

    adjusted = deltas * speed_scale[:, None, None]
    t = np.linspace(0.0, 1.0, base_futures.shape[1], dtype=np.float32)
    adjusted[..., 0] = adjusted[..., 0] + lat_bias[:, None] * t[None, :]
    max_step = max_speed * dt
    step_norm = np.linalg.norm(adjusted, axis=-1, keepdims=True).clip(min=1e-4)
    adjusted = adjusted * np.minimum(max_step / step_norm, 1.0)
    realized = start[:, None, :] + np.cumsum(adjusted, axis=1)
    return np.where((relevance >= threshold)[:, None, None], realized, base_futures)


def scene_bounds(boxes, agent_trajs, ego_trajs):
    points = [np.array([[0.0, 0.0]], dtype=np.float32)]
    if boxes is not None and len(boxes):
        points.append(boxes[:, :2])
    if agent_trajs is not None and agent_trajs.size:
        points.append(agent_trajs.reshape(-1, 2))
    for traj in ego_trajs:
        if traj is not None:
            points.append(traj.reshape(-1, 2))
    pts = np.concatenate(points, axis=0)
    lo = np.nanpercentile(pts, 2, axis=0) - 8.0
    hi = np.nanpercentile(pts, 98, axis=0) + 8.0
    return lo, hi


def save_candidate_panels(pts, out_path, top_idx, score_thr=0.25, max_agents=8):
    boxes, agent_trajs, scores, _ = agent_futures_from_result(pts)
    if boxes is None or agent_trajs is None:
        return False
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    risks = risk_arrays(pts)
    total = risks.get("total", np.zeros(1, dtype=np.float32))
    rels = pts.get("cf_interaction_relevance", [])
    responses = pts.get("cf_response_meta_actions", [])
    scenes = pts.get("cf_counterfactual_scenes")
    base_ego = ego_traj(pts.get("ego_fut_preds")) if pts.get("ego_fut_preds") is not None else None

    panel_width = 4.0
    panel_height = 5.5
    fig, axes = plt.subplots(1, len(top_idx), figsize=(panel_width * len(top_idx), panel_height), squeeze=False)
    # fig.patch.set_facecolor("black")
    ego_trajs_for_bounds = [base_ego] + [
        candidate_ego_for_plot(pts, int(i), future_steps=agent_trajs.shape[1], base_ego=base_ego)
        for i in top_idx
    ]
    lo, hi = scene_bounds(boxes, agent_trajs, ego_trajs_for_bounds)
    lines = map_polylines(pts)

    for ax, idx in zip(axes.reshape(-1), top_idx):
        idx = int(idx)
        rel = np.zeros(len(boxes), dtype=np.float32)
        if rels and idx < len(rels):
            rel = np.squeeze(to_numpy(rels[idx])).astype(np.float32)[: len(boxes)]
        speed_labels, path_labels, speed_probs, path_probs = response_labels_for_candidate(responses, idx, len(boxes))
        score_mask = scores >= score_thr
        relevant_rank = np.argsort(-rel)
        selected_agents = [i for i in relevant_rank if score_mask[i] and rel[i] >= 0.35][:max_agents]
        if not selected_agents:
            selected_agents = [i for i in relevant_rank if score_mask[i]][: min(3, max_agents)]

        realized = agent_trajs
        if scenes and idx < len(scenes):
            scene_agents = to_numpy(scenes[idx].get("agent_futures"))
            scene_agents = np.squeeze(scene_agents).astype(np.float32)
            if scene_agents.ndim == 3:
                realized = scene_agents
        elif responses and idx < len(responses):
            resp = responses[idx]
            speed_labels = np.squeeze(to_numpy(resp.get("speed"))).astype(np.int64)
            path_labels = np.squeeze(to_numpy(resp.get("path"))).astype(np.int64)
            speed_probs = to_numpy(resp.get("speed_probs"))
            path_probs = to_numpy(resp.get("path_probs"))
            if speed_probs is not None:
                speed_probs = np.squeeze(speed_probs).astype(np.float32)[: len(boxes)]
            if path_probs is not None:
                path_probs = np.squeeze(path_probs).astype(np.float32)[: len(boxes)]
            realized = realize_agent_futures_np(agent_trajs, boxes, speed_labels, path_labels, rel)

        for line in lines[:150]:
            if len(line) > 1:
                ax.plot(line[:, 0], line[:, 1], color="0.85", linewidth=0.6, alpha=0.8)
        for i in selected_agents:
            draw_agent_rect(
                ax,
                boxes[i, :2],
                length=float(boxes[i, 3]) if boxes.shape[1] > 3 else 4.2,
                width=float(boxes[i, 4]) if boxes.shape[1] > 4 else 1.8,
                yaw=float(boxes[i, 6]) if boxes.shape[1] > 6 else 0.0,
                color="black",
                alpha=0.75,
                linewidth=1.4,
            )
            ax.plot(agent_trajs[i, :, 0], agent_trajs[i, :, 1], "--", color="0.55", linewidth=1.0, alpha=0.75)
            ax.plot(realized[i, :, 0], realized[i, :, 1], "-", color=plt.cm.magma(min(1.0, float(rel[i]))), linewidth=2.0)
            ax.scatter(agent_trajs[i, -1, 0], agent_trajs[i, -1, 1], marker="x", color="0.45", s=18, linewidths=1.1)
            ax.scatter(realized[i, -1, 0], realized[i, -1, 1], marker="o", color=plt.cm.magma(min(1.0, float(rel[i]))), s=18)
            if np.linalg.norm(realized[i, -1] - agent_trajs[i, -1]) > 0.15:
                ax.annotate(
                    "",
                    xy=(realized[i, -1, 0], realized[i, -1, 1]),
                    xytext=(agent_trajs[i, -1, 0], agent_trajs[i, -1, 1]),
                    arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.0, alpha=0.8),
                )
            draw_future_rectangles(ax, agent_trajs[i], boxes[i], color="0.55", alpha=0.35, linestyle="--", linewidth=0.8)
            draw_future_rectangles(
                ax,
                realized[i],
                boxes[i],
                color=plt.cm.magma(min(1.0, float(rel[i]))),
                alpha=0.85,
                linestyle="-",
                linewidth=1.1,
            )
            label = "{}:{:.2f}".format(i, rel[i])
            if speed_labels is not None and path_labels is not None and i < len(speed_labels) and i < len(path_labels):
                label += "\n{} / {}".format(
                    label_name(speed_labels[i], SPEED_META_ACTIONS)[:10],
                    label_name(path_labels[i], PATH_META_ACTIONS)[:10],
                )
            # ax.text(boxes[i, 0], boxes[i, 1], label, fontsize=6.5)

        cand_ego = candidate_ego_for_plot(pts, idx, future_steps=agent_trajs.shape[1], base_ego=base_ego)
        if cand_ego is not None:
            ax.plot(cand_ego[:, 0], cand_ego[:, 1], color="tab:red", linewidth=2.5, label="ego candidate")
        ax.scatter([0], [0], c="black", marker="*", s=55)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        risk_text = "risk {:.2f}".format(float(total[idx])) if idx < len(total) else ""
        ax.set_title("#{} {}\n{}".format(idx, stored_candidate_meta(pts, idx), risk_text), fontsize=9)
        # draw_response_table(ax, selected_agents, rel, speed_labels, path_labels, speed_probs, path_probs)

    for ax in axes.reshape(-1)[len(top_idx) :]:
        ax.axis("off")
    # fig.suptitle(
    #     "Candidate-conditioned agent futures: dashed gray/x=MindDrive base, solid color/o=interaction-response realized future",
    #     fontsize=13,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return True


def plot_case(pts, info, out_path, top_k=6, score_thr=0.25):
    boxes, agent_trajs, scores, labels = agent_futures_from_result(pts)
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    risks = risk_arrays(pts)
    total = risks.get("total", np.zeros(1, dtype=np.float32))
    selected_idx = int(np.argmin(total))
    top_idx = np.argsort(total)[: min(top_k, len(total))]
    base_ego = ego_traj(pts.get("ego_fut_preds")) if pts.get("ego_fut_preds") is not None else None
    selected_ego = ego_traj(pts.get("cf_selected_ego_future"))
    rels = pts.get("cf_interaction_relevance", [])
    rel = np.zeros(len(boxes), dtype=np.float32)
    if rels and selected_idx < len(rels):
        rel = np.squeeze(to_numpy(rels[selected_idx])).astype(np.float32)[: len(boxes)]
    score_mask = scores >= score_thr
    relevant_mask = score_mask & (rel >= 0.5)
    lines = map_polylines(pts)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.2, 1.0])
    ax_scene = fig.add_subplot(gs[:, 0])
    ax_risk = fig.add_subplot(gs[0, 1])
    ax_inter = fig.add_subplot(gs[1, 1])

    for line in lines[:250]:
        if len(line) > 1:
            ax_scene.plot(line[:, 0], line[:, 1], color="0.78", linewidth=0.8, alpha=0.8, zorder=1)
    if boxes is not None:
        for i, box in enumerate(boxes):
            if not score_mask[i]:
                continue
            color = plt.cm.magma(min(1.0, float(rel[i])))
            draw_box(ax_scene, box, color=color, alpha=0.75 if relevant_mask[i] else 0.28)
            if agent_trajs is not None:
                lw = 1.8 if relevant_mask[i] else 0.7
                ax_scene.plot(agent_trajs[i, :, 0], agent_trajs[i, :, 1], color=color, alpha=0.85, linewidth=lw)
    scenes = pts.get("cf_counterfactual_scenes")
    if scenes and selected_idx < len(scenes):
        cf_agents = to_numpy(scenes[selected_idx].get("agent_futures"))
        cf_agents = np.squeeze(cf_agents).astype(np.float32)
        if cf_agents.ndim == 3:
            for i in np.where(relevant_mask[: cf_agents.shape[0]])[0]:
                ax_scene.plot(
                    cf_agents[i, :, 0],
                    cf_agents[i, :, 1],
                    color="tab:cyan",
                    linewidth=2.4,
                    alpha=0.9,
                    label="selected CF agent future" if i == np.where(relevant_mask[: cf_agents.shape[0]])[0][0] else None,
                )
    if base_ego is not None:
        ax_scene.plot(base_ego[:, 0], base_ego[:, 1], "--", color="tab:blue", linewidth=2.5, label="base ego")
    for idx in top_idx:
        cand = candidate_ego_for_plot(pts, int(idx), future_steps=len(selected_ego), base_ego=base_ego)
        if cand is None:
            continue
        color = "tab:red" if int(idx) == selected_idx else "0.45"
        alpha = 0.95 if int(idx) == selected_idx else 0.35
        lw = 2.8 if int(idx) == selected_idx else 1.1
        ax_scene.plot(cand[:, 0], cand[:, 1], color=color, linewidth=lw, alpha=alpha)
    ax_scene.plot(selected_ego[:, 0], selected_ego[:, 1], "-", color="tab:red", linewidth=3.0, label="selected CF ego")
    ax_scene.scatter([0], [0], c="black", s=70, marker="*", label="ego now", zorder=5)
    ax_scene.set_title(
        "Scene input + final trajectory\n{} frame {}".format(info.get("folder", ""), info.get("frame_idx", "NA")),
        fontsize=12,
    )
    ax_scene.set_xlabel("x / lateral-like axis")
    ax_scene.set_ylabel("y / longitudinal-like axis")
    ax_scene.axis("equal")
    ax_scene.grid(True, alpha=0.2)
    ax_scene.legend(loc="upper right")

    labels_y = [candidate_meta(i) for i in top_idx]
    colors = ["tab:red" if i == selected_idx else ("tab:blue" if i == 0 else "0.45") for i in top_idx]
    ax_risk.barh(np.arange(len(top_idx)), total[top_idx], color=colors)
    ax_risk.set_yticks(np.arange(len(top_idx)), labels_y, fontsize=8)
    ax_risk.invert_yaxis()
    ax_risk.set_title("Top candidate risks")
    ax_risk.set_xlabel("total risk, lower is better")
    for y, i in enumerate(top_idx):
        ax_risk.text(total[i], y, "  #{:02d} {:.3f}".format(i, total[i]), va="center", fontsize=8)

    scatter = ax_inter.scatter(boxes[:, 0], boxes[:, 1], c=rel, s=np.clip(scores * 60, 6, 80), cmap="magma", vmin=0, vmax=1)
    for rank, i in enumerate(np.argsort(-rel)[:10]):
        if not score_mask[i]:
            continue
        ax_inter.text(boxes[i, 0], boxes[i, 1], str(i), fontsize=7)
    ax_inter.scatter([0], [0], c="black", marker="*", s=60)
    ax_inter.set_title("Interaction relevance for selected candidate #{}".format(selected_idx))
    ax_inter.set_xlabel("x")
    ax_inter.set_ylabel("y")
    ax_inter.axis("equal")
    ax_inter.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax_inter, label="rho")

    summary = response_summary(pts, selected_idx, rel, score_mask)
    text = [
        "selected: #{} {}".format(selected_idx, candidate_meta(selected_idx)),
        "relevant agents: {}".format(summary.get("num_relevant", 0)),
    ]
    for key in ("collision", "ttc", "interaction", "progress", "nominal", "total"):
        if key in risks and selected_idx < len(risks[key]):
            text.append("{}: {:.4f}".format(key, float(risks[key][selected_idx])))
    if "cf_counterfactual_scenes" not in pts:
        text.append("agent futures: base futures, with CF relevance/response diagnostics")
    fig.text(0.59, 0.03, "\n".join(text), fontsize=9, family="monospace", va="bottom")

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return selected_idx, top_idx.tolist(), summary


def write_risk_csv(path, risks, top_idx, pts=None):
    keys = ["candidate", "meta_action"] + [k for k in risks.keys()]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for idx in top_idx:
            row = {
                "candidate": int(idx),
                "meta_action": stored_candidate_meta(pts, int(idx)) if pts is not None else candidate_meta(int(idx)),
            }
            for key, value in risks.items():
                row[key] = float(value[idx]) if idx < len(value) else ""
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="MindDrive result pkl with CF diagnostics.")
    parser.add_argument("--infos", required=True, help="B2D info pkl aligned with the result order.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--index", type=int, default=None, help="Sample index. Omit to auto-pick a high-gain CF case.")
    parser.add_argument("--scene-folder", default=None, help="Exact B2D scene folder, e.g. v1/xxx_Townxx_Routexx_Weatherxx.")
    parser.add_argument("--scenario-name", default=None, help="Substring matched against info['folder'], e.g. VanillaNonSignalizedTurn.")
    parser.add_argument("--frame-idx", type=int, default=None, help="Frame index inside the matched scenario/scene.")
    parser.add_argument("--all-frames", action="store_true", help="Visualize all frames matching --scene-folder/--scenario-name.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--candidate-indices",
        default=None,
        help="Comma-separated candidate ids for candidate panels. Defaults to top-k by risk.",
    )
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--data-root", default="data/bench2drive")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    items = result_items(load_pickle(args.results))
    infos = load_infos(args.infos)
    resolved_indices = resolve_case_indices(
        infos,
        args.index,
        args.scene_folder,
        args.scenario_name,
        args.frame_idx,
        args.all_frames,
    )
    if resolved_indices == [None]:
        resolved_indices = [select_case(items, None)]
    summaries = []
    for idx in resolved_indices:
        if idx < 0 or idx >= len(items):
            raise IndexError("Resolved index {} is outside result length {}".format(idx, len(items)))
        summary = visualize_one_case(args, items, infos, idx)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    if len(summaries) > 1:
        summary_path = os.path.join(args.out_dir, "all_frames_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)
        print("Wrote {} frame summaries to {}".format(len(summaries), summary_path))


def visualize_one_case(args, items, infos, idx):
    pts = get_pts_bbox(items[idx])
    info = infos[idx]

    safe_folder = str(info.get("folder", "unknown")).replace("/", "__")
    frame = info.get("frame_idx")
    if frame is None:
        prefix = os.path.join(args.out_dir, "case_{:05d}".format(idx))
    else:
        prefix = os.path.join(args.out_dir, "{}__frame_{:04d}__idx_{:05d}".format(safe_folder, int(frame), idx))
    camera_ok = save_camera_montage(info, prefix + "_input_cameras.png", args.data_root)
    selected_idx, top_idx, response = plot_case(pts, info, prefix + "_counterfactual_case.png", args.top_k, args.score_thr)
    panel_idx = top_idx
    if args.candidate_indices:
        panel_idx = [int(x) for x in args.candidate_indices.split(",") if x.strip()]
    panels_ok = save_candidate_panels(pts, prefix + "_candidate_panels.png", panel_idx, args.score_thr)
    risks = risk_arrays(pts)
    write_risk_csv(prefix + "_risks.csv", risks, top_idx, pts)
    summary = {
        "sample_index": idx,
        "folder": info.get("folder"),
        "frame_idx": info.get("frame_idx"),
        "selected_candidate": selected_idx,
        "selected_candidate_meta": stored_candidate_meta(pts, selected_idx),
        "selected_meta_action": str(pts.get("cf_selected_meta_action", "")),
        "top_candidates": [{"idx": int(i), "meta": stored_candidate_meta(pts, int(i)), "total_risk": float(risks["total"][i])} for i in top_idx],
        "response_summary": response,
        "camera_montage_saved": camera_ok,
        "candidate_panels_saved": panels_ok,
        "has_counterfactual_scenes": "cf_counterfactual_scenes" in pts,
    }
    with open(prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main()
