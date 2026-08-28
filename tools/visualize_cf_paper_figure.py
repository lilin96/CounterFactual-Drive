#!/usr/bin/env python
"""Create a paper-style counterfactual candidate planning figure.

The layout is tailored for one MindDrive counterfactual sample:
  - top-left: six camera inputs;
  - top-middle: interaction graph with relevance-coded ego-agent edges;
  - middle-left column: seven candidate-conditioned BEV scenes;
  - middle-right column: radar risk chart for each candidate;
  - right column: final selected BEV and projected ego trajectory on CAM_FRONT.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle
from mmcv.fileio.io import load

try:
    from PIL import Image
except ImportError:
    Image = None

from mmcv.models.counterfactual.meta_action_labels import PATH_META_ACTIONS, SPEED_META_ACTIONS
from tools.visualize_counterfactual_case_study import (
    CAM_ORDER,
    agent_futures_from_result,
    candidate_ego_for_plot,
    draw_agent_rect,
    draw_box,
    ego_traj,
    get_pts_bbox,
    label_name,
    load_infos,
    map_polylines,
    realize_agent_futures_np,
    response_labels_for_candidate,
    result_items,
    risk_arrays,
    scene_bounds,
    stored_candidate_meta,
    to_numpy,
)


RISK_KEYS = ("collision", "ttc", "interaction", "map_rule", "comfort", "progress")


def short_meta_action(text, max_len=22):
    aliases = {
        "maintain moderate speed": "moderate",
        "maintain slow speed": "slow",
        "maintain fast speed": "fast",
        "slow down rapidly": "rapid slow",
        "change lane left": "lane left",
        "change lane right": "lane right",
    }
    text = str(text)
    for src, dst in aliases.items():
        text = text.replace(src, dst)
    return text if len(text) <= max_len else text[: max_len - 1] + "."


def resolve_case_index(infos, index=None, scene_folder=None, scenario_name=None, frame_idx=None):
    if index is not None:
        return index
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
        raise ValueError("Matched {} frames. Pass --frame-idx. First matches: {}".format(len(matches), examples))
    return matches[0]


def resolve_image_path(path, data_root):
    if not path:
        return None
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    candidates = [os.path.join(data_root, path), os.path.join("data/bench2drive", path), path]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def candidate_indices_from_args(pts, text, count=7):
    risks = risk_arrays(pts)
    total = risks.get("total")
    if text:
        return [int(x) for x in text.split(",") if x.strip()]
    if total is None:
        return list(range(count))
    return np.argsort(total)[: min(count, len(total))].astype(int).tolist()


def class_is_pedestrian(label, name=None):
    if name is not None and "ped" in str(name).lower() or name is not None and "walker" in str(name).lower():
        return True
    # B2D configs usually map pedestrian to label 1. Keep this as a visual-only heuristic.
    return int(label) == 1


def draw_agent_shape(ax, box, label=-1, rel=0.0, color=None, linewidth=1.5, alpha=0.9):
    color = color or plt.cm.magma(float(np.clip(rel, 0.0, 1.0)))
    if class_is_pedestrian(label):
        radius = 0.85
        ax.add_patch(Circle((float(box[0]), float(box[1])), radius, fill=False, color=color, linewidth=linewidth, alpha=alpha))
    else:
        draw_agent_rect(
            ax,
            box[:2],
            length=float(box[3]) if len(box) > 3 else 4.2,
            width=float(box[4]) if len(box) > 4 else 1.8,
            yaw=float(box[6]) if len(box) > 6 else 0.0,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


def draw_camera_montage(ax, info, data_root):
    ax.axis("off")
    if Image is None:
        ax.text(0.5, 0.5, "PIL unavailable", ha="center", va="center")
        return
    sensors = info.get("sensors", {})
    thumbs = []
    titles = []
    for cam in CAM_ORDER:
        cam_info = sensors.get(cam, {})
        path = resolve_image_path(cam_info.get("data_path"), data_root)
        if path is None:
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((320, 180))
        thumbs.append(np.asarray(img))
        titles.append(cam.replace("CAM_", ""))
    if not thumbs:
        ax.text(0.5, 0.5, "camera images unavailable", ha="center", va="center")
        return
    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=ax.get_subplotspec(), wspace=0.02, hspace=0.12)
    fig = ax.figure
    ax.remove()
    for i in range(6):
        sub_ax = fig.add_subplot(sub[i // 3, i % 3])
        sub_ax.axis("off")
        if i < len(thumbs):
            sub_ax.imshow(thumbs[i])
            sub_ax.set_title(titles[i], fontsize=8, pad=1)


def select_relevant_agents(pts, boxes, scores, candidate_idx, top_k=8, score_thr=0.25):
    rels = pts.get("cf_interaction_relevance", [])
    if rels and candidate_idx < len(rels):
        rel = np.squeeze(to_numpy(rels[candidate_idx])).astype(np.float32)[: len(boxes)]
    else:
        rel = np.zeros(len(boxes), dtype=np.float32)
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    valid = scores >= score_thr
    order = [i for i in np.argsort(-rel) if valid[i]][:top_k]
    return np.asarray(order, dtype=np.int64), rel


def draw_interaction_graph(ax, pts, boxes, agent_trajs, scores, labels, selected_idx, score_thr=0.25, top_k=10):
    selected_agents, rel = select_relevant_agents(pts, boxes, scores, selected_idx, top_k=top_k, score_thr=score_thr)
    ax.set_title("Interaction graph: nodes + relevance edges", fontsize=10)
    for line in map_polylines(pts)[:120]:
        if len(line) > 1:
            ax.plot(line[:, 0], line[:, 1], color="0.88", linewidth=0.45, alpha=0.7)
    ax.scatter([0], [0], c="tab:red", marker="*", s=95, zorder=5)
    ax.text(0, 0, "ego", fontsize=7, ha="left", va="bottom")
    for i in selected_agents:
        color = plt.cm.magma(float(np.clip(rel[i], 0.0, 1.0)))
        lw = 0.4 + 4.0 * float(np.clip(rel[i], 0.0, 1.0))
        ax.plot([0, boxes[i, 0]], [0, boxes[i, 1]], color=color, linewidth=lw, alpha=0.65, zorder=1)
        draw_agent_shape(ax, boxes[i], labels[i] if labels is not None and i < len(labels) else -1, rel[i], color=color)
        ax.text(boxes[i, 0], boxes[i, 1], "{}\nρ={:.2f}".format(i, rel[i]), fontsize=6.5, ha="center", va="center")
        if agent_trajs is not None and i < len(agent_trajs):
            ax.plot(agent_trajs[i, :, 0], agent_trajs[i, :, 1], color=color, linewidth=0.9, alpha=0.55)
    # Local agent-agent edges for nearby relevant nodes.
    for a_pos, i in enumerate(selected_agents):
        for j in selected_agents[a_pos + 1 :]:
            dist = float(np.linalg.norm(boxes[i, :2] - boxes[j, :2]))
            if dist > 12.0:
                continue
            edge_alpha = max(0.12, 1.0 - dist / 12.0)
            ax.plot([boxes[i, 0], boxes[j, 0]], [boxes[i, 1], boxes[j, 1]], color="tab:blue", linewidth=0.7, alpha=edge_alpha)
    ego_refs = [candidate_ego_for_plot(pts, selected_idx, future_steps=agent_trajs.shape[1] if agent_trajs is not None else 6)]
    lo, hi = scene_bounds(boxes[selected_agents] if len(selected_agents) else boxes[:1], agent_trajs[selected_agents] if agent_trajs is not None and len(selected_agents) else None, ego_refs)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)


def candidate_scene_agents(pts, idx, fallback, boxes=None):
    scenes = pts.get("cf_counterfactual_scenes")
    if scenes and idx < len(scenes):
        fut = to_numpy(scenes[idx].get("agent_futures"))
        if fut is not None:
            fut = np.squeeze(fut).astype(np.float32)
            if fut.ndim == 3:
                return fut, "saved"
    responses = pts.get("cf_response_meta_actions", [])
    rels = pts.get("cf_interaction_relevance", [])
    if boxes is not None and fallback is not None and responses and rels and idx < len(responses) and idx < len(rels):
        speed_labels, path_labels, _, _ = response_labels_for_candidate(responses, idx, len(boxes))
        rel = np.squeeze(to_numpy(rels[idx])).astype(np.float32)[: len(boxes)]
        if speed_labels is not None and path_labels is not None:
            return realize_agent_futures_np(fallback, boxes, speed_labels, path_labels, rel), "reconstructed"
    return fallback, "base"


def draw_candidate_scene(ax, pts, idx, boxes, agent_trajs, scores, labels, bounds, score_thr=0.25, max_agents=8):
    realized, realized_source = candidate_scene_agents(pts, idx, agent_trajs, boxes)
    _, rel = select_relevant_agents(pts, boxes, scores, idx, top_k=max_agents, score_thr=score_thr)
    score_arr = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    valid = score_arr >= score_thr
    motion = np.zeros(len(boxes), dtype=np.float32)
    change = np.zeros(len(boxes), dtype=np.float32)
    if realized is not None and len(realized):
        motion[: len(realized)] = np.linalg.norm(realized[:, -1, :2] - realized[:, 0, :2], axis=-1)
        if agent_trajs is not None:
            n = min(len(realized), len(agent_trajs))
            change[:n] = np.linalg.norm(realized[:n, :, :2] - agent_trajs[:n, :, :2], axis=-1).max(axis=-1)
    rank_score = rel + 0.15 * np.clip(motion / 12.0, 0.0, 1.0) + 0.25 * np.clip(change / 3.0, 0.0, 1.0)
    rel_agents = np.asarray([i for i in np.argsort(-rank_score) if valid[i]][:max_agents], dtype=np.int64)
    speed_labels, path_labels, _, _ = response_labels_for_candidate(pts.get("cf_response_meta_actions", []), idx, len(boxes))
    for line in map_polylines(pts)[:80]:
        if len(line) > 1:
            ax.plot(line[:, 0], line[:, 1], color="0.9", linewidth=0.35, alpha=0.75)
    for i in rel_agents:
        color = plt.cm.magma(float(np.clip(rel[i], 0.0, 1.0)))
        draw_agent_shape(ax, boxes[i], labels[i] if labels is not None and i < len(labels) else -1, rel[i], color=color, linewidth=0.9)
        if agent_trajs is not None:
            ax.plot(agent_trajs[i, :, 0], agent_trajs[i, :, 1], "--", color="0.58", linewidth=0.6, alpha=0.55)
        if realized is not None and i < len(realized):
            ax.plot(
                realized[i, :, 0],
                realized[i, :, 1],
                color=color,
                linewidth=2.1,
                alpha=0.94,
                marker="o",
                markersize=2.8,
                zorder=4,
            )
            ax.scatter(realized[i, 0, 0], realized[i, 0, 1], marker="s", s=15, color=color, edgecolor="black", linewidth=0.25, zorder=5)
            ax.scatter(realized[i, -1, 0], realized[i, -1, 1], marker="X", s=22, color=color, edgecolor="black", linewidth=0.25, zorder=5)
            if np.linalg.norm(realized[i, -1, :2] - realized[i, 0, :2]) > 0.75:
                ax.annotate(
                    "",
                    xy=(realized[i, -1, 0], realized[i, -1, 1]),
                    xytext=(realized[i, 0, 0], realized[i, 0, 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.85),
                    zorder=5,
                )
        label = "{}:{:.2f}".format(i, rel[i])
        if speed_labels is not None and path_labels is not None and i < len(speed_labels):
            label += " {}|{}".format(label_name(speed_labels[i], SPEED_META_ACTIONS)[:4], label_name(path_labels[i], PATH_META_ACTIONS)[:4])
        ax.text(boxes[i, 0], boxes[i, 1], label, fontsize=5.2)
    ego = candidate_ego_for_plot(pts, idx, future_steps=agent_trajs.shape[1] if agent_trajs is not None else 6)
    if ego is not None:
        ax.plot(ego[:, 0], ego[:, 1], color="tab:red", linewidth=2.0)
    ax.scatter([0], [0], marker="*", c="black", s=28, zorder=5)
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.12)
    ax.tick_params(labelsize=5, length=1)
    suffix = "" if realized_source == "saved" else " ({})".format(realized_source)
    ax.set_title("#{} {}{}".format(idx, short_meta_action(stored_candidate_meta(pts, idx), max_len=18), suffix), fontsize=6.0, pad=1.5)


def progress_score_for_candidate(pts, idx, future_steps=6):
    """Return candidate final-progress ratio, clipped to [0, 1].

    Progress is displayed as achieved forward endpoint / target forward
    endpoint, rather than min-max normalized risk. The target is MindDrive's
    original selected ego future, matching the base_ego_future used by the risk
    scorer when computing progress loss.
    """
    cand = candidate_ego_for_plot(pts, idx, future_steps=future_steps)
    base_source = pts.get("decision_expert_ego_fut_preds")
    if base_source is None:
        base_source = pts.get("ego_fut_preds")
    base = ego_traj(base_source)
    if cand is None or base is None or len(cand) == 0 or len(base) == 0:
        return None
    target = float(base[-1, 1])
    achieved = float(cand[-1, 1])
    if abs(target) < 1e-3:
        return 1.0 if abs(achieved) < 1e-3 else 0.0
    return float(np.clip(achieved / target, 0.0, 1.0))


def normalized_risk_values(risks, idx, keys, progress_score=None):
    values = []
    labels = []
    for key in keys:
        if key == "progress" and progress_score is not None:
            values.append(float(np.clip(progress_score, 0.0, 1.0)))
            labels.append("progress")
            continue
        if key not in risks or idx >= len(risks[key]):
            continue
        arr = np.asarray(risks[key], dtype=np.float32)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        denom = hi - lo
        normalized_risk = 0.0 if denom < 1e-6 else (float(arr[idx]) - lo) / denom
        # Radar charts show normalized goodness scores: higher is better.
        # The model still uses raw risk costs where lower is better.
        val = 1.0 - normalized_risk
        values.append(float(np.clip(val, 0.0, 1.0)))
        labels.append(key)
    if not values:
        values = [0.0]
        labels = ["risk"]
    return labels, values


def draw_radar(ax, pts, risks, idx, selected_idx, future_steps=6):
    labels, values = normalized_risk_values(
        risks,
        idx,
        RISK_KEYS,
        progress_score=progress_score_for_candidate(pts, idx, future_steps=future_steps),
    )
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]
    color = "tab:red" if idx == selected_idx else "tab:blue"
    ax.plot(angles, values, color=color, linewidth=1.3)
    ax.fill(angles, values, color=color, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=4.8)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels([".5", "1"], fontsize=5)
    ax.set_ylim(0, 1)
    total = risks.get("total")
    title = "#{}".format(idx)
    if total is not None and idx < len(total):
        title += " raw {:.2f}".format(float(total[idx]))
    ax.set_title(title, fontsize=6.2, pad=1.5)


def draw_final_bev(ax, pts, info, boxes, agent_trajs, scores, labels, selected_idx, score_thr=0.25):
    rel_agents, rel = select_relevant_agents(pts, boxes, scores, selected_idx, top_k=12, score_thr=score_thr)
    selected_ego = ego_traj(pts.get("cf_selected_ego_future"))
    base_source = pts.get("decision_expert_ego_fut_preds")
    if base_source is None:
        base_source = pts.get("ego_fut_preds")
    base_ego = ego_traj(base_source)
    realized, _ = candidate_scene_agents(pts, selected_idx, agent_trajs, boxes)
    for line in map_polylines(pts)[:200]:
        if len(line) > 1:
            ax.plot(line[:, 0], line[:, 1], color="0.82", linewidth=0.55, alpha=0.8)
    for i in rel_agents:
        color = plt.cm.magma(float(np.clip(rel[i], 0.0, 1.0)))
        draw_agent_shape(ax, boxes[i], labels[i] if labels is not None and i < len(labels) else -1, rel[i], color=color, linewidth=1.2)
        if realized is not None and i < len(realized):
            ax.plot(realized[i, :, 0], realized[i, :, 1], color=color, linewidth=1.8, alpha=0.9, marker="o", markersize=2.6)
            ax.scatter(realized[i, -1, 0], realized[i, -1, 1], marker="X", s=28, color=color, edgecolor="black", linewidth=0.3)
    if base_ego is not None:
        ax.plot(base_ego[:, 0], base_ego[:, 1], "--", color="tab:blue", linewidth=1.6, label="base")
    if selected_ego is not None:
        ax.plot(selected_ego[:, 0], selected_ego[:, 1], color="tab:red", linewidth=2.6, label="selected")
    ax.scatter([0], [0], marker="*", c="black", s=45)
    lo, hi = scene_bounds(boxes[rel_agents] if len(rel_agents) else boxes[:1], realized[rel_agents] if realized is not None and len(rel_agents) else agent_trajs, [selected_ego, base_ego])
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15)
    ax.set_title("Final selected BEV", fontsize=10)
    ax.legend(fontsize=6, loc="upper right")


def ego_xy_to_camera_uv(ego_xy, cam_info):
    if ego_xy is None or cam_info is None:
        return None
    intrinsic = np.asarray(cam_info.get("intrinsic"), dtype=np.float32)
    cam2ego = np.asarray(cam_info.get("cam2ego"), dtype=np.float32)
    if intrinsic.shape != (3, 3) or cam2ego.shape != (4, 4):
        return None
    ego2cam = np.linalg.inv(cam2ego)
    # MindDrive planning xy is plotted as lateral, longitudinal. Convert it to
    # an approximate ego 3D point: x_forward=longitudinal, y_left=lateral.
    pts_ego = np.ones((len(ego_xy), 4), dtype=np.float32)
    pts_ego[:, 0] = ego_xy[:, 1]
    pts_ego[:, 1] = -ego_xy[:, 0]
    pts_ego[:, 2] = 0.0
    pts_cam = (ego2cam @ pts_ego.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > 0.1
    if not valid.any():
        return None
    uvw = (intrinsic @ pts_cam[valid].T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv


def draw_front_projection(ax, info, pts, data_root):
    ax.axis("off")
    cam_info = info.get("sensors", {}).get("CAM_FRONT", {})
    path = resolve_image_path(cam_info.get("data_path"), data_root)
    if Image is None or path is None:
        ax.text(0.5, 0.5, "CAM_FRONT unavailable", ha="center", va="center")
        return
    img = Image.open(path).convert("RGB")
    ax.imshow(img)
    selected_ego = ego_traj(pts.get("cf_selected_ego_future"))
    base_source = pts.get("decision_expert_ego_fut_preds")
    if base_source is None:
        base_source = pts.get("ego_fut_preds")
    base_ego = ego_traj(base_source)
    for traj, color, label in ((base_ego, "tab:blue", "base"), (selected_ego, "tab:red", "selected")):
        uv = ego_xy_to_camera_uv(traj, cam_info)
        if uv is None:
            continue
        h, w = np.asarray(img).shape[:2]
        mask = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv = uv[mask]
        if len(uv) == 0:
            continue
        ax.plot(uv[:, 0], uv[:, 1], color=color, linewidth=3.0, marker="o", markersize=3, label=label)
    ax.set_title("Selected trajectory on CAM_FRONT", fontsize=10)
    ax.legend(fontsize=6, loc="lower left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--scene-folder", default=None)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--candidate-indices", default=None, help="Comma-separated ids. Defaults to 7 lowest-risk candidates.")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--data-root", default="data/bench2drive")
    args = parser.parse_args()

    items = result_items(load(args.results))
    infos = load_infos(args.infos)
    idx = resolve_case_index(infos, args.index, args.scene_folder, args.scenario_name, args.frame_idx)
    pts = get_pts_bbox(items[idx])
    info = infos[idx]

    boxes, agent_trajs, scores, labels = agent_futures_from_result(pts)
    if boxes is None or agent_trajs is None:
        raise ValueError("Selected result does not contain boxes_3d/trajs_3d.")
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    labels = np.asarray(labels if labels is not None else np.zeros(len(boxes)), dtype=np.int64)
    risks = risk_arrays(pts)
    total = risks.get("total")
    selected_idx = int(np.nanargmin(total)) if total is not None and len(total) else 0
    candidate_ids = candidate_indices_from_args(pts, args.candidate_indices, count=7)[:7]

    ego_refs = [candidate_ego_for_plot(pts, i, future_steps=agent_trajs.shape[1]) for i in candidate_ids]
    lo, hi = scene_bounds(boxes, agent_trajs, ego_refs)
    bounds = (lo, hi)

    fig = plt.figure(figsize=(30, 17))
    gs = GridSpec(
        4,
        7,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        height_ratios=[1.25, 1.15, 1.15, 1.35],
        wspace=0.2,
        hspace=0.34,
    )

    ax_cam = fig.add_subplot(gs[0, 0:3])
    draw_camera_montage(ax_cam, info, args.data_root)
    ax_graph = fig.add_subplot(gs[0, 3:7])
    draw_interaction_graph(ax_graph, pts, boxes, agent_trajs, scores, labels, selected_idx, args.score_thr)

    for col, cand_idx in enumerate(candidate_ids):
        ax_scene = fig.add_subplot(gs[1, col])
        draw_candidate_scene(ax_scene, pts, cand_idx, boxes, agent_trajs, scores, labels, bounds, args.score_thr)
        ax_radar = fig.add_subplot(gs[2, col], projection="polar")
        draw_radar(ax_radar, pts, risks, cand_idx, selected_idx, future_steps=agent_trajs.shape[1])

    ax_final = fig.add_subplot(gs[3, 0:4])
    draw_final_bev(ax_final, pts, info, boxes, agent_trajs, scores, labels, selected_idx, args.score_thr)
    ax_front = fig.add_subplot(gs[3, 4:7])
    draw_front_projection(ax_front, info, pts, args.data_root)

    title = "{} | frame {} | selected #{} {}".format(
        os.path.basename(str(info.get("folder", ""))),
        info.get("frame_idx"),
        selected_idx,
        short_meta_action(stored_candidate_meta(pts, selected_idx), max_len=32),
    )
    fig.suptitle(title, fontsize=15, y=0.992)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
