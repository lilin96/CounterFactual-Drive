#!/usr/bin/env python
"""Visualize only the seven candidate-conditioned BEV scenes.

This is a larger standalone version of the second row in
``visualize_cf_paper_figure.py``. The visual style follows the BEV outputs in
``vis_tools/visualize.py``: dark background, high-contrast boxes/trajectories,
and larger text labels.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from mmcv.fileio.io import load

from mmcv.models.counterfactual.meta_action_labels import PATH_META_ACTIONS, SPEED_META_ACTIONS
from tools.visualize_counterfactual_case_study import (
    agent_futures_from_result,
    candidate_ego_for_plot,
    draw_agent_rect,
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


def short_meta_action(text, max_len=30):
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


def candidate_indices_from_args(pts, text, count=7):
    risks = risk_arrays(pts)
    total = risks.get("total")
    if text:
        return [int(x) for x in text.split(",") if x.strip()]
    if total is None:
        return list(range(count))
    return np.argsort(total)[: min(count, len(total))].astype(int).tolist()


def class_is_pedestrian(label):
    # MindDrive B2D detection labels:
    # 0 car, 1 van, 2 truck, 3 bicycle, 4 traffic_sign, 5 traffic_cone,
    # 6 traffic_light, 7 pedestrian, 8 others.
    return int(label) == 7


def draw_agent_shape(ax, box, label=-1, color="#ff8800", linewidth=2.0, alpha=0.95):
    if class_is_pedestrian(label):
        ax.add_patch(
            Circle(
                (float(box[0]), float(box[1])),
                0.95,
                fill=False,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=5,
            )
        )
        return
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


def select_candidate_agents(
    pts,
    boxes,
    scores,
    idx,
    agent_trajs,
    realized,
    ego,
    top_k=10,
    score_thr=0.25,
    relevance_thr=0.5,
    change_thr=0.15,
    max_context=5,
):
    """Split agents into affected, contextual, and low-confidence CF groups.

    Only valid detections can be presented as affected agents.  Motion of an
    otherwise unrelated actor is deliberately excluded from the affected-agent
    ranking: it is useful for scene context, but is not evidence of a
    candidate-conditioned response.
    """
    rels = pts.get("cf_interaction_relevance", [])
    if rels and idx < len(rels):
        rel = np.squeeze(to_numpy(rels[idx])).astype(np.float32)[: len(boxes)]
    else:
        rel = np.zeros(len(boxes), dtype=np.float32)
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    valid = scores >= score_thr
    change = np.zeros(len(boxes), dtype=np.float32)
    if realized is not None and agent_trajs is not None:
        n = min(len(realized), len(agent_trajs), len(boxes))
        if n:
            change[:n] = np.linalg.norm(
                realized[:n, :, :2] - agent_trajs[:n, :, :2], axis=-1
            ).max(axis=-1)

    cf_affected = (rel >= relevance_thr) | (change >= change_thr)
    affected_ids = np.flatnonzero(valid & cf_affected)
    affected_rank = rel[affected_ids] + 0.5 * np.clip(change[affected_ids] / 3.0, 0.0, 1.0)
    affected_ids = affected_ids[np.argsort(-affected_rank)][:top_k]

    context_ids = np.flatnonzero(valid & ~cf_affected)
    if len(context_ids) and max_context > 0:
        if ego is not None and np.asarray(ego).size:
            ego_xy = np.asarray(ego, dtype=np.float32).reshape(-1, 2)
            distance = np.linalg.norm(
                boxes[context_ids, None, :2] - ego_xy[None, :, :], axis=-1
            ).min(axis=1)
            # Detection confidence breaks ties between similarly close actors.
            order = np.lexsort((-scores[context_ids], distance))
        else:
            order = np.argsort(-scores[context_ids])
        context_ids = context_ids[order[:max_context]]
    else:
        context_ids = np.asarray([], dtype=np.int64)

    low_conf_cf_ids = np.flatnonzero(~valid & cf_affected)
    low_conf_rank = rel[low_conf_cf_ids] + 0.5 * np.clip(change[low_conf_cf_ids] / 3.0, 0.0, 1.0)
    low_conf_cf_ids = low_conf_cf_ids[np.argsort(-low_conf_rank)][:top_k]
    return affected_ids, context_ids, low_conf_cf_ids, rel, change


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


def style_bev_ax(ax):
    ax.set_facecolor("black")
    ax.grid(True, color="white", alpha=0.10, linewidth=0.5)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.35)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def draw_candidate_panel(
    ax,
    pts,
    idx,
    boxes,
    agent_trajs,
    scores,
    labels,
    bounds,
    score_thr,
    max_agents,
    show_text=False,
    relevance_thr=0.5,
    change_thr=0.15,
    max_context=5,
    show_low_confidence_cf=False,
):
    realized, source = candidate_scene_agents(pts, idx, agent_trajs, boxes)
    ego = candidate_ego_for_plot(pts, idx, future_steps=agent_trajs.shape[1] if agent_trajs is not None else 6)
    affected_agents, context_agents, low_conf_cf_agents, rel, change = select_candidate_agents(
        pts,
        boxes,
        scores,
        idx,
        agent_trajs,
        realized,
        ego,
        top_k=max_agents,
        score_thr=score_thr,
        relevance_thr=relevance_thr,
        change_thr=change_thr,
        max_context=max_context,
    )
    speed_labels, path_labels, _, _ = response_labels_for_candidate(pts.get("cf_response_meta_actions", []), idx, len(boxes))

    style_bev_ax(ax)
    for line in map_polylines(pts)[:120]:
        if len(line) > 1:
            ax.plot(line[:, 0], line[:, 1], color="white", linewidth=0.7, alpha=0.22, zorder=1)

    # Context actors establish scene geometry without implying CF relevance.
    for i in context_agents:
        draw_agent_shape(ax, boxes[i], labels[i] if labels is not None and i < len(labels) else -1, color="#8c8c8c", linewidth=1.0, alpha=0.55)
        if agent_trajs is not None:
            ax.plot(agent_trajs[i, :, 0], agent_trajs[i, :, 1], "--", color="#a0a0a0", linewidth=0.8, alpha=0.35, zorder=2)

    annotation_rows = []
    for i in affected_agents:
        color = plt.cm.plasma(float(np.clip(rel[i], 0.0, 1.0)))
        draw_agent_shape(ax, boxes[i], labels[i] if labels is not None and i < len(labels) else -1, color=color, linewidth=1.8)
        if agent_trajs is not None:
            ax.plot(agent_trajs[i, :, 0], agent_trajs[i, :, 1], "--", color="white", linewidth=1.1, alpha=0.45, zorder=2)
        if realized is not None and i < len(realized):
            ax.plot(
                realized[i, :, 0],
                realized[i, :, 1],
                color=color,
                linewidth=2.8,
                alpha=0.98,
                marker="o",
                markersize=4.5,
                zorder=4,
            )
            ax.scatter(realized[i, 0, 0], realized[i, 0, 1], marker="s", s=34, color=color, edgecolor="white", linewidth=0.5, zorder=5)
            ax.scatter(realized[i, -1, 0], realized[i, -1, 1], marker="X", s=54, color=color, edgecolor="white", linewidth=0.5, zorder=5)
            if np.linalg.norm(realized[i, -1, :2] - realized[i, 0, :2]) > 0.75:
                ax.annotate(
                    "",
                    xy=(realized[i, -1, 0], realized[i, -1, 1]),
                    xytext=(realized[i, 0, 0], realized[i, 0, 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.1, alpha=0.9),
                    zorder=5,
                )

        if show_text:
            response = ""
            if speed_labels is not None and path_labels is not None and i < len(speed_labels):
                response = "{} | {}".format(
                    label_name(speed_labels[i], SPEED_META_ACTIONS)[:8],
                    label_name(path_labels[i], PATH_META_ACTIONS)[:8],
                )
            annotation_rows.append(
                (int(i), color, "A{}  s={:.2f}  ρ={:.2f}  Δ={:.2f}m  {}".format(
                    int(i), float(scores[i]), float(rel[i]), float(change[i]), response
                ))
            )
            marker_xy = realized[i, -1, :2] if realized is not None and i < len(realized) else boxes[i, :2]
            ax.annotate(
                "A{}".format(int(i)),
                xy=(float(marker_xy[0]), float(marker_xy[1])),
                xytext=(4, 4),
                textcoords="offset points",
                color="white",
                fontsize=7.5,
                weight="bold",
                bbox=dict(facecolor="black", edgecolor=color, alpha=0.72, boxstyle="round,pad=0.12"),
                zorder=8,
            )

    # Put detailed labels into fixed, vertically separated rows.  Keeping the
    # long text out of data coordinates prevents nearby actors and trajectories
    # from producing unreadable overlapping annotation boxes.
    if show_text:
        for row, (_, color, text) in enumerate(annotation_rows[:max_agents]):
            ax.text(
                0.02,
                0.98 - row * 0.075,
                text,
                transform=ax.transAxes,
                color="white",
                fontsize=6.8,
                ha="left",
                va="top",
                bbox=dict(facecolor="black", edgecolor=color, alpha=0.78, boxstyle="round,pad=0.15"),
                zorder=10,
            )

    if show_low_confidence_cf:
        for i in low_conf_cf_agents:
            color = "#ff3355"
            draw_agent_shape(
                ax,
                boxes[i],
                labels[i] if labels is not None and i < len(labels) else -1,
                color=color,
                linewidth=1.2,
                alpha=0.65,
            )
            if agent_trajs is not None:
                ax.plot(agent_trajs[i, :, 0], agent_trajs[i, :, 1], "--", color="#aaaaaa", linewidth=0.8, alpha=0.35, zorder=2)
            if realized is not None and i < len(realized):
                ax.plot(realized[i, :, 0], realized[i, :, 1], ":", color=color, linewidth=1.8, alpha=0.8, zorder=3)
            ax.text(
                boxes[i, 0],
                boxes[i, 1],
                "Q{} low-conf\ns={:.2f} ρ={:.2f} Δ={:.2f}m".format(
                    int(i), float(scores[i]), float(rel[i]), float(change[i])
                ),
                color=color,
                fontsize=7.5,
                ha="center",
                va="bottom",
                zorder=8,
            )

    if ego is not None:
        ax.plot(ego[:, 0], ego[:, 1], color="#00ff66", linewidth=3.0, marker="o", markersize=4.0, label="ego candidate", zorder=6)
        ax.scatter(ego[-1, 0], ego[-1, 1], color="#00ff66", marker="*", s=85, edgecolor="white", linewidth=0.5, zorder=7)

    ax.scatter([0], [0], marker="*", c="#00ff66", s=95, edgecolor="white", linewidth=0.6, zorder=8)
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)

    total = risk_arrays(pts).get("total")
    risk_text = "raw risk {:.2f}".format(float(total[idx])) if total is not None and idx < len(total) else "risk N/A"
    ax.set_title(
        "#{} | affected {} | context {}\n{}{}".format(
            idx,
            len(affected_agents),
            len(context_agents),
            risk_text,
            " | low-conf CF {}".format(len(low_conf_cf_agents)) if low_conf_cf_agents.size else "",
        ),
        fontsize=10.5,
        pad=7,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--scene-folder", default=None)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--candidate-indices", default="0,1,2,3,4,5,6")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--max-agents", type=int, default=10)
    parser.add_argument("--max-context", type=int, default=5)
    parser.add_argument("--relevance-thr", type=float, default=0.5)
    parser.add_argument("--change-thr", type=float, default=0.15, help="Minimum max CF/base trajectory difference in metres.")
    parser.add_argument("--show-low-confidence-cf", action="store_true")
    parser.add_argument("--show-text", action="store_true", help="Show per-agent id/relevance/response labels.")
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
    candidate_ids = candidate_indices_from_args(pts, args.candidate_indices, count=7)[:7]

    ego_refs = [candidate_ego_for_plot(pts, i, future_steps=agent_trajs.shape[1]) for i in candidate_ids]
    lo, hi = scene_bounds(boxes, agent_trajs, ego_refs)
    bounds = (lo, hi)

    # fig, axes = plt.subplots(1, len(candidate_ids), figsize=(5.8 * len(candidate_ids), 6.3), squeeze=False)
    fig, axes = plt.subplots(1, len(candidate_ids), figsize=(16, 6.3), squeeze=False)
    fig.patch.set_facecolor("black")
    for ax, cand_idx in zip(axes.reshape(-1), candidate_ids):
        draw_candidate_panel(
            ax,
            pts,
            cand_idx,
            boxes,
            agent_trajs,
            scores,
            labels,
            bounds,
            args.score_thr,
            args.max_agents,
            show_text=args.show_text,
            relevance_thr=args.relevance_thr,
            change_thr=args.change_thr,
            max_context=args.max_context,
            show_low_confidence_cf=args.show_low_confidence_cf,
        )

    for ax in axes.reshape(-1)[len(candidate_ids) :]:
        ax.axis("off")
    title = "{} | frame {} | candidate-conditioned BEV scenes".format(
        os.path.basename(str(info.get("folder", ""))),
        info.get("frame_idx"),
    )
    # fig.suptitle(title, color="white", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
