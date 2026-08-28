#!/usr/bin/env python
"""Visualize only the counterfactual interaction graph for one frame.

The figure intentionally omits per-agent text labels by default. Vehicles are
drawn as rectangles, pedestrians as circles, ego-agent edges are relevance
coded, and nearby selected agents are connected by thin local edges.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from mmcv.fileio.io import load

from tools.visualize_cf_paper_figure import (
    resolve_case_index,
)
from tools.visualize_counterfactual_case_study import (
    agent_futures_from_result,
    candidate_ego_for_plot,
    draw_agent_rect,
    get_pts_bbox,
    load_infos,
    map_polylines,
    result_items,
    scene_bounds,
    to_numpy,
)


VEHICLE_LABELS = {0, 1, 2, 3}  # car, van, truck, bicycle
PEDESTRIAN_LABELS = {7}
DETECTION_CLASS_NAMES = (
    "car",
    "van",
    "truck",
    "bicycle",
    "traffic_sign",
    "traffic_cone",
    "traffic_light",
    "pedestrian",
    "others",
)


def is_vehicle_label(label):
    return int(label) in VEHICLE_LABELS


def is_pedestrian_label(label):
    return int(label) in PEDESTRIAN_LABELS


def is_dynamic_agent_label(label):
    return is_vehicle_label(label) or is_pedestrian_label(label)


def draw_dynamic_agent_shape(ax, box, label, color, linewidth=2.0, alpha=0.95):
    if is_pedestrian_label(label):
        ax.add_patch(
            Circle(
                (float(box[0]), float(box[1])),
                0.85,
                fill=False,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=6,
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


def relevance_for_candidate(pts, candidate_idx, num_agents):
    rels = pts.get("cf_interaction_relevance", [])
    if rels and candidate_idx < len(rels):
        rel = np.squeeze(to_numpy(rels[candidate_idx])).astype(np.float32)
        return rel[:num_agents]
    return np.zeros(num_agents, dtype=np.float32)


def select_dynamic_agents(
    pts,
    boxes,
    scores,
    labels,
    candidate_idx,
    agent_type="all",
    top_k=8,
    score_thr=0.25,
    filter_by_score=False,
):
    """Filter dynamic classes first, then rank by candidate relevance.

    MindDrive B2D detection labels use:
      0 car, 1 van, 2 truck, 3 bicycle,
      4 traffic_sign, 5 traffic_cone, 6 traffic_light,
      7 pedestrian, 8 others.
    """
    rel = relevance_for_candidate(pts, candidate_idx, len(boxes))
    labels = np.asarray(labels if labels is not None else np.full(len(boxes), -1), dtype=np.int64)
    if filter_by_score:
        scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)

    keep = []
    for i in range(len(boxes)):
        if filter_by_score and scores[i] < score_thr:
            continue
        if agent_type == "vehicle" and not is_vehicle_label(labels[i]):
            continue
        if agent_type == "pedestrian" and not is_pedestrian_label(labels[i]):
            continue
        if agent_type == "all" and not is_dynamic_agent_label(labels[i]):
            continue
        keep.append(i)
    keep = np.asarray(keep, dtype=np.int64)
    if len(keep) == 0:
        return keep, rel
    ranked = keep[np.argsort(-rel[keep])]
    return ranked[:top_k], rel


def selected_candidate_index(pts, candidate_idx=None):
    if candidate_idx is not None:
        return int(candidate_idx)
    risks = pts.get("cf_risk_scores", {})
    total = risks.get("total") if isinstance(risks, dict) else None
    if total is None:
        return int(pts.get("cf_selected_candidate_idx", 0) or 0)
    total = np.squeeze(to_numpy(total)).astype(np.float32)
    return int(np.nanargmin(total)) if len(total) else 0


def draw_graph(
    ax,
    pts,
    boxes,
    agent_trajs,
    scores,
    labels,
    candidate_idx,
    score_thr=0.25,
    top_k=12,
    show_text=False,
    show_map=True,
    agent_type="all",
    filter_by_score=False,
):
    selected_agents, rel = select_dynamic_agents(
        pts,
        boxes,
        scores,
        labels,
        candidate_idx,
        agent_type=agent_type,
        top_k=top_k,
        score_thr=score_thr,
        filter_by_score=filter_by_score,
    )

    if show_map:
        for line in map_polylines(pts)[:160]:
            if len(line) > 1:
                ax.plot(line[:, 0], line[:, 1], color="0.86", linewidth=0.55, alpha=0.75, zorder=0)

    # Ego node.
    ax.scatter([0], [0], c="#e41a1c", marker="*", s=210, edgecolor="white", linewidth=0.7, zorder=7)

    # Ego-agent relevance edges and agent nodes.
    for i in selected_agents:
        relevance = float(np.clip(rel[i], 0.0, 1.0))
        color = plt.cm.magma(relevance)
        linewidth = 0.8 + 5.2 * relevance
        ax.plot(
            [0, boxes[i, 0]],
            [0, boxes[i, 1]],
            color=color,
            linewidth=linewidth,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )
        if agent_trajs is not None and i < len(agent_trajs):
            ax.plot(
                agent_trajs[i, :, 0],
                agent_trajs[i, :, 1],
                color=color,
                linewidth=1.5,
                alpha=0.60,
                marker="o",
                markersize=2.2,
                zorder=3,
            )
        draw_dynamic_agent_shape(
            ax,
            boxes[i],
            labels[i] if labels is not None and i < len(labels) else -1,
            color=color,
            linewidth=2.0,
            alpha=0.95,
        )
        if show_text:
            ax.text(
                boxes[i, 0],
                boxes[i, 1],
                "{} {:.2f}".format(i, relevance),
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
                zorder=8,
            )

    # Local agent-agent edges: nearby selected nodes only.
    for pos, i in enumerate(selected_agents):
        for j in selected_agents[pos + 1 :]:
            dist = float(np.linalg.norm(boxes[i, :2] - boxes[j, :2]))
            if dist > 12.0:
                continue
            alpha = max(0.12, 1.0 - dist / 12.0)
            ax.plot(
                [boxes[i, 0], boxes[j, 0]],
                [boxes[i, 1], boxes[j, 1]],
                color="#377eb8",
                linewidth=1.0,
                alpha=alpha,
                zorder=1,
            )

    ego = candidate_ego_for_plot(pts, candidate_idx, future_steps=agent_trajs.shape[1] if agent_trajs is not None else 6)
    if ego is not None:
        ax.plot(ego[:, 0], ego[:, 1], color="#e41a1c", linewidth=2.6, alpha=0.95, zorder=5)
        ax.scatter(ego[-1, 0], ego[-1, 1], color="#e41a1c", marker="X", s=70, edgecolor="white", linewidth=0.6, zorder=6)

    ego_refs = [ego] if ego is not None else []
    lo, hi = scene_bounds(
        boxes[selected_agents] if len(selected_agents) else boxes[:1],
        agent_trajs[selected_agents] if agent_trajs is not None and len(selected_agents) else None,
        ego_refs,
    )
    pad = np.array([4.0, 4.0], dtype=np.float32)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.14)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Lateral / x", fontsize=12)
    ax.set_ylabel("Forward / y", fontsize=12)

    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=plt.cm.magma)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("relevance", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--scene-folder", default=None)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--candidate-idx", type=int, default=None, help="Default: selected lowest-risk candidate.")
    parser.add_argument("--score-thr", type=float, default=0.25, help="Only used when --filter-by-score is set.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--filter-by-score",
        action="store_true",
        help="Also filter dynamic agents by detection score. Default: disabled.",
    )
    parser.add_argument("--show-text", action="store_true", help="Show compact agent id/relevance labels.")
    parser.add_argument("--hide-map", action="store_true")
    parser.add_argument(
        "--agent-type",
        choices=["all", "vehicle", "pedestrian"],
        default="all",
        help="Filter displayed agent nodes. Default: all.",
    )
    parser.add_argument("--figsize", default="5,8", help="Figure size as width,height.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--print-label-stats", action="store_true", help="Print label id/name counts for the selected frame.")
    args = parser.parse_args()

    items = result_items(load(args.results))
    infos = load_infos(args.infos)
    idx = resolve_case_index(infos, args.index, args.scene_folder, args.scenario_name, args.frame_idx)
    pts = get_pts_bbox(items[idx])
    boxes, agent_trajs, scores, labels = agent_futures_from_result(pts)
    if boxes is None:
        raise ValueError("Selected result does not contain boxes_3d.")
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    # print(scores)
    labels = np.asarray(labels if labels is not None else np.zeros(len(boxes)), dtype=np.int64)
    candidate_idx = selected_candidate_index(pts, args.candidate_idx)
    if args.print_label_stats:
        valid = np.ones_like(scores, dtype=bool)
        if args.filter_by_score:
            valid = scores >= args.score_thr
        unique, counts = np.unique(labels[valid], return_counts=True)
        header = "label counts"
        if args.filter_by_score:
            header += " with score_thr >= {}".format(args.score_thr)
        print(header + ":")
        for label, count in zip(unique.tolist(), counts.tolist()):
            name = DETECTION_CLASS_NAMES[label] if 0 <= label < len(DETECTION_CLASS_NAMES) else "unknown"
            print("  {} ({}) : {}".format(label, name, count))

    width, height = [float(x) for x in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(5, 8))
    draw_graph(
        ax,
        pts,
        boxes,
        agent_trajs,
        scores,
        labels,
        candidate_idx,
        score_thr=args.score_thr,
        top_k=args.top_k,
        show_text=args.show_text,
        show_map=not args.hide_map,
        agent_type=args.agent_type,
        filter_by_score=args.filter_by_score,
    )
    info = infos[idx]
    # ax.set_title(
    #     "{} | frame {} | candidate #{}".format(
    #         os.path.basename(str(info.get("folder", ""))),
    #         info.get("frame_idx"),
    #         candidate_idx,
    #     ),
    #     fontsize=11,
    #     pad=8,
    # )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
