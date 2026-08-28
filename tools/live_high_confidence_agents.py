#!/usr/bin/env python
"""Run live MindDrive inference and visualize high-confidence current agents."""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.datasets import build_dataset
from mmcv.parallel import collate
from mmcv.utils import DictAction

from tools.live_cf_candidate_scenes_only import (
    build_live_model,
    prepare_cfg,
    resolve_sample_index,
)
from tools.visualize_cf_candidate_scenes_only import draw_agent_shape, style_bev_ax
from tools.visualize_counterfactual_case_study import (
    agent_futures_from_result,
    get_pts_bbox,
    label_name,
    load_infos,
    map_polylines,
)


DETECTION_CLASSES = (
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
DYNAMIC_AGENT_LABELS = (0, 1, 2, 3, 7, 8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--scene-folder", default=None)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--infos", default=None, help="Defaults to cfg.data.test.ann_file.")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--max-agents", type=int, default=None, help="Keep the highest scoring N agents.")
    parser.add_argument("--include-static", action="store_true", help="Also show signs, cones, and traffic lights.")
    parser.add_argument("--show-future", action="store_true", help="Draw the base predicted future as a dashed line.")
    parser.add_argument("--figsize", default="10,9")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def current_scene_bounds(boxes, selected_ids, agent_trajs=None, margin=10.0):
    points = [np.asarray([[0.0, 0.0]], dtype=np.float32)]
    if len(selected_ids):
        points.append(boxes[selected_ids, :2])
        if agent_trajs is not None:
            points.append(agent_trajs[selected_ids, :, :2].reshape(-1, 2))
    points = np.concatenate(points, axis=0)
    lo = np.nanmin(points, axis=0) - margin
    hi = np.nanmax(points, axis=0) + margin
    span = np.maximum(hi - lo, 20.0)
    center = 0.5 * (lo + hi)
    return center - 0.5 * span, center + 0.5 * span


def draw_high_confidence_agents(
    pts,
    info,
    out_path,
    score_thr=0.25,
    max_agents=None,
    show_future=False,
    include_static=False,
    figsize="10,9",
    dpi=220,
):
    boxes, agent_trajs, scores, labels = agent_futures_from_result(pts)
    if boxes is None:
        raise ValueError("Live result does not contain boxes_3d.")
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    labels = np.asarray(labels if labels is not None else np.full(len(boxes), -1), dtype=np.int64)

    valid = scores >= score_thr
    if not include_static:
        valid &= np.isin(labels, DYNAMIC_AGENT_LABELS)
    selected_ids = np.flatnonzero(valid)
    selected_ids = selected_ids[np.argsort(-scores[selected_ids])]
    if max_agents is not None:
        selected_ids = selected_ids[:max_agents]

    width, height = [float(x) for x in figsize.split(",")]
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    fig.patch.set_facecolor("black")
    style_bev_ax(ax)
    for line in map_polylines(pts)[:120]:
        if len(line) > 1:
            ax.plot(line[:, 0], line[:, 1], color="white", linewidth=0.8, alpha=0.25, zorder=1)

    cmap = plt.get_cmap("turbo")
    denom = max(len(selected_ids) - 1, 1)
    agents = []
    for rank, agent_id in enumerate(selected_ids):
        agent_id = int(agent_id)
        color = cmap(rank / denom)
        label = int(labels[agent_id]) if agent_id < len(labels) else -1
        class_name = label_name(label, DETECTION_CLASSES)
        draw_agent_shape(ax, boxes[agent_id], label=label, color=color, linewidth=2.2, alpha=0.98)
        ax.scatter(
            boxes[agent_id, 0], boxes[agent_id, 1], s=22, color=color,
            edgecolor="white", linewidth=0.5, zorder=6,
        )
        if show_future and agent_trajs is not None and agent_id < len(agent_trajs):
            ax.plot(
                agent_trajs[agent_id, :, 0], agent_trajs[agent_id, :, 1], "--",
                color=color, linewidth=1.5, alpha=0.75, zorder=3,
            )
        ax.text(
            boxes[agent_id, 0],
            boxes[agent_id, 1],
            "A{} {}\ns={:.3f}".format(agent_id, class_name, float(scores[agent_id])),
            color="white",
            fontsize=9,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", edgecolor=color, alpha=0.78, boxstyle="round,pad=0.18"),
            zorder=8,
        )
        agents.append(
            dict(
                agent_id=agent_id,
                score=float(scores[agent_id]),
                label=label,
                class_name=class_name,
                box=boxes[agent_id].tolist(),
                current_xy=boxes[agent_id, :2].tolist(),
            )
        )

    ax.scatter([0], [0], marker="*", c="#00ff66", s=130, edgecolor="white", linewidth=0.7, zorder=9)
    lo, hi = current_scene_bounds(boxes, selected_ids, agent_trajs if show_future else None)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title(
        "High-confidence dynamic agents: {} | score >= {:.2f}\nframe {}".format(
            len(selected_ids), score_thr, info.get("frame_idx")
        ),
        fontsize=12,
        pad=9,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return agents


def main():
    args = parse_args()
    cfg = prepare_cfg(args)
    infos = load_infos(args.infos or cfg.data.test.ann_file)
    sample_index = resolve_sample_index(
        infos, args.index, args.scene_folder, args.scenario_name, args.frame_idx
    )
    dataset = build_dataset(cfg.data.test)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError("index {} out of dataset length {}".format(sample_index, len(dataset)))
    model = build_live_model(cfg, args.checkpoint, args.device_id)
    data = collate([dataset[sample_index]], samples_per_gpu=1)
    with torch.no_grad():
        result = model(data, return_loss=False)
    bbox_result = result["bbox_results"][0] if isinstance(result, dict) and "bbox_results" in result else result[0]
    pts = get_pts_bbox(bbox_result)
    info = infos[sample_index]
    agents = draw_high_confidence_agents(
        pts,
        info,
        args.out,
        score_thr=args.score_thr,
        max_agents=args.max_agents,
        show_future=args.show_future,
        include_static=args.include_static,
        figsize=args.figsize,
        dpi=args.dpi,
    )
    summary = dict(
        source="live_model_forward",
        config=args.config,
        checkpoint=args.checkpoint,
        sample_index=sample_index,
        folder=info.get("folder"),
        frame_idx=info.get("frame_idx"),
        score_threshold=args.score_thr,
        include_static=args.include_static,
        num_high_confidence_agents=len(agents),
        agents=agents,
        out=args.out,
    )
    summary_path = os.path.splitext(args.out)[0] + "_agents.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", args.out)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
