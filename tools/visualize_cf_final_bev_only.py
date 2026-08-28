#!/usr/bin/env python
"""Draw only ``draw_final_bev`` from ``visualize_cf_paper_figure.py``.

This script intentionally reuses the paper-figure implementation directly so
the standalone BEV view stays visually consistent with the full paper figure.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mmcv.fileio.io import load

from tools.visualize_cf_paper_figure import draw_final_bev, resolve_case_index
from tools.visualize_counterfactual_case_study import (
    agent_futures_from_result,
    get_pts_bbox,
    load_infos,
    result_items,
    risk_arrays,
)


def selected_candidate_index(pts, candidate_idx=None):
    if candidate_idx is not None:
        return int(candidate_idx)
    risks = risk_arrays(pts)
    total = risks.get("total")
    if total is not None and len(total):
        return int(np.nanargmin(total))
    return int(pts.get("cf_selected_candidate_idx", 0) or 0)


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
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--figsize", default="5,6", help="Figure size as width,height.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--print-case", action="store_true")
    args = parser.parse_args()

    items = result_items(load(args.results))
    infos = load_infos(args.infos)
    idx = resolve_case_index(infos, args.index, args.scene_folder, args.scenario_name, args.frame_idx)
    if idx >= len(items):
        raise IndexError("Resolved info index {} but results only contain {} items.".format(idx, len(items)))

    pts = get_pts_bbox(items[idx])
    info = infos[idx]
    boxes, agent_trajs, scores, labels = agent_futures_from_result(pts)
    if boxes is None or agent_trajs is None:
        raise ValueError("Selected result does not contain boxes_3d/trajs_3d.")
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    labels = np.asarray(labels if labels is not None else np.zeros(len(boxes)), dtype=np.int64)
    selected_idx = selected_candidate_index(pts, args.candidate_idx)

    if args.print_case:
        print("resolved_index:", idx)
        print("folder:", info.get("folder", ""))
        print("frame_idx:", info.get("frame_idx", ""))
        print("selected_candidate_idx:", selected_idx)

    width, height = [float(x) for x in args.figsize.split(",")]
    fig, ax = plt.subplots(figsize=(width, height))
    draw_final_bev(ax, pts, info, boxes, agent_trajs, scores, labels, selected_idx, args.score_thr)
    # fig.suptitle(
    #     "{} | frame {}".format(os.path.basename(str(info.get("folder", ""))), info.get("frame_idx")),
    #     fontsize=12,
    #     y=0.995,
    # )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
