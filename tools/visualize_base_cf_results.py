"""Visualize base MindDrive and counterfactual result pickles in BEV.

Example:
    python tools/visualize_base_cf_results.py \
      --base work_dirs/cf_test/results_base_first10.pkl \
      --cf work_dirs/cf_test/results_cf_first10.pkl \
      --out-dir work_dirs/cf_test/vis \
      --max-samples 10
"""

import argparse
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.storage
from mmcv.fileio.io import load

torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


SPEED_NAMES = [
    "maintain moderate speed",
    "stop",
    "maintain slow speed",
    "speed up",
    "slow down",
    "maintain fast speed",
    "slow down rapidly",
]

PATH_NAMES = [
    "lanefollow",
    "straight",
    "turn left",
    "change lane left",
    "turn right",
    "change lane right",
]


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "tensor"):
        return to_numpy(x.tensor)
    if hasattr(x, "center"):
        return to_numpy(x.center)
    return np.asarray(x)


def load_results(path):
    data = load(path)
    if isinstance(data, dict) and "bbox_results" in data:
        return data["bbox_results"]
    if isinstance(data, list):
        return data
    raise TypeError("Unsupported result format: {}".format(type(data)))


def get_pts(sample):
    if isinstance(sample, dict) and "pts_bbox" in sample:
        return sample["pts_bbox"]
    return sample


def squeeze_traj(x):
    arr = to_numpy(x)
    if arr is None:
        return None
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr[..., :2]


def bev_plot_xy(points):
    """Convert MindDrive local (lateral, forward) coords to plot (forward, lateral)."""
    arr = np.asarray(points)
    return arr[..., 1], arr[..., 0]


def draw_boxes(ax, pts, score_thr=0.35, max_boxes=80):
    boxes = pts.get("boxes_3d")
    scores = to_numpy(pts.get("scores_3d"))
    centers = to_numpy(boxes)
    if centers is None or scores is None:
        return
    centers = centers[:, :2]
    keep = np.where(scores >= score_thr)[0][:max_boxes]
    if keep.size == 0:
        return
    x, y = bev_plot_xy(centers[keep])
    ax.scatter(x, y, s=10, c="#777777", alpha=0.55, label="agents")


def draw_map(ax, pts, max_lanes=50):
    lanes = to_numpy(pts.get("map_pts_3d"))
    scores = to_numpy(pts.get("map_scores_3d"))
    if lanes is None:
        return
    if scores is not None:
        order = np.argsort(-scores)[:max_lanes]
        lanes = lanes[order]
    else:
        lanes = lanes[:max_lanes]
    for lane in lanes:
        lane = lane[..., :2]
        x, y = bev_plot_xy(lane)
        ax.plot(x, y, color="#b7b7b7", linewidth=0.8, alpha=0.45)


def draw_traj(ax, traj, label, color, marker="o"):
    traj = squeeze_traj(traj)
    if traj is None or traj.size == 0:
        return
    x, y = bev_plot_xy(traj)
    ax.plot(x, y, color=color, marker=marker, linewidth=2.0, markersize=3.5, label=label)
    ax.scatter([0], [0], c="black", s=28, marker="x", label="ego now")


def format_meta(meta):
    if isinstance(meta, dict):
        return "{} / {}".format(meta.get("speed", "?"), meta.get("path", "?"))
    return str(meta)


def risk_vector(cf_pts):
    risks = cf_pts.get("cf_risk_scores")
    if not isinstance(risks, dict) or "total" not in risks:
        return None, None
    total = np.asarray(to_numpy(risks["total"]))
    if total.ndim == 2:
        total = total[0]
    idx = int(np.argmin(total))
    terms = {}
    for key, val in risks.items():
        arr = np.asarray(to_numpy(val))
        if arr.ndim == 2:
            arr = arr[0]
        if arr.size > idx:
            terms[key] = float(arr[idx])
    return idx, terms


def visualize_pair(base_sample, cf_sample, out_path, index):
    base_pts = get_pts(base_sample)
    cf_pts = get_pts(cf_sample) if cf_sample is not None else {}

    fig = plt.figure(figsize=(13, 6), dpi=150)
    ax = fig.add_subplot(1, 2, 1)
    draw_map(ax, base_pts)
    draw_boxes(ax, base_pts)
    draw_traj(ax, base_pts.get("ego_fut_preds"), "base ego", "#1f77b4")

    cf_traj = cf_pts.get("cf_selected_ego_future")
    if cf_traj is not None:
        draw_traj(ax, cf_traj, "cf selected", "#d62728", marker="s")

    ax.set_title("Sample {} BEV".format(index))
    ax.set_xlim(-35, 80)
    ax.set_ylim(-35, 35)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("forward / longitudinal (m)")
    ax.set_ylabel("left / lateral (m)")

    ax2 = fig.add_subplot(1, 2, 2)
    selected_idx, risks = risk_vector(cf_pts)
    if risks:
        names = [k for k in ["collision", "ttc", "interaction", "map_rule", "comfort", "progress", "nominal", "total"] if k in risks]
        values = [risks[k] for k in names]
        ax2.barh(names, values, color="#d62728", alpha=0.75)
        ax2.set_title("CF selected risk, candidate {}".format(selected_idx))
        ax2.grid(True, axis="x", linewidth=0.3, alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "No counterfactual risk fields found", ha="center", va="center")
        ax2.set_axis_off()

    base_speed = base_pts.get("speed_value")
    base_path = base_pts.get("path_value")
    base_meta = ""
    if base_speed is not None and base_path is not None:
        base_meta = "base: {} / {}".format(SPEED_NAMES[int(base_speed)], PATH_NAMES[int(base_path)])
    cf_meta = cf_pts.get("cf_selected_meta_action")
    if cf_meta is not None:
        cf_meta = "cf: " + format_meta(cf_meta)
    else:
        cf_meta = "cf: missing"
    fig.suptitle("{}\n{}".format(base_meta, cf_meta), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--cf", required=True)
    parser.add_argument("--out-dir", default="work_dirs/cf_test/vis")
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.base):
        raise FileNotFoundError(args.base)
    if not os.path.exists(args.cf):
        raise FileNotFoundError(args.cf)

    os.makedirs(args.out_dir, exist_ok=True)
    base = load_results(args.base)
    cf = load_results(args.cf)
    n = min(len(base), len(cf), args.max_samples)
    for i in range(n):
        out_path = os.path.join(args.out_dir, "sample_{:03d}.png".format(i))
        visualize_pair(base[i], cf[i], out_path, i)
        print(out_path)


if __name__ == "__main__":
    main()
