"""Visualize one MindDrive/CF result sequence across all tested timesteps.

Example:
    python tools/visualize_cf_sequence.py \
      --base work_dirs/cf_test/results_base_first10.pkl \
      --cf work_dirs/cf_test/results_cf_mini_iter500_first10.pkl \
      --infos data/infos/b2d_infos_first10.pkl \
      --out-dir work_dirs/cf_test/vis_iter500_sequence
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


def load_result_list(path):
    data = load(path)
    if isinstance(data, dict) and "bbox_results" in data:
        return data["bbox_results"]
    if isinstance(data, list):
        return data
    raise TypeError("Unsupported result format: {}".format(type(data)))


def pts(sample):
    return sample.get("pts_bbox", sample)


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


def draw_map(ax, sample_pts, max_lanes=35):
    lanes = to_numpy(sample_pts.get("map_pts_3d"))
    scores = to_numpy(sample_pts.get("map_scores_3d"))
    if lanes is None:
        return
    if scores is not None:
        lanes = lanes[np.argsort(-scores)[:max_lanes]]
    else:
        lanes = lanes[:max_lanes]
    for lane in lanes:
        lane = np.asarray(lane)[..., :2]
        x, y = bev_plot_xy(lane)
        ax.plot(x, y, color="#b8b8b8", linewidth=0.6, alpha=0.45)


def draw_agents(ax, sample_pts, score_thr=0.35, max_boxes=60):
    boxes = sample_pts.get("boxes_3d")
    scores = to_numpy(sample_pts.get("scores_3d"))
    centers = to_numpy(boxes)
    if centers is None or scores is None:
        return
    centers = centers[:, :2]
    keep = np.where(scores >= score_thr)[0][:max_boxes]
    if keep.size:
        x, y = bev_plot_xy(centers[keep])
        ax.scatter(x, y, s=6, c="#666666", alpha=0.45)


def draw_traj(ax, traj, color, label=None, marker="o"):
    traj = squeeze_traj(traj)
    if traj is None or traj.size == 0:
        return
    x, y = bev_plot_xy(traj)
    ax.plot(x, y, color=color, marker=marker, linewidth=1.6, markersize=2.8, label=label)


def selected_risk(sample_pts):
    risks = sample_pts.get("cf_risk_scores")
    if not isinstance(risks, dict) or "total" not in risks:
        return np.nan
    total = np.asarray(to_numpy(risks["total"]))
    if total.ndim == 2:
        total = total[0]
    return float(np.min(total)) if total.size else np.nan


def selected_meta(sample_pts):
    meta = sample_pts.get("cf_selected_meta_action")
    if isinstance(meta, dict):
        return "{}/{}".format(meta.get("speed", "?"), meta.get("path", "?"))
    return str(meta) if meta is not None else "missing"


def frame_title(info, idx):
    if not info:
        return "t={}".format(idx)
    frame_idx = info.get("frame_idx", idx)
    return "t={} frame={}".format(idx, frame_idx)


def draw_frame(ax, base_pts, cf_pts, title):
    draw_map(ax, base_pts)
    draw_agents(ax, base_pts)
    draw_traj(ax, base_pts.get("ego_fut_preds"), "#1f77b4", "base", marker="o")
    draw_traj(ax, cf_pts.get("cf_selected_ego_future"), "#d62728", "cf", marker="s")
    ax.scatter([0], [0], c="black", s=18, marker="x")
    ax.set_title(title, fontsize=8)
    ax.set_xlim(-35, 80)
    ax.set_ylim(-35, 35)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.35)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--cf", required=True)
    parser.add_argument("--infos", default=None)
    parser.add_argument("--out-dir", default="work_dirs/cf_test/vis_sequence")
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = load_result_list(args.base)
    cf = load_result_list(args.cf)
    infos = load(args.infos) if args.infos else None
    n = min(len(base), len(cf), args.max_frames)

    cols = 2
    rows = int(np.ceil(n / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.6 * rows), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    risks = []
    metas = []

    for i in range(n):
        base_pts = pts(base[i])
        cf_pts = pts(cf[i])
        info = infos[i] if isinstance(infos, list) and i < len(infos) else None
        risk = selected_risk(cf_pts)
        meta = selected_meta(cf_pts)
        risks.append(risk)
        metas.append(meta)
        title = "{} risk={:.3f}\n{}".format(frame_title(info, i), risk, meta)
        draw_frame(axes[i], base_pts, cf_pts, title)

        single_path = os.path.join(args.out_dir, "frame_{:03d}.png".format(i))
        single_fig, single_ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)
        draw_frame(single_ax, base_pts, cf_pts, title)
        single_ax.legend(loc="upper right", fontsize=7)
        single_fig.tight_layout()
        single_fig.savefig(single_path)
        plt.close(single_fig)

    for ax in axes[n:]:
        ax.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Base vs counterfactual selected ego future across timesteps", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    sequence_path = os.path.join(args.out_dir, "sequence_all_timesteps.png")
    fig.savefig(sequence_path)
    plt.close(fig)

    risk_fig, risk_ax = plt.subplots(1, 1, figsize=(9, 3.5), dpi=150)
    risk_ax.plot(np.arange(n), risks, marker="o", color="#d62728")
    risk_ax.set_xlabel("timestep")
    risk_ax.set_ylabel("selected total risk")
    risk_ax.grid(True, linewidth=0.3, alpha=0.4)
    risk_ax.set_title("Counterfactual selected risk over time")
    risk_path = os.path.join(args.out_dir, "selected_risk_over_time.png")
    risk_fig.tight_layout()
    risk_fig.savefig(risk_path)
    plt.close(risk_fig)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for i, (risk, meta) in enumerate(zip(risks, metas)):
            f.write("t={:03d} risk={:.6f} meta={}\n".format(i, risk, meta))

    print(sequence_path)
    print(risk_path)
    print(summary_path)


if __name__ == "__main__":
    main()
