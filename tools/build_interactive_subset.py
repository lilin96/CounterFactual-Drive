"""Build an interaction-heavy subset from MindDrive/Bench2Drive infos.

Two selection modes are supported:
1. With --cf-results: use candidate-0 base risk terms emitted by the
   counterfactual head. This is preferred because it uses predicted futures.
2. Without --cf-results: use current-frame GT box distance as a coarse proxy.

The output is an infos pkl with the same item structure as the input, so it can
be used as data.test.ann_file or data.val.ann_file.

Examples:
    python tools/build_interactive_subset.py \
      --infos data/infos/b2d_infos_val.pkl \
      --cf-results work_dirs/cf_test/results_cf_val.pkl \
      --out data/infos/b2d_infos_val_interactive.pkl \
      --max-samples 2000

    python tools/build_interactive_subset.py \
      --infos data/infos/b2d_infos_val.pkl \
      --out data/infos/b2d_infos_val_interactive_proxy.pkl \
      --distance-thr 12.0
"""

import argparse
import os
import pickle

import numpy as np
import torch
from mmcv.fileio.io import load


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


def pts(sample):
    return sample.get("pts_bbox", sample)


def base_risk(sample_pts, term):
    risks = sample_pts.get("cf_risk_scores")
    if not isinstance(risks, dict) or term not in risks:
        return np.nan
    arr = np.asarray(to_numpy(risks[term]), dtype=np.float64)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    arr = arr.reshape(-1)
    return float(arr[0]) if arr.size else np.nan


def proxy_min_current_distance(info):
    boxes = info.get("gt_boxes")
    if boxes is None:
        return np.nan
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.nan
    centers = boxes[:, :2]
    return float(np.linalg.norm(centers, axis=-1).min())


def select_with_results(infos, results, args):
    selected = []
    records = []
    n = min(len(infos), len(results))
    for i in range(n):
        sample_pts = pts(results[i])
        total = base_risk(sample_pts, "total")
        collision = base_risk(sample_pts, "collision")
        ttc = base_risk(sample_pts, "ttc")
        interaction = base_risk(sample_pts, "interaction")
        keep = (
            total >= args.total_risk_thr
            or collision >= args.collision_risk_thr
            or ttc >= args.ttc_risk_thr
            or interaction >= args.interaction_risk_thr
        )
        if keep:
            selected.append(infos[i])
            records.append((i, total, collision, ttc, interaction))
    records.sort(key=lambda x: (x[1], x[4], x[3]), reverse=True)
    if args.max_samples is not None:
        keep_indices = set(r[0] for r in records[: args.max_samples])
        selected = [infos[i] for i in range(n) if i in keep_indices]
        records = records[: args.max_samples]
    return selected, records


def select_with_proxy(infos, args):
    selected = []
    records = []
    for i, info in enumerate(infos):
        min_dist = proxy_min_current_distance(info)
        keep = np.isfinite(min_dist) and min_dist <= args.distance_thr
        if keep:
            selected.append(info)
            records.append((i, min_dist))
    records.sort(key=lambda x: x[1])
    if args.max_samples is not None:
        keep_indices = set(r[0] for r in records[: args.max_samples])
        selected = [infos[i] for i in range(len(infos)) if i in keep_indices]
        records = records[: args.max_samples]
    return selected, records


def write_records(records, path, with_results):
    with open(path, "w") as f:
        if with_results:
            f.write("index,total,collision,ttc,interaction\n")
            for r in records:
                f.write("{},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(*r))
        else:
            f.write("index,min_current_distance\n")
            for r in records:
                f.write("{},{:.6f}\n".format(*r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cf-results", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--distance-thr", type=float, default=12.0)
    parser.add_argument("--total-risk-thr", type=float, default=40.0)
    parser.add_argument("--collision-risk-thr", type=float, default=0.1)
    parser.add_argument("--ttc-risk-thr", type=float, default=20.0)
    parser.add_argument("--interaction-risk-thr", type=float, default=10.0)
    args = parser.parse_args()

    with open(args.infos, "rb") as f:
        infos = pickle.load(f)

    if args.cf_results:
        selected, records = select_with_results(infos, load_results(args.cf_results), args)
        mode = "risk"
    else:
        selected, records = select_with_proxy(infos, args)
        mode = "distance_proxy"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(selected, f)
    stats_path = args.out + ".csv"
    write_records(records, stats_path, with_results=bool(args.cf_results))

    print("mode={}".format(mode))
    print("input_samples={}".format(len(infos)))
    print("selected_samples={}".format(len(selected)))
    print(args.out)
    print(stats_path)


if __name__ == "__main__":
    main()
