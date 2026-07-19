"""Plot risk comparison between MindDrive base and CF selected trajectories.

The counterfactual output stores risk for every ego candidate. In the current
MindDrive integration, candidate 0 is the original MindDrive ego trajectory
prepended before generated meta-action candidates. The selected candidate is
the minimum-risk CF choice. This script compares those two risks with the same
rule-based risk terms.

Example:
    python tools/plot_base_cf_risk_comparison.py \
      --cf work_dirs/cf_test/results_cf_mini_abstrajfix_500iter_first10.pkl \
      --out-dir work_dirs/cf_test/risk_compare_abstrajfix_500iter
"""

import argparse
import csv
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


RISK_TERMS = ["collision", "ttc", "interaction", "map_rule", "comfort", "progress", "nominal", "total"]


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
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


def risk_array(sample_pts, term):
    risks = sample_pts.get("cf_risk_scores")
    if not isinstance(risks, dict) or term not in risks:
        return None
    arr = np.asarray(to_numpy(risks[term]), dtype=np.float64)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.reshape(-1)


def selected_index(sample_pts):
    total = risk_array(sample_pts, "total")
    if total is None or total.size == 0:
        return None
    return int(np.argmin(total))


def selected_meta(sample_pts):
    meta = sample_pts.get("cf_selected_meta_action")
    if isinstance(meta, dict):
        return "{}/{}".format(meta.get("speed", "?"), meta.get("path", "?"))
    return str(meta) if meta is not None else ""


def collect_rows(results, max_samples=None):
    n = len(results) if max_samples is None else min(len(results), max_samples)
    rows = []
    for i in range(n):
        sample_pts = pts(results[i])
        sel_idx = selected_index(sample_pts)
        if sel_idx is None:
            continue
        row = {"sample": i, "selected_idx": sel_idx, "selected_meta": selected_meta(sample_pts)}
        for term in RISK_TERMS:
            arr = risk_array(sample_pts, term)
            if arr is None or arr.size <= sel_idx:
                row["base_" + term] = np.nan
                row["ours_" + term] = np.nan
            else:
                row["base_" + term] = float(arr[0])
                row["ours_" + term] = float(arr[sel_idx])
        rows.append(row)
    return rows


def write_csv(rows, path):
    fieldnames = ["sample", "selected_idx", "selected_meta"]
    for term in RISK_TERMS:
        fieldnames.extend(["base_" + term, "ours_" + term])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_total(rows, path):
    xs = np.asarray([r["sample"] for r in rows])
    base = np.asarray([r["base_total"] for r in rows])
    ours = np.asarray([r["ours_total"] for r in rows])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=160)
    ax.plot(xs, base, marker="o", linewidth=1.8, label="Base MindDrive", color="#1f77b4")
    ax.plot(xs, ours, marker="s", linewidth=1.8, label="Ours CF reranked", color="#d62728")
    ax.fill_between(xs, ours, base, where=base >= ours, color="#d62728", alpha=0.12, interpolate=True)
    ax.set_xlabel("sample / timestep")
    ax.set_ylabel("total risk")
    ax.set_title("Planning Risk: Base vs Counterfactual Reranking")
    ax.grid(True, linewidth=0.3, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_terms(rows, path):
    base_means = []
    ours_means = []
    names = []
    for term in RISK_TERMS:
        base = np.asarray([r["base_" + term] for r in rows], dtype=np.float64)
        ours = np.asarray([r["ours_" + term] for r in rows], dtype=np.float64)
        if np.all(np.isnan(base)) or np.all(np.isnan(ours)):
            continue
        names.append(term)
        base_means.append(np.nanmean(base))
        ours_means.append(np.nanmean(ours))

    y = np.arange(len(names))
    height = 0.36
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), dpi=160)
    ax.barh(y - height / 2, base_means, height, label="Base MindDrive", color="#1f77b4", alpha=0.82)
    ax.barh(y + height / 2, ours_means, height, label="Ours CF reranked", color="#d62728", alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("mean risk over samples")
    ax.set_title("Mean Risk Term Comparison")
    ax.grid(True, axis="x", linewidth=0.3, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary(rows, path):
    base_total = np.asarray([r["base_total"] for r in rows], dtype=np.float64)
    ours_total = np.asarray([r["ours_total"] for r in rows], dtype=np.float64)
    improvement = base_total - ours_total
    rel = improvement / np.maximum(base_total, 1e-6)
    with open(path, "w") as f:
        f.write("num_samples={}\n".format(len(rows)))
        f.write("base_total_mean={:.6f}\n".format(float(np.nanmean(base_total))))
        f.write("ours_total_mean={:.6f}\n".format(float(np.nanmean(ours_total))))
        f.write("absolute_improvement_mean={:.6f}\n".format(float(np.nanmean(improvement))))
        f.write("relative_improvement_mean={:.2f}%\n".format(float(np.nanmean(rel) * 100.0)))
        f.write("\nselected_meta_counts:\n")
        counts = {}
        for row in rows:
            counts[row["selected_meta"]] = counts.get(row["selected_meta"], 0) + 1
        for key, value in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write("{}: {}\n".format(key, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", required=True, help="Counterfactual result pkl containing cf_risk_scores.")
    parser.add_argument("--out-dir", default="work_dirs/cf_test/risk_compare")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = collect_rows(load_result_list(args.cf), max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("No cf_risk_scores found in {}".format(args.cf))

    csv_path = os.path.join(args.out_dir, "risk_comparison.csv")
    total_path = os.path.join(args.out_dir, "total_risk_base_vs_ours.png")
    terms_path = os.path.join(args.out_dir, "mean_risk_terms_base_vs_ours.png")
    summary_path = os.path.join(args.out_dir, "summary.txt")

    write_csv(rows, csv_path)
    plot_total(rows, total_path)
    plot_terms(rows, terms_path)
    write_summary(rows, summary_path)

    print(total_path)
    print(terms_path)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
