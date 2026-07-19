"""Evaluate counterfactual reranking risks against the MindDrive base plan.

This evaluator uses the risk tensors emitted by CounterfactualReasoningHead.
By convention, candidate 0 is the original MindDrive ego future and the
selected candidate is the counterfactual reranked trajectory. The script
therefore compares base-vs-ours under the exact same risk scorer.

It does not assume ground-truth counterfactual supervision. If future GT fields
are later available in the infos, GT metrics can be added separately without
changing this internal-risk evaluation.

Example:
    python tools/eval_counterfactual_risk.py \
      --cf work_dirs/cf_test/results_cf_mini_abstrajfix_500iter_first10.pkl \
      --out-dir work_dirs/cf_test/eval_abstrajfix_500iter \
      --max-samples 10
"""

import argparse
import csv
import io
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.storage
from mmcv.fileio.io import load


torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


RISK_TERMS = ["collision", "ttc", "interaction", "map_rule", "comfort", "progress", "nominal", "total"]
RELATIVE_EPS = 0.1


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
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


def squeeze_traj(x):
    arr = to_numpy(x)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr[..., :2]


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
    return int(np.nanargmin(total))


def selected_meta(sample_pts):
    meta = sample_pts.get("cf_selected_meta_action")
    if isinstance(meta, dict):
        return "{}/{}".format(meta.get("speed", "?"), meta.get("path", "?"))
    return str(meta) if meta is not None else ""


def final_progress(traj):
    traj = squeeze_traj(traj)
    if traj is None or traj.size == 0:
        return np.nan
    return float(traj[-1, 1])


def mean_accel_norm(traj):
    traj = squeeze_traj(traj)
    if traj is None or traj.shape[0] < 2:
        return np.nan
    deltas = np.diff(traj, axis=0, prepend=np.zeros((1, 2), dtype=traj.dtype))
    accel = np.diff(deltas, axis=0, prepend=np.zeros((1, 2), dtype=traj.dtype))
    return float(np.linalg.norm(accel, axis=-1).mean())


def nominal_deviation(base_traj, ours_traj):
    base = squeeze_traj(base_traj)
    ours = squeeze_traj(ours_traj)
    if base is None or ours is None:
        return np.nan
    t = min(base.shape[0], ours.shape[0])
    if t == 0:
        return np.nan
    return float(np.linalg.norm(base[:t] - ours[:t], axis=-1).mean())


def collect_rows(results, max_samples=None):
    n = len(results) if max_samples is None else min(len(results), max_samples)
    rows = []
    skipped = 0
    for i in range(n):
        sample_pts = pts(results[i])
        idx = selected_index(sample_pts)
        if idx is None:
            skipped += 1
            continue
        row = {
            "sample": i,
            "selected_idx": idx,
            "selected_meta": selected_meta(sample_pts),
            "base_progress_final": final_progress(sample_pts.get("ego_fut_preds")),
            "ours_progress_final": final_progress(sample_pts.get("cf_selected_ego_future")),
            "base_comfort_proxy": mean_accel_norm(sample_pts.get("ego_fut_preds")),
            "ours_comfort_proxy": mean_accel_norm(sample_pts.get("cf_selected_ego_future")),
            "trajectory_deviation": nominal_deviation(sample_pts.get("ego_fut_preds"), sample_pts.get("cf_selected_ego_future")),
        }
        for term in RISK_TERMS:
            arr = risk_array(sample_pts, term)
            if arr is None or arr.size <= idx:
                row["base_" + term] = np.nan
                row["ours_" + term] = np.nan
                row["delta_" + term] = np.nan
            else:
                base_val = float(arr[0])
                ours_val = float(arr[idx])
                row["base_" + term] = base_val
                row["ours_" + term] = ours_val
                row["delta_" + term] = base_val - ours_val
        rows.append(row)
    return rows, skipped


def bootstrap_ci(values, num_bootstrap=2000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return [np.nan, np.nan]
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(num_bootstrap, values.size))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(rows, skipped):
    summary = {"num_samples": len(rows), "num_skipped": skipped, "risk_terms": {}, "planning": {}}
    for term in RISK_TERMS:
        base = np.asarray([r["base_" + term] for r in rows], dtype=np.float64)
        ours = np.asarray([r["ours_" + term] for r in rows], dtype=np.float64)
        delta = base - ours
        rel = np.full_like(delta, np.nan, dtype=np.float64)
        valid_rel = np.abs(base) > RELATIVE_EPS
        rel[valid_rel] = delta[valid_rel] / np.abs(base[valid_rel])
        summary["risk_terms"][term] = {
            "base_mean": float(np.nanmean(base)),
            "ours_mean": float(np.nanmean(ours)),
            "delta_mean": float(np.nanmean(delta)),
            "delta_ci95": bootstrap_ci(delta),
            "relative_delta_mean_pct": float(np.nanmean(rel) * 100.0) if not np.all(np.isnan(rel)) else None,
            "improved_fraction": float(np.nanmean(delta > 0)),
        }

    for key in ["progress_final", "comfort_proxy"]:
        base = np.asarray([r["base_" + key] for r in rows], dtype=np.float64)
        ours = np.asarray([r["ours_" + key] for r in rows], dtype=np.float64)
        delta = ours - base
        summary["planning"][key] = {
            "base_mean": float(np.nanmean(base)),
            "ours_mean": float(np.nanmean(ours)),
            "ours_minus_base_mean": float(np.nanmean(delta)),
            "ours_minus_base_ci95": bootstrap_ci(delta),
        }

    dev = np.asarray([r["trajectory_deviation"] for r in rows], dtype=np.float64)
    summary["planning"]["trajectory_deviation_mean"] = float(np.nanmean(dev))
    summary["planning"]["trajectory_deviation_ci95"] = bootstrap_ci(dev)
    summary["selected_meta_counts"] = dict(Counter(r["selected_meta"] for r in rows))
    return summary


def write_csv(rows, path):
    fieldnames = [
        "sample",
        "selected_idx",
        "selected_meta",
        "base_progress_final",
        "ours_progress_final",
        "base_comfort_proxy",
        "ours_comfort_proxy",
        "trajectory_deviation",
    ]
    for term in RISK_TERMS:
        fieldnames.extend(["base_" + term, "ours_" + term, "delta_" + term])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary, path):
    def pct_or_na(value):
        return "N/A" if value is None or np.isnan(value) else "{:.2f}%".format(value)

    with open(path, "w") as f:
        f.write("# Counterfactual Risk Evaluation\n\n")
        f.write("Samples: {}  \nSkipped: {}\n\n".format(summary["num_samples"], summary["num_skipped"]))
        f.write("## Risk Terms\n\n")
        f.write("| Term | Base ↓ | Ours ↓ | Δ Base-Ours ↑ | 95% CI | Rel. Δ | Improved |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for term in RISK_TERMS:
            item = summary["risk_terms"][term]
            ci = item["delta_ci95"]
            f.write(
                "| {} | {:.4f} | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] | {} | {:.1f}% |\n".format(
                    term,
                    item["base_mean"],
                    item["ours_mean"],
                    item["delta_mean"],
                    ci[0],
                    ci[1],
                    pct_or_na(item["relative_delta_mean_pct"]),
                    item["improved_fraction"] * 100.0,
                )
            )
        f.write("\n## Planning Proxies\n\n")
        f.write("| Metric | Base | Ours | Ours-Base | 95% CI |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for key in ["progress_final", "comfort_proxy"]:
            item = summary["planning"][key]
            ci = item["ours_minus_base_ci95"]
            f.write(
                "| {} | {:.4f} | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |\n".format(
                    key, item["base_mean"], item["ours_mean"], item["ours_minus_base_mean"], ci[0], ci[1]
                )
            )
        ci = summary["planning"]["trajectory_deviation_ci95"]
        f.write(
            "| trajectory_deviation | 0.0000 | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |\n".format(
                summary["planning"]["trajectory_deviation_mean"],
                summary["planning"]["trajectory_deviation_mean"],
                ci[0],
                ci[1],
            )
        )
        f.write("\n## Selected Meta-Actions\n\n")
        for key, val in sorted(summary["selected_meta_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
            f.write("- {}: {}\n".format(key, val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cf", required=True, help="Counterfactual result pkl containing cf_risk_scores.")
    parser.add_argument("--out-dir", default="work_dirs/cf_test/eval_counterfactual")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows, skipped = collect_rows(load_results(args.cf), max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("No evaluable samples found in {}".format(args.cf))
    summary = summarize(rows, skipped)

    csv_path = os.path.join(args.out_dir, "per_sample_metrics.csv")
    json_path = os.path.join(args.out_dir, "summary.json")
    md_path = os.path.join(args.out_dir, "summary.md")
    write_csv(rows, csv_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_markdown(summary, md_path)

    print(csv_path)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
