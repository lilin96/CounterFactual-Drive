#!/usr/bin/env python
"""Rank validation samples for counterfactual case-study visualization."""

import argparse
import csv
import json
import os

import numpy as np
from mmcv.fileio.io import load

from tools.visualize_counterfactual_case_study import get_pts_bbox, load_infos, result_items, risk_arrays, to_numpy


def label_diversity(response_actions, candidate_indices):
    if not response_actions:
        return 0.0
    speed_sets = []
    path_sets = []
    for idx in candidate_indices:
        if idx >= len(response_actions):
            continue
        resp = response_actions[idx]
        speed = np.squeeze(to_numpy(resp.get("speed"))).astype(np.int64)
        path = np.squeeze(to_numpy(resp.get("path"))).astype(np.int64)
        speed_sets.append(speed)
        path_sets.append(path)
    if len(speed_sets) < 2:
        return 0.0
    speed_arr = np.stack(speed_sets, axis=0)
    path_arr = np.stack(path_sets, axis=0)
    speed_change = (speed_arr != speed_arr[:1]).mean()
    path_change = (path_arr != path_arr[:1]).mean()
    return float(0.5 * (speed_change + path_change))


def relevance_stats(relevance_all):
    if not relevance_all:
        return 0.0, 0.0, 0.0
    rel = []
    for item in relevance_all:
        arr = np.squeeze(to_numpy(item)).astype(np.float32)
        if arr.ndim == 0:
            continue
        rel.append(arr)
    if len(rel) < 2:
        return 0.0, 0.0, 0.0
    rel = np.stack(rel, axis=0)
    max_by_candidate = np.nanmax(rel, axis=1)
    agent_span = np.nanmax(rel, axis=0) - np.nanmin(rel, axis=0)
    return float(np.nanmax(max_by_candidate)), float(np.nanstd(max_by_candidate)), float(np.nanmax(agent_span))


def score_sample(pts):
    risks = risk_arrays(pts)
    total = risks.get("total")
    if total is None or len(total) < 2:
        return None
    total = np.asarray(total, dtype=np.float32)
    selected = int(np.nanargmin(total))
    candidate_indices = np.argsort(total)[: min(12, len(total))].tolist()
    rel_max, rel_std, rel_agent_span = relevance_stats(pts.get("cf_interaction_relevance", []))
    div = label_diversity(pts.get("cf_response_meta_actions", []), candidate_indices)
    risk_range = float(np.nanmax(total) - np.nanmin(total))
    risk_std = float(np.nanstd(total))
    gain = float(total[0] - np.nanmin(total))
    # Prefer scenes where risk changes, relevance changes, and response labels vary.
    score = np.log1p(max(risk_range, 0.0)) + 4.0 * rel_agent_span + 2.0 * rel_std + 2.0 * div + 0.5 * np.log1p(max(gain, 0.0))
    return {
        "score": float(score),
        "selected_candidate": selected,
        "base_total": float(total[0]),
        "selected_total": float(total[selected]),
        "risk_range": risk_range,
        "risk_std": risk_std,
        "base_minus_selected": gain,
        "rel_max": rel_max,
        "rel_std": rel_std,
        "rel_agent_span": rel_agent_span,
        "response_diversity": div,
        "top_candidates": ",".join(str(x) for x in candidate_indices),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-rel", type=float, default=0.35)
    parser.add_argument("--min-risk-range", type=float, default=50.0)
    return parser.parse_args()


def main():
    args = parse_args()
    results = load(args.results)
    items = result_items(results)
    infos = load_infos(args.infos)
    rows = []
    for idx, item in enumerate(items):
        pts = get_pts_bbox(item)
        stat = score_sample(pts)
        if stat is None:
            continue
        if stat["rel_max"] < args.min_rel or stat["risk_range"] < args.min_risk_range:
            continue
        info = infos[idx]
        stat.update(
            {
                "sample_index": idx,
                "folder": info.get("folder"),
                "town": info.get("town_name"),
                "frame_idx": info.get("frame_idx"),
            }
        )
        rows.append(stat)
    rows.sort(key=lambda r: r["score"], reverse=True)
    rows = rows[: args.top_n]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows[:10], indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
