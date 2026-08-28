#!/usr/bin/env python
"""Aggregate open- and closed-loop paper metrics into CSV/Markdown tables."""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="paper_results")
    parser.add_argument("--experiments", default="M0,A0,A1,A2,A3,A4")
    parser.add_argument("--out", default=None, help="Output prefix (default: ROOT/paper_summary).")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def nested(data, *keys):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def mean_std(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    return mean, math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def route_records(payload):
    checkpoint = payload.get("_checkpoint", payload) if isinstance(payload, dict) else {}
    records = checkpoint.get("records", []) if isinstance(checkpoint, dict) else []
    return records if isinstance(records, list) else []


def global_scores(payload):
    checkpoint = payload.get("_checkpoint", payload) if isinstance(payload, dict) else {}
    record = checkpoint.get("global_record", {}) if isinstance(checkpoint, dict) else {}
    scores = record.get("scores_mean", {}) if isinstance(record, dict) else {}
    return {
        "driving_score": scores.get("score_composed"),
        "route_completion": scores.get("score_route"),
        "infraction_penalty": scores.get("score_penalty"),
    }


def closed_loop_metrics(exp_dir):
    seed_metrics = []
    perfect = total = 0
    for path in sorted((exp_dir / "closed_loop").glob("seed_*/result.json")):
        payload = read_json(path)
        seed_metrics.append(global_scores(payload))
        records = route_records(payload)
        total += len(records)
        perfect += sum(str(row.get("status", "")).lower() == "perfect" for row in records)
    result = {"closed_loop_seeds": len(seed_metrics)}
    for key in ("driving_score", "route_completion", "infraction_penalty"):
        mean, std = mean_std([row.get(key) for row in seed_metrics])
        result[key + "_mean"] = mean
        result[key + "_std"] = std
    # Keep the definition explicit: this is the fraction of route records whose
    # official evaluator status is exactly "Perfect", not merely completed.
    result["perfect_route_rate"] = 100.0 * perfect / total if total else None
    result["closed_loop_route_records"] = total
    return result


def open_loop_metrics(exp_dir):
    trajectory = read_json(exp_dir / "open_loop" / "trajectory" / "trajectory_interaction_metrics.json") or {}
    risk = read_json(exp_dir / "open_loop" / "risk" / "summary.json") or {}
    return {
        "ego_base_ade": nested(trajectory, "ego_base_ade", "mean"),
        "ego_base_fde": nested(trajectory, "ego_base_fde", "mean"),
        "ego_cf_ade": nested(trajectory, "ego_cf_ade", "mean"),
        "ego_cf_fde": nested(trajectory, "ego_cf_fde", "mean"),
        "agent_ade": nested(trajectory, "agent_ade", "mean"),
        "agent_fde": nested(trajectory, "agent_fde", "mean"),
        "interaction_rel_acc": nested(trajectory, "interaction_rel_acc", "mean"),
        "response_speed_acc": nested(trajectory, "response_speed_acc", "mean"),
        "response_path_acc": nested(trajectory, "response_path_acc", "mean"),
        "risk_collision": nested(risk, "risk_terms", "collision", "ours_mean"),
        "risk_interaction": nested(risk, "risk_terms", "interaction", "ours_mean"),
        "risk_comfort": nested(risk, "risk_terms", "comfort", "ours_mean"),
        "selected_progress": nested(risk, "risk_terms", "progress", "ours_mean"),
    }


def fmt(value):
    if value is None:
        return "--"
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return str(value)


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    ids = [x.strip() for x in args.experiments.split(",") if x.strip()]
    rows = []
    for exp_id in ids:
        exp_dir = root / exp_id
        metadata = read_json(exp_dir / "experiment.json") or {}
        row = {"experiment": exp_id, "name": metadata.get("name", exp_id)}
        row.update(open_loop_metrics(exp_dir))
        row.update(closed_loop_metrics(exp_dir))
        rows.append(row)

    out = Path(args.out).resolve() if args.out else root / "paper_summary"
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["experiment", "name"]
    with open(str(out) + ".csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with open(str(out) + ".md", "w") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row.get(key)) for key in columns) + " |\n")
    print("wrote", str(out) + ".csv")
    print("wrote", str(out) + ".md")


if __name__ == "__main__":
    main()
