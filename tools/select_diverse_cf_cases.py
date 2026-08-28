#!/usr/bin/env python
"""Select diverse counterfactual case-study samples by scenario type."""

import argparse
import csv
import json
import os

from mmcv.fileio.io import load

from tools.find_counterfactual_case_candidates import score_sample
from tools.visualize_counterfactual_case_study import get_pts_bbox, load_infos, result_items


def scenario_type(folder):
    name = os.path.basename(str(folder).rstrip("/"))
    if "_Town" in name:
        return name.split("_Town", 1)[0]
    parts = name.split("_")
    return parts[0] if parts else name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--command-file", default=None, help="Optional shell script with visualization commands.")
    parser.add_argument("--vis-out-dir", default="work_dirs/vis_cf_diverse_10")
    parser.add_argument("--vis-script", default="tools/visualize_counterfactual_case_study.py")
    parser.add_argument("--paper-figure", action="store_true", help="Generate commands for tools/visualize_cf_paper_figure.py style single-image output.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-rel", type=float, default=0.2)
    parser.add_argument("--min-risk-range", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-indices", default="0,1,2,3,4,5,6")
    parser.add_argument("--data-root", default="data/bench2drive")
    return parser.parse_args()


def main():
    args = parse_args()
    items = result_items(load(args.results))
    infos = load_infos(args.infos)

    candidates = []
    for idx, item in enumerate(items):
        pts = get_pts_bbox(item)
        stat = score_sample(pts)
        if stat is None:
            continue
        if stat["rel_max"] < args.min_rel or stat["risk_range"] < args.min_risk_range:
            continue
        info = infos[idx]
        folder = info.get("folder")
        stat.update(
            {
                "sample_index": idx,
                "folder": folder,
                "scenario_type": scenario_type(folder),
                "town": info.get("town_name"),
                "frame_idx": info.get("frame_idx"),
            }
        )
        candidates.append(stat)

    candidates.sort(key=lambda r: r["score"], reverse=True)
    selected = []
    used_types = set()
    for row in candidates:
        if row["scenario_type"] in used_types:
            continue
        selected.append(row)
        used_types.add(row["scenario_type"])
        if len(selected) >= args.top_n:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "sample_index",
        "scenario_type",
        "folder",
        "town",
        "frame_idx",
        "score",
        "selected_candidate",
        "base_total",
        "selected_total",
        "risk_range",
        "rel_max",
        "rel_agent_span",
        "response_diversity",
        "top_candidates",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k) for k in fieldnames})
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(selected, f, indent=2)

    if args.command_file:
        os.makedirs(os.path.dirname(args.command_file) or ".", exist_ok=True)
        with open(args.command_file, "w") as f:
            f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            f.write("export PYTHONPATH=${PYTHONPATH:-.}:.\n\n")
            for row in selected:
                out_dir = os.path.join(args.vis_out_dir, row["scenario_type"])
                if args.paper_figure:
                    out_path = os.path.join(out_dir, "paper_candidate_planning.png")
                    f.write(
                        "mkdir -p {out_dir}\n"
                        "python tools/visualize_cf_paper_figure.py "
                        "--results {results} --infos {infos} --out {out_path} "
                        "--index {idx} --candidate-indices {cand} --data-root {data_root}\n".format(
                            out_dir=out_dir,
                            results=args.results,
                            infos=args.infos,
                            out_path=out_path,
                            idx=row["sample_index"],
                            cand=args.candidate_indices,
                            data_root=args.data_root,
                        )
                    )
                else:
                    f.write(
                        "python {vis_script} "
                        "--results {results} --infos {infos} --out-dir {out_dir} "
                        "--index {idx} --top-k {top_k} --candidate-indices {cand} --data-root {data_root}\n".format(
                            vis_script=args.vis_script,
                            results=args.results,
                            infos=args.infos,
                            out_dir=out_dir,
                            idx=row["sample_index"],
                            top_k=args.top_k,
                            cand=args.candidate_indices,
                            data_root=args.data_root,
                        )
                    )
        os.chmod(args.command_file, 0o755)

    print(json.dumps(selected, indent=2))
    print("wrote", args.out)
    if args.command_file:
        print("wrote", args.command_file)


if __name__ == "__main__":
    main()
