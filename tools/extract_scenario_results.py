#!/usr/bin/env python
"""Extract one scenario from MindDrive result and info pickle files.

This keeps results and infos aligned, so downstream visualization scripts can
read the smaller pkl files without loading the full validation result each time.
"""

import argparse
import csv
import os

from mmcv.fileio.io import dump, load

from tools.visualize_counterfactual_case_study import load_infos, result_items


def matched_indices(infos, scenario_name=None, scene_folder=None, frame_idx=None):
    indices = []
    for i, info in enumerate(infos):
        folder = str(info.get("folder", ""))
        if scene_folder is not None and folder != scene_folder:
            continue
        if scenario_name is not None and scenario_name not in folder:
            continue
        if frame_idx is not None and int(info.get("frame_idx", -1)) != int(frame_idx):
            continue
        indices.append(i)
    return indices


def save_index_csv(indices, infos, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset_index", "original_index", "folder", "frame_idx"])
        for subset_i, original_i in enumerate(indices):
            info = infos[original_i]
            writer.writerow([subset_i, original_i, info.get("folder", ""), info.get("frame_idx", "")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Full MindDrive results.pkl.")
    parser.add_argument("--infos", required=True, help="Full b2d_infos_val.pkl or list pkl.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenario-name", default=None, help="Substring matched against info['folder'].")
    parser.add_argument("--scene-folder", default=None, help="Exact match against info['folder'].")
    parser.add_argument("--frame-idx", type=int, default=None, help="Optional single frame filter.")
    args = parser.parse_args()

    if args.scenario_name is None and args.scene_folder is None:
        raise ValueError("Pass --scenario-name or --scene-folder.")

    print("loading infos:", args.infos)
    infos = load_infos(args.infos)
    indices = matched_indices(infos, args.scenario_name, args.scene_folder, args.frame_idx)
    if not indices:
        raise ValueError(
            "No frames matched scenario_name={!r}, scene_folder={!r}, frame_idx={!r}".format(
                args.scenario_name, args.scene_folder, args.frame_idx
            )
        )

    print("loading results:", args.results)
    results = result_items(load(args.results))
    n = min(len(infos), len(results))
    indices = [i for i in indices if i < n]
    if not indices:
        raise ValueError("Matched infos are outside result length. infos={}, results={}".format(len(infos), len(results)))

    subset_infos = [infos[i] for i in indices]
    subset_results = [results[i] for i in indices]

    os.makedirs(args.out_dir, exist_ok=True)
    results_out = os.path.join(args.out_dir, "results.pkl")
    infos_out = os.path.join(args.out_dir, "infos.pkl")
    index_out = os.path.join(args.out_dir, "index.csv")

    dump(subset_results, results_out)
    dump(subset_infos, infos_out)
    save_index_csv(indices, infos, index_out)

    print("matched_frames:", len(indices))
    print("first_original_index:", indices[0])
    print("last_original_index:", indices[-1])
    print("wrote", results_out)
    print("wrote", infos_out)
    print("wrote", index_out)


if __name__ == "__main__":
    main()
