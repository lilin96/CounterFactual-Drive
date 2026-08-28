#!/usr/bin/env python
"""Check whether MindDrive result pkl files contain cf_counterfactual_scenes."""

import argparse

from mmcv.fileio.io import load


def result_items(results):
    if isinstance(results, dict) and "bbox_results" in results:
        return results["bbox_results"]
    return results


def get_pts_bbox(item):
    if isinstance(item, dict) and "pts_bbox" in item:
        return item["pts_bbox"]
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--check-samples", type=int, default=10)
    args = parser.parse_args()

    for path in args.paths:
        try:
            data = load(path)
            items = result_items(data)
            total = len(items) if hasattr(items, "__len__") else 0
            checked = min(total, args.check_samples)
            scene_lens = []
            cf_keys = []
            for i in range(checked):
                pts = get_pts_bbox(items[i])
                if isinstance(pts, dict) and not cf_keys:
                    cf_keys = [k for k in pts.keys() if k.startswith("cf_")]
                scenes = pts.get("cf_counterfactual_scenes") if isinstance(pts, dict) else None
                scene_lens.append(len(scenes) if scenes else 0)
            has_scenes = any(x > 0 for x in scene_lens)
            print(
                "{}\t{}\tsamples={}\tchecked={}\tscene_lens_first={}\tcf_keys={}".format(
                    "YES" if has_scenes else "NO",
                    path,
                    total,
                    checked,
                    scene_lens,
                    cf_keys,
                ),
                flush=True,
            )
        except Exception as exc:
            print("ERR\t{}\t{}: {}".format(path, type(exc).__name__, exc), flush=True)


if __name__ == "__main__":
    main()
