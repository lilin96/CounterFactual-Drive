#!/usr/bin/env python
"""Run one live MindDrive forward pass and visualize CF candidate scenes.

Unlike ``visualize_counterfactual_case_study.py``, this script does not read a
previous result pkl. It loads the model checkpoint, fetches one dataset sample,
runs inference once, and visualizes the in-memory model output. It also forces
``counterfactual_head.save_counterfactual_scenes=True`` so candidate-conditioned
realized agent futures come directly from the model forward.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from torch.nn import DataParallel

from mmcv.datasets import build_dataset, replace_ImageToTensor
from mmcv.models import build_model
from mmcv.parallel import collate
from mmcv.utils import Config, DictAction, load_checkpoint

from adzoo.minddrive.test import custom_wrap_fp16_model
from tools.visualize_counterfactual_case_study import (
    get_pts_bbox,
    load_infos,
    plot_case,
    risk_arrays,
    save_camera_montage,
    save_candidate_panels,
    write_risk_csv,
)


DEFAULT_DIVERSE_CANDIDATES = "0,1,2,3,4,5,6"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="adzoo/minddrive/configs/minddrive_qwen2_05B_infer_counterfactual.py")
    parser.add_argument("--checkpoint", default="work_dirs/cf_meta_aug_base_20k/latest.pth")
    parser.add_argument("--index", type=int, default=12095)
    parser.add_argument(
        "--scene-folder",
        default=None,
        help="Scene folder, e.g. v1/VehicleTurningRoutePedestrian_Town15_Route481_Weather19.",
    )
    parser.add_argument("--frame-idx", type=int, default=None, help="Frame index inside --scene-folder.")
    parser.add_argument("--out-dir", default="work_dirs/analysis_cf_base_20k/live_case_study")
    parser.add_argument("--data-root", default="data/bench2drive")
    parser.add_argument("--infos", default=None, help="Info pkl for camera montage. Defaults to cfg.data.test.ann_file.")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-indices", default=DEFAULT_DIVERSE_CANDIDATES)
    parser.add_argument("--fuse-conv-bn", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def resolve_sample_index(infos, index=None, scene_folder=None, frame_idx=None):
    if scene_folder is None:
        return index
    matches = []
    for i, info in enumerate(infos):
        if info.get("folder") != scene_folder:
            continue
        if frame_idx is not None and info.get("frame_idx") != frame_idx:
            continue
        matches.append(i)
    if not matches:
        raise ValueError("No sample found for scene_folder={!r}, frame_idx={!r}".format(scene_folder, frame_idx))
    if frame_idx is None and len(matches) > 1:
        raise ValueError(
            "Scene {!r} has {} frames. Please pass --frame-idx. First few frame_idx: {}".format(
                scene_folder, len(matches), [infos[i].get("frame_idx") for i in matches[:10]]
            )
        )
    return matches[0]


def prepare_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if cfg.get("close_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        raise TypeError("This live script expects cfg.data.test to be a dataset dict.")
    if "counterfactual_head" not in cfg.model or cfg.model.counterfactual_head is None:
        raise RuntimeError("Config must define model.counterfactual_head for live CF visualization.")
    cfg.model.counterfactual_head.save_counterfactual_scenes = True
    return cfg


def build_live_model(cfg, checkpoint_path):
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        custom_wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    model = DataParallel(model, device_ids=[0])
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = prepare_cfg(args)
    infos_path = args.infos or cfg.data.test.ann_file
    infos = load_infos(infos_path)
    sample_index = args.index
    # sample_index = resolve_sample_index(infos, args.index, args.scene_folder, args.frame_idx)
    dataset = build_dataset(cfg.data.test)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError("index {} out of dataset length {}".format(sample_index, len(dataset)))
    model = build_live_model(cfg, args.checkpoint)

    data = collate([dataset[sample_index]], samples_per_gpu=1)
    with torch.no_grad():
        result = model(data, return_loss=False)
    bbox_result = result["bbox_results"][0] if isinstance(result, dict) and "bbox_results" in result else result[0]
    pts = get_pts_bbox(bbox_result)
    if "cf_counterfactual_scenes" not in pts:
        raise RuntimeError("Live model output still has no cf_counterfactual_scenes; check config override.")

    info = infos[sample_index]

    prefix = os.path.join(args.out_dir, "live_case_{:05d}".format(sample_index))
    camera_ok = save_camera_montage(info, prefix + "_input_cameras.png", args.data_root)
    selected_idx, top_idx, response = plot_case(pts, info, prefix + "_counterfactual_case.png", args.top_k, args.score_thr)
    panel_idx = [int(x) for x in args.candidate_indices.split(",") if x.strip()] if args.candidate_indices else top_idx
    if selected_idx not in panel_idx:
        panel_idx = panel_idx + [selected_idx]
    panels_ok = save_candidate_panels(pts, prefix + "_candidate_panels.png", panel_idx, args.score_thr)
    risks = risk_arrays(pts)
    write_risk_csv(prefix + "_risks.csv", risks, panel_idx)

    summary = {
        "source": "live_model_forward",
        "ego_candidate_source": pts.get("cf_ego_candidate_source", "unknown"),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "sample_index": sample_index,
        "folder": info.get("folder"),
        "frame_idx": info.get("frame_idx"),
        "selected_candidate": selected_idx,
        "panel_candidates": panel_idx,
        "camera_montage_saved": camera_ok,
        "candidate_panels_saved": panels_ok,
        "has_counterfactual_scenes": "cf_counterfactual_scenes" in pts,
        "response_summary": response,
    }
    with open(prefix + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
