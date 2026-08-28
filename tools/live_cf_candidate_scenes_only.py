#!/usr/bin/env python
"""Run live MindDrive inference and draw candidate-conditioned BEV panels.

Data/model execution follows ``tools/live_counterfactual_case_study.py``:
  - build dataset from config;
  - fetch one sample by index or scene/frame;
  - run one model forward pass;
  - use in-memory counterfactual outputs.

Visualization follows ``tools/visualize_cf_candidate_scenes_only.py``:
  - dark BEV background;
  - one row of candidate-conditioned scenes;
  - ego candidate trajectory and realized agent futures.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import DataParallel

from mmcv.datasets import build_dataset, replace_ImageToTensor
from mmcv.models import build_model
from mmcv.parallel import collate
from mmcv.utils import Config, DictAction, load_checkpoint

from adzoo.minddrive.test import custom_wrap_fp16_model
from tools.visualize_cf_candidate_scenes_only import (
    candidate_indices_from_args,
    draw_candidate_panel,
)
from tools.visualize_counterfactual_case_study import (
    agent_futures_from_result,
    candidate_ego_for_plot,
    get_pts_bbox,
    load_infos,
    risk_arrays,
    scene_bounds,
    to_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--scene-folder", default=None, help="Exact match against info['folder'].")
    parser.add_argument("--scenario-name", default=None, help="Substring match against info['folder'].")
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--infos", default=None, help="Defaults to cfg.data.test.ann_file.")
    parser.add_argument("--candidate-indices", default="0,1,2,3,4,5,6")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--max-agents", type=int, default=10)
    parser.add_argument("--max-context", type=int, default=5)
    parser.add_argument("--relevance-thr", type=float, default=0.5)
    parser.add_argument("--change-thr", type=float, default=0.15, help="Minimum max CF/base trajectory difference in metres.")
    parser.add_argument("--show-low-confidence-cf", action="store_true")
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--figsize", default="16,6.3", help="Figure size as width,height.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--use-action-expert-candidates",
        action="store_true",
        default=True,
        help="Use MindDrive Action Expert 7-way ego candidates. Enabled by default.",
    )
    parser.add_argument(
        "--use-rule-fallback-candidates",
        action="store_true",
        help="Disable Action Expert candidates and use CF rule fallback candidates.",
    )
    parser.add_argument("--debug-candidate-diff", action="store_true", help="Print candidate output difference diagnostics.")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    return parser.parse_args()


def resolve_sample_index(infos, index=None, scene_folder=None, scenario_name=None, frame_idx=None):
    if index is not None:
        return int(index)
    matches = []
    for i, info in enumerate(infos):
        folder = str(info.get("folder", ""))
        if scene_folder is not None and folder != scene_folder:
            continue
        if scenario_name is not None and scenario_name not in folder:
            continue
        if frame_idx is not None and int(info.get("frame_idx", -1)) != int(frame_idx):
            continue
        matches.append(i)
    if not matches:
        raise ValueError(
            "No sample found for scene_folder={!r}, scenario_name={!r}, frame_idx={!r}".format(
                scene_folder, scenario_name, frame_idx
            )
        )
    if frame_idx is None and len(matches) > 1:
        examples = [(i, infos[i].get("folder"), infos[i].get("frame_idx")) for i in matches[:10]]
        raise ValueError("Matched {} frames. Pass --frame-idx or --index. First matches: {}".format(len(matches), examples))
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
    if not isinstance(cfg.data.test, dict):
        raise TypeError("This script expects cfg.data.test to be a dataset dict.")
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    if "counterfactual_head" not in cfg.model or cfg.model.counterfactual_head is None:
        raise RuntimeError("Config must define model.counterfactual_head.")
    cfg.model.counterfactual_head.save_counterfactual_scenes = True
    return cfg


def build_live_model(cfg, checkpoint_path, device_id=0):
    if not torch.cuda.is_available():
        raise RuntimeError("Live MindDrive inference requires CUDA in this code path.")
    torch.cuda.set_device(device_id)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        custom_wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    model = DataParallel(model.cuda(device_id), device_ids=[device_id])
    model.eval()
    return model


def candidate_diff_summary(pts, candidate_ids=None, top_agents=8):
    scenes = pts.get("cf_counterfactual_scenes", [])
    rels = pts.get("cf_interaction_relevance", [])
    responses = pts.get("cf_response_meta_actions", [])
    if candidate_ids is None:
        candidate_ids = list(range(len(scenes)))
    candidate_ids = [int(i) for i in candidate_ids if int(i) < len(scenes)]
    summary = {
        "num_scenes": len(scenes),
        "candidate_ids": candidate_ids,
        "ego_endpoints": {},
        "ego_pairwise_max_abs_diff": {},
        "agent_pairwise_max_abs_diff": {},
        "relevance_pairwise_max_abs_diff": {},
        "speed_prob_pairwise_max_abs_diff": {},
        "path_prob_pairwise_max_abs_diff": {},
    }
    if not candidate_ids:
        return summary

    ego = {}
    agent = {}
    rel = {}
    speed_prob = {}
    path_prob = {}
    for idx in candidate_ids:
        scene = scenes[idx]
        ego_arr = np.squeeze(to_numpy(scene.get("ego_future"))).astype(np.float32)
        agent_arr = np.squeeze(to_numpy(scene.get("agent_futures"))).astype(np.float32)
        ego[idx] = ego_arr
        agent[idx] = agent_arr
        summary["ego_endpoints"][str(idx)] = ego_arr[-1].tolist() if ego_arr.size else []
        if idx < len(rels):
            rel[idx] = np.squeeze(to_numpy(rels[idx])).astype(np.float32)
        if idx < len(responses):
            resp = responses[idx]
            if resp.get("speed_probs") is not None:
                speed_prob[idx] = np.squeeze(to_numpy(resp.get("speed_probs"))).astype(np.float32)
            if resp.get("path_probs") is not None:
                path_prob[idx] = np.squeeze(to_numpy(resp.get("path_probs"))).astype(np.float32)

    ref = candidate_ids[0]
    if ref in rel and rel[ref].size:
        ranked_agents = np.argsort(-rel[ref])[:top_agents]
    else:
        ranked_agents = np.arange(min(top_agents, agent[ref].shape[0])) if ref in agent and agent[ref].ndim >= 3 else np.asarray([], dtype=np.int64)

    for idx in candidate_ids[1:]:
        key = "{}-{}".format(ref, idx)
        summary["ego_pairwise_max_abs_diff"][key] = float(np.max(np.abs(ego[ref] - ego[idx]))) if ego[ref].shape == ego[idx].shape else None
        if agent[ref].shape == agent[idx].shape:
            if ranked_agents.size:
                diff = np.max(np.abs(agent[ref][ranked_agents] - agent[idx][ranked_agents]))
            else:
                diff = np.max(np.abs(agent[ref] - agent[idx]))
            summary["agent_pairwise_max_abs_diff"][key] = float(diff)
        if ref in rel and idx in rel and rel[ref].shape == rel[idx].shape:
            summary["relevance_pairwise_max_abs_diff"][key] = float(np.max(np.abs(rel[ref] - rel[idx])))
        if ref in speed_prob and idx in speed_prob and speed_prob[ref].shape == speed_prob[idx].shape:
            summary["speed_prob_pairwise_max_abs_diff"][key] = float(np.max(np.abs(speed_prob[ref] - speed_prob[idx])))
        if ref in path_prob and idx in path_prob and path_prob[ref].shape == path_prob[idx].shape:
            summary["path_prob_pairwise_max_abs_diff"][key] = float(np.max(np.abs(path_prob[ref] - path_prob[idx])))
    summary["top_agents_by_ref_relevance"] = ranked_agents.astype(int).tolist()
    return summary


def draw_live_candidate_panels(
    pts,
    info,
    out_path,
    candidate_text,
    score_thr,
    max_agents,
    show_text,
    figsize,
    dpi,
    relevance_thr=0.5,
    change_thr=0.15,
    max_context=5,
    show_low_confidence_cf=False,
):
    boxes, agent_trajs, scores, labels = agent_futures_from_result(pts)
    if boxes is None or agent_trajs is None:
        raise ValueError("Live result does not contain boxes_3d/trajs_3d.")
    scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
    labels = np.asarray(labels if labels is not None else np.zeros(len(boxes)), dtype=np.int64)
    candidate_ids = candidate_indices_from_args(pts, candidate_text, count=7)[:7]
    ego_refs = [candidate_ego_for_plot(pts, i, future_steps=agent_trajs.shape[1]) for i in candidate_ids]
    bounds = scene_bounds(boxes, agent_trajs, ego_refs)

    width, height = [float(x) for x in figsize.split(",")]
    fig, axes = plt.subplots(1, len(candidate_ids), figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor("black")
    for ax, cand_idx in zip(axes.reshape(-1), candidate_ids):
        draw_candidate_panel(
            ax,
            pts,
            cand_idx,
            boxes,
            agent_trajs,
            scores,
            labels,
            bounds,
            score_thr,
            max_agents,
            show_text=show_text,
            relevance_thr=relevance_thr,
            change_thr=change_thr,
            max_context=max_context,
            show_low_confidence_cf=show_low_confidence_cf,
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.98], w_pad=1.0)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return candidate_ids


def main():
    args = parse_args()
    if args.use_rule_fallback_candidates:
        os.environ["MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES"] = "0"
    elif args.use_action_expert_candidates:
        os.environ["MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES"] = "1"

    cfg = prepare_cfg(args)
    infos_path = args.infos or cfg.data.test.ann_file
    infos = load_infos(infos_path)
    sample_index = resolve_sample_index(infos, args.index, args.scene_folder, args.scenario_name, args.frame_idx)

    dataset = build_dataset(cfg.data.test)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError("index {} out of dataset length {}".format(sample_index, len(dataset)))
    model = build_live_model(cfg, args.checkpoint, args.device_id)

    data = collate([dataset[sample_index]], samples_per_gpu=1)
    with torch.no_grad():
        result = model(data, return_loss=False)
    bbox_result = result["bbox_results"][0] if isinstance(result, dict) and "bbox_results" in result else result[0]
    pts = get_pts_bbox(bbox_result)
    if "cf_counterfactual_scenes" not in pts:
        raise RuntimeError("Model output has no cf_counterfactual_scenes; check save_counterfactual_scenes override.")

    info = infos[sample_index]
    candidate_ids = draw_live_candidate_panels(
        pts,
        info,
        args.out + '/candidate_index_{}.png'.format(sample_index),
        args.candidate_indices,
        args.score_thr,
        args.max_agents,
        args.show_text,
        args.figsize,
        args.dpi,
        relevance_thr=args.relevance_thr,
        change_thr=args.change_thr,
        max_context=args.max_context,
        show_low_confidence_cf=args.show_low_confidence_cf,
    )
    risks = risk_arrays(pts)
    total = risks.get("total")
    selected_idx = int(np.nanargmin(total)) if total is not None and len(total) else 0
    diff_summary = candidate_diff_summary(pts, candidate_ids, top_agents=args.max_agents)
    summary = {
        "source": "live_model_forward",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "sample_index": sample_index,
        "folder": info.get("folder"),
        "frame_idx": info.get("frame_idx"),
        "selected_candidate": selected_idx,
        "candidate_indices": candidate_ids,
        "out": args.out,
        "has_counterfactual_scenes": "cf_counterfactual_scenes" in pts,
        "ego_candidate_source": pts.get("cf_ego_candidate_source", "unknown"),
        "num_valid_agents": int(np.squeeze(to_numpy(pts.get("cf_num_valid_agents", 0)))),
        "num_decoded_agents": int(len(to_numpy(pts.get("scores_3d")))) if pts.get("scores_3d") is not None else 0,
        "candidate_diff": diff_summary,
    }
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if args.debug_candidate_diff:
        print("candidate_diff:")
        print(json.dumps(diff_summary, indent=2))
    print("wrote", args.out)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
