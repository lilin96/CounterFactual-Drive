#!/usr/bin/env python
"""Print candidate-conditioned counterfactual agent responses.

Example:
  python tools/print_cf_agent_response.py \
      --results work_dirs/eval_cf_replace_decision/results.pkl \
      --infos data/infos/b2d_infos_val.pkl \
      --scenario-name VanillaNonSignalizedTurnEncounterStopsign_Town12_Route979_Weather9 \
      --frame-idx 15 \
      --top-k 8

The script reads saved MindDrive test outputs. It does not run the model; use a
result pkl generated with counterfactual diagnostics enabled.
"""

import argparse
import json

import numpy as np
from mmcv.fileio.io import load

from mmcv.models.counterfactual.meta_action_labels import PATH_META_ACTIONS, SPEED_META_ACTIONS


def to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def result_items(results):
    if isinstance(results, dict) and "bbox_results" in results:
        return results["bbox_results"]
    return results


def get_pts_bbox(item):
    if isinstance(item, dict) and "pts_bbox" in item:
        return item["pts_bbox"]
    return item


def load_infos(path):
    data = load(path)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data


def resolve_case_index(infos, index=None, scene_folder=None, scenario_name=None, frame_idx=None):
    if index is not None:
        return index
    matches = []
    for i, info in enumerate(infos):
        folder = info.get("folder", "")
        if scene_folder is not None and folder != scene_folder:
            continue
        if scenario_name is not None and scenario_name not in folder:
            continue
        if frame_idx is not None and info.get("frame_idx") != frame_idx:
            continue
        matches.append(i)
    if not matches:
        raise ValueError(
            "No case found for scene_folder={!r}, scenario_name={!r}, frame_idx={!r}".format(
                scene_folder, scenario_name, frame_idx
            )
        )
    if frame_idx is None and len(matches) > 1:
        examples = [(i, infos[i].get("folder"), infos[i].get("frame_idx")) for i in matches[:20]]
        raise ValueError(
            "Matched {} frames. Please pass --frame-idx. First matches:\n{}".format(
                len(matches), json.dumps(examples, indent=2)
            )
        )
    return matches[0]


def parse_indices(text):
    if text is None or text.strip() == "":
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def label_name(label, names):
    label = int(label)
    if 0 <= label < len(names):
        return names[label]
    return "unknown({})".format(label)


def top_probs(prob, names, n=3):
    if prob is None:
        return ""
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 1 or prob.size == 0:
        return ""
    order = np.argsort(-prob)[:n]
    return ", ".join("{}:{:.3f}".format(label_name(i, names), float(prob[i])) for i in order)


def ego_traj(x):
    if x is None:
        return None
    arr = to_numpy(x)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim == 3:
        arr = arr[0]
    return arr[..., :2]


def fallback_candidate_ego(candidate_idx, future_steps=6, dt=0.5):
    """Fallback rule ego candidate for old results without saved CF scenes."""
    if candidate_idx <= 0:
        return None
    j = int(candidate_idx) - 1
    path_count = len(PATH_META_ACTIONS)
    speed_idx = j // path_count
    path_idx = j % path_count
    if speed_idx >= len(SPEED_META_ACTIONS):
        return None
    t = np.arange(1, future_steps + 1, dtype=np.float32) * dt
    base_v = np.asarray([5.0, 0.0, 2.0, 7.0, 3.5, 9.0, 1.5], dtype=np.float32)[speed_idx]
    lateral = np.zeros_like(t)
    longitudinal = base_v * t
    path = PATH_META_ACTIONS[path_idx]
    if path == "turn left":
        lateral = 0.5 * t * t
    elif path == "turn right":
        lateral = -0.5 * t * t
    elif path == "change lane left":
        lateral = np.linspace(0.0, 3.5, future_steps, dtype=np.float32)
    elif path == "change lane right":
        lateral = np.linspace(0.0, -3.5, future_steps, dtype=np.float32)
    lateral = lateral * min(base_v / 5.0, 1.0)
    return np.stack([lateral, longitudinal], axis=-1)


def candidate_ego_future(pts, candidate_idx, future_steps=6):
    """Return ego future for one candidate.

    Prefer ``cf_counterfactual_scenes[k]["ego_future"]`` because that is the
    exact ego candidate consumed by the counterfactual head. If missing, fall
    back to the selected/base ego trajectory for candidate 0 or to the old rule
    candidate generator for candidate ids > 0.
    """
    scenes = pts.get("cf_counterfactual_scenes")
    if scenes and candidate_idx < len(scenes):
        cand = ego_traj(scenes[candidate_idx].get("ego_future"))
        if cand is not None and cand.size:
            return cand, "counterfactual_scene"
    if candidate_idx == 0:
        base = ego_traj(pts.get("decision_expert_ego_fut_preds"))
        if base is None:
            base = ego_traj(pts.get("ego_fut_preds"))
        if base is not None:
            return base, "minddrive_selected"
    fallback = fallback_candidate_ego(candidate_idx, future_steps=future_steps)
    if fallback is not None:
        return fallback, "rule_fallback"
    return None, "missing"


def candidate_meta(pts, candidate_idx):
    metas = pts.get("cf_candidate_meta_actions")
    if isinstance(metas, (list, tuple)) and 0 <= candidate_idx < len(metas):
        meta = metas[candidate_idx]
        if isinstance(meta, dict):
            speed = meta.get("speed", "")
            path = meta.get("path", "")
            if speed or path:
                return "{} / {}".format(speed, path).strip(" /")
            speed_token = meta.get("speed_token", "")
            path_token = meta.get("path_token", "")
            if speed_token or path_token:
                return "{} / {}".format(speed_token, path_token).strip(" /")
        return str(meta)
    return "candidate {}".format(candidate_idx)


def box_centers(pts):
    boxes = to_numpy(pts.get("boxes_3d"))
    scores = to_numpy(pts.get("scores_3d"))
    labels = to_numpy(pts.get("labels_3d"))
    if boxes is None:
        return None, scores, labels
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    return boxes[:, :2], scores, labels


def base_agent_futures(pts):
    """Decode MindDrive base surrounding-agent futures to absolute xy.

    Returns:
        np.ndarray | None: Shape (N, T, 2). These futures are not conditioned on
        a counterfactual ego candidate.
    """
    centers, _, _ = box_centers(pts)
    trajs = to_numpy(pts.get("trajs_3d"))
    if centers is None or trajs is None:
        return None
    trajs = np.asarray(trajs, dtype=np.float32)
    if trajs.ndim == 4:
        trajs = trajs[:, 0]
    elif trajs.ndim == 3 and trajs.shape[-1] != 2:
        trajs = trajs[:, 0]
    if trajs.ndim == 2:
        trajs = trajs.reshape(trajs.shape[0], -1, 2)
    n = min(len(centers), len(trajs))
    return np.cumsum(trajs[:n, :, :2], axis=1) + centers[:n, None, :]


def candidate_agent_futures(pts, candidate_idx):
    """Return agent futures for one candidate.

    Prefer saved counterfactual scenes because those are the actual realized
    agent futures after interaction response prediction. If the result file
    does not contain them, fall back to base MindDrive futures.
    """
    scenes = pts.get("cf_counterfactual_scenes")
    if scenes and candidate_idx < len(scenes):
        fut = to_numpy(scenes[candidate_idx].get("agent_futures"))
        if fut is not None:
            fut = np.squeeze(fut).astype(np.float32)
            if fut.ndim == 4:
                fut = fut[0]
            if fut.ndim == 3 and fut.shape[-1] >= 2:
                return fut[..., :2], "counterfactual_realized"
    base = base_agent_futures(pts)
    if base is not None:
        return base, "base_minddrive_fallback"
    return None, "missing"


def format_traj(traj, precision=2):
    if traj is None:
        return ""
    traj = np.asarray(traj, dtype=np.float32)
    fmt = "{{:.{}f}}".format(int(precision))
    return "[" + ", ".join("({}, {})".format(fmt.format(float(x)), fmt.format(float(y))) for x, y in traj[:, :2]) + "]"


def risk_total_for_candidate(pts, candidate_idx):
    risks = pts.get("cf_risk_scores", {})
    total = risks.get("total")
    if total is None:
        return None
    total = np.squeeze(to_numpy(total)).astype(np.float32)
    if total.ndim == 0:
        total = total.reshape(1)
    if candidate_idx >= len(total):
        return None
    return float(total[candidate_idx])


def selected_candidate_idx(pts):
    risks = pts.get("cf_risk_scores", {})
    total = risks.get("total")
    if total is None:
        return None
    total = np.squeeze(to_numpy(total)).astype(np.float32)
    if total.ndim == 0:
        total = total.reshape(1)
    return int(np.nanargmin(total)) if total.size else None


def response_for_candidate(pts, candidate_idx):
    responses = pts.get("cf_response_meta_actions")
    relevances = pts.get("cf_interaction_relevance")
    if not responses:
        raise KeyError("Result does not contain cf_response_meta_actions. Re-run test with counterfactual diagnostics enabled.")
    if not relevances:
        raise KeyError("Result does not contain cf_interaction_relevance. Re-run test with counterfactual diagnostics enabled.")
    if candidate_idx >= len(responses) or candidate_idx >= len(relevances):
        raise IndexError(
            "candidate {} is out of range: responses={}, relevances={}".format(
                candidate_idx, len(responses), len(relevances)
            )
        )
    resp = responses[candidate_idx]
    rel = np.squeeze(to_numpy(relevances[candidate_idx])).astype(np.float32)
    speed = np.squeeze(to_numpy(resp.get("speed"))).astype(np.int64)
    path = np.squeeze(to_numpy(resp.get("path"))).astype(np.int64)
    speed_probs = to_numpy(resp.get("speed_probs"))
    path_probs = to_numpy(resp.get("path_probs"))
    if speed_probs is not None:
        speed_probs = np.squeeze(speed_probs).astype(np.float32)
    if path_probs is not None:
        path_probs = np.squeeze(path_probs).astype(np.float32)
    return rel, speed, path, speed_probs, path_probs


def print_candidate(
    pts,
    candidate_idx,
    top_k,
    agent_indices=None,
    prob_top_n=3,
    print_trajectories=False,
    print_ego_trajectory=False,
    traj_precision=2,
):
    rel, speed, path, speed_probs, path_probs = response_for_candidate(pts, candidate_idx)
    centers, scores, labels = box_centers(pts)
    futures, future_source = candidate_agent_futures(pts, candidate_idx)
    n = min(len(rel), len(speed), len(path))
    if futures is not None:
        n = min(n, len(futures))
    if agent_indices is None:
        order = np.argsort(-rel[:n])[: min(top_k, n)]
    else:
        order = np.asarray([i for i in agent_indices if 0 <= i < n], dtype=np.int64)

    risk = risk_total_for_candidate(pts, candidate_idx)
    risk_text = " risk_total={:.4f}".format(risk) if risk is not None else ""
    print("\n[candidate {}] {}{} future_source={}".format(candidate_idx, candidate_meta(pts, candidate_idx), risk_text, future_source))
    if print_ego_trajectory:
        future_steps = int(futures.shape[1]) if futures is not None and futures.ndim >= 3 else 6
        ego, ego_source = candidate_ego_future(pts, candidate_idx, future_steps=future_steps)
        print("ego_source:", ego_source)
        print("ego_future_xy:", format_traj(ego, traj_precision))
    print("agent  rel    score  cls  center_x center_y  speed_argmax                 path_argmax                speed_top_probs | path_top_probs")
    print("-" * 150)
    for agent_idx in order:
        sx = sy = float("nan")
        if centers is not None and agent_idx < len(centers):
            sx, sy = float(centers[agent_idx, 0]), float(centers[agent_idx, 1])
        score = float(scores[agent_idx]) if scores is not None and agent_idx < len(scores) else float("nan")
        cls = int(labels[agent_idx]) if labels is not None and agent_idx < len(labels) else -1
        sp_prob = speed_probs[agent_idx] if speed_probs is not None and agent_idx < len(speed_probs) else None
        pa_prob = path_probs[agent_idx] if path_probs is not None and agent_idx < len(path_probs) else None
        print(
            "{:>5d}  {:>5.3f}  {:>5.3f}  {:>3d}  {:>8.2f} {:>8.2f}  {:<28s} {:<25s} {} | {}".format(
                int(agent_idx),
                float(rel[agent_idx]),
                score,
                cls,
                sx,
                sy,
                label_name(speed[agent_idx], SPEED_META_ACTIONS),
                label_name(path[agent_idx], PATH_META_ACTIONS),
                top_probs(sp_prob, SPEED_META_ACTIONS, prob_top_n),
                top_probs(pa_prob, PATH_META_ACTIONS, prob_top_n),
            )
        )
        if print_trajectories:
            traj = futures[agent_idx] if futures is not None and agent_idx < len(futures) else None
            print("       future_xy:", format_traj(traj, traj_precision))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="MindDrive results.pkl with CF outputs.")
    parser.add_argument("--infos", required=True, help="B2D infos pkl aligned with results.")
    parser.add_argument("--index", type=int, default=None, help="Direct sample index.")
    parser.add_argument("--scene-folder", default=None, help="Exact info['folder'].")
    parser.add_argument("--scenario-name", default=None, help="Substring matched against info['folder'].")
    parser.add_argument("--frame-idx", type=int, default=None, help="Frame index inside the selected scenario.")
    parser.add_argument("--candidate-indices", default=None, help="Comma-separated candidate ids. Default: all candidates.")
    parser.add_argument("--agent-indices", default=None, help="Comma-separated agent ids. Default: top-k by relevance.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of relevant agents to print per candidate.")
    parser.add_argument("--prob-top-n", type=int, default=3, help="Number of probability entries to print per label head.")
    parser.add_argument("--print-trajectories", action="store_true", help="Print per-agent future xy for each candidate.")
    parser.add_argument("--print-ego-trajectories", action="store_true", help="Print ego future xy for each candidate.")
    parser.add_argument("--traj-precision", type=int, default=2, help="Decimal places for printed trajectories.")
    args = parser.parse_args()

    items = result_items(load(args.results))
    infos = load_infos(args.infos)
    idx = resolve_case_index(infos, args.index, args.scene_folder, args.scenario_name, args.frame_idx)
    if idx < 0 or idx >= len(items):
        raise IndexError("Resolved index {} is outside result length {}".format(idx, len(items)))

    pts = get_pts_bbox(items[idx])
    info = infos[idx]
    responses = pts.get("cf_response_meta_actions")
    if not responses:
        raise KeyError("No cf_response_meta_actions in selected result.")

    candidate_indices = parse_indices(args.candidate_indices)
    if candidate_indices is None:
        candidate_indices = list(range(len(responses)))
    agent_indices = parse_indices(args.agent_indices)

    selected = selected_candidate_idx(pts)
    print("sample_index:", idx)
    print("folder:", info.get("folder"))
    print("frame_idx:", info.get("frame_idx"))
    print("selected_candidate:", selected)
    if selected is not None:
        print("selected_candidate_meta:", candidate_meta(pts, selected))
    print("num_candidates:", len(responses))
    print("trajectory_note: counterfactual_realized = saved agent futures after response realizer; base_minddrive_fallback = original trajs_3d because cf_counterfactual_scenes is absent.")
    print("ego_note: counterfactual_scene = exact ego candidate used by CF head; rule_fallback = old synthetic candidate because saved scenes are absent.")

    for candidate_idx in candidate_indices:
        print_candidate(
            pts,
            candidate_idx,
            args.top_k,
            agent_indices,
            args.prob_top_n,
            args.print_trajectories,
            args.print_ego_trajectories,
            args.traj_precision,
        )


if __name__ == "__main__":
    main()
