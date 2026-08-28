#!/usr/bin/env python
"""Run reproducible Counterfactual MindDrive paper experiments.

This runner orchestrates existing project entry points.  It never edits source
configs in place and defaults to refusing to overwrite completed artifacts.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmcv.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="configs/cf_paper_eval_matrix.py")
    parser.add_argument("--experiment", default="A0", help="ID, comma-separated IDs, or all")
    parser.add_argument(
        "--stage",
        choices=["list", "validate", "smoke", "open_loop", "analyze", "closed_loop_jobs", "all"],
        default="validate",
    )
    parser.add_argument("--gpu", default="0", help="One visible physical GPU for inference.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint; required for A1 until configured.")
    parser.add_argument("--routes", default=None, help="Official Bench2Drive route XML for closed_loop_jobs.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_matrix(path):
    path = (REPO_ROOT / path).resolve() if not os.path.isabs(path) else Path(path)
    spec = importlib.util.spec_from_file_location("cf_paper_eval_matrix", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.COMMON), dict(module.EXPERIMENTS), path


def select_experiments(text, experiments):
    ids = list(experiments) if text == "all" else [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in ids if x not in experiments]
    if unknown:
        raise KeyError("Unknown experiment IDs: {}".format(unknown))
    return ids


def abs_path(path):
    if path is None:
        return None
    return str((REPO_ROOT / path).resolve()) if not os.path.isabs(path) else path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_experiment(exp_id, raw, common, args):
    exp = dict(raw)
    exp["id"] = exp_id
    exp["config"] = abs_path(exp["config"])
    exp["checkpoint"] = abs_path(args.checkpoint or exp.get("checkpoint"))
    exp["infos"] = abs_path(common["infos"])
    exp["output_root"] = abs_path(args.output_root or common["output_root"])
    exp["seed"] = args.seed
    exp["gpu"] = args.gpu
    exp["routes"] = abs_path(args.routes or common.get("official_routes"))
    exp["closed_loop_repetitions"] = int(common["closed_loop_repetitions"])
    exp["closed_loop_seeds"] = list(common.get("seeds", [0, 1, 2]))
    exp["base_port"] = int(common.get("base_port", 30000))
    exp["base_tm_port"] = int(common.get("base_tm_port", 50000))
    exp["smoke_start"] = int(common["smoke_start"])
    exp["smoke_end"] = int(common["smoke_end"])
    return exp


def validate_experiment(exp):
    errors = []
    for key in ("config", "infos"):
        if not exp.get(key) or not os.path.isfile(exp[key]):
            errors.append("{} does not exist: {}".format(key, exp.get(key)))
    if not exp.get("checkpoint"):
        errors.append("checkpoint is unset (A1 requires --checkpoint with a separately trained no-meta model)")
    elif not os.path.isfile(exp["checkpoint"]):
        errors.append("checkpoint does not exist: {}".format(exp["checkpoint"]))
    if errors:
        raise RuntimeError("{} validation failed:\n- {}".format(exp["id"], "\n- ".join(errors)))

    cfg = Config.fromfile(exp["config"])
    head = cfg.model.get("counterfactual_head") if exp["has_counterfactual"] else None
    if exp["has_counterfactual"] and head is None:
        raise RuntimeError("{} expects counterfactual_head but config has none".format(exp["id"]))
    if head is not None:
        actual_meta = bool(head.get("use_ego_meta_embedding", False))
        actual_response = bool(head.get("use_candidate_conditioned_agent_response", True))
        if actual_meta != exp["meta_embedding"]:
            raise RuntimeError("{} meta mismatch: matrix={} config={}".format(exp["id"], exp["meta_embedding"], actual_meta))
        if actual_response != exp["agent_response"]:
            raise RuntimeError("{} response mismatch: matrix={} config={}".format(exp["id"], exp["agent_response"], actual_response))
        if exp["candidate_source"] == "rule_speed_k7" and head.get("rule_candidate_mode") != "speed_only":
            raise RuntimeError("A3 config must set rule_candidate_mode='speed_only'")
    return cfg


def experiment_env(exp):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(exp["gpu"])
    env["MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES"] = "1" if exp["use_action_expert_candidates"] else "0"
    env["MINDDRIVE_CF_REPLACE_DECISION"] = "1" if exp["replace_decision"] else "0"
    return env


def exp_dir(exp):
    return Path(exp["output_root"]) / exp["id"]


def snapshot(exp, matrix_path):
    root = exp_dir(exp)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "experiment.json", "w") as handle:
        json.dump(exp, handle, indent=2, sort_keys=True)
    shutil.copy2(exp["config"], root / "config.py")
    with open(root / "artifacts.txt", "w") as handle:
        handle.write("matrix={}\n".format(matrix_path))
        handle.write("config={}\n".format(exp["config"]))
        handle.write("checkpoint={}\n".format(exp["checkpoint"]))
        handle.write("checkpoint_sha256={}\n".format(sha256(exp["checkpoint"])))
        if exp.get("routes") and os.path.isfile(exp["routes"]):
            handle.write("routes={}\n".format(exp["routes"]))
            handle.write("routes_sha256={}\n".format(sha256(exp["routes"])))
    git = subprocess.run(["git", "status", "--short"], cwd=str(REPO_ROOT), text=True, capture_output=True)
    (root / "git_status.txt").write_text(git.stdout + git.stderr)


def command_text(command, env):
    prefixes = [
        "CUDA_VISIBLE_DEVICES={}".format(shlex.quote(env["CUDA_VISIBLE_DEVICES"])),
        "MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES={}".format(env["MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES"]),
        "MINDDRIVE_CF_REPLACE_DECISION={}".format(env["MINDDRIVE_CF_REPLACE_DECISION"]),
    ]
    return " ".join(prefixes + [shlex.join(command)])


def run_command(command, env, log_path, args):
    print(command_text(command, env), flush=True)
    if args.dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        result = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError("Command failed ({}); see {}".format(result.returncode, log_path))


def ensure_target(path, args):
    if not path.exists():
        return True
    if args.overwrite:
        return True
    if args.resume:
        print("resume/skip existing", path)
        return False
    raise FileExistsError("Output exists: {} (use --resume or --overwrite)".format(path))


def stage_smoke(exp, args):
    if not exp["has_counterfactual"]:
        print("{}: baseline has no CF diagnostics; validate only".format(exp["id"]))
        return
    out = exp_dir(exp) / "smoke" / "scan.json"
    if not ensure_target(out, args):
        return
    cmd = [
        args.python, "tools/scan_live_cf_candidate_differences.py",
        "--config", exp["config"], "--checkpoint", exp["checkpoint"],
        "--infos", exp["infos"], "--start", str(exp["smoke_start"]),
        "--end", str(exp["smoke_end"]), "--out", str(out),
        "--seed", str(exp["seed"]), "--device-id", "0", "--progress-every", "5",
    ]
    run_command(cmd, experiment_env(exp), out.parent / "run.log", args)
    if not args.dry_run:
        payload = json.loads(out.read_text())
        if payload.get("failures"):
            raise RuntimeError("{} smoke failures: {}".format(exp["id"], payload["failures"]))
        expected = exp.get("expected_candidates")
        if expected is not None:
            for row in payload["records"]:
                # affected_counts has exactly one entry per candidate.
                if len(row["affected_counts"]) != expected:
                    raise RuntimeError("{} expected K={}, got {} at index {}".format(
                        exp["id"], expected, len(row["affected_counts"]), row["index"]))
        if exp["id"] == "A2" and any(row["agent_max_diff"] != 0.0 for row in payload["records"]):
            raise RuntimeError("A2 invariant failed: candidate agent futures are not unified")


def stage_open_loop(exp, args):
    out = exp_dir(exp) / "open_loop" / "results.pkl"
    if not ensure_target(out, args):
        return
    cmd = [
        args.python, "adzoo/minddrive/test.py", exp["config"], exp["checkpoint"],
        "--out", str(out), "--seed", str(exp["seed"]), "--deterministic",
    ]
    run_command(cmd, experiment_env(exp), out.parent / "run.log", args)


def stage_analyze(exp, args):
    results = exp_dir(exp) / "open_loop" / "results.pkl"
    if not results.exists() and not args.dry_run:
        raise FileNotFoundError("Run open_loop first: {}".format(results))
    if not exp["has_counterfactual"]:
        print("{}: skipping CF-specific analyzers".format(exp["id"]))
        return
    commands = [
        ([args.python, "tools/analyze_cf_predictions.py", "--results", str(results), "--infos", exp["infos"],
          "--out-dir", str(results.parent / "trajectory")], results.parent / "analyze_trajectory.log"),
        ([args.python, "tools/eval_counterfactual_risk.py", "--cf", str(results),
          "--out-dir", str(results.parent / "risk")], results.parent / "analyze_risk.log"),
        ([args.python, "tools/plot_interaction_roc.py", "--results", str(results), "--infos", exp["infos"],
          "--out-dir", str(results.parent / "interaction_roc")], results.parent / "analyze_interaction.log"),
    ]
    for command, log in commands:
        run_command(command, experiment_env(exp), log, args)


def stage_closed_loop_jobs(exp, args):
    if not exp.get("routes") or not os.path.isfile(exp["routes"]):
        raise RuntimeError("closed_loop_jobs requires --routes <official evaluation XML>")
    root = exp_dir(exp) / "closed_loop"
    if args.dry_run:
        print("would generate closed-loop jobs in", root)
        return
    root.mkdir(parents=True, exist_ok=True)
    generated_cfg = root / "resolved_carla_config.py"
    generated_cfg.write_text(
        "_base_ = [{}]\nload_from = {!r}\n".format(repr(exp["config"]), exp["checkpoint"])
    )
    jobs = []
    for offset, seed in enumerate(exp["closed_loop_seeds"]):
        seed_dir = root / "seed_{}".format(seed)
        result_json = seed_dir / "result.json"
        jobs.append(dict(
            seed=seed,
            gpu=exp["gpu"],
            port=exp["base_port"] + offset * 300,
            tm_port=exp["base_tm_port"] + offset * 300,
            routes=exp["routes"],
            result_json=str(result_json),
            repetitions=exp["closed_loop_repetitions"],
        ))
    with open(root / "jobs.json", "w") as handle:
        json.dump(jobs, handle, indent=2)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd {}".format(shlex.quote(str(REPO_ROOT)))]
    for job in jobs:
        seed_dir = Path(job["result_json"]).parent
        lines.extend([
            "mkdir -p {}".format(shlex.quote(str(seed_dir))),
            "CUDA_VISIBLE_DEVICES={} MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES={} MINDDRIVE_CF_REPLACE_DECISION={} {} adzoo/minddrive/rollout.py {} --routes={} --checkpoint={} --port={} --traffic_manager_port={} --traffic-manager-seed={} --seed={} --deterministic --repetitions={} --resume --use_carla > {} 2>&1".format(
                shlex.quote(str(job["gpu"])),
                "1" if exp["use_action_expert_candidates"] else "0",
                "1" if exp["replace_decision"] else "0",
                shlex.quote(args.python),
                shlex.quote(str(generated_cfg)),
                shlex.quote(job["routes"]),
                shlex.quote(job["result_json"]),
                job["port"], job["tm_port"], job["seed"], job["seed"], job["repetitions"],
                shlex.quote(str(seed_dir / "run.log")),
            ),
        ])
    script = root / "run_jobs.sh"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    print("wrote", script)
    print("Review CARLA_ROOT/PYTHONPATH and run jobs one at a time unless separate GPUs/ports are assigned.")


def main():
    args = parse_args()
    common, experiments, matrix_path = load_matrix(args.matrix)
    if args.stage == "list":
        for key, value in experiments.items():
            print("{}: {}".format(key, value["name"]))
        return
    ids = select_experiments(args.experiment, experiments)
    for exp_id in ids:
        exp = resolved_experiment(exp_id, experiments[exp_id], common, args)
        validate_experiment(exp)
        print(json.dumps(exp, indent=2, sort_keys=True))
        if args.stage == "validate":
            continue
        if not args.dry_run:
            snapshot(exp, matrix_path)
        stages = ["smoke", "open_loop", "analyze"] if args.stage == "all" else [args.stage]
        for stage in stages:
            globals()["stage_" + stage](exp, args)


if __name__ == "__main__":
    main()
