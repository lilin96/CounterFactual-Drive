#!/usr/bin/env python
"""Train the counterfactual head with ego meta-action augmentation.

This wrapper keeps the original MindDrive training script untouched.  It simply
selects the meta-augmented counterfactual config and forwards all extra CLI
arguments to ``adzoo/minddrive/train.py``.
"""

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="adzoo/minddrive/configs/minddrive_qwen2_05b_train_cf_meta_aug.py",
    )
    parser.add_argument("--checkpoint", default="ckpts/minddrive_rltrain.pth")
    parser.add_argument("--work-dir", default="work_dirs/cf_meta_aug_base_20k")
    parser.add_argument("--launcher", default="none", choices=["none", "pytorch", "slurm", "mpi"])
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to adzoo/minddrive/train.py, e.g. --cfg-options runner.max_iters=5000",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        "-u",
        "adzoo/minddrive/train.py",
        args.config,
        "--work-dir",
        args.work_dir,
    ]
    if args.launcher != "none":
        cmd += ["--launcher", args.launcher]
    if args.checkpoint:
        cmd += ["--load_from", args.checkpoint]
    cmd += args.extra_args

    env = os.environ.copy()
    print("Running:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
