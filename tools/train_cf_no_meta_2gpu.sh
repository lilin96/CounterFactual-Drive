#!/usr/bin/env bash
set -euo pipefail

cd /home/lilin/MindDrive

# Physical GPUs 0 and 3 become local CUDA ranks 0 and 1.
export CUDA_VISIBLE_DEVICES=0,3
export OMP_NUM_THREADS=8
export PYTHONPATH=/home/lilin/MindDrive:${PYTHONPATH:-}

# These switches do not participate in forward_train, but keeping them fixed
# documents the candidate/decision policy used by the corresponding A1 eval.
export MINDDRIVE_CF_REPLACE_DECISION=1
export MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=1

# Parameter summary is always printed; set this to 1 only when all individual
# parameter names are needed in the log.
export MINDDRIVE_PRINT_PARAM_NAMES=0
export MINDDRIVE_LOG_FULL_MODEL=0

# WandB is enabled by the config.  Run `wandb login` first, or export
# WANDB_API_KEY outside this script.  Set WANDB_MODE=offline if required.
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=/home/lilin/MindDrive/work_dirs/wandb

mkdir -p /home/lilin/MindDrive/work_dirs/cf_no_meta_2gpu_effbs4_20ksteps
mkdir -p "${WANDB_DIR}"

/home/lilin/.conda/envs/MindDrive/bin/python -m torch.distributed.run \
  --nproc-per-node=2 \
  --master-port=29517 \
  adzoo/minddrive/train.py \
  adzoo/minddrive/configs/minddrive_qwen2_05b_train_cf_no_meta_2gpu.py \
  --launcher pytorch \
  --work-dir work_dirs/cf_no_meta_2gpu_effbs4_20ksteps \
  --load_from ckpts/minddrive_rltrain.pth \
  --seed 0 \
  --deterministic \
  2>&1 | tee work_dirs/cf_no_meta_2gpu_effbs4_20ksteps/train_console.log
