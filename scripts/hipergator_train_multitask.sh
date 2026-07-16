#!/bin/bash
#SBATCH --job-name=probe-mt
#SBATCH --output=probe_multitask_%j.out
#SBATCH --error=probe_multitask_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --requeue
#
# HyperGator: auto-resume multitask PROBE after TIMEOUT / preemption.
# Edit paths below, then:  sbatch scripts/hipergator_train_multitask.sh
#
# Training rewrites output_dir/last_checkpoint.pt after every epoch
# (atomic replace). Re-submit the same job (or rely on --requeue) and
# this script will call --resume automatically when that file exists.

set -euo pipefail

# ---- edit these ----
REPO_DIR="${REPO_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/probe_mace_multitask_outputs}"
# Optional: persist MACE cache across jobs (recommended on HPG)
# MACE_CACHE_DIR="${OUTPUT_DIR}/mace_cache"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. --enable-cueq --lambda-force-mol 0.3
# --------------------

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

module load conda 2>/dev/null || true
# module load cuda 2>/dev/null || true
# conda activate your_env

LAST_CKPT="${OUTPUT_DIR}/last_checkpoint.pt"

CMD=(python train_mace_multitask.py --output-dir "$OUTPUT_DIR")

if [[ -f "$LAST_CKPT" ]]; then
  echo "[$(date)] Resuming from $LAST_CKPT"
  CMD+=(--resume "$LAST_CKPT")
else
  echo "[$(date)] No checkpoint found — starting fresh"
fi

if [[ -n "${MACE_CACHE_DIR:-}" ]]; then
  CMD+=(--mace-cache-dir "$MACE_CACHE_DIR")
fi

# shellcheck disable=SC2206
CMD+=($EXTRA_ARGS)

echo "[$(date)] Running: ${CMD[*]}"
"${CMD[@]}"
echo "[$(date)] Done."
