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
#
#   1. Edit CONDA_ENV / paths below
#   2. sbatch scripts/hipergator_train_multitask.sh
#
# Re-submit the same job after TIMEOUT; if last_checkpoint.pt exists,
# training resumes automatically.

set -euo pipefail

# ---- edit these ----
REPO_DIR="${REPO_DIR:-$PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/probe_mace_multitask_outputs}"

# Path to your conda env (HPG recommended: put env/bin on PATH)
# Examples:
#   CONDA_ENV=/blue/yourgroup/$USER/conda/envs/probe
#   CONDA_ENV=/home/$USER/.conda/envs/probe
CONDA_ENV="${CONDA_ENV:-/blue/yourgroup/$USER/conda/envs/probe}"

# Optional: persist MACE cache across jobs (recommended)
# export MACE_CACHE_DIR="${OUTPUT_DIR}/mace_cache"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # e.g. --enable-cueq --lambda-force-mol 0.3
# --------------------

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

# Avoid nounset failures in conda activate.d MKL hooks (set -u).
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

module purge
module load conda
# module load cuda/12.4.0   # uncomment if your env needs a CUDA module

if [[ ! -d "${CONDA_ENV}/bin" ]]; then
  echo "ERROR: CONDA_ENV bin not found: ${CONDA_ENV}/bin" >&2
  echo "Set CONDA_ENV to your env path before sbatch." >&2
  exit 1
fi
# UFRC best practice for jobs: prepend env/bin instead of conda activate
export PATH="${CONDA_ENV}/bin:${PATH}"
hash -r
echo "[$(date)] Using python: $(which python)"
python -V

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
