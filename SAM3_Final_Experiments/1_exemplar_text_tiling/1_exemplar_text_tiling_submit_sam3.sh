#!/bin/bash -l
#
#  1_exemplar_text_tiling_submit_sam3.sh  --  SLURM batch script for the
#      experiment 1_exemplar_text_tiling
#      (SAM3 SINGLE-exemplar-prompted Rumex detection, tiling ON,
#       cluster port of 1_exemplar_text_tiling.ipynb)
#
#  Every composed tile gets the exemplar box(es) AND the text prompt TEXT_PROMPT
#  in the same forward pass. Same pipeline as 1_exemplar_with_tiling otherwise,
#  so the two form a clean pair differing only in the text prompt. The operating
#  point is 0.30 (equal to the inference threshold), as in the notebook.
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  1_exemplar_text_tiling_run_sam3.sh; PHASE 2 (the offline evaluation) runs once at the end,
#  in the same job, over everything that reached disk.
#
#  Resubmitting after the 24 h walltime is safe and expected: every finished
#  (image, anchor) pair is listed in the shard manifests and is skipped before
#  any GPU work happens.

#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="1ex_text_tiling_sam3"
#SBATCH --output=1_exemplar_text_tiling_sam3_%j.out
#SBATCH --error=1_exemplar_text_tiling_sam3_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=hassan@pixtell.ch
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR="$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3"

# Everything the job writes goes to scratch.
cwd="${SCRATCH}/experiments/sam3"
mkdir -p "${cwd}"
cd "${cwd}"

chmod +x "${SCRIPT_DIR}/1_exemplar_text_tiling_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch 1_exemplar_text_tiling_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-1_exemplar_text_tiling}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (1 exemplar, tiling ON) -------------------
export N_EXEMPLARS="${N_EXEMPLARS:-1}"
export TEXT_PROMPT="${TEXT_PROMPT:-Rumex obtusifolius}"   # sent with the boxes, same call
export USE_TILING="${USE_TILING:-1}"
export TILE_SIZE="${TILE_SIZE:-1000}"
export OVERLAP="${OVERLAP:-150}"
export THRESHOLD="${THRESHOLD:-0.30}"             # SAM3 runs ONCE at this score
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DTYPE="${DTYPE:-bfloat16}"
# The notebook keeps the operating point EQUAL to the inference threshold.
export OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompts=${N_EXEMPLARS} exemplar(s) + text  tiling=${USE_TILING}  dtype=${DTYPE}  gpus=${NUM_GPUS}"
echo "text_prompt='${TEXT_PROMPT}'"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# 1_exemplar_text_tiling_run_sam3.sh sets. Do NOT rely on the compute node reaching PyPI -- it cannot. If the
# image's transformers is too old for Sam3Model, build a dedicated image
# (1_exemplar_text_tiling_HOW_TO_RUN.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/1_exemplar_text_tiling_run_sam3.sh run"
