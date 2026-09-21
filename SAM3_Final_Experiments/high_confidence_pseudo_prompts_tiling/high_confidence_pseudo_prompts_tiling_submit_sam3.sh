#!/bin/bash -l
#
#  high_confidence_pseudo_prompts_tiling_submit_sam3.sh  --  SLURM batch script for the
#      experiment high_confidence_pseudo_prompts_tiling

#
#  Two SAM3 rounds per IMAGE (not per anchor): round 1 tiles the image with a
#  3-crop strip (size-based S / M / L GT plants), then round 2 tiles it again with
#  a 5-crop strip that adds the 2 highest-confidence round-1 predictions, cropped
#  from the full image. The operating point is 0.30, as the notebook sets it.
#
#  ~2 x n_tiles forward passes per image (about 140 at 1000/150), so this is the
#  most expensive of the pseudo-prompt pair - but still per image, not per GT box.
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  high_confidence_pseudo_prompts_tiling_run_sam3.sh; PHASE 2 (the offline evaluation) runs once at the end,
#  in the same job, over everything that reached disk.
#
#  Resubmitting after the 24 h walltime is safe and expected: every finished image
#  is listed in the shard manifests and is skipped before any GPU work happens.

#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="hc_pseudo_tiling"
#SBATCH --output=high_confidence_pseudo_prompts_tiling_sam3_%j.out
#SBATCH --error=high_confidence_pseudo_prompts_tiling_sam3_%j.err
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

chmod +x "${SCRIPT_DIR}/high_confidence_pseudo_prompts_tiling_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch high_confidence_pseudo_prompts_tiling_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-high_confidence_pseudo_prompts_tiling}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (1 exemplar, tiling ON) -------------------
export K_EXEMPLARS="${K_EXEMPLARS:-3}"            # size-based GT crops (S / M / L)
export N_SELF_PROMPTS="${N_SELF_PROMPTS:-2}"      # round-1 predictions cropped into round 2
export SELF_PROMPT_EXCLUDE_IOU="${SELF_PROMPT_EXCLUDE_IOU:-0.50}"
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
echo "round1=${K_EXEMPLARS} size-based GT crops  round2=+${N_SELF_PROMPTS} self-prompt crops"
echo "tiling=${USE_TILING} (${TILE_SIZE}/${OVERLAP})  dtype=${DTYPE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# high_confidence_pseudo_prompts_tiling_run_sam3.sh sets. Do NOT rely on the compute node reaching PyPI -- it cannot. If the
# image's transformers is too old for Sam3Model, build a dedicated image
# (high_confidence_pseudo_prompts_tiling_HOW_TO_RUN.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/high_confidence_pseudo_prompts_tiling_run_sam3.sh run"
