#!/bin/bash -l
#
#  pos_neg_pseudo_prompts_tiling_submit_sam3.sh  --  SLURM batch script for
#      experiment pos_neg_pseudo_prompts_tiling
#      (SAM3, two rounds per image: S/M/L GT crops, then + positive / negative
#       self-prompt crops from SAM3's own round-1 predictions, tiling ON,
#       cluster port of pos_neg_pseudo_prompts_tiling.ipynb)
#
#  The image is split into overlapping tiles (1000px, 150px overlap). An exemplar
#  strip is composed above each tile (local background and feathering) and the
#  tile + strip is sent to SAM3 in batches of 4. ROUND 1: the strip holds the 3
#  size-based GT plants (smallest / median-closest / largest). From the NMS-merged
#  round-1 detections that do not sit on a prompt plant, the 2 highest-confidence
#  boxes become POSITIVE crops (label 1) and the 2 lowest-confidence boxes below
#  0.50 become NEGATIVE crops (label 0). ROUND 2: every tile again with S, M, L +
#  those crops; its output is final (skipped if no self-prompt exists).
#  Detections are filtered by target-region and plausibility, then mapped back to
#  full image coords. Evaluated on BOTH archives (AGS_Multi_Rumex + AgsSpringRumex).
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  pos_neg_pseudo_prompts_tiling_run_sam3.sh; PHASE 2 (the offline evaluation,
#  including the qualitative figures) runs once at the end, in the same job, over
#  everything that reached disk.
#
#  Resubmitting after the walltime is safe and expected: every finished image is
#  listed in the shard manifests and is skipped before any GPU work happens.
#
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="posneg_pseudo_tile_sam3"
#SBATCH --output=pos_neg_pseudo_prompts_tiling_sam3_%j.out
#SBATCH --error=pos_neg_pseudo_prompts_tiling_sam3_%j.err
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

chmod +x "${SCRIPT_DIR}/pos_neg_pseudo_prompts_tiling_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-pos_neg_pseudo_prompts_tiling}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (S/M/L + 2 pos / 2 neg self-prompts, tiling ON) ---
export N_SELF_POSITIVES="${N_SELF_POSITIVES:-2}"
export N_SELF_NEGATIVES="${N_SELF_NEGATIVES:-2}"
export SELF_PROMPT_EXCLUDE_IOU="${SELF_PROMPT_EXCLUDE_IOU:-0.50}"
export UNRELIABLE_MAX_SCORE="${UNRELIABLE_MAX_SCORE:-0.50}"
export USE_TILING="${USE_TILING:-true}"
export TILE_SIZE="${TILE_SIZE:-1000}"
export OVERLAP="${OVERLAP:-150}"
export CACHE_TILES_IN_MEMORY="${CACHE_TILES_IN_MEMORY:-true}"

export THRESHOLD="${THRESHOLD:-0.30}"             # CONFIDENCE_THRESHOLD (SAM3 + operating point)
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DTYPE="${DTYPE:-bfloat16}"

export STRIP_MARGIN="${STRIP_MARGIN:-6}"
export FEATHER_WIDTH="${FEATHER_WIDTH:-8}"
export BACKGROUND_BLUR_RADIUS="${BACKGROUND_BLUR_RADIUS:-1.5}"
export MAX_STRIP_HEIGHT_FRACTION="${MAX_STRIP_HEIGHT_FRACTION:-none}"

export MIN_FILL_RATIO="${MIN_FILL_RATIO:-0.15}"
export MAX_AREA_FRACTION="${MAX_AREA_FRACTION:-0.80}"
export EDGE_MARGIN="${EDGE_MARGIN:-5}"
export TILE_REGION_MIN_FRACTION="${TILE_REGION_MIN_FRACTION:-0.50}"

# The notebook uses ONE confidence threshold (THRESHOLD above) for inference,
# self-prompt selection and the operating point.
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "round 1: S/M/L GT crops | round 2: + ${N_SELF_POSITIVES} pos / ${N_SELF_NEGATIVES} neg self-prompt crops  tiling=${USE_TILING} (tile=${TILE_SIZE}, overlap=${OVERLAP})"
echo "dtype=${DTYPE}  batch_size=${BATCH_SIZE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${THRESHOLD}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# pos_neg_pseudo_prompts_tiling_run_sam3.sh sets. Do NOT rely on the compute node reaching
# PyPI -- it cannot. If the image's transformers is too old for Sam3Model, build a dedicated
# image (pos_neg_pseudo_prompts_tiling_HOW_TO_RUN.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/pos_neg_pseudo_prompts_tiling_run_sam3.sh run"