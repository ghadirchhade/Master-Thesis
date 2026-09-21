#!/bin/bash -l
#
#  pos_neg_exemplars_text_tiling_submit_sam3.sh  --  SLURM batch script for
#      experiment pos_neg_exemplars_text_tiling
#      (SAM3 prompted with K positive GT boxes + J generated negative boxes + text,
#       tiling ON,
#       cluster port of pos_neg_exemplars_text_tiling.ipynb)
#
#  The image is split into overlapping tiles (1000px, 150px overlap). For each 
#  (image, anchor), an exemplar strip is composed above each tile (2 positives, 
#  then 3 negatives, with local background and feathering). The tile + strip is 
#  sent to SAM3 in batches of 4 together with the text prompt "Rumex obtusifolius".
#  Detections are filtered by target-region and plausibility, then mapped back to 
#  full image coords.
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  pos_neg_exemplars_text_tiling_run_sam3.sh; PHASE 2 (the offline evaluation,
#  including the qualitative figures) runs once at the end, in the same job, over
#  everything that reached disk.
#
#  Resubmitting after the walltime is safe and expected: every finished
#  (image, anchor) pair is listed in the shard manifests and is skipped before
#  any GPU work happens.
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="postile_text_sam3"
#SBATCH --output=pos_neg_exemplars_text_tiling_sam3_%j.out
#SBATCH --error=pos_neg_exemplars_text_tiling_sam3_%j.err
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

chmod +x "${SCRIPT_DIR}/pos_neg_exemplars_text_tiling_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch pos_neg_exemplars_text_tiling_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-pos_neg_exemplars_text_tiling}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (2 pos + 3 neg + text, tiling ON) --------
export K_POSITIVES="${K_POSITIVES:-2}"
export J_NEGATIVES="${J_NEGATIVES:-3}"
export TEXT_PROMPT="${TEXT_PROMPT:-Rumex obtusifolius}"
export USE_TILING="${USE_TILING:-true}"
export TILE_SIZE="${TILE_SIZE:-1000}"
export OVERLAP="${OVERLAP:-150}"
export CACHE_TILES_IN_MEMORY="${CACHE_TILES_IN_MEMORY:-true}"

export NEG_NUM_CANDIDATES="${NEG_NUM_CANDIDATES:-2000}"
export NEG_MIN_GAP_PX="${NEG_MIN_GAP_PX:-10}"
export NEG_MAX_GAP_FACTOR="${NEG_MAX_GAP_FACTOR:-1.0}"

export THRESHOLD="${THRESHOLD:-0.30}"             # SAM3 runs at this score
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DTYPE="${DTYPE:-bfloat16}"

export STRIP_MARGIN="${STRIP_MARGIN:-6}"
export FEATHER_WIDTH="${FEATHER_WIDTH:-8}"
export BACKGROUND_BLUR_RADIUS="${BACKGROUND_BLUR_RADIUS:-1.5}"

export MIN_FILL_RATIO="${MIN_FILL_RATIO:-0.15}"
export MAX_AREA_FRACTION="${MAX_AREA_FRACTION:-0.80}"
export EDGE_MARGIN="${EDGE_MARGIN:-5}"
export TILE_REGION_MIN_FRACTION="${TILE_REGION_MIN_FRACTION:-0.50}"

# The notebook keeps the operating point EQUAL to the inference threshold.
export OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompts=${K_POSITIVES} pos + ${J_NEGATIVES} neg + text '${TEXT_PROMPT}'  tiling=ON (tile=${TILE_SIZE}, overlap=${OVERLAP})"
echo "dtype=${DTYPE}  batch_size=${BATCH_SIZE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# pos_neg_exemplars_text_tiling_run_sam3.sh sets. Do NOT rely on the compute node reaching
# PyPI -- it cannot. If the image's transformers is too old for Sam3Model, build a dedicated
# image (pos_neg_exemplars_text_tiling_HOW_TO_RUN.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/pos_neg_exemplars_text_tiling_run_sam3.sh run"