#!/bin/bash -l
#
#  text_only_with_tiling_submit_sam3.sh  --  SLURM batch script for experiment
#      text_only_with_tiling
#      (SAM3 TEXT-ONLY prompted Rumex detection, tiling ON, NO visual prompts,
#       cluster port of text_only_with_tiling.ipynb)
#
#  Every UAV image is split into overlapping tiles (TILE_SIZE=1000, OVERLAP=150).
#  Every RAW tile is sent to SAM3 in batches of BATCH_SIZE=4 with ONLY a text
#  prompt (e.g. "Rumex obtusifolius"). No exemplar box, no visual prompts,
#  no exemplar strip, no plausibility filter. Tile detections are shifted back
#  to original-image coordinates, and duplicates from overlapping tiles are
#  merged by NMS (offline).
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  text_only_with_tiling_run_sam3.sh; PHASE 2 (the offline evaluation, including
#  the qualitative figures) runs once at the end, in the same job, over
#  everything that reached disk.
#
#  Resubmitting after the walltime is safe and expected: every finished image
#  is listed in the shard manifests and is skipped before any GPU work happens.
#
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="txt_only_tile_sam3"
#SBATCH --output=text_only_with_tiling_sam3_%j.out
#SBATCH --error=text_only_with_tiling_sam3_%j.err
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

chmod +x "${SCRIPT_DIR}/text_only_with_tiling_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch text_only_with_tiling_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-text_only_with_tiling}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (text only, tiling ON) --------------------
export TEXT_PROMPT="${TEXT_PROMPT:-Rumex obtusifolius}"
export N_EXEMPLARS="${N_EXEMPLARS:-0}"
export TILE_SIZE="${TILE_SIZE:-1000}"             # tile width/height in pixels
export OVERLAP="${OVERLAP:-150}"                  # overlap between neighbouring tiles
export BATCH_SIZE="${BATCH_SIZE:-4}"              # tiles per forward pass
export CACHE_TILES_IN_MEMORY="${CACHE_TILES_IN_MEMORY:-True}"
export THRESHOLD="${THRESHOLD:-0.30}"             # SAM3 runs ONCE per tile at this score
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"
export DTYPE="${DTYPE:-bfloat16}"

# The notebook keeps the operating point EQUAL to the inference threshold.
export OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "text_prompt='${TEXT_PROMPT}'  prompts=${N_EXEMPLARS}  tiling=ON  tile_size=${TILE_SIZE}  overlap=${OVERLAP}  batch_size=${BATCH_SIZE}  dtype=${DTYPE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# text_only_with_tiling_run_sam3.sh sets. Do NOT rely on the compute node reaching PyPI -- it
# cannot. If the image's transformers is too old for Sam3Model, build a dedicated image
# (text_only_with_tiling_HOW_TO_RUN.md, "Problem A").

srun \
  --environment=yolo26 \
  --container-workdir="$PWD" \
  --cpu-bind=cores \
  bash -c "${SCRIPT_DIR}/text_only_with_tiling_run_sam3.sh run"