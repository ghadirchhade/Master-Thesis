#!/bin/bash -l
#
#  k_diverse_exemplars_tiling_submit_sam3.sh  --  SLURM batch script for the
#      experiment k_diverse_exemplars_tiling
#      (SAM3 prompted with K=3 GT boxes chosen by max-min DINOv2 embedding
#       diversity, tiling ON,
#       cluster port of k_diverse_exemplars_tiling.ipynb)
#
#  Once per image, at full resolution, every GT box is cropped (+10% context ->
#  square padding -> 224x224) and embedded with DINOv2; the triple whose closest
#  pair is as far apart as possible is selected. Those 3 crops are then pasted
#  into an exemplar strip above EVERY overlapping tile and their boxes inside the
#  strip are the SAM3 positive prompts. The selection is deterministic, so there
#  is exactly ONE run per image.
#
#  One node, 4 GPUs. PHASE 1 (DINOv2 + tiled SAM3 inference) is split across the
#  4 GPUs by k_diverse_exemplars_tiling_run_sam3.sh; PHASE 2 (the offline
#  evaluation, including the size-group table and the qualitative figures) runs
#  once at the end, in the same job, over everything that reached disk.
#
#  Resubmitting after the walltime is safe and expected: every finished image is
#  listed in the shard manifests and is skipped before any GPU work happens.

#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="kdiv_tiling_sam3"
#SBATCH --output=k_diverse_exemplars_tiling_sam3_%j.out
#SBATCH --error=k_diverse_exemplars_tiling_sam3_%j.err
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

chmod +x "${SCRIPT_DIR}/k_diverse_exemplars_tiling_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch k_diverse_exemplars_tiling_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-k_diverse_exemplars_tiling}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"   # holds SAM3 *and* DINOv2
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"    # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (3 diversity exemplars, tiling ON) --------
export K_EXEMPLARS="${K_EXEMPLARS:-3}"
export USE_TILING="${USE_TILING:-1}"
export TILE_SIZE="${TILE_SIZE:-1000}"
export OVERLAP="${OVERLAP:-150}"
export THRESHOLD="${THRESHOLD:-0.30}"             # SAM3 runs ONCE per tile at this score
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DTYPE="${DTYPE:-bfloat16}"                 # SAM3 dtype
# DINOv2 exemplar embeddings (CELL 14). fp32: the cosine distances are compared
# directly, so the embeddings are kept at full precision as in the notebook.
export DINOV2_MODEL_ID="${DINOV2_MODEL_ID:-facebook/dinov2-large}"
export DINOV2_DTYPE="${DINOV2_DTYPE:-float32}"
export DINOV2_BATCH_SIZE="${DINOV2_BATCH_SIZE:-16}"
export CROP_CONTEXT="${CROP_CONTEXT:-0.10}"
export DINOV2_INPUT_SIZE="${DINOV2_INPUT_SIZE:-224}"
export DIVERSITY_EXACT_MAX_GT="${DIVERSITY_EXACT_MAX_GT:-300}"
export EXEMPLAR_CROP_FOR_STRIP="${EXEMPLAR_CROP_FOR_STRIP:-embedding_crop}"

# The notebook keeps the operating point EQUAL to the inference threshold.
export OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompts=${K_EXEMPLARS} (max-min cosine diversity, ${DINOV2_MODEL_ID})"
echo "tiling=${USE_TILING}  tile=${TILE_SIZE}  overlap=${OVERLAP}  batch=${BATCH_SIZE}"
echo "sam3_dtype=${DTYPE}  dinov2_dtype=${DINOV2_DTYPE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# k_diverse_exemplars_tiling_run_sam3.sh sets. Do NOT rely on the compute node reaching PyPI
# -- it cannot, and it cannot reach the HuggingFace hub either, which is why BOTH
# facebook/sam3 and facebook/dinov2-large must already be in $HF_HOME. If the image's
# transformers is too old for Sam3Model, build a dedicated image
# (k_diverse_exemplars_tiling_HOW_TO_RUN.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/k_diverse_exemplars_tiling_run_sam3.sh run"