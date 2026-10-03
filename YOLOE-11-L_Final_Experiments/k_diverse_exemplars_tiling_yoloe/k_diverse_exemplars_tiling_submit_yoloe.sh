#!/bin/bash -l
#
#  k_diverse_exemplars_tiling_submit_yoloe.sh  --  SLURM batch script for the
#      experiment k_diverse_exemplars_tiling_yoloe
#      (YOLOE prompted with K=3 GT boxes chosen by max-min DINOv2 embedding
#       diversity (D1/D2/D3), tiling ON (1000 px tiles, 150 px overlap), ONE run
#       per image; YOLOE version of the SAM3 notebook k_diverse_exemplars_tiling)
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  k_diverse_exemplars_tiling_run_yoloe.sh; PHASE 2 (the offline evaluation) runs once at the end,
#  in the same job, over everything that reached disk.
#
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="kdiv_tiling_yoloe"
#SBATCH --output=k_diverse_exemplars_tiling_yoloe_%j.out
#SBATCH --error=k_diverse_exemplars_tiling_yoloe_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=hassan@pixtell.ch
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR="$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe"

# Everything the job writes goes to scratch.
cwd="${SCRATCH}/experiments/yoloe"
mkdir -p "${cwd}"
cd "${cwd}"

chmod +x "${SCRIPT_DIR}/k_diverse_exemplars_tiling_run_yoloe.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  USE_FP16=0 sbatch k_diverse_exemplars_tiling_submit_yoloe.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-k_diverse_exemplars_tiling_yoloe}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export YOLOE_WEIGHTS_NAME="${YOLOE_WEIGHTS_NAME:-yoloe-11l-seg.pt}"
export YOLOE_WEIGHTS="${YOLOE_WEIGHTS:-${SCRATCH}/yoloe_weights/${YOLOE_WEIGHTS_NAME}}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (DINOv2-diverse exemplars, tiling ON) ------
export K_EXEMPLARS="${K_EXEMPLARS:-3}"             # D1 + D2 + D3
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"   # holds DINOv2
# DINOv2 exemplar embeddings (CELL 14). fp32: the cosine distances are compared
# directly, so they are kept at full precision.
export DINOV2_MODEL_ID="${DINOV2_MODEL_ID:-facebook/dinov2-large}"
export DINOV2_DTYPE="${DINOV2_DTYPE:-float32}"
export DINOV2_BATCH_SIZE="${DINOV2_BATCH_SIZE:-16}"
export USE_TILING="${USE_TILING:-1}"
export TILE_SIZE="${TILE_SIZE:-1000}"
export OVERLAP="${OVERLAP:-150}"
export IMGSZ="${IMGSZ:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export THRESHOLD="${THRESHOLD:-0.30}"             # YOLOE runs ONCE at this score
export PREDICT_NMS_IOU="${PREDICT_NMS_IOU:-0.90}" # in-predictor NMS, permissive
export USE_FP16="${USE_FP16:-1}"

# The notebook keeps the operating point EQUAL to the inference threshold.
export OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompts=${K_EXEMPLARS} (max-min cosine diversity, ${DINOV2_MODEL_ID})  tiling=${USE_TILING} (${TILE_SIZE}/${OVERLAP})  fp16=${USE_FP16}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "weights=${YOLOE_WEIGHTS}"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + ultralytics (which contains YOLOE). Packages the image does
# not ship (supervision) are picked up from $PYEXTRA via PYTHONPATH, which
# k_diverse_exemplars_tiling_run_yoloe.sh sets. Do NOT rely on the compute node reaching PyPI or
# GitHub -- it cannot. If the image's ultralytics is too old for YOLOE visual prompts,
# see k_diverse_exemplars_tiling_HOW_TO_RUN_yoloe.md, "Problem A".
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/k_diverse_exemplars_tiling_run_yoloe.sh run"
