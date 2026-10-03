#!/bin/bash -l
#
#  high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh  --  SLURM batch script for the
#      experiment high_confidence_pseudo_prompts_no_tiling_yoloe
#      (YOLOE visual-prompted Rumex detection, TWO rounds per image: S/M/L GT
#       exemplars, then S/M/L + the highest-confidence round-1 predictions as
#       self-prompts; NO tiling: whole image resized to MAX_DIM,
#       cluster port of E01_single_image_pseudo_prompts_YOLOE_11_no_tiling.ipynb)
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh; PHASE 2 (the offline evaluation) runs once at the end,
#  in the same job, over everything that reached disk.
#
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="hcpp_notile_yoloe"
#SBATCH --output=high_confidence_pseudo_prompts_no_tiling_yoloe_%j.out
#SBATCH --error=high_confidence_pseudo_prompts_no_tiling_yoloe_%j.err
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

chmod +x "${SCRIPT_DIR}/high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  USE_FP16=0 sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-high_confidence_pseudo_prompts_no_tiling_yoloe}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export YOLOE_WEIGHTS_NAME="${YOLOE_WEIGHTS_NAME:-yoloe-11l-seg.pt}"
export YOLOE_WEIGHTS="${YOLOE_WEIGHTS:-${SCRATCH}/yoloe_weights/${YOLOE_WEIGHTS_NAME}}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (S/M/L + self-prompts, NO tiling) ----------
export N_SELF_PROMPTS="${N_SELF_PROMPTS:-2}"
export SELF_PROMPT_EXCLUDE_IOU="${SELF_PROMPT_EXCLUDE_IOU:-0.50}"
export MAX_DIM="${MAX_DIM:-1024}"                 # whole image resized to this
export IMGSZ="${IMGSZ:-${MAX_DIM}}"
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
echo "prompts=S/M/L + ${N_SELF_PROMPTS} self-prompts  no tiling (MAX_DIM=${MAX_DIM})  fp16=${USE_FP16}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "weights=${YOLOE_WEIGHTS}"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + ultralytics (which contains YOLOE). Packages the image does
# not ship (supervision) are picked up from $PYEXTRA via PYTHONPATH, which
# high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh sets. Do NOT rely on the compute node reaching PyPI or
# GitHub -- it cannot. If the image's ultralytics is too old for YOLOE visual prompts,
# see high_confidence_pseudo_prompts_no_tiling_HOW_TO_RUN_yoloe.md, "Problem A".
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh run"
