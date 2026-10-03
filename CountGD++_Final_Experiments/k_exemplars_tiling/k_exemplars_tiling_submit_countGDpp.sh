#!/bin/bash -l
#
#  k_exemplars_tiling_submit_countGDpp.sh  --  SLURM batch script for the
#      experiment k_exemplars_tiling_countGDpp
#      (CountGD++ k-EXEMPLAR-prompted Rumex detection, k = 3, tiling ON: the k
#       crops are k boxes in ONE exemplar mosaic on real background texture)
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  k_exemplars_tiling_run_countGDpp.sh (one shard per GPU, ONE tile per forward
#  pass); PHASE 2 (the offline evaluation) runs once at the end, in the same job,
#  over everything that reached disk.
#
#  Before the first submission: 'download' on a login node, then 'build' inside
#  the container (k_exemplars_tiling_HOW_TO_RUN_countGDpp.md, Steps 3-5).
#  Both are shared by every CountGD++ experiment: if they were already done for
#  1_exemplars_tiling_countGDpp, just submit.
#

#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="kex_tiling_cgdpp"
#SBATCH --output=k_exemplars_tiling_countGDpp_%j.out
#SBATCH --error=k_exemplars_tiling_countGDpp_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=hassan@pixtell.ch
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR="$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp"

# Everything the job writes goes to scratch.
cwd="${SCRATCH}/experiments/countgdpp"
mkdir -p "${cwd}"
cd "${cwd}"

chmod +x "${SCRIPT_DIR}/k_exemplars_tiling_run_countGDpp.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  NUM_GPUS=2 sbatch k_exemplars_tiling_submit_countGDpp.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-k_exemplars_tiling_countGDpp}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export COUNTGD_REPO="${COUNTGD_REPO:-${SCRATCH}/CountGDPlusPlus}"
export COUNTGD_PYEXTRA="${COUNTGD_PYEXTRA:-${SCRATCH}/pyextra_countgdpp}"   # transformers<5, timm, ...
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (k = 3 exemplars, tiling ON) --------------
export N_EXEMPLARS="${N_EXEMPLARS:-3}"
export STRIP_MARGIN="${STRIP_MARGIN:-6}"                   # exemplar mosaic layout
export FEATHER_WIDTH="${FEATHER_WIDTH:-8}"                 # (same values as the SAM3 strip)
export BACKGROUND_BLUR_RADIUS="${BACKGROUND_BLUR_RADIUS:-1.5}"
export USE_TILING="${USE_TILING:-1}"
export TILE_SIZE="${TILE_SIZE:-1000}"
export OVERLAP="${OVERLAP:-150}"
export THRESHOLD="${THRESHOLD:-0.30}"             # CountGD++ runs ONCE at this score
export BATCH_SIZE="${BATCH_SIZE:-1}"              # fixed: one tile per forward pass
export DTYPE="${DTYPE:-float32}"                  # the notebook (USE_FP16 = False)
export TEXT_PROMPT="${TEXT_PROMPT:-}"             # exemplar only
export EXEMPLAR_SCALE_MODE="${EXEMPLAR_SCALE_MODE:-match_tile}"
# The notebook keeps the operating point EQUAL to the inference threshold.
export OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompts=${N_EXEMPLARS} (one exemplar mosaic)  tiling=${USE_TILING}  dtype=${DTYPE}  batch=${BATCH_SIZE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "countgd_repo=${COUNTGD_REPO}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch. Everything CountGD++ adds on top (transformers<5, timm, addict,
# yapf, pycocotools, termcolor, supervision) is picked up from $COUNTGD_PYEXTRA via
# PYTHONPATH, which k_exemplars_tiling_run_countGDpp.sh sets. The compiled deformable-
# attention op was built against THIS container's torch ('build' mode); with another
# container, rebuild it. Do NOT rely on the compute node reaching PyPI -- it cannot.
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/k_exemplars_tiling_run_countGDpp.sh run"
