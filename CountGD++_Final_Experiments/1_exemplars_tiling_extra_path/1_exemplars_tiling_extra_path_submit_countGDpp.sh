#!/bin/bash -l
#
#  1_exemplars_tiling_extra_path_submit_countGDpp.sh  --  SLURM batch script for the
#      experiment 1_exemplars_tiling_extra_path_countGDpp
#      (CountGD++ SINGLE-exemplar-prompted Rumex detection, tiling ON with
#       1536 px tiles / 384 px overlap + the downsampled global-context pass)
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  1_exemplars_tiling_extra_path_run_countGDpp.sh (one shard per GPU, ONE tile per forward
#  pass); PHASE 2 (the offline evaluation) runs once at the end, in the same job,
#  over everything that reached disk. A shard stopped by the RAM guard exits with
#  code 3 (FAIL email): just resubmit, it resumes.
#
#  Before the first submission: 'download' on a login node, then 'build' inside
#  the container (1_exemplars_tiling_extra_path_HOW_TO_RUN_countGDpp.md, Steps 3-5).
#  Both are shared by every CountGD++ experiment: if they were already done for
#  1_exemplars_tiling_countGDpp, just submit.
#

#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="1ex_tilxp_cgdpp"
#SBATCH --output=1_exemplars_tiling_extra_path_countGDpp_%j.out
#SBATCH --error=1_exemplars_tiling_extra_path_countGDpp_%j.err
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

chmod +x "${SCRIPT_DIR}/1_exemplars_tiling_extra_path_run_countGDpp.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  NUM_GPUS=2 sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-1_exemplars_tiling_extra_path_countGDpp}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export COUNTGD_REPO="${COUNTGD_REPO:-${SCRATCH}/CountGDPlusPlus}"
export COUNTGD_PYEXTRA="${COUNTGD_PYEXTRA:-${SCRATCH}/pyextra_countgdpp}"   # transformers<5, timm, ...
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (1 exemplar, tiling + extra path) ---------
export N_EXEMPLARS="${N_EXEMPLARS:-1}"
export USE_TILING="${USE_TILING:-1}"
export TILE_SIZE="${TILE_SIZE:-1536}"
export OVERLAP="${OVERLAP:-384}"
export ADD_GLOBAL_CONTEXT_PASS="${ADD_GLOBAL_CONTEXT_PASS:-1}"   # the "extra path"
export GLOBAL_DOWNSCALE="${GLOBAL_DOWNSCALE:-2}"
export MEM_STOP_THRESHOLD_PCT="${MEM_STOP_THRESHOLD_PCT:-70}"    # RAM guard
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
echo "prompts=${N_EXEMPLARS}  tiling=${USE_TILING}  dtype=${DTYPE}  batch=${BATCH_SIZE}  gpus=${NUM_GPUS}"
echo "tile=${TILE_SIZE}  overlap=${OVERLAP}  global_pass=${ADD_GLOBAL_CONTEXT_PASS} (x1/${GLOBAL_DOWNSCALE})  mem_stop=${MEM_STOP_THRESHOLD_PCT}%"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "countgd_repo=${COUNTGD_REPO}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch. Everything CountGD++ adds on top (transformers<5, timm, addict,
# yapf, pycocotools, termcolor, supervision) is picked up from $COUNTGD_PYEXTRA via
# PYTHONPATH, which 1_exemplars_tiling_extra_path_run_countGDpp.sh sets. The compiled deformable-
# attention op was built against THIS container's torch ('build' mode); with another
# container, rebuild it. Do NOT rely on the compute node reaching PyPI -- it cannot.
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/1_exemplars_tiling_extra_path_run_countGDpp.sh run"
