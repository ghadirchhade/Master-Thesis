#!/bin/bash -l
#
#  text_only_no_tiling_submit_countGDpp.sh  --  SLURM batch script for the
#      experiment text_only_no_tiling_countGDpp
#      (CountGD++ TEXT-ONLY prompt " rumex obtusifolius ": no exemplar, ONE run
#       per image, all_gt evaluation; NO tiling: the whole image at CountGD++'s
#       own resize)
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  text_only_no_tiling_run_countGDpp.sh (one shard per GPU, ONE tile per forward
#  pass); PHASE 2 (the offline evaluation) runs once at the end, in the same job,
#  over everything that reached disk.
#
#  Before the first submission: 'download' on a login node, then 'build' inside
#  the container (text_only_no_tiling_HOW_TO_RUN_countGDpp.md, Steps 3-5).
#  Both are shared by every CountGD++ experiment: if they were already done for
#  1_exemplars_tiling_countGDpp, just submit.
#

#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="txt_notile_cgdpp"
#SBATCH --output=text_only_no_tiling_countGDpp_%j.out
#SBATCH --error=text_only_no_tiling_countGDpp_%j.err
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

chmod +x "${SCRIPT_DIR}/text_only_no_tiling_run_countGDpp.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  NUM_GPUS=2 sbatch text_only_no_tiling_submit_countGDpp.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-text_only_no_tiling_countGDpp}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export COUNTGD_REPO="${COUNTGD_REPO:-${SCRATCH}/CountGDPlusPlus}"
export COUNTGD_PYEXTRA="${COUNTGD_PYEXTRA:-${SCRATCH}/pyextra_countgdpp}"   # transformers<5, timm, ...
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (text only, NO tiling) -------------------
export THRESHOLD="${THRESHOLD:-0.30}"             # CountGD++ runs ONCE at this score
export BATCH_SIZE="${BATCH_SIZE:-1}"              # fixed: one image per forward pass
export DTYPE="${DTYPE:-float32}"                  # the notebook (USE_FP16 = False)
export TEXT_PROMPT="${TEXT_PROMPT:- rumex obtusifolius }"   # the ONLY prompt
export MAX_AREA_FRACTION="${MAX_AREA_FRACTION:-0.80}"     # plausibility filter, as in the
export EDGE_MARGIN="${EDGE_MARGIN:-5}"                    # other CountGD++ experiments
# The notebook keeps the operating point EQUAL to the inference threshold.
export OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompt=text only '${TEXT_PROMPT}' (one run per image)  NO tiling (whole image, CountGD++ resize)  dtype=${DTYPE}  batch=${BATCH_SIZE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "countgd_repo=${COUNTGD_REPO}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch. Everything CountGD++ adds on top (transformers<5, timm, addict,
# yapf, pycocotools, termcolor, supervision) is picked up from $COUNTGD_PYEXTRA via
# PYTHONPATH, which text_only_no_tiling_run_countGDpp.sh sets. The compiled deformable-
# attention op was built against THIS container's torch ('build' mode); with another
# container, rebuild it. Do NOT rely on the compute node reaching PyPI -- it cannot.
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/text_only_no_tiling_run_countGDpp.sh run"
