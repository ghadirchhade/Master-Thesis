#!/bin/bash -l
#
#  E02_2c_submit_sam3.sh  --  SLURM batch script for experiment E02_2c
#      (SAM3 exemplar-prompted Rumex detection with 1536/384 TILES and an
#       optional GLOBAL-CONTEXT PASS, cluster port of E02_2c.ipynb)
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  E02_2c_run_sam3.sh; PHASE 2 (the offline evaluation) runs once at the end,
#  in the same job, over everything that reached disk.
#
#  Resubmitting after the 24 h walltime is safe and expected: every finished
#  (image, anchor) pair is listed in the shard manifests and is skipped before
#  any GPU work happens.

#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="E02_2c_sam3"
#SBATCH --output=E02_2c_sam3_%j.out
#SBATCH --error=E02_2c_sam3_%j.err
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

chmod +x "${SCRIPT_DIR}/E02_2c_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch E02_2c_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-E02_2c}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (E02_2c: 3 exemplars, 1536/384 tiles + global pass) ---
export N_EXEMPLARS="${N_EXEMPLARS:-3}"
export USE_TILING="${USE_TILING:-1}"
export TILE_SIZE="${TILE_SIZE:-1536}"             # E02_2c tile size
export OVERLAP="${OVERLAP:-384}"                  # E02_2c: 25 percent overlap
export ADD_GLOBAL_CONTEXT_PASS="${ADD_GLOBAL_CONTEXT_PASS:-1}"   # extra whole-image pass
export GLOBAL_DOWNSCALE="${GLOBAL_DOWNSCALE:-2}"
export MEM_STOP_THRESHOLD_PCT="${MEM_STOP_THRESHOLD_PCT:-70}"    # clean stop, then resubmit
export THRESHOLD="${THRESHOLD:-0.30}"             # SAM3 runs ONCE at this score
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export DTYPE="${DTYPE:-bfloat16}"
export BEST_CONFIDENCE="${BEST_CONFIDENCE:-0.40}" # frozen operating point
export BEST_NMS_IOU="${BEST_NMS_IOU:-0.40}"       # frozen operating point
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompts=${N_EXEMPLARS}  tile=${TILE_SIZE}/${OVERLAP}  dtype=${DTYPE}  gpus=${NUM_GPUS}"
echo "global_pass=${ADD_GLOBAL_CONTEXT_PASS} (downscale=${GLOBAL_DOWNSCALE})  mem_stop=${MEM_STOP_THRESHOLD_PCT}%"
echo "operating point: confidence=${BEST_CONFIDENCE}  NMS IoU=${BEST_NMS_IOU} (frozen)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# E02_2c_run_sam3.sh sets. Do NOT rely on the compute node reaching PyPI -- it cannot. If the
# image's transformers is too old for Sam3Model, build a dedicated image
# (E02_2c_HOW_TO_RUN.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/E02_2c_run_sam3.sh run"
