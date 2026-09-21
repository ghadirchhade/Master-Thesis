#!/bin/bash -l
#
#  k_exemplars_no_tiling_submit_sam3.sh  --  SLURM batch script for experiment
#      k_exemplars_no_tiling
#      (SAM3 k-exemplar-prompted Rumex detection, tiling OFF,
#       cluster port of k_exemplars_no_tiling.ipynb)
#
#  The whole image is sent to SAM3 in ONE pass (longest side -> 1024 px) with k
#  GT boxes of the same image as direct positive box prompts. No tiles, no
#  exemplar strip, no plausibility filter.
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  k_exemplars_no_tiling_run_sam3.sh; PHASE 2 (the offline evaluation, including
#  the qualitative figures) runs once at the end, in the same job, over
#  everything that reached disk.
#
#  Resubmitting after the walltime is safe and expected: every finished
#  (image, anchor) pair is listed in the shard manifests and is skipped before
#  any GPU work happens.
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="kex_notile_sam3"
#SBATCH --output=k_exemplars_no_tiling_sam3_%j.out
#SBATCH --error=k_exemplars_no_tiling_sam3_%j.err
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

chmod +x "${SCRIPT_DIR}/k_exemplars_no_tiling_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch k_exemplars_no_tiling_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-k_exemplars_no_tiling}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (k exemplars, tiling OFF) ----------------
export N_EXEMPLARS="${N_EXEMPLARS:-3}"
export MAX_DIM="${MAX_DIM:-1024}"                 # longest side fed to SAM3
export THRESHOLD="${THRESHOLD:-0.30}"             # SAM3 runs ONCE at this score
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"
export DTYPE="${DTYPE:-bfloat16}"

# The notebook keeps the operating point EQUAL to the inference threshold.
export OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompts=${N_EXEMPLARS}  tiling=OFF  max_dim=${MAX_DIM}  dtype=${DTYPE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${OPERATING_CONFIDENCE}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# k_exemplars_no_tiling_run_sam3.sh sets. Do NOT rely on the compute node reaching PyPI -- it
# cannot. If the image's transformers is too old for Sam3Model, build a dedicated image
# (k_exemplars_no_tiling_HOW_TO_RUN.md, "Problem A").
srun \
  --environment=yolo26 \
  --container-workdir="$PWD" \
  --cpu-bind=cores \
  bash -c "${SCRIPT_DIR}/k_exemplars_no_tiling_run_sam3.sh run"