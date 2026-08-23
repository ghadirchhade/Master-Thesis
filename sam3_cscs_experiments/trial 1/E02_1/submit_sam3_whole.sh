#!/bin/bash -l
 
#  submit_sam3_whole.sh  --  SLURM batch script for the SAM3 WHOLE-IMAGE Rumex experiment
 
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="sam3_whole"
#SBATCH --output=sam3_whole_%j.out
#SBATCH --error=sam3_whole_%j.err
#SBATCH --time=12:00:00
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
 
chmod +x "${SCRIPT_DIR}/run_sam3_whole.sh"
 
# Defaults for this job; any of these can be overridden from the submitting shell.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-E02_1}"
export PRESET="${PRESET:-multi}"                  # multi = 3 exemplar boxes, single = 1
export MAX_DIM="${MAX_DIM:-1024}"
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"
 
echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}  preset=${PRESET}  max_dim=${MAX_DIM}"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="
 
# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# run_sam3_whole.sh sets. Do NOT rely on the compute node reaching PyPI -- it cannot. If the
# image's transformers is too old for Sam3Model, build a dedicated image
# (HOW_TO_RUN_whole.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/run_sam3_whole.sh run"
 

