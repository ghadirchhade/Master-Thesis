#!/bin/bash -l
 
#  submit_sam3_single.sh  --  SLURM batch script for the SAM3 TILED SINGLE-BOX Rumex
#  experiment (E01_2: 1 exemplar crop, tiling ON, no batching)
 
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="sam3_single"
#SBATCH --output=sam3_single_%j.out
#SBATCH --error=sam3_single_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=hassan@pixtell.ch
#SBATCH --mail-type=BEGIN,END,FAIL
 
# --time is 24h, same as E02_2: this experiment does the same ~70 forward passes per anchor,
# only with a one-crop strip instead of a three-crop one. Expect a runtime in the same
# ballpark, marginally faster. If it overruns, just resubmit -- the job resumes from the
# shard CSVs.
 
set -euo pipefail
 
SCRIPT_DIR="$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3"
 
# Everything the job writes goes to scratch.
cwd="${SCRATCH}/experiments/sam3"
mkdir -p "${cwd}"
cd "${cwd}"
 
chmod +x "${SCRIPT_DIR}/run_sam3_single.sh"
 
# Defaults for this job; any of these can be overridden from the submitting shell.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-E01_2}"
export PRESET="${PRESET:-single}"                 # single = 1 exemplar crop, multi = 3
export THRESHOLD="${THRESHOLD:-0.3}"
export NMS_IOU="${NMS_IOU:-0.5}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"
 
echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}  preset=${PRESET}  conf=${THRESHOLD}  nms_iou=${NMS_IOU}"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="
 
# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# run_sam3_single.sh sets. Do NOT rely on the compute node reaching PyPI -- it cannot. If the
# image's transformers is too old for Sam3Model, build a dedicated image
# (HOW_TO_RUN_single.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/run_sam3_single.sh run"
 

