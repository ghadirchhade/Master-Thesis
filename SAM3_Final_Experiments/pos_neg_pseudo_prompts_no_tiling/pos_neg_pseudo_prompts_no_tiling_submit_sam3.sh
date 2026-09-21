#!/bin/bash -l
#
#  pos_neg_pseudo_prompts_no_tiling_submit_sam3.sh  --  SLURM batch script for
#      experiment pos_neg_pseudo_prompts_no_tiling
#      (SAM3, two rounds per image: S/M/L GT boxes, then + positive / negative
#       self-prompts from SAM3's own round-1 predictions, NO text, tiling OFF,
#       cluster port of pos_neg_pseudo_prompts_no_tiling.ipynb)
#
#  The whole image is sent to SAM3 (longest side -> 1024 px), no tiles, no strip.
#  ROUND 1 uses the 3 size-based GT boxes (smallest / median-closest / largest) as
#  positive box prompts. From the NMS-merged round-1 detections that do not sit on a
#  prompt plant, the 2 highest-confidence boxes become POSITIVE and the 2 lowest-
#  confidence boxes below 0.50 become NEGATIVE box prompts. ROUND 2 runs S/M/L + those
#  in ONE forward pass and is the final output (skipped if no self-prompt exists).
#  Evaluated on BOTH archives (AGS_Multi_Rumex + AgsSpringRumex).
#
#  One node, 4 GPUs. PHASE 1 (inference) is split across the 4 GPUs by
#  pos_neg_pseudo_prompts_no_tiling_run_sam3.sh; PHASE 2 (the offline evaluation,
#  including the qualitative figures) runs once at the end, in the same job, over
#  everything that reached disk.
#
#  Resubmitting after the walltime is safe and expected: every finished image is
#  listed in the shard manifests and is skipped before any GPU work happens.
#SBATCH --no-requeue
#SBATCH --account="go077"
#SBATCH --job-name="posneg_pseudo_notile_sam3"
#SBATCH --output=pos_neg_pseudo_prompts_no_tiling_sam3_%j.out
#SBATCH --error=pos_neg_pseudo_prompts_no_tiling_sam3_%j.err
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

chmod +x "${SCRIPT_DIR}/pos_neg_pseudo_prompts_no_tiling_run_sam3.sh"

# Defaults for this job; any of these can be overridden from the submitting shell,
# e.g.  DTYPE=float16 sbatch pos_neg_pseudo_prompts_no_tiling_submit_sam3.sh
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-pos_neg_pseudo_prompts_no_tiling}"
export DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"   # holds `supervision`
export NUM_GPUS="${NUM_GPUS:-4}"

# --- notebook CELL 3 configuration (S/M/L + 2 pos / 2 neg self-prompts, tiling OFF) ---
export N_SELF_POSITIVES="${N_SELF_POSITIVES:-2}"
export N_SELF_NEGATIVES="${N_SELF_NEGATIVES:-2}"
export SELF_PROMPT_EXCLUDE_IOU="${SELF_PROMPT_EXCLUDE_IOU:-0.50}"
export UNRELIABLE_MAX_SCORE="${UNRELIABLE_MAX_SCORE:-0.50}"
export MAX_DIM="${MAX_DIM:-1024}"                 # longest side fed to SAM3
export THRESHOLD="${THRESHOLD:-0.30}"             # CONFIDENCE_THRESHOLD (SAM3 + operating point)
export MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"
export DTYPE="${DTYPE:-bfloat16}"

export NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"
export EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
export PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

echo "===== SLURM JOB ====="
echo "job_id=${SLURM_JOB_ID:-?}  node=$(hostname)"
echo "experiment=${EXPERIMENT_NAME}"
echo "round 1: S/M/L GT boxes | round 2: + ${N_SELF_POSITIVES} pos / ${N_SELF_NEGATIVES} neg self-prompts (NO text)  tiling=OFF  max_dim=${MAX_DIM}"
echo "dtype=${DTYPE}  gpus=${NUM_GPUS}"
echo "operating point: confidence=${THRESHOLD}  NMS IoU=${NMS_IOU_THRESHOLD} (fixed)"
echo "dataset_root=${DATASET_ROOT}"
echo "scratch_cwd=${cwd}"
echo "====================="

# --environment=yolo26 selects the CSCS Container Engine EDF (~/.edf/yolo26.toml) that
# provides CUDA + PyTorch + transformers. Packages the image does not ship (supervision, and
# possibly a newer transformers) are picked up from $PYEXTRA via PYTHONPATH, which
# pos_neg_pseudo_prompts_no_tiling_run_sam3.sh sets. Do NOT rely on the compute node reaching
# PyPI -- it cannot. If the image's transformers is too old for Sam3Model, build a dedicated
# image (pos_neg_pseudo_prompts_no_tiling_HOW_TO_RUN.md, "Problem A").
srun \
    --environment=yolo26 \
    --container-workdir="$PWD" \
    --cpu-bind=cores \
    bash -c "${SCRIPT_DIR}/pos_neg_pseudo_prompts_no_tiling_run_sam3.sh run"