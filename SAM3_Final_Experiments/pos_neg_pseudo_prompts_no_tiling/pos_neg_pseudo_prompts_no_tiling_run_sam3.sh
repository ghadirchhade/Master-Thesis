#!/bin/bash
#
# pos_neg_pseudo_prompts_no_tiling_run_sam3.sh  ==>  dispatcher for experiment
#     pos_neg_pseudo_prompts_no_tiling
#     (SAM3, TWO rounds per image: S/M/L GT boxes, then + positive / negative
#      self-prompts taken from SAM3's own round-1 predictions, NO text,
#      tiling OFF, cluster port of pos_neg_pseudo_prompts_no_tiling.ipynb)
#
# This experiment sends the WHOLE image to SAM3 (downscaled so that its longest
# side is MAX_DIM=1024), no tiles, no exemplar strip:
#   ROUND 1  the K_EXEMPLARS=3 size-based GT boxes of the same image are the
#            POSITIVE box prompts (S = smallest, M = closest to the median area,
#            L = largest). Deterministic -> ONE run per image, no anchors.
#   SELF-PROMPTS  the round-1 detections are NMS-merged (NMS_IOU_THRESHOLD=0.40),
#            boxes sitting on a prompt plant (IoU >= SELF_PROMPT_EXCLUDE_IOU=0.50)
#            are dropped, then
#              N_SELF_POSITIVES=2 highest-confidence boxes -> POSITIVE (label 1)
#              N_SELF_NEGATIVES=2 lowest-confidence boxes below
#                UNRELIABLE_MAX_SCORE=0.50 -> NEGATIVE (label 0)
#   ROUND 2  S/M/L + the positive self-prompts (label 1) + the negative
#            self-prompts (label 0) in ONE forward pass. Its output is FINAL.
#            Skipped when no self-prompt exists (round 1 is then final).
# At most two SAM3 forward passes per image.
#
# Sets up the environment, decides where everything lives under $SCRATCH and
# launches pos_neg_pseudo_prompts_no_tiling_infer_sam3.py.
#
# MODES
#   download    Pre-fetch facebook/sam3 into $HF_HOME and install `supervision`
#               into $PYEXTRA (plus wheels for the evaluation packages).
#               RUN THIS ON A LOGIN NODE FIRST: CSCS compute nodes have no
#               internet, and facebook/sam3 is a GATED repo, so you must
#               (a) accept the licence at https://huggingface.co/facebook/sam3
#               with the account owning $HF_TOKEN, and (b) export HF_TOKEN.
#   dryrun      Discover the dataset and print the run plan. No GPU, no model.
#               Use it to sanity-check DATASET_ROOT and to estimate the number
#               of SAM3 forward passes before burning an allocation.
#   run         (default) PHASE 1 + PHASE 2. Launches NUM_GPUS shard processes,
#               one per GPU, waits for all of them, then runs the offline
#               evaluation once over everything that reached disk.
#
# NOTE: all experiments share $HF_HOME and $PYEXTRA, so the download mode only
#       has to be run once for all of them.
#
#   evaluate    PHASE 2 only. Rebuilds every metric, the pooled AP, the
#               confusion matrices and the qualitative figures from the cached
#               NPZ files. No GPU, no model, SAM3 is never loaded. Repeat as
#               often as you like.

set -euo pipefail

# ----------------------------- environment -----------------------------------
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/pos_neg_pseudo_prompts_no_tiling_infer_sam3.py"

# ------------------- paths and experiment identity ---------------------------
EXPERIMENT_ROOT="${SCRATCH}/experiments"
SAM3_ROOT="${EXPERIMENT_ROOT}/sam3"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pos_neg_pseudo_prompts_no_tiling}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME//\//-}}"     # a "/" would become a sub-folder
OUTPUT_DIR="${OUTPUT_DIR:-${SAM3_ROOT}/${RUN_NAME}}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"

# ---------------- experiment configuration (notebook CELL 3) -----------------
# K_EXEMPLARS=3 (S/M/L), SIZE_MEASURE=area and USE_TILING=False are fixed by the
# design of the experiment and are therefore not variables here.
N_SELF_POSITIVES="${N_SELF_POSITIVES:-2}"               # round-1 top boxes -> POSITIVE in round 2
N_SELF_NEGATIVES="${N_SELF_NEGATIVES:-2}"               # round-1 lowest unreliable boxes -> NEGATIVE
SELF_PROMPT_EXCLUDE_IOU="${SELF_PROMPT_EXCLUDE_IOU:-0.50}"  # not eligible if on a S/M/L prompt plant
UNRELIABLE_MAX_SCORE="${UNRELIABLE_MAX_SCORE:-0.50}"    # negatives only below this confidence
MAX_DIM="${MAX_DIM:-1024}"                              # longest side fed to SAM3 (no tiling)

# CONFIDENCE_THRESHOLD: ONE value, used inside SAM3 (both rounds), for the
# self-prompt selection AND as the operating point (exactly like the notebook).
THRESHOLD="${THRESHOLD:-0.30}"                   # CONFIDENCE_THRESHOLD
MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"         # MASK_THRESHOLD
DTYPE="${DTYPE:-bfloat16}"                       # notebook used the Colab default (fp32)

NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"   # round-1 self-prompt NMS + offline NMS
EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

# --------------------------- qualitative plot --------------------------------
PLOT_EVALUATION_MODE="${PLOT_EVALUATION_MODE:-all_gt}"
PLOT_MIN_GT_BOXES="${PLOT_MIN_GT_BOXES:-7}"
PLOT_MAX_DISPLAY_DIM="${PLOT_MAX_DISPLAY_DIM:-2048}"

# HuggingFace cache. Must live on $SCRATCH: $HOME is small and the weights are several GB.
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
MODEL_ID="${MODEL_ID:-facebook/sam3}"
PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"
export PYTHONPATH="${PYEXTRA}${PYTHONPATH:+:${PYTHONPATH}}"
WHEELS="${WHEELS:-${SCRATCH}/wheels}"

# ------------------------------ GPU count ------------------------------------
if [[ -n "${SLURM_GPUS_PER_TASK:-}" ]]; then
    DEFAULT_GPUS="${SLURM_GPUS_PER_TASK}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    DEFAULT_GPUS="$(nvidia-smi -L | wc -l)"
else
    DEFAULT_GPUS=1
fi
NUM_GPUS="${NUM_GPUS:-${DEFAULT_GPUS}}"
[[ "${NUM_GPUS}" -lt 1 ]] && NUM_GPUS=1

# Mode dispatch: first positional argument, everything after it goes to python.
MODE="run"
if [[ $# -gt 0 ]]; then
    MODE="$1"
    shift
fi

echo "===== pos_neg_pseudo_prompts_no_tiling SAM3 ${MODE^^} ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(command -v python || command -v python3)"
echo "script_dir=${SCRIPT_DIR}"
echo "experiment=${EXPERIMENT_NAME}"
echo "round 1: 3 size-based GT boxes (S/M/L)  |  round 2: + ${N_SELF_POSITIVES} pos / ${N_SELF_NEGATIVES} neg self-prompts (NO text)"
echo "self-prompts: exclude IoU=${SELF_PROMPT_EXCLUDE_IOU}  unreliable < ${UNRELIABLE_MAX_SCORE}"
echo "tiling=OFF  max_dim=${MAX_DIM}"
echo "confidence_threshold=${THRESHOLD} (SAM3 both rounds + operating point)  nms_iou=${NMS_IOU_THRESHOLD}"
echo "dtype=${DTYPE}"
echo "dataset_root=${DATASET_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "model_id=${MODEL_ID}"
echo "hf_home=${HF_HOME}"
echo "pyextra=${PYEXTRA}"
echo "num_gpus=${NUM_GPUS}"
echo "================================"

mkdir -p "${EXPERIMENT_ROOT}" "${SAM3_ROOT}" "${OUTPUT_DIR}" "${HF_HOME}" "${PYEXTRA}"

# ---------------------------- python debug -----------------------------------
echo "===== PYTHON DEBUG ====="
python --version
python -m pip --version || true
python -m pip show transformers || true

python - <<'PY'
import os, site, sys
print("sys.executable:", sys.executable)
print("sys.path:")
for p in sys.path:
    print("  ", p)
try:
    print("site.getsitepackages():")
    for p in site.getsitepackages():
        print("  ", p)
except Exception as exc:
    print("  (unavailable:", exc, ")")
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

python - <<'PY'
import sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| device count:", torch.cuda.device_count())
    print("bfloat16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a")
except Exception as exc:
    print("torch import FAILED:", exc)
try:
    import transformers
    print("transformers:", transformers.__version__, transformers.__file__)
    from transformers import Sam3Model, Sam3Processor  # noqa: F401
    print("Sam3Model import: OK")
except Exception as exc:
    print("transformers/Sam3Model import FAILED:", exc)
try:
    import supervision
    print("supervision:", supervision.__version__, supervision.__file__)
    from supervision.metrics import MeanAveragePrecision  # noqa: F401
    print("MeanAveragePrecision import: OK")
except Exception as exc:
    print("supervision import FAILED:", exc)
    print("  -> inference will refuse to start. Run './pos_neg_pseudo_prompts_no_tiling_run_sam3.sh download' on a login node.")
try:
    import pandas
    print("pandas:", pandas.__version__, "-> PHASE 2 import: OK")
except Exception as exc:
    print("pandas import FAILED:", exc)
    print("  -> PHASE 1 still works, but the evaluation cannot run. See 'Problem B'.")
try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__, "-> confusion-matrix + qualitative PNGs: OK")
except Exception as exc:
    print("matplotlib import FAILED:", exc)
    print("  -> the CSVs are still written, only the PNGs are skipped.")
PY
echo "========================"

# ------------- shared python arguments (notebook CELL 3 values) --------------
COMMON_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --model-id "${MODEL_ID}"
    --n-self-positives "${N_SELF_POSITIVES}"
    --n-self-negatives "${N_SELF_NEGATIVES}"
    --self-prompt-exclude-iou "${SELF_PROMPT_EXCLUDE_IOU}"
    --unreliable-max-score "${UNRELIABLE_MAX_SCORE}"
    --max-dim "${MAX_DIM}"
    --confidence-threshold "${THRESHOLD}"
    --mask-threshold "${MASK_THRESHOLD}"
    --dtype "${DTYPE}"
    --nms-iou-threshold "${NMS_IOU_THRESHOLD}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --plot-evaluation-mode "${PLOT_EVALUATION_MODE}"
    --plot-min-gt-boxes "${PLOT_MIN_GT_BOXES}"
    --plot-max-display-dim "${PLOT_MAX_DISPLAY_DIM}"
)

# The evaluation needs to know where things are, which operating point is used,
# which prompt setting the runs were made with (the manifest is filtered on
# PROMPT_TYPE, built from N_SELF_POSITIVES / N_SELF_NEGATIVES) and where the
# dataset is, because notebook CELL 27 reopens the selected images to draw the
# qualitative figures.
EVAL_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --n-self-positives "${N_SELF_POSITIVES}"
    --n-self-negatives "${N_SELF_NEGATIVES}"
    --self-prompt-exclude-iou "${SELF_PROMPT_EXCLUDE_IOU}"
    --unreliable-max-score "${UNRELIABLE_MAX_SCORE}"
    --max-dim "${MAX_DIM}"
    --confidence-threshold "${THRESHOLD}"
    --mask-threshold "${MASK_THRESHOLD}"
    --nms-iou-threshold "${NMS_IOU_THRESHOLD}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --plot-evaluation-mode "${PLOT_EVALUATION_MODE}"
    --plot-min-gt-boxes "${PLOT_MIN_GT_BOXES}"
    --plot-max-display-dim "${PLOT_MAX_DISPLAY_DIM}"
)

case "${MODE}" in
    download)
        # Runs on a LOGIN node (internet + your HF token). Populates $HF_HOME so
        # the compute node can load the model completely offline.
        if [[ -z "${HF_TOKEN:-}" ]]; then
            echo "ERROR: HF_TOKEN is not set. facebook/sam3 is a gated repo:"
            echo "  1) accept the licence at https://huggingface.co/facebook/sam3"
            echo "  2) export HF_TOKEN=hf_xxx"
            exit 1
        fi

        echo "--- 1/3 : model weights -> ${HF_HOME} ---"
        python - <<PY
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="${MODEL_ID}",
    token=os.environ["HF_TOKEN"],
    # weights + config + processor only
    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "*.model", "*.py"],
)
print("Snapshot cached at:", path)
PY

        echo "--- 2/3 : supervision -> ${PYEXTRA} ---"
        pip install --target "${PYEXTRA}" --no-deps --upgrade supervision
        python - <<'PY'
import sys
print("checking the freshly installed package is importable ...")
import supervision
from supervision.metrics import MeanAveragePrecision  # noqa: F401
print("supervision", supervision.__version__, "->", supervision.__file__)
PY

        echo "--- 3/3 : wheels for the PHASE 2 packages -> ${WHEELS} ---"
        # PHASE 2 needs pandas (metric tables) and matplotlib (confusion-matrix
        # and qualitative PNGs). Most containers already have them; if the dryrun
        # says they are missing, install them INSIDE the container from these
        # wheels, always with --no-deps so numpy / pillow are never shadowed.
        mkdir -p "${WHEELS}"
        pip download --no-deps -d "${WHEELS}" \
            pandas matplotlib supervision \
            contourpy cycler fonttools kiwisolver pyparsing packaging \
            python-dateutil pytz tzdata six || \
            echo "  (warning: some wheels could not be downloaded; see 'Problem B')"
        ls -1 "${WHEELS}" | head -20

        echo "Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1)."
        echo "NOTE: this check ran with the LOGIN node's python. Verify inside the container"
        echo "      with: ./pos_neg_pseudo_prompts_no_tiling_run_sam3.sh dryrun   (the PYTHON DEBUG block prints the result)."
        ;;

    dryrun)
        # No model, no GPU: just proves the dataset layout is understood and
        # prints the cost of the experiment.
        python "${PY_SCRIPT}" "${COMMON_ARGS[@]}" --dry-run "$@"
        ;;

    run)
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

        if [[ ! -d "${HF_HOME}" || -z "$(ls -A "${HF_HOME}" 2>/dev/null)" ]]; then
            echo "WARNING: ${HF_HOME} looks empty and we are running offline."
            echo "         Run './pos_neg_pseudo_prompts_no_tiling_run_sam3.sh download' on a login node first."
        fi

        echo "===== PHASE 1 : SHARDED INFERENCE ====="
        echo "Launching ${NUM_GPUS} shard process(es) ..."
        PIDS=()
        for ((i = 0; i < NUM_GPUS; i++)); do
            LOG="${OUTPUT_DIR}/shard${i}.log"
            CUDA_VISIBLE_DEVICES="${i}" \
            python "${PY_SCRIPT}" \
                "${COMMON_ARGS[@]}" \
                --num-shards "${NUM_GPUS}" \
                --shard-index "${i}" \
                --no-evaluate \
                "$@" > >(tee "${LOG}") 2>&1 &
            PIDS+=("$!")
            echo "  shard ${i} -> pid ${PIDS[-1]}  (log: ${LOG})"
        done

        # Wait for every shard and remember whether any of them failed.
        FAILED=0
        for idx in "${!PIDS[@]}"; do
            if wait "${PIDS[$idx]}"; then
                echo "  shard ${idx} finished OK"
            else
                rc=$?
                echo "  shard ${idx} FAILED (exit ${rc})"
                FAILED=1
            fi
        done

        echo "===== PHASE 2 : OFFLINE EVALUATION ====="
        # Always evaluate: even a partially failed run should produce usable
        # metrics from whatever NPZ files made it to disk.
        python "${PY_SCRIPT}" "${EVAL_ARGS[@]}" --evaluate-only

        if [[ "${FAILED}" -ne 0 ]]; then
            echo "At least one shard failed -- see the per-shard logs above."
            exit 1
        fi
        echo "All shards completed. Results in ${OUTPUT_DIR}"
        ;;

    evaluate)
        # PHASE 2 only. Never loads SAM3, so it runs fine on a login node or in a
        # small CPU allocation.
        python "${PY_SCRIPT}" "${EVAL_ARGS[@]}" --evaluate-only "$@"
        ;;

    *)
        echo "Unknown mode: ${MODE} (expected: download|dryrun|run|evaluate)"
        exit 1
        ;;
esac