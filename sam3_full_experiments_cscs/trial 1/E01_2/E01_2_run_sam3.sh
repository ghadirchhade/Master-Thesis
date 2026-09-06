#!/bin/bash
#
# E01_2_run_sam3.sh  ==>  dispatcher for experiment E01_2
#     (SAM3 SINGLE-exemplar-prompted Rumex detection, tiling ON,
#      cluster port of E01_2.ipynb)
#
# E01_2 is the same pipeline as E02_2 with ONE visual prompt instead of three
# (N_EXEMPLARS=1). Everything else is identical, so the two experiments are
# directly comparable.
#
# Sets up the environment, decides where everything lives under $SCRATCH and
# launches E01_2_infer_sam3.py.
#
# MODES
#   download    Pre-fetch facebook/sam3 into $HF_HOME and install `supervision`
#               into $PYEXTRA (plus wheels for the evaluation packages).
#               RUN THIS ON A LOGIN NODE FIRST: CSCS compute nodes have no
#               internet, and facebook/sam3 is a GATED repo, so you must
#               (a) accept the licence at https://huggingface.co/facebook/sam3
#               with the account owning $HF_TOKEN, and (b) export HF_TOKEN.
#   dryrun      Discover the dataset and print the run plan. No GPU, no model.
#               Use it to sanity-check DATASET_ROOT and to estimate the runtime
#               before burning an allocation.
#   run         (default) PHASE 1 + PHASE 2. Launches NUM_GPUS shard processes,
#               one per GPU, waits for all of them, then runs the offline
#               evaluation once over everything that reached disk.
#
# NOTE: all three experiments share $HF_HOME and $PYEXTRA, so the download mode
#       only has to be run once for all of them.
#
#   evaluate    PHASE 2 only. Rebuilds every metric, the pooled AP and the
#               confusion matrices from the cached NPZ files. No GPU, no model,
#               SAM3 is never loaded. Repeat as often as you like.

set -euo pipefail

# ----------------------------- environment -----------------------------------
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/E01_2_infer_sam3.py"

# ------------------- paths and experiment identity ---------------------------
EXPERIMENT_ROOT="${SCRATCH}/experiments"
SAM3_ROOT="${EXPERIMENT_ROOT}/sam3"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-E01_2}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SAM3_ROOT}/${RUN_NAME}}"

DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"

# ---------------- experiment configuration (notebook CELL 3) -----------------
N_EXEMPLARS="${N_EXEMPLARS:-1}"                  # E01_2: ONE visual prompt (the anchor)
USE_TILING="${USE_TILING:-1}"                    # 1 = overlapping tiles, 0 = whole image
TILE_SIZE="${TILE_SIZE:-1000}"
OVERLAP="${OVERLAP:-150}"

THRESHOLD="${THRESHOLD:-0.30}"                   # SAM3_INFERENCE_THRESHOLD
MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"         # MASK_THRESHOLD
BATCH_SIZE="${BATCH_SIZE:-4}"                    # BATCH_SIZE
DTYPE="${DTYPE:-bfloat16}"                       # notebook used fp16 on a T4

STRIP_MARGIN="${STRIP_MARGIN:-6}"
FEATHER_WIDTH="${FEATHER_WIDTH:-8}"
BACKGROUND_BLUR_RADIUS="${BACKGROUND_BLUR_RADIUS:-1.5}"

MIN_FILL_RATIO="${MIN_FILL_RATIO:-0.15}"
MAX_AREA_FRACTION="${MAX_AREA_FRACTION:-0.80}"
EDGE_MARGIN="${EDGE_MARGIN:-5}"
TILE_REGION_MIN_FRACTION="${TILE_REGION_MIN_FRACTION:-0.50}"

EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"
BEST_CONFIDENCE="${BEST_CONFIDENCE:-0.40}"       # frozen operating point, no sweep
BEST_NMS_IOU="${BEST_NMS_IOU:-0.40}"             # frozen operating point, no sweep

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

echo "===== E01_2 SAM3 ${MODE^^} ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(command -v python || command -v python3)"
echo "script_dir=${SCRIPT_DIR}"
echo "experiment=${EXPERIMENT_NAME}"
echo "n_exemplars=${N_EXEMPLARS}  use_tiling=${USE_TILING}  tile=${TILE_SIZE}  overlap=${OVERLAP}"
echo "sam3_threshold=${THRESHOLD}  operating_point=conf:${BEST_CONFIDENCE}/nms:${BEST_NMS_IOU}"
echo "dtype=${DTYPE}  batch_size=${BATCH_SIZE}"
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
    print("  -> inference will refuse to start. Run './E01_2_run_sam3.sh download' on a login node.")
try:
    import pandas
    print("pandas:", pandas.__version__, "-> PHASE 2 import: OK")
except Exception as exc:
    print("pandas import FAILED:", exc)
    print("  -> PHASE 1 still works, but the evaluation cannot run. See 'Problem B'.")
try:
    import matplotlib
    print("matplotlib:", matplotlib.__version__, "-> confusion-matrix PNGs: OK")
except Exception as exc:
    print("matplotlib import FAILED:", exc)
    print("  -> confusion-matrix CSVs are still written, only the PNGs are skipped.")
PY
echo "========================"

# ------------- shared python arguments (notebook CELL 3 values) --------------
COMMON_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --model-id "${MODEL_ID}"
    --n-exemplars "${N_EXEMPLARS}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
    --threshold "${THRESHOLD}"
    --mask-threshold "${MASK_THRESHOLD}"
    --batch-size "${BATCH_SIZE}"
    --dtype "${DTYPE}"
    --strip-margin "${STRIP_MARGIN}"
    --feather-width "${FEATHER_WIDTH}"
    --background-blur-radius "${BACKGROUND_BLUR_RADIUS}"
    --min-fill-ratio "${MIN_FILL_RATIO}"
    --max-area-fraction "${MAX_AREA_FRACTION}"
    --edge-margin "${EDGE_MARGIN}"
    --tile-region-min-fraction "${TILE_REGION_MIN_FRACTION}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --best-confidence "${BEST_CONFIDENCE}"
    --best-nms-iou "${BEST_NMS_IOU}"
)
if [[ "${USE_TILING}" == "1" ]]; then
    COMMON_ARGS+=(--tiling)
else
    COMMON_ARGS+=(--no-tiling)
fi

# The evaluation only needs to know where things are and which operating point
# was frozen; it never touches the dataset or the model.
EVAL_ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --n-exemplars "${N_EXEMPLARS}"
    --threshold "${THRESHOLD}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --best-confidence "${BEST_CONFIDENCE}"
    --best-nms-iou "${BEST_NMS_IOU}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
)
if [[ "${USE_TILING}" == "1" ]]; then
    EVAL_ARGS+=(--tiling)
else
    EVAL_ARGS+=(--no-tiling)
fi

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
    # weights + config + processor only; skip anything we do not need
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
        # PNGs). Most containers already have them; if the dryrun says they are
        # missing, install them INSIDE the container from these wheels, always
        # with --no-deps so that numpy / pillow are never shadowed.
        mkdir -p "${WHEELS}"
        pip download --no-deps -d "${WHEELS}" \
            pandas matplotlib supervision \
            contourpy cycler fonttools kiwisolver pyparsing packaging \
            python-dateutil pytz tzdata six || \
            echo "  (warning: some wheels could not be downloaded; see 'Problem B')"
        ls -1 "${WHEELS}" | head -20

        echo "Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1)."
        echo "NOTE: this check ran with the LOGIN node's python. Verify inside the container"
        echo "      with: ./E01_2_run_sam3.sh dryrun   (the PYTHON DEBUG block prints the result)."
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
            echo "         Run './E01_2_run_sam3.sh download' on a login node first."
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
                echo "  shard ${idx} FAILED (exit $?)"
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
