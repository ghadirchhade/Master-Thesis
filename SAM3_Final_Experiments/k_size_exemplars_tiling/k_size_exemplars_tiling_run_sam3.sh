#!/bin/bash
#
# k_size_exemplars_tiling_run_sam3.sh  ==>  dispatcher for experiment
#     k_size_exemplars_tiling
#     (SAM3 prompted with the K_EXEMPLARS=3 size-based GT plants S / M / L of the
#      image, cropped into an exemplar strip, tiling ON,
#      cluster port of k_size_exemplars_tiling.ipynb)
#
# This experiment splits the UAV image into overlapping tiles (TILE_SIZE=1000,
# OVERLAP=150). The 3 size-based GT plants of the image (S = smallest,
# M = closest to the median area, L = largest) are cropped from the full image and
# pasted into an exemplar strip above each tile (local background and feathering).
# The tile + strip is sent to SAM3 in batches of BATCH_SIZE=4 with the strip boxes
# as positive box prompts. The choice is deterministic -> ONE run per image, no
# anchors. Detections are filtered by target-region (>=50% area in tile) and
# plausibility (fill ratio >=15%, max area <=80%, edge margin >=5px), then mapped
# back to full image coords; NMS is applied offline in PHASE 2.
#
# Sets up the environment, decides where everything lives under $SCRATCH and
# launches k_size_exemplars_tiling_infer_sam3.py.
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
#               before burning an allocation (it counts the real tiles and
#               forward passes of every image of the shard, CPU only).
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
PY_SCRIPT="${SCRIPT_DIR}/k_size_exemplars_tiling_infer_sam3.py"

# ------------------- paths and experiment identity ---------------------------
EXPERIMENT_ROOT="${SCRATCH}/experiments"
SAM3_ROOT="${EXPERIMENT_ROOT}/sam3"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-k_size_exemplars_tiling}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME//\//-}}"     # a "/" would become a sub-folder
OUTPUT_DIR="${OUTPUT_DIR:-${SAM3_ROOT}/${RUN_NAME}}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"

# ---------------- experiment configuration (notebook CELL 3) -----------------
# K_EXEMPLARS=3 (S/M/L), SIZE_MEASURE=area and PROMPT_TYPE=size_SML are fixed by
# the design of the experiment and are therefore not variables here.
USE_TILING="${USE_TILING:-true}"                 # true -> overlapping tiles
TILE_SIZE="${TILE_SIZE:-1000}"                   # tile edge length in original-image pixels
OVERLAP="${OVERLAP:-150}"                        # overlap between neighbouring tiles in pixels
CACHE_TILES_IN_MEMORY="${CACHE_TILES_IN_MEMORY:-true}" # cache tiles in RAM

# --------------------------- SAM3 inference ----------------------------------
# CONFIDENCE_THRESHOLD: ONE value, used inside SAM3 (once per image x tile) AND as
# the operating point (exactly like the notebook).
THRESHOLD="${THRESHOLD:-0.30}"                   # CONFIDENCE_THRESHOLD
MASK_THRESHOLD="${MASK_THRESHOLD:-0.40}"         # MASK_THRESHOLD
BATCH_SIZE="${BATCH_SIZE:-4}"                    # composed tiles per forward pass
DTYPE="${DTYPE:-bfloat16}"                       # cluster default (GH200/H100)

# ------------------------- exemplar strip layout ------------------------------
STRIP_MARGIN="${STRIP_MARGIN:-6}"                # px between exemplar crops inside the strip
FEATHER_WIDTH="${FEATHER_WIDTH:-8}"              # px of soft alpha blending around each exemplar crop
BACKGROUND_BLUR_RADIUS="${BACKGROUND_BLUR_RADIUS:-1.5}" # light Gaussian blur applied to the sampled background
MAX_STRIP_HEIGHT_FRACTION="${MAX_STRIP_HEIGHT_FRACTION:-none}" # optional strip-height clamp (none = disabled)

# ------------------- plausibility filter (unchanged thresholds) ---------------
MIN_FILL_RATIO="${MIN_FILL_RATIO:-0.15}"         # >=15% of the box area must be covered by the mask
MAX_AREA_FRACTION="${MAX_AREA_FRACTION:-0.80}"   # a detection may not cover >80% of a tile
EDGE_MARGIN="${EDGE_MARGIN:-5}"                  # boxes with width or height <=5 px are discarded

# ---------------- target-region membership rule (CELL 13 fix) -----------------
TILE_REGION_MIN_FRACTION="${TILE_REGION_MIN_FRACTION:-0.50}" # majority rule

# ------------------------------ evaluation ------------------------------------
EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"
NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"   # applied offline in PHASE 2

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

echo "===== k_size_exemplars_tiling SAM3 ${MODE^^} ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(command -v python || command -v python3)"
echo "script_dir=${SCRIPT_DIR}"
echo "experiment=${EXPERIMENT_NAME}"
echo "prompts=3 size-based GT crops (S/M/L) in the strip, positive only"
echo "tiling=${USE_TILING} (tile=${TILE_SIZE}, overlap=${OVERLAP})"
echo "confidence_threshold=${THRESHOLD} (SAM3 + operating point)  nms_iou=${NMS_IOU_THRESHOLD}"
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
    print("  -> inference will refuse to start. Run './k_size_exemplars_tiling_run_sam3.sh download' on a login node.")

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
    --use-tiling "${USE_TILING}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
    --cache-tiles-in-memory "${CACHE_TILES_IN_MEMORY}"
    --confidence-threshold "${THRESHOLD}"
    --mask-threshold "${MASK_THRESHOLD}"
    --batch-size "${BATCH_SIZE}"
    --dtype "${DTYPE}"
    --strip-margin "${STRIP_MARGIN}"
    --feather-width "${FEATHER_WIDTH}"
    --background-blur-radius "${BACKGROUND_BLUR_RADIUS}"
    --max-strip-height-fraction "${MAX_STRIP_HEIGHT_FRACTION}"
    --min-fill-ratio "${MIN_FILL_RATIO}"
    --max-area-fraction "${MAX_AREA_FRACTION}"
    --edge-margin "${EDGE_MARGIN}"
    --tile-region-min-fraction "${TILE_REGION_MIN_FRACTION}"
    --nms-iou-threshold "${NMS_IOU_THRESHOLD}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --plot-evaluation-mode "${PLOT_EVALUATION_MODE}"
    --plot-min-gt-boxes "${PLOT_MIN_GT_BOXES}"
    --plot-max-display-dim "${PLOT_MAX_DISPLAY_DIM}"
)

# The evaluation needs to know where things are, which operating point is used,
# which prompt setting the runs were made with (the manifest is filtered on
# PROMPT_TYPE = size_SML) and where the dataset is, because notebook CELL 30
# reopens the selected images to draw the qualitative figures.
EVAL_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --use-tiling "${USE_TILING}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
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
        echo "      with: ./k_size_exemplars_tiling_run_sam3.sh dryrun   (the PYTHON DEBUG block prints the result)."
        ;;

    dryrun)
        # No model, no GPU: just proves the dataset layout is understood and
        # prints the cost of the experiment (real tiles + forward passes).
        python "${PY_SCRIPT}" "${COMMON_ARGS[@]}" --dry-run "$@"
        ;;

    run)
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
        if [[ ! -d "${HF_HOME}" || -z "$(ls -A "${HF_HOME}" 2>/dev/null)" ]]; then
            echo "WARNING: ${HF_HOME} looks empty and we are running offline."
            echo "         Run './k_size_exemplars_tiling_run_sam3.sh download' on a login node first."
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