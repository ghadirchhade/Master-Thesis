#!/bin/bash
#
# k_exemplars_with_tiling_extra_path_run_yoloe.sh  ==>  dispatcher for k_exemplars_with_tiling_extra_path_yoloe
#     (YOLOE K-EXEMPLAR visual-prompted Rumex detection, tiling ON (1536/384)
#      + downsampled global-context pass ("extra path"),
#      cluster port of E01_single_image_several_exemplar_YOLOE_11_extra_path_tiling.ipynb)
#
# MODES
#   download    Pre-fetch yoloe-11l-seg.pt into $YOLOE_WEIGHTS_DIR and install
#               `supervision` into $PYEXTRA (plus wheels for the evaluation
#               packages). RUN THIS ON A LOGIN NODE FIRST: CSCS compute nodes
#               have no internet. YOLOE is NOT gated: no HF token is needed.
#   dryrun      Discover the dataset and print the run plan. No GPU, no model.
#               Use it to sanity-check DATASET_ROOT and to estimate the runtime
#               before burning an allocation.
#   run         (default) PHASE 1 + PHASE 2. Launches NUM_GPUS shard processes,
#               one per GPU, waits for all of them, then runs the offline
#               evaluation once over everything that reached disk.
#
# NOTE: the SAM3 experiments and the YOLOE experiments share $PYEXTRA, $WHEELS
#       and $SCRATCH/yoloe_weights, so the download mode only has to be done once.
#
#   evaluate    PHASE 2 only. Rebuilds every metric, the pooled AP, the
#               confusion matrices and the qualitative figures from the cached
#               NPZ files. No GPU, no model, YOLOE is never loaded. Repeat as
#               often as you like.

set -euo pipefail

# ----------------------------- environment -----------------------------------
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/k_exemplars_with_tiling_extra_path_infer_yoloe.py"

# ------------------- paths and experiment identity ---------------------------
EXPERIMENT_ROOT="${SCRATCH}/experiments"
YOLOE_ROOT="${EXPERIMENT_ROOT}/yoloe"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-k_exemplars_with_tiling_extra_path_yoloe}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${YOLOE_ROOT}/${RUN_NAME}}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"

# ---------------- experiment configuration (notebook CELL 3) -----------------
N_EXEMPLARS="${N_EXEMPLARS:-3}"                  # k visual prompts (anchor + k-1 random)
USE_TILING="${USE_TILING:-1}"                    # 1 = overlapping tiles, 0 = whole image
TILE_SIZE="${TILE_SIZE:-1536}"
OVERLAP="${OVERLAP:-384}"
GLOBAL_CONTEXT_PASS="${GLOBAL_CONTEXT_PASS:-1}"  # ADD_GLOBAL_CONTEXT_PASS = True (extra path)
GLOBAL_DOWNSCALE="${GLOBAL_DOWNSCALE:-2}"        # GLOBAL_DOWNSCALE
IMGSZ="${IMGSZ:-1024}"                           # IMGSZ (E01 value; tiles fed at 0.67x)
THRESHOLD="${THRESHOLD:-0.30}"                   # YOLOE_INFERENCE_THRESHOLD
PREDICT_NMS_IOU="${PREDICT_NMS_IOU:-0.90}"       # YOLOE_PREDICT_NMS_IOU (permissive)
MAX_DET="${MAX_DET:-300}"                        # MAX_DET per tile
MASK_BINARISE="${MASK_BINARISE:-0.50}"           # MASK_BINARISE
BATCH_SIZE="${BATCH_SIZE:-4}"                    # BATCH_SIZE
USE_FP16="${USE_FP16:-1}"                        # USE_FP16 = True (notebook)
MIN_FILL_RATIO="${MIN_FILL_RATIO:-0.15}"
MAX_AREA_FRACTION="${MAX_AREA_FRACTION:-0.80}"
EDGE_MARGIN="${EDGE_MARGIN:-5}"
EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

# The notebook keeps the operating point EQUAL to the inference threshold, so a
# single confidence value (0.30) governs both the YOLOE pass and every
# precision / recall / F1 / IoU1 / IoU2 number. No sweep is performed.
OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"   # applied offline in PHASE 2

# --------------------------- qualitative plot --------------------------------
PLOT_EVALUATION_MODE="${PLOT_EVALUATION_MODE:-all_gt}"
PLOT_MIN_GT_BOXES="${PLOT_MIN_GT_BOXES:-7}"
PLOT_MAX_DISPLAY_DIM="${PLOT_MAX_DISPLAY_DIM:-2048}"

# YOLOE checkpoint. Must live on $SCRATCH: compute nodes cannot download it.
YOLOE_WEIGHTS_NAME="${YOLOE_WEIGHTS_NAME:-yoloe-11l-seg.pt}"
YOLOE_WEIGHTS_DIR="${YOLOE_WEIGHTS_DIR:-${SCRATCH}/yoloe_weights}"
YOLOE_WEIGHTS="${YOLOE_WEIGHTS:-${YOLOE_WEIGHTS_DIR}/${YOLOE_WEIGHTS_NAME}}"
YOLOE_WEIGHTS_URL="${YOLOE_WEIGHTS_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/${YOLOE_WEIGHTS_NAME}}"

# Ultralytics writes a settings file; keep it off the small $HOME.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${SCRATCH}/ultralytics_config}"

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

echo "===== k_exemplar_tiling YOLOE ${MODE^^} ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(command -v python || command -v python3)"
echo "script_dir=${SCRIPT_DIR}"
echo "experiment=${EXPERIMENT_NAME}"
echo "n_exemplars=${N_EXEMPLARS}  use_tiling=${USE_TILING}  tile=${TILE_SIZE}  overlap=${OVERLAP}"
echo "global_pass=${GLOBAL_CONTEXT_PASS}  global_downscale=${GLOBAL_DOWNSCALE}"
echo "yoloe_threshold=${THRESHOLD}  predict_nms=${PREDICT_NMS_IOU}  operating_point=conf:${OPERATING_CONFIDENCE}/nms:${NMS_IOU_THRESHOLD}"
echo "imgsz=${IMGSZ}  fp16=${USE_FP16}  batch_size=${BATCH_SIZE}"
echo "dataset_root=${DATASET_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "weights=${YOLOE_WEIGHTS}"
echo "yolo_config_dir=${YOLO_CONFIG_DIR}"
echo "pyextra=${PYEXTRA}"
echo "num_gpus=${NUM_GPUS}"
echo "================================"

mkdir -p "${EXPERIMENT_ROOT}" "${YOLOE_ROOT}" "${OUTPUT_DIR}" "${YOLOE_WEIGHTS_DIR}" \
         "${YOLO_CONFIG_DIR}" "${PYEXTRA}"

# ---------------------------- python debug -----------------------------------
echo "===== PYTHON DEBUG ====="
python --version
python -m pip --version || true
python -m pip show ultralytics || true

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
print("YOLO_CONFIG_DIR:", os.environ.get("YOLO_CONFIG_DIR"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

YOLOE_WEIGHTS="${YOLOE_WEIGHTS}" python - <<'PY'
import os, sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| device count:", torch.cuda.device_count())
    print("float16 supported:", torch.cuda.is_available())
except Exception as exc:
    print("torch import FAILED:", exc)
try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__, ultralytics.__file__)
    from ultralytics import YOLOE  # noqa: F401
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    print("YOLOE import: OK")
    print("YOLOEVPSegPredictor.get_vpe:",
          "OK" if hasattr(YOLOEVPSegPredictor, "get_vpe") else "MISSING (ultralytics too old)")
    from ultralytics.cfg import DEFAULT_CFG_DICT
    print("precision argument:",
          "quantize=16 (as in the notebook)" if "quantize" in DEFAULT_CFG_DICT
          else "half=True (older ultralytics; used automatically)")
except Exception as exc:
    print("ultralytics/YOLOE import FAILED:", exc)
w = os.environ.get("YOLOE_WEIGHTS", "")
print("weights file:", w, "->", "FOUND" if os.path.isfile(w) else "MISSING (run the download mode)")
try:
    import supervision
    print("supervision:", supervision.__version__, supervision.__file__)
    from supervision.metrics import MeanAveragePrecision  # noqa: F401
    print("MeanAveragePrecision import: OK")
except Exception as exc:
    print("supervision import FAILED:", exc)
    print("  -> inference will refuse to start. Run './k_exemplars_with_tiling_extra_path_run_yoloe.sh download' on a login node.")
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
    --weights "${YOLOE_WEIGHTS}"
    --n-exemplars "${N_EXEMPLARS}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
    --imgsz "${IMGSZ}"
    --threshold "${THRESHOLD}"
    --predict-nms-iou "${PREDICT_NMS_IOU}"
    --max-det "${MAX_DET}"
    --mask-binarise "${MASK_BINARISE}"
    --batch-size "${BATCH_SIZE}"
    --min-fill-ratio "${MIN_FILL_RATIO}"
    --max-area-fraction "${MAX_AREA_FRACTION}"
    --edge-margin "${EDGE_MARGIN}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --nms-iou-threshold "${NMS_IOU_THRESHOLD}"
    --operating-confidence "${OPERATING_CONFIDENCE}"
    --plot-evaluation-mode "${PLOT_EVALUATION_MODE}"
    --plot-min-gt-boxes "${PLOT_MIN_GT_BOXES}"
    --plot-max-display-dim "${PLOT_MAX_DISPLAY_DIM}"
)

if [[ "${USE_TILING}" == "1" ]]; then
    COMMON_ARGS+=(--tiling)
else
    COMMON_ARGS+=(--no-tiling)
fi

COMMON_ARGS+=(--global-downscale "${GLOBAL_DOWNSCALE}")
if [[ "${GLOBAL_CONTEXT_PASS}" == "1" ]]; then
    COMMON_ARGS+=(--global-pass)
else
    COMMON_ARGS+=(--no-global-pass)
fi

if [[ "${USE_FP16}" == "1" ]]; then
    COMMON_ARGS+=(--fp16)
else
    COMMON_ARGS+=(--no-fp16)
fi

# The evaluation needs to know where things are, which operating point is used
# and where the dataset is, because the qualitative figures reopen the selected
# images.
EVAL_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --weights "${YOLOE_WEIGHTS}"
    --n-exemplars "${N_EXEMPLARS}"
    --imgsz "${IMGSZ}"
    --threshold "${THRESHOLD}"
    --predict-nms-iou "${PREDICT_NMS_IOU}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --nms-iou-threshold "${NMS_IOU_THRESHOLD}"
    --operating-confidence "${OPERATING_CONFIDENCE}"
    --plot-evaluation-mode "${PLOT_EVALUATION_MODE}"
    --plot-min-gt-boxes "${PLOT_MIN_GT_BOXES}"
    --plot-max-display-dim "${PLOT_MAX_DISPLAY_DIM}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
)

if [[ "${USE_TILING}" == "1" ]]; then
    EVAL_ARGS+=(--tiling)
else
    EVAL_ARGS+=(--no-tiling)
fi

EVAL_ARGS+=(--global-downscale "${GLOBAL_DOWNSCALE}")
if [[ "${GLOBAL_CONTEXT_PASS}" == "1" ]]; then
    EVAL_ARGS+=(--global-pass)
else
    EVAL_ARGS+=(--no-global-pass)
fi

case "${MODE}" in
    download)
        # Runs on a LOGIN node (internet). Puts the checkpoint on $SCRATCH so the
        # compute node can load the model completely offline.
        echo "--- 1/3 : YOLOE weights -> ${YOLOE_WEIGHTS} ---"
        if [[ -s "${YOLOE_WEIGHTS}" ]]; then
            echo "Already present, skipped."
        elif curl -fL --retry 3 -o "${YOLOE_WEIGHTS}.part" "${YOLOE_WEIGHTS_URL}"; then
            mv "${YOLOE_WEIGHTS}.part" "${YOLOE_WEIGHTS}"
        else
            rm -f "${YOLOE_WEIGHTS}.part"
            echo "  curl failed; trying Ultralytics' own downloader ..."
            (cd "${YOLOE_WEIGHTS_DIR}" && python - <<PY
from ultralytics import YOLOE
YOLOE("${YOLOE_WEIGHTS_NAME}")   # downloads into the current folder
PY
            )
        fi
        if [[ ! -s "${YOLOE_WEIGHTS}" ]]; then
            echo "ERROR: ${YOLOE_WEIGHTS} is missing. Download ${YOLOE_WEIGHTS_URL}"
            echo "       manually and copy it there."
            exit 1
        fi
        ls -l "${YOLOE_WEIGHTS}"
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
        # and qualitative PNGs). Most containers already have them; if the dryrun says they are
        # missing, install them INSIDE the container from these wheels, always
        # with --no-deps so that numpy / pillow are never shadowed.
        mkdir -p "${WHEELS}"
        pip download --no-deps -d "${WHEELS}" \
            pandas matplotlib supervision \
            contourpy cycler fonttools kiwisolver pyparsing packaging \
            python-dateutil pytz tzdata six || \
            echo "  (warning: some wheels could not be downloaded; see 'Problem B')"
        ls -1 "${WHEELS}" | head -20
        echo "Done. Compute nodes can now run offline."
        echo "NOTE: this check ran with the LOGIN node's python. Verify inside the container"
        echo "      with: ./k_exemplars_with_tiling_extra_path_run_yoloe.sh dryrun   (the PYTHON DEBUG block prints the result)."
        ;;

    dryrun)
        # No model, no GPU: just proves the dataset layout is understood and
        # prints the cost of the experiment.
        python "${PY_SCRIPT}" "${COMMON_ARGS[@]}" --dry-run "$@"
        ;;

    run)
        # Stop Ultralytics from trying to reach the internet on the compute node.
        export YOLO_OFFLINE="${YOLO_OFFLINE:-1}"
        if [[ ! -s "${YOLOE_WEIGHTS}" ]]; then
            echo "ERROR: ${YOLOE_WEIGHTS} does not exist and the compute node is offline."
            echo "       Run './k_exemplars_with_tiling_extra_path_run_yoloe.sh download' on a login node first."
            exit 1
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
        # PHASE 2 only. Never loads YOLOE, so it runs fine on a login node or in a
        # small CPU allocation.
        python "${PY_SCRIPT}" "${EVAL_ARGS[@]}" --evaluate-only "$@"
        ;;

    *)
        echo "Unknown mode: ${MODE} (expected: download|dryrun|run|evaluate)"
        exit 1
        ;;
esac
