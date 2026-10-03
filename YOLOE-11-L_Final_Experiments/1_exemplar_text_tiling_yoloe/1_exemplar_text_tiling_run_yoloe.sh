#!/bin/bash
#
# 1_exemplar_text_tiling_run_yoloe.sh  ==>  dispatcher for 1_exemplar_text_tiling_yoloe
#     (YOLOE prompted with ONE exemplar (the anchor) AND a text prompt, fused
#      into one class embedding, tiling ON; cluster port of
#      E01_single_image_1_exemplar_text_YOLOE_11_tiling.ipynb, the YOLOE version
#      of the SAM3 notebook 1_exemplar_text_tiling)
#
# MODES
#   download    Pre-fetch yoloe-11l-seg.pt AND the YOLOE text encoder into
#               $YOLOE_WEIGHTS_DIR, install `supervision` and the CLIP tokenizer
#               into $PYEXTRA (plus wheels for the evaluation packages). RUN
#               THIS ON A LOGIN NODE FIRST: CSCS compute nodes have no internet.
#               YOLOE is NOT gated: no HF token is needed.
#   dryrun      Discover the dataset and print the run plan. No GPU, no model.
#               Use it to sanity-check DATASET_ROOT and to estimate the runtime
#               before burning an allocation.
#   textpe      Only compute + cache the text embedding (needs the container;
#               'run' does this automatically as its first step).
#   run         (default) text embedding (once) + PHASE 1 + PHASE 2. Launches NUM_GPUS shard processes,
#               one per GPU, waits for all of them, then runs the offline
#               evaluation once over everything that reached disk.
#
# NOTE: the SAM3 experiments and this one share $PYEXTRA and $WHEELS, so the
#       supervision part of the download mode only has to be done once.
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
PY_SCRIPT="${SCRIPT_DIR}/1_exemplar_text_tiling_infer_yoloe.py"

# ------------------- paths and experiment identity ---------------------------
EXPERIMENT_ROOT="${SCRATCH}/experiments"
YOLOE_ROOT="${EXPERIMENT_ROOT}/yoloe"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-1_exemplar_text_tiling_yoloe}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${YOLOE_ROOT}/${RUN_NAME}}"
DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"

# ---------------- experiment configuration (notebook CELL 3) -----------------
N_EXEMPLARS="${N_EXEMPLARS:-1}"                  # 1 visual prompt: the anchor only
TEXT_PROMPT="${TEXT_PROMPT:-Rumex obtusifolius}" # text prompt fused with the exemplar
FUSION_ALPHA="${FUSION_ALPHA:-0.5}"              # e = norm(a*e_visual + (1-a)*e_text)
USE_TILING="${USE_TILING:-1}"                    # 1 = overlapping tiles, 0 = whole image
TILE_SIZE="${TILE_SIZE:-1000}"
OVERLAP="${OVERLAP:-150}"
IMGSZ="${IMGSZ:-1024}"                           # IMGSZ (>= TILE_SIZE, multiple of 32)
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

# YOLOE text encoder (MobileCLIP). Ultralytics downloads it on the first text
# prompt; the download mode puts it next to the weights, where the python script
# looks for it. The text embedding itself is cached in TEXT_PE_DIR (computed once).
TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME:-mobileclip_blt.ts}"
TEXT_ENCODER_URL="${TEXT_ENCODER_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/${TEXT_ENCODER_NAME}}"
TEXT_PE_DIR="${TEXT_PE_DIR:-${YOLOE_WEIGHTS_DIR}/text_pe}"
TEXT_SLUG="$(printf '%s' "${TEXT_PROMPT}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
TEXT_PE_CACHE="${TEXT_PE_CACHE:-${TEXT_PE_DIR}/${YOLOE_WEIGHTS_NAME%.pt}__${TEXT_SLUG}.pt}"

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

echo "===== 1_exemplar_text_tiling YOLOE ${MODE^^} ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(command -v python || command -v python3)"
echo "script_dir=${SCRIPT_DIR}"
echo "experiment=${EXPERIMENT_NAME}"
echo "n_exemplars=${N_EXEMPLARS}  use_tiling=${USE_TILING}  tile=${TILE_SIZE}  overlap=${OVERLAP}"
echo "text_prompt='${TEXT_PROMPT}'  fusion_alpha=${FUSION_ALPHA}"
echo "text_pe_cache=${TEXT_PE_CACHE}"
echo "yoloe_threshold=${THRESHOLD}  predict_nms=${PREDICT_NMS_IOU}  operating_point=conf:${OPERATING_CONFIDENCE}/nms:${NMS_IOU_THRESHOLD}"
echo "imgsz=${IMGSZ}  fp16=${USE_FP16}  batch_size=${BATCH_SIZE}"
echo "dataset_root=${DATASET_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "weights=${YOLOE_WEIGHTS}"
echo "yolo_config_dir=${YOLO_CONFIG_DIR}"
echo "pyextra=${PYEXTRA}"
echo "num_gpus=${NUM_GPUS}"
echo "================================"

mkdir -p "${EXPERIMENT_ROOT}" "${YOLOE_ROOT}" "${OUTPUT_DIR}" "${YOLOE_WEIGHTS_DIR}" "${TEXT_PE_DIR}" \
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

YOLOE_WEIGHTS="${YOLOE_WEIGHTS}" TEXT_PE_CACHE="${TEXT_PE_CACHE}" TEXT_ENCODER_NAME="${TEXT_ENCODER_NAME}" python - <<'PY'
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
enc = os.path.join(os.path.dirname(w), os.environ.get("TEXT_ENCODER_NAME", ""))
print("text encoder file:", enc, "->", "FOUND" if os.path.isfile(enc) else "MISSING (see 'Problem C')")
try:
    import clip  # noqa: F401  (tokenizer used by the YOLOE text encoder)
    print("CLIP tokenizer import: OK")
except Exception as exc:
    print("CLIP tokenizer import FAILED:", exc, "(needed only to compute the text embedding once)")
t = os.environ.get("TEXT_PE_CACHE", "")
print("text embedding cache:", t, "->", "FOUND" if os.path.isfile(t) else "not yet (the run mode prepares it)")
try:
    import supervision
    print("supervision:", supervision.__version__, supervision.__file__)
    from supervision.metrics import MeanAveragePrecision  # noqa: F401
    print("MeanAveragePrecision import: OK")
except Exception as exc:
    print("supervision import FAILED:", exc)
    print("  -> inference will refuse to start. Run './1_exemplar_text_tiling_run_yoloe.sh download' on a login node.")
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
    --text-prompt "${TEXT_PROMPT}"
    --fusion-alpha "${FUSION_ALPHA}"
    --text-pe-cache "${TEXT_PE_CACHE}"
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
    --text-prompt "${TEXT_PROMPT}"
    --fusion-alpha "${FUSION_ALPHA}"
    --text-pe-cache "${TEXT_PE_CACHE}"
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

case "${MODE}" in
    download)
        # Runs on a LOGIN node (internet). Puts the checkpoint on $SCRATCH so the
        # compute node can load the model completely offline.
        echo "--- 1/4 : YOLOE weights -> ${YOLOE_WEIGHTS} ---"
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
        echo "--- 2/4 : YOLOE text encoder + CLIP tokenizer (for the text prompt) ---"
        # Needed ONCE, to compute the text embedding before the shards start.
        # Best effort: if it fails, see 'Problem C' (compute the embedding in Colab).
        TEXT_ENCODER_PATH="${YOLOE_WEIGHTS_DIR}/${TEXT_ENCODER_NAME}"
        if [[ -s "${TEXT_ENCODER_PATH}" ]]; then
            echo "Text encoder already present, skipped."
        elif curl -fL --retry 3 -o "${TEXT_ENCODER_PATH}.part" "${TEXT_ENCODER_URL}"; then
            mv "${TEXT_ENCODER_PATH}.part" "${TEXT_ENCODER_PATH}"
        else
            rm -f "${TEXT_ENCODER_PATH}.part"
            echo "  WARNING: could not download ${TEXT_ENCODER_URL} (see 'Problem C')."
        fi
        ls -l "${TEXT_ENCODER_PATH}" 2>/dev/null || true
        pip install --target "${PYEXTRA}" --no-deps --upgrade \
            "git+https://github.com/ultralytics/CLIP.git" ftfy wcwidth || \
            echo "  WARNING: CLIP tokenizer install failed (see 'Problem C')."
        echo "--- 3/4 : supervision -> ${PYEXTRA} ---"
        pip install --target "${PYEXTRA}" --no-deps --upgrade supervision
        python - <<'PY'
import sys
print("checking the freshly installed package is importable ...")
import supervision
from supervision.metrics import MeanAveragePrecision  # noqa: F401
print("supervision", supervision.__version__, "->", supervision.__file__)
PY
        echo "--- 4/4 : wheels for the PHASE 2 packages -> ${WHEELS} ---"
        # PHASE 2 needs pandas (metric tables) and matplotlib (confusion-matrix
        # and qualitative PNGs). Most containers already have them; if the dryrun says they are
        # missing, install them INSIDE the container from these wheels, always
        # with --no-deps so that numpy / pillow are never shadowed.
        mkdir -p "${WHEELS}"
        pip download --no-deps -d "${WHEELS}" \
            pandas matplotlib supervision \
            contourpy cycler fonttools kiwisolver pyparsing packaging \
            python-dateutil pytz tzdata six regex || \
            echo "  (warning: some wheels could not be downloaded; see 'Problem B')"
        ls -1 "${WHEELS}" | head -20
        echo "Done. Compute nodes can now run offline."
        echo "NOTE: this check ran with the LOGIN node's python. Verify inside the container"
        echo "      with: ./1_exemplar_text_tiling_run_yoloe.sh dryrun   (the PYTHON DEBUG block prints the result)."
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
            echo "       Run './1_exemplar_text_tiling_run_yoloe.sh download' on a login node first."
            exit 1
        fi

        echo "===== STEP 0 : TEXT EMBEDDING (computed once, cached) ====="
        # e_text is identical for every image -> computed ONCE here, the shards
        # only load it. Uses GPU 0; a cached file is simply verified.
        CUDA_VISIBLE_DEVICES=0 python "${PY_SCRIPT}" "${COMMON_ARGS[@]}" --prepare-text-pe

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

    textpe)
        # Only compute / verify the cached text embedding (inside the container).
        export YOLO_OFFLINE="${YOLO_OFFLINE:-1}"
        python "${PY_SCRIPT}" "${COMMON_ARGS[@]}" --prepare-text-pe "$@"
        ;;

    evaluate)
        # PHASE 2 only. Never loads YOLOE, so it runs fine on a login node or in a
        # small CPU allocation.
        python "${PY_SCRIPT}" "${EVAL_ARGS[@]}" --evaluate-only "$@"
        ;;

    *)
        echo "Unknown mode: ${MODE} (expected: download|dryrun|textpe|run|evaluate)"
        exit 1
        ;;
esac
