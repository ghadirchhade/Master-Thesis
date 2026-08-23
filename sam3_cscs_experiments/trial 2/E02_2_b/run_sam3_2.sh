#!/bin/bash

#  run_sam3_2.sh  ==>  dispatcher for the SAM3 TILED MULTI-BOX + GLOBAL PASS Rumex
#  experiment (E02_2_b)
#
#  Sets up the environment, decides where everything lives under $SCRATCH, and launches
#  infer_sam3_2.py -- the port of the E02_2 Colab notebook: tiling ON (2000 px tiles,
#  800 px overlap), THREE exemplar crops in the strip, plus one extra forward pass on the
#  whole image downscaled by 2, all merged with a single NMS. No tile batching.
#
#  MODES
#    download    Pre-fetch facebook/sam3 into $HF_HOME and install `supervision` into
#                $PYEXTRA. RUN THIS ON A LOGIN NODE FIRST: CSCS compute nodes have no
#                internet, and facebook/sam3 is a GATED repo, so you must (a) accept the
#                licence at https://huggingface.co/facebook/sam3 with the account owning
#                $HF_TOKEN, and (b) export HF_TOKEN before calling.
#    dryrun      Discover the dataset and print the run plan. No GPU, no model. Use this to
#                sanity-check --dataset-root and to estimate runtime before burning an
#                allocation.
#    run         (default) The real thing. Launches NUM_GPUS shard processes, one per GPU,
#                waits for all of them, then aggregates.
#    aggregate   Merge existing shard CSVs and rebuild every summary. No inference.


set -euo pipefail

# Environment
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths and experiment identity
EXPERIMENT_ROOT="${SCRATCH}/experiments"
SAM3_ROOT="${EXPERIMENT_ROOT}/sam3"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-E02_2_b}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SAM3_ROOT}/${RUN_NAME}}"


DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"

# Experiment configuration
PRESET="${PRESET:-multi}"               # multi = 3 exemplar crops (E02_2_b), single = 1
TILE_SIZE="${TILE_SIZE:-2000}"          # notebook value
OVERLAP="${OVERLAP:-800}"               # notebook value
THRESHOLD="${THRESHOLD:-0.3}"           # SAM3 confidence threshold
MASK_THRESHOLD="${MASK_THRESHOLD:-0.4}" # mask binarisation, as in the notebook
IOU_THRESHOLD="${IOU_THRESHOLD:-0.5}"   # TP matching threshold
NMS_IOU="${NMS_IOU:-0.5}"               # merging detections across tiles + global pass
GLOBAL_PASS="${GLOBAL_PASS:-1}"         # 1 = extra downscaled whole-image pass (notebook)
GLOBAL_DOWNSCALE="${GLOBAL_DOWNSCALE:-2}"
DTYPE="${DTYPE:-float32}"
SAVE_MASKS="${SAVE_MASKS:-0}"
CACHE_TILES="${CACHE_TILES:-1}"         # 1 = crop the tiles once per image and reuse them

# HuggingFace cache. Must live on $SCRATCH: $HOME is small and the weights are several GB.
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
MODEL_ID="${MODEL_ID:-facebook/sam3}"


PYEXTRA="${PYEXTRA:-${SCRATCH}/pyextra}"
export PYTHONPATH="${PYEXTRA}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${SLURM_GPUS_PER_TASK:-}" ]]; then
    DEFAULT_GPUS="${SLURM_GPUS_PER_TASK}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    DEFAULT_GPUS="$(nvidia-smi -L | wc -l)"
else
    DEFAULT_GPUS=1
fi
NUM_GPUS="${NUM_GPUS:-${DEFAULT_GPUS}}"
[[ "${NUM_GPUS}" -lt 1 ]] && NUM_GPUS=1

# Mode dispatch: first positional argument, everything after it is forwarded to python
MODE="run"
if [[ $# -gt 0 ]]; then
    MODE="$1"
    shift
fi

echo "===== SAM3 MULTI-BOX + GLOBAL PASS ${MODE^^} ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(command -v python || command -v python3)"
echo "script_dir=${SCRIPT_DIR}"
echo "experiment=${EXPERIMENT_NAME}  preset=${PRESET}"
echo "dataset_root=${DATASET_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "model_id=${MODEL_ID}"
echo "hf_home=${HF_HOME}"
echo "pyextra=${PYEXTRA}"
echo "tile=${TILE_SIZE} overlap=${OVERLAP} conf=${THRESHOLD} nms_iou=${NMS_IOU} cache_tiles=${CACHE_TILES}"
echo "global_pass=${GLOBAL_PASS} global_downscale=${GLOBAL_DOWNSCALE}"
echo "num_gpus=${NUM_GPUS}"
echo "=================================================="

mkdir -p "${EXPERIMENT_ROOT}" "${SAM3_ROOT}" "${OUTPUT_DIR}" "${HF_HOME}" "${PYEXTRA}"

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
except Exception as exc:
    print("torch import FAILED:", exc)
try:
    import transformers
    print("transformers:", transformers.__version__, transformers.__file__)
    from transformers import Sam3Model  # noqa: F401
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
    print("  -> inference will refuse to start. Run './run_sam3_2.sh download' on a login node.")
PY
echo "========================"

# Shared python arguments
COMMON_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --preset "${PRESET}"
    --model-id "${MODEL_ID}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
    --threshold "${THRESHOLD}"
    --mask-threshold "${MASK_THRESHOLD}"
    --iou-threshold "${IOU_THRESHOLD}"
    --nms-iou "${NMS_IOU}"
    --global-downscale "${GLOBAL_DOWNSCALE}"
    --dtype "${DTYPE}"
)
if [[ "${GLOBAL_PASS}" == "1" ]]; then
    COMMON_ARGS+=(--global-pass)
else
    COMMON_ARGS+=(--no-global-pass)
fi
if [[ "${SAVE_MASKS}" == "1" ]]; then
    COMMON_ARGS+=(--save-masks)
fi
if [[ "${CACHE_TILES}" == "1" ]]; then
    COMMON_ARGS+=(--cache-tiles)
else
    COMMON_ARGS+=(--no-cache-tiles)
fi

case "${MODE}" in

    download)
        # Runs on a LOGIN node (internet + your HF token). Populates $HF_HOME so the compute
        # node can load the model completely offline.
        if [[ -z "${HF_TOKEN:-}" ]]; then
            echo "ERROR: HF_TOKEN is not set. facebook/sam3 is a gated repo:"
            echo "  1) accept the licence at https://huggingface.co/facebook/sam3"
            echo "  2) export HF_TOKEN=hf_xxx"
            exit 1
        fi
        echo "--- 1/2 : model weights -> ${HF_HOME} ---"
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

        echo "--- 2/2 : supervision -> ${PYEXTRA} ---"
        pip install --target "${PYEXTRA}" --no-deps --upgrade supervision
        python - <<'PY'
import sys
print("checking the freshly installed package is importable ...")
import supervision
from supervision.metrics import MeanAveragePrecision  # noqa: F401
print("supervision", supervision.__version__, "->", supervision.__file__)
PY
        echo "Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1)."
        echo "NOTE: this check ran with the LOGIN node's python. Verify inside the container"
        echo "      with: ./run_sam3_2.sh dryrun   (the PYTHON DEBUG block prints the result)."
        ;;

    dryrun)
        # No model, no GPU: just proves the dataset layout is understood and prints the cost.
        python "${SCRIPT_DIR}/infer_sam3_2.py" "${COMMON_ARGS[@]}" --dry-run "$@"
        ;;

    run)
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
        if [[ ! -d "${HF_HOME}" || -z "$(ls -A "${HF_HOME}" 2>/dev/null)" ]]; then
            echo "WARNING: ${HF_HOME} looks empty and we are running offline."
            echo "         Run './run_sam3_2.sh download' on a login node first."
        fi

        echo "Launching ${NUM_GPUS} shard process(es) ..."
        PIDS=()
        for ((i = 0; i < NUM_GPUS; i++)); do
            LOG="${OUTPUT_DIR}/shard${i}.log"
            CUDA_VISIBLE_DEVICES="${i}" \
            python "${SCRIPT_DIR}/infer_sam3_2.py" \
                "${COMMON_ARGS[@]}" \
                --num-shards "${NUM_GPUS}" \
                --shard-index "${i}" \
                --no-aggregate \
                "$@" > >(tee "${LOG}") 2>&1 &
            PIDS+=("$!")
            echo "  shard ${i} -> pid ${PIDS[-1]}  (log: ${LOG})"
        done

        # Wait for every shard and remember whether any of them failed
        FAILED=0
        for idx in "${!PIDS[@]}"; do
            if wait "${PIDS[$idx]}"; then
                echo "  shard ${idx} finished OK"
            else
                echo "  shard ${idx} FAILED (exit $?)"
                FAILED=1
            fi
        done

        echo "===== AGGREGATION ====="
        # Always aggregate: even a partially failed run should produce usable summaries of
        # whatever rows made it to disk.
        python "${SCRIPT_DIR}/infer_sam3_2.py" \
            --output-dir "${OUTPUT_DIR}" \
            --experiment-name "${EXPERIMENT_NAME}" \
            --aggregate-only

        if [[ "${FAILED}" -ne 0 ]]; then
            echo "At least one shard failed -- see the per-shard logs above."
            exit 1
        fi
        echo "All shards completed. Results in ${OUTPUT_DIR}"
        ;;

    aggregate)
        python "${SCRIPT_DIR}/infer_sam3_2.py" \
            --output-dir "${OUTPUT_DIR}" \
            --experiment-name "${EXPERIMENT_NAME}" \
            --aggregate-only "$@"
        ;;

    *)
        echo "Unknown mode: ${MODE} (expected: download|dryrun|run|aggregate)"
        exit 1
        ;;
esac