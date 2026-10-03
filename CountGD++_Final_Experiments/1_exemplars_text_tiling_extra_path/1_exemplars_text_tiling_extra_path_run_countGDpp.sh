#!/bin/bash
#
# 1_exemplars_text_tiling_extra_path_run_countGDpp.sh  ==>  dispatcher for 1_exemplars_text_tiling_extra_path_countGDpp
#     (CountGD++ SINGLE-exemplar + TEXT prompted Rumex detection, tiling ON with
#      1536 px tiles / 384 px overlap, PLUS the "extra path": a downsampled
#      whole-image global-context pass merged into the tiled detections before NMS.
#      Text prompt " rumex obtusifolius " sent with the exemplar in every pass)
#
# MODES
#   download    LOGIN NODE ONLY (needs internet). CSCS compute nodes have no
#               internet, so everything CountGD++ needs is fetched onto $SCRATCH:
#                 1. git clone of the CountGD++ repository  -> $COUNTGD_REPO
#                 2. login-node tools (gdown, huggingface_hub) -> $COUNTGD_PYTOOLS
#                 3. countgd_plusplus.pth (Google Drive, gdown) -> $COUNTGD_CKPT
#                 4. bert-base-uncased (public, no token)      -> $HF_HOME
#                 5. wheels of the extra python packages       -> $COUNTGD_WHEELS
#   build       INSIDE THE CONTAINER, on a GPU node (no internet needed):
#                 1. installs those wheels into $COUNTGD_PYEXTRA
#                 2. writes <repo>/checkpoints/bert-base-uncased (the repo's
#                    download_bert.py, run offline from the HF cache)
#                 3. compiles the MultiScaleDeformableAttention CUDA op. If that
#                    fails, inference uses the pure-PyTorch fallback automatically.
#   dryrun      Discover the dataset and print the run plan. No GPU, no model.
#               Use it to sanity-check DATASET_ROOT and to estimate the runtime
#               before burning an allocation.
#   run         (default) PHASE 1 + PHASE 2. Launches NUM_GPUS shard processes,
#               one per GPU, waits for all of them, then runs the offline
#               evaluation once over everything that reached disk.
#   evaluate    PHASE 2 only. Rebuilds every metric, the pooled AP, the
#               confusion matrices and the qualitative figures from the cached
#               NPZ files. No GPU, no model, CountGD++ is never loaded. Repeat as
#               often as you like.
#
# NOTE: CountGD++ needs transformers<5, while SAM3 needs a recent transformers.
#       Its packages therefore live in their OWN folder ($COUNTGD_PYEXTRA), never
#       in the $SCRATCH/pyextra shared by the SAM3 experiments.
# NOTE: every CountGD++ experiment shares $COUNTGD_REPO, $COUNTGD_PYEXTRA,
#       $COUNTGD_WHEELS and $HF_HOME, so 'download' and 'build' only have to be
#       run once for all of them (e.g. already done for 1_exemplars_tiling_countGDpp).

set -euo pipefail

# ----------------------------- environment -----------------------------------
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/1_exemplars_text_tiling_extra_path_infer_countGDpp.py"

# ------------------- paths and experiment identity ---------------------------
EXPERIMENT_ROOT="${SCRATCH}/experiments"
COUNTGD_ROOT="${EXPERIMENT_ROOT}/countgdpp"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-1_exemplars_text_tiling_extra_path_countGDpp}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${COUNTGD_ROOT}/${RUN_NAME}}"

DATASET_ROOT="${DATASET_ROOT:-${SCRATCH}/overney/dataset}"

# ------------------------- CountGD++ code + weights --------------------------
COUNTGD_REPO="${COUNTGD_REPO:-${SCRATCH}/CountGDPlusPlus}"
COUNTGD_REPO_URL="${COUNTGD_REPO_URL:-https://github.com/niki-amini-naieni/CountGDPlusPlus.git}"
COUNTGD_CKPT="${COUNTGD_CKPT:-${COUNTGD_REPO}/checkpoints/countgd_plusplus.pth}"
COUNTGD_CKPT_ID="${COUNTGD_CKPT_ID:-1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs}"   # Google Drive id from the README
OPS_DIR="${COUNTGD_REPO}/models/GroundingDINO/ops"

# ---------------- experiment configuration (notebook CELL 3) -----------------
N_EXEMPLARS="${N_EXEMPLARS:-1}"                  # ONE visual prompt (the anchor)
USE_TILING="${USE_TILING:-1}"                    # 1 = overlapping tiles, 0 = whole image
TILE_SIZE="${TILE_SIZE:-1536}"                   # extra-path notebook: 1536 (not 1000)
OVERLAP="${OVERLAP:-384}"                        # plants <= 384 px are whole in >= 1 tile

# ------------------- extra path: global-context pass (CELL 16) ---------------
ADD_GLOBAL_CONTEXT_PASS="${ADD_GLOBAL_CONTEXT_PASS:-1}"   # 1 = ON (notebook default), 0 = tiles only
GLOBAL_DOWNSCALE="${GLOBAL_DOWNSCALE:-2}"                 # whole image / 2, then CountGD++ resize

# ------------------------------ safety net (CELL 3) --------------------------
MEM_STOP_THRESHOLD_PCT="${MEM_STOP_THRESHOLD_PCT:-70}"    # stop a shard cleanly above this RAM %

THRESHOLD="${THRESHOLD:-0.30}"                   # CONFIDENCE_THRESHOLD inside CountGD++ (default 0.23)
BATCH_SIZE="${BATCH_SIZE:-1}"                    # FIXED: CountGD++ = one tile per forward pass
DTYPE="${DTYPE:-float32}"                        # USE_FP16 = False (the notebook)
TEXT_PROMPT="${TEXT_PROMPT:- rumex obtusifolius }"   # sent WITH the exemplar (positive prompt)
EXEMPLAR_SCALE_MODE="${EXEMPLAR_SCALE_MODE:-match_tile}"
MODEL_SHORT_SIDE="${MODEL_SHORT_SIDE:-800}"
MODEL_MAX_SIZE="${MODEL_MAX_SIZE:-1333}"
SEED="${SEED:-42}"

MAX_AREA_FRACTION="${MAX_AREA_FRACTION:-0.80}"
EDGE_MARGIN="${EDGE_MARGIN:-5}"

EVAL_IOU_THRESHOLD="${EVAL_IOU_THRESHOLD:-0.50}"
PROMPT_IGNORE_IOU="${PROMPT_IGNORE_IOU:-0.50}"

# The notebook keeps the operating point EQUAL to the inference threshold, so a
# single confidence value (0.30) governs both the CountGD++ pass and every
# precision / recall / F1 / IoU1 / IoU2 / count_abs_error number. No sweep.
OPERATING_CONFIDENCE="${OPERATING_CONFIDENCE:-0.30}"
NMS_IOU_THRESHOLD="${NMS_IOU_THRESHOLD:-0.40}"   # applied offline in PHASE 2

# --------------------------- qualitative plot --------------------------------
PLOT_EVALUATION_MODE="${PLOT_EVALUATION_MODE:-all_gt}"
PLOT_MIN_GT_BOXES="${PLOT_MIN_GT_BOXES:-7}"
PLOT_MAX_DISPLAY_DIM="${PLOT_MAX_DISPLAY_DIM:-2048}"

# HuggingFace cache: only bert-base-uncased (CountGD++'s text encoder) is needed.
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"

# ---------------------- extra python packages --------------------------------
COUNTGD_PYEXTRA="${COUNTGD_PYEXTRA:-${SCRATCH}/pyextra_countgdpp}"
export PYTHONPATH="${COUNTGD_PYEXTRA}${PYTHONPATH:+:${PYTHONPATH}}"
COUNTGD_WHEELS="${COUNTGD_WHEELS:-${SCRATCH}/wheels_countgdpp}"
COUNTGD_PYTOOLS="${COUNTGD_PYTOOLS:-${SCRATCH}/pytools_countgdpp}"   # login node only
CONTAINER_ARCH="${CONTAINER_ARCH:-aarch64}"                          # GH200 = ARM
CONTAINER_PYVERS="${CONTAINER_PYVERS:-3.10 3.11 3.12 3.13}"          # wheels for each

# Notebook CELL 1 pip line ("transformers<5" addict yapf==0.40.1 timm pycocotools
# termcolor supervision), plus what transformers 4.x and yapf 0.40.1 need at
# import time. Installed with --no-deps, so torch / torchvision / numpy / pillow /
# opencv / scipy always come from the container and are never shadowed.
COUNTGD_PACKAGES=(
    "transformers<5"                 # CountGD++'s BERT wrapper needs get_extended_attention_mask
    "huggingface_hub>=0.34,<1.0"     # required by transformers 4.x
    "tokenizers>=0.22,<=0.23"        # required by transformers 4.x
    "safetensors>=0.4.3"
    "regex"
    "addict"                         # the repo's config loader (SLConfig)
    "yapf==0.40.1"                   # SLConfig breaks with newer yapf
    "platformdirs>=3.5.1"            # yapf 0.40.1 deps
    "tomli>=2.0.1"
    "importlib_metadata>=6.6.0"
    "zipp>=3.20"
    "timm"
    "pycocotools"
    "termcolor"
    "supervision"                    # AP50 / AP50:95
    "defusedxml"                     # supervision dep
)

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

echo "===== 1_exemplars_text_tiling_extra_path CountGD++ ${MODE^^} ====="
echo "host=$(hostname)"
echo "pwd=$(pwd)"
echo "python=$(command -v python || command -v python3)"
echo "script_dir=${SCRIPT_DIR}"
echo "experiment=${EXPERIMENT_NAME}"
echo "n_exemplars=${N_EXEMPLARS}  use_tiling=${USE_TILING}  tile=${TILE_SIZE}  overlap=${OVERLAP}"
echo "global_pass=${ADD_GLOBAL_CONTEXT_PASS}  global_downscale=${GLOBAL_DOWNSCALE}  mem_stop=${MEM_STOP_THRESHOLD_PCT}%"
echo "countgd_threshold=${THRESHOLD}  operating_point=conf:${OPERATING_CONFIDENCE}/nms:${NMS_IOU_THRESHOLD}"
echo "dtype=${DTYPE}  batch_size=${BATCH_SIZE}  exemplar_scale=${EXEMPLAR_SCALE_MODE}  text_prompt='${TEXT_PROMPT}'"
echo "dataset_root=${DATASET_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "countgd_repo=${COUNTGD_REPO}"
echo "countgd_ckpt=${COUNTGD_CKPT}"
echo "hf_home=${HF_HOME}"
echo "countgd_pyextra=${COUNTGD_PYEXTRA}"
echo "num_gpus=${NUM_GPUS}"
echo "================================"

mkdir -p "${EXPERIMENT_ROOT}" "${COUNTGD_ROOT}" "${OUTPUT_DIR}" "${HF_HOME}" "${COUNTGD_PYEXTRA}"

# ---------------------------- python debug -----------------------------------
echo "===== PYTHON DEBUG ====="
python --version
python -m pip --version || true
COUNTGD_REPO="${COUNTGD_REPO}" COUNTGD_CKPT="${COUNTGD_CKPT}" python - <<'PY'
import glob, os, sys
print("sys.executable:", sys.executable)
print("sys.path:")
for p in sys.path:
    print("  ", p)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available(),
          "| device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0),
              "| capability:", torch.cuda.get_device_capability(0))
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as exc:
    print("torch import FAILED:", exc)
try:
    import transformers
    from transformers import BertModel, AutoTokenizer  # noqa: F401
    ok = (int(transformers.__version__.split(".")[0]) < 5
          and hasattr(BertModel, "get_extended_attention_mask"))
    print("transformers:", transformers.__version__, transformers.__file__)
    print("transformers<5 for CountGD++:", "OK" if ok else
          "FAILED (CountGD++ needs transformers<5 -> run the 'build' mode)")
except Exception as exc:
    print("transformers import FAILED:", exc)
for name in ["timm", "addict", "yapf", "pycocotools", "termcolor", "scipy", "cv2"]:
    try:
        module = __import__(name)
        print(f"{name}: {getattr(module, '__version__', '?')} -> OK")
    except Exception as exc:
        print(f"{name} import FAILED: {exc}")
try:
    import psutil
    print("psutil:", psutil.__version__, "-> RAM guard: OK",
          f"(system RAM in use now: {psutil.virtual_memory().percent:.0f}%)")
except Exception as exc:
    print("psutil import FAILED:", exc)
    print("  -> inference still runs, but the RAM guard (MEM_STOP_THRESHOLD_PCT) is disabled.")
try:
    import supervision
    from supervision.metrics import MeanAveragePrecision  # noqa: F401
    print("supervision:", supervision.__version__, "-> MeanAveragePrecision import: OK")
except Exception as exc:
    print("supervision import FAILED:", exc)
    print("  -> inference will refuse to start. Run 'download' (login node) then 'build'.")
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

repo, ckpt = os.environ["COUNTGD_REPO"], os.environ["COUNTGD_CKPT"]
print("CountGD++ repo:", repo, "->",
      "OK" if os.path.isfile(os.path.join(repo, "cfg_app.py")) else "MISSING (run 'download')")
if os.path.isfile(ckpt):
    print(f"checkpoint: {ckpt} ({os.path.getsize(ckpt) / 1e9:.2f} GB) -> OK")
else:
    print(f"checkpoint: {ckpt} -> MISSING (run 'download')")
bert = os.path.join(repo, "checkpoints", "bert-base-uncased")
print("BERT folder:", bert, "->",
      "OK" if os.path.isfile(os.path.join(bert, "config.json")) else "MISSING (run 'build')")
so_files = glob.glob(os.path.join(repo, "models", "GroundingDINO", "ops", "build", "lib*",
                                  "MultiScaleDeformableAttention*.so"))
if so_files:
    try:
        import torch  # noqa: F401  (the op needs torch loaded first)
        sys.path.insert(0, os.path.dirname(so_files[0]))
        import MultiScaleDeformableAttention  # noqa: F401
        print("deformable attention: compiled CUDA op -> OK")
    except Exception as exc:
        print("deformable attention: compiled op found but NOT importable (", exc,
              ") -> pure-PyTorch fallback")
else:
    print("deformable attention: no compiled op -> pure-PyTorch fallback (slower, same computation)")
PY
echo "========================"

# ------------- shared python arguments (notebook CELL 3 values) --------------
COMMON_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --countgd-repo "${COUNTGD_REPO}"
    --checkpoint "${COUNTGD_CKPT}"
    --n-exemplars "${N_EXEMPLARS}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
    --threshold "${THRESHOLD}"
    --batch-size "${BATCH_SIZE}"
    --dtype "${DTYPE}"
    --text-prompt "${TEXT_PROMPT}"
    --exemplar-scale-mode "${EXEMPLAR_SCALE_MODE}"
    --model-short-side "${MODEL_SHORT_SIDE}"
    --model-max-size "${MODEL_MAX_SIZE}"
    --seed "${SEED}"
    --max-area-fraction "${MAX_AREA_FRACTION}"
    --edge-margin "${EDGE_MARGIN}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --nms-iou-threshold "${NMS_IOU_THRESHOLD}"
    --operating-confidence "${OPERATING_CONFIDENCE}"
    --plot-evaluation-mode "${PLOT_EVALUATION_MODE}"
    --plot-min-gt-boxes "${PLOT_MIN_GT_BOXES}"
    --plot-max-display-dim "${PLOT_MAX_DISPLAY_DIM}"
    --global-downscale "${GLOBAL_DOWNSCALE}"
    --mem-stop-threshold-pct "${MEM_STOP_THRESHOLD_PCT}"
)
if [[ "${USE_TILING}" == "1" ]]; then
    COMMON_ARGS+=(--tiling)
else
    COMMON_ARGS+=(--no-tiling)
fi
if [[ "${ADD_GLOBAL_CONTEXT_PASS}" == "1" ]]; then
    COMMON_ARGS+=(--global-pass)
else
    COMMON_ARGS+=(--no-global-pass)
fi

# The evaluation needs to know where things are, which operating point is used
# and where the dataset is, because the qualitative figures reopen the selected
# images (and redraw the exemplar exactly as it was fed to CountGD++).
EVAL_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --n-exemplars "${N_EXEMPLARS}"
    --threshold "${THRESHOLD}"
    --text-prompt "${TEXT_PROMPT}"
    --exemplar-scale-mode "${EXEMPLAR_SCALE_MODE}"
    --model-short-side "${MODEL_SHORT_SIDE}"
    --model-max-size "${MODEL_MAX_SIZE}"
    --eval-iou-threshold "${EVAL_IOU_THRESHOLD}"
    --prompt-ignore-iou "${PROMPT_IGNORE_IOU}"
    --nms-iou-threshold "${NMS_IOU_THRESHOLD}"
    --operating-confidence "${OPERATING_CONFIDENCE}"
    --plot-evaluation-mode "${PLOT_EVALUATION_MODE}"
    --plot-min-gt-boxes "${PLOT_MIN_GT_BOXES}"
    --plot-max-display-dim "${PLOT_MAX_DISPLAY_DIM}"
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
    --global-downscale "${GLOBAL_DOWNSCALE}"
)
if [[ "${USE_TILING}" == "1" ]]; then
    EVAL_ARGS+=(--tiling)
else
    EVAL_ARGS+=(--no-tiling)
fi
if [[ "${ADD_GLOBAL_CONTEXT_PASS}" == "1" ]]; then
    EVAL_ARGS+=(--global-pass)
else
    EVAL_ARGS+=(--no-global-pass)
fi

case "${MODE}" in

    download)
        # Runs on a LOGIN node (internet). The download tools live in their own
        # folder and are put on PYTHONPATH only for this mode.
        export PYTHONPATH="${COUNTGD_PYTOOLS}"
        mkdir -p "${COUNTGD_PYTOOLS}" "${COUNTGD_WHEELS}"

        echo "--- 1/5 : CountGD++ repository -> ${COUNTGD_REPO} ---"
        if [[ -d "${COUNTGD_REPO}/.git" ]]; then
            echo "already cloned"
        else
            git clone "${COUNTGD_REPO_URL}" "${COUNTGD_REPO}"
        fi
        echo "commit: $(git -C "${COUNTGD_REPO}" rev-parse HEAD)"

        echo "--- 2/5 : login-node tools (gdown, huggingface_hub) -> ${COUNTGD_PYTOOLS} ---"
        python -m pip install --quiet --target "${COUNTGD_PYTOOLS}" --upgrade gdown huggingface_hub

        echo "--- 3/5 : CountGD++ checkpoint -> ${COUNTGD_CKPT} ---"
        mkdir -p "$(dirname "${COUNTGD_CKPT}")"
        if [[ -f "${COUNTGD_CKPT}" && "$(stat -c %s "${COUNTGD_CKPT}")" -gt 1000000000 ]]; then
            echo "already downloaded"
        else
            python -m gdown "https://drive.google.com/uc?id=${COUNTGD_CKPT_ID}" -O "${COUNTGD_CKPT}"
        fi
        ls -lh "${COUNTGD_CKPT}"

        echo "--- 4/5 : bert-base-uncased (text encoder) -> ${HF_HOME} ---"
        python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="bert-base-uncased",
    # config + tokenizer + safetensors weights only (the repo also holds TF / Flax / ONNX copies)
    allow_patterns=["*.json", "*.txt", "model.safetensors"],
)
print("Snapshot cached at:", path)
PY

        echo "--- 5/5 : wheels of the extra packages -> ${COUNTGD_WHEELS} ---"
        # One download per container Python version, because tokenizers / regex /
        # safetensors / pycocotools are compiled wheels (pure-python wheels are
        # downloaded once and reused). --no-deps: exactly this list, nothing else.
        for pyver in ${CONTAINER_PYVERS}; do
            echo "  python ${pyver} / ${CONTAINER_ARCH}"
            for pkg in "${COUNTGD_PACKAGES[@]}"; do
                python -m pip download --quiet --no-deps --only-binary=:all: \
                    --implementation cp --python-version "${pyver}" \
                    --platform "manylinux2014_${CONTAINER_ARCH}" \
                    --platform "manylinux_2_17_${CONTAINER_ARCH}" \
                    --platform "manylinux_2_28_${CONTAINER_ARCH}" \
                    -d "${COUNTGD_WHEELS}" "${pkg}" \
                    || echo "    (warning: no wheel for '${pkg}' on python ${pyver}; see 'Problem B')"
            done
        done
        ls -1 "${COUNTGD_WHEELS}"

        echo "Done. Compute nodes can now run offline."
        echo "NEXT: open an interactive session INSIDE the container (HOW_TO_RUN Step 4) and run"
        echo "      ./1_exemplars_text_tiling_extra_path_run_countGDpp.sh build"
        ;;

    build)
        # Runs INSIDE the container on a GPU node: the packages must match the
        # container's Python, and setup.py refuses to compile without a GPU.
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        echo "--- 1/3 : extra packages -> ${COUNTGD_PYEXTRA} ---"
        # --no-deps --no-index: only the wheels fetched by 'download'. torch /
        # numpy / pillow come from the container and must never be shadowed.
        python -m pip install --target "${COUNTGD_PYEXTRA}" --no-deps --no-index \
            --find-links "${COUNTGD_WHEELS}" --upgrade "${COUNTGD_PACKAGES[@]}"
        python - <<'PY'
import addict, pycocotools, termcolor, timm, transformers, yapf  # noqa: F401
import supervision
from supervision.metrics import MeanAveragePrecision  # noqa: F401
from transformers import AutoTokenizer, BertModel  # noqa: F401
assert int(transformers.__version__.split(".")[0]) < 5, transformers.__version__
assert hasattr(BertModel, "get_extended_attention_mask")
print("transformers", transformers.__version__, "| timm", timm.__version__,
      "| supervision", supervision.__version__, "-> OK")
PY

        echo "--- 2/3 : BERT text encoder -> ${COUNTGD_REPO}/checkpoints/bert-base-uncased ---"
        if [[ -f "${COUNTGD_REPO}/checkpoints/bert-base-uncased/config.json" ]]; then
            echo "already present"
        else
            # the repo's own download_bert.py (notebook CELL 1, step 5), run
            # OFFLINE from the HF cache that 'download' filled
            (cd "${COUNTGD_REPO}" && python download_bert.py)
        fi
        ls -1 "${COUNTGD_REPO}/checkpoints/bert-base-uncased"

        echo "--- 3/3 : MultiScaleDeformableAttention CUDA op ---"
        # One-line fix for the compile error seen on Colab with a recent torch:
        #   AT_DISPATCH_FLOATING_TYPES(value.type(), ...) -> (value.scalar_type(), ...)
        CU_FILE="${OPS_DIR}/src/cuda/ms_deform_attn_cuda.cu"
        sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' \
            "${CU_FILE}"
        grep -n "AT_DISPATCH_FLOATING_TYPES" "${CU_FILE}" || true

        CUDA_ARCH="$(python -c 'import torch; m, n = torch.cuda.get_device_capability(0); print(f"{m}.{n}")' \
                     2>/dev/null || echo "")"
        BUILD_LOG="${OPS_DIR}/build_countgdpp.log"
        if [[ -z "${CUDA_ARCH}" ]]; then
            echo "No GPU visible -> cannot compile (setup.py needs one)."
            echo "Inference will use the pure-PyTorch fallback (same maths, slower)."
        elif (cd "${OPS_DIR}" && TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" FORCE_CUDA=1 \
                MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}" python setup.py build) > "${BUILD_LOG}" 2>&1; then
            echo "Compiled for sm_${CUDA_ARCH/./} (log: ${BUILD_LOG})"
        else
            echo "COMPILATION FAILED -> inference will use the pure-PyTorch fallback (same maths, slower)."
            echo "Last lines of ${BUILD_LOG}:"
            tail -n 25 "${BUILD_LOG}"
        fi
        ls -1 "${OPS_DIR}"/build/lib*/MultiScaleDeformableAttention*.so 2>/dev/null \
            || echo "(no compiled .so)"

        echo "Build done. Verify with: ./1_exemplars_text_tiling_extra_path_run_countGDpp.sh dryrun"
        echo "            (the PYTHON DEBUG block above prints the state of every piece)."
        ;;

    dryrun)
        # No model, no GPU: just proves the dataset layout is understood and
        # prints the cost of the experiment.
        python "${PY_SCRIPT}" "${COMMON_ARGS[@]}" --dry-run "$@"
        ;;

    run)
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
        if [[ ! -f "${COUNTGD_CKPT}" ]]; then
            echo "WARNING: ${COUNTGD_CKPT} not found and we are running offline."
            echo "         Run './1_exemplars_text_tiling_extra_path_run_countGDpp.sh download' on a login node first."
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
            RC=0
            wait "${PIDS[$idx]}" || RC=$?
            if [[ "${RC}" -eq 0 ]]; then
                echo "  shard ${idx} finished OK"
            elif [[ "${RC}" -eq 3 ]]; then
                echo "  shard ${idx} STOPPED by the RAM guard (> ${MEM_STOP_THRESHOLD_PCT}% RAM) -- resubmit to resume"
                FAILED=1
            else
                echo "  shard ${idx} FAILED (exit ${RC})"
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
        # PHASE 2 only. Never loads CountGD++, so it runs fine on a login node or
        # in a small CPU allocation.
        python "${PY_SCRIPT}" "${EVAL_ARGS[@]}" --evaluate-only "$@"
        ;;

    *)
        echo "Unknown mode: ${MODE} (expected: download|build|dryrun|run|evaluate)"
        exit 1
        ;;
esac
