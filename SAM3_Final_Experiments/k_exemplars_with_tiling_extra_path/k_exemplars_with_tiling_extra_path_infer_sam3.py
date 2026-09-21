#!/usr/bin/env python3
# =============================================================================
#  k_exemplars_with_tiling_extra_path_infer_sam3.py
# =============================================================================
#  Cluster port of k_exemplars_with_tiling_extra_path.ipynb (same pipeline, same
#  functions, same comments), run on BOTH archives (AGS_Multi_Rumex and
#  AgsSpringRumex) instead of only AGS_Multi_Rumex.

#
#  PIPELINE
#    every GT box of an image is an ANCHOR once; for every (image, anchor):
#      the anchor + (N_EXEMPLARS - 1) seeded random other GT boxes of the SAME
#      image are the visual prompts                                   (CELL 6)
#      the exemplar crops are pasted into a strip above EVERY tile (local
#      background + feathering; the tile itself is never modified)  (CELL 12-13)
#      SAM3 over all tiles in batches, strip boxes = positive prompts (CELL 15)
#      target-region + plausibility filter, tile -> image coordinates (CELL 14)
#      EXTRA PATH: the same strip + SAM3 + filters on the whole image
#      downscaled by GLOBAL_DOWNSCALE=2; its detections are rescaled back and
#      pooled with the tiled ones (tile_id = -1) BEFORE NMS           (CELL 16-17)
#
#  The run is split into two phases:
#
#    PHASE 1 - INFERENCE   (notebook CELL 19, GPU)
#        for every image x every anchor: the steps above, then the PRE-NMS
#        detections are saved to NPZ (CELL 18).
#        Sharded: one process per GPU, round-robin over the (deterministically
#        sorted) image list. Each shard writes its own manifest, so the phase is
#        crash-safe and resumable. The resume key is (image_ID, anchor_idx).
#
#    PHASE 2 - EVALUATION  (notebook CELL 20 ... CELL 32, no GPU)
#        load the NPZ files, apply the provenance-aware NMS at NMS_IOU_THRESHOLD =
#        0.40 and evaluate in both modes (all_gt / held_out) at
#        CONFIDENCE_THRESHOLD = 0.30: run-level, image-level and experiment-level
#        metrics, the pooled dataset AP, the confusion matrices and the
#        qualitative figures. Never touches SAM3.

#  MODES
#    (default)          sharded inference, then the evaluation
#    --dry-run          discover the dataset, print the cost, exit. No GPU.
#    --no-evaluate      inference only (used by the per-GPU shard processes)
#    --evaluate-only    PHASE 2 only (used once after all shards finished)
# =============================================================================

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

# torch / transformers / supervision / pandas / matplotlib are imported
# lazily inside the functions that need them, so the PHASE 2 evaluation and the
# dry run work on a CPU node without torch.

# UAV orthomosaics are very large; disable PIL's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None


# =============================================================================
# CELL 3 - CONFIGURATION
# =============================================================================

# ------------------------- experiment identity -------------------------------
EXPERIMENT_NAME = "k_exemplars_with_tiling_extra_path"   # k exemplars + tiles + global pass
N_EXEMPLARS     = 3           # k: 3 = multiple visual prompts, 1 = single visual prompt
USE_TILING      = True        # True  -> overlapping tiles
                              # False -> whole image treated as one single tile
PROMPT_TYPE = "multiple" if N_EXEMPLARS > 1 else "single"

# ------------------------------- dataset -------------------------------------
# CLUSTER CHANGE: the notebook pointed at ONE folder
#   IMAGES_ROOT = ".../dataset/AGS_Multi_Rumex/images"
#   LABELS_ROOT = ".../dataset/AGS_Multi_Rumex/annotations_yolo"
#   RUMEX_CLASS_ID = 0
# On the cluster the two archives live side by side under --dataset-root and use
# DIFFERENT class ids inside their YOLO files, so the class id is a property of
# the archive, not a global constant.
ARCHIVES: dict[str, int] = {
    "AGS_Multi_Rumex": 0,
    "AgsSpringRumex": 2,
}
IGNORED_ARCHIVES = ("AGS_Multiple_Fields", "AGS_Multiple_Fields_Embeddings")
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
NON_LABEL_FILES = {"darknet.labels", "classes.txt", "obj.names"}

# ------------------------------- outputs -------------------------------------
# notebook: RESULTS_ROOT = f"/content/drive/MyDrive/master_thesis/results_new1/{EXPERIMENT_NAME}"
# cluster : RESULTS_ROOT = --output-dir  (default: $SCRATCH/experiments/sam3/<EXPERIMENT_NAME>,
#           chosen by k_exemplars_with_tiling_extra_path_run_sam3.sh)

# ------------------------------- tiling --------------------------------------
# Neighbouring tiles overlap by OVERLAP pixels, so every plant whose width AND
# height are <= OVERLAP (384 px) lies completely inside at least one tile.
# Larger plants can be cut by every tile; the global-context pass below still
# sees them whole.
TILE_SIZE = 1536
OVERLAP   = 384
CACHE_TILES_IN_MEMORY = True  # True  -> crop all tiles once per image and keep the
                              #          PIL images in RAM (CPU) for all anchor runs
                              # False -> crop tiles on demand from the (already open)
                              #          full image; slower but uses less RAM

# --------------------------- SAM3 inference ----------------------------------
# The confidence threshold is FIXED from the start. It is used both
#   (a) inside SAM3 post-processing (SAM3 runs EXACTLY ONCE per
#       (image_ID, anchor_idx, tile) and once per global pass), and
#   (b) as the operating point for precision / recall / F1 / IoU1 / IoU2.
# Because (a) == (b), AP and the operating-point metrics use the same predictions.
CONFIDENCE_THRESHOLD = 0.30
MASK_THRESHOLD = 0.40         # SAM3 mask binarisation (only used transiently)
BATCH_SIZE = 4                # composed tiles per forward pass.
                              # TILE_SIZE=1536 tiles are larger than the
                              # E02_2 reference's 1000px tiles, so each one
                              # uses more GPU memory -- lower this if you hit
                              # CUDA OOM, raise it if your GPU has headroom.
# USE_FP16 = True             # half precision inference on the GPU
                              # (notebook, Colab T4)
                              # CLUSTER: replaced by --dtype (bfloat16 by default).
KEEP_MASKS = False            # masks are NEVER stored (RAM / GPU / disk / CSV / NPZ).
                              # They are only used transiently to compute the
                              # mask-fill ratio of the plausibility filter.

# ------------------------- exemplar strip layout ------------------------------
STRIP_MARGIN = 6              # px between exemplar crops inside the strip
FEATHER_WIDTH = 8             # px of soft alpha blending around each exemplar crop
BACKGROUND_BLUR_RADIUS = 1.5  # light Gaussian blur applied to the sampled background
MAX_STRIP_HEIGHT_FRACTION = None
# ^ Optional extra clamp on the strip HEIGHT, expressed as a fraction of the tile
#   height. Disabled (None) by default, same as E02_2.

# ------------------- plausibility filter (aligned to E02_2) -------------------
MIN_FILL_RATIO   = 0.15       # >=15 % of the box area must be covered by the mask
MAX_AREA_FRACTION = 0.80      # a detection may not cover >80 % of a tile (was 0.60
                              # in this notebook's old code -> aligned to E02_2)
EDGE_MARGIN      = 5          # boxes with width or height <=5 px are discarded

# ---------------- target-region membership rule (bug-fixed, as in E02_2) ------
# A prediction made on the composed image (exemplar strip on top + tile below) is
# kept only if at least this fraction of its AREA lies inside the real tile region.
# Majority rule (0.50): the box belongs to whichever region holds most of its area.
# (Replaces this notebook's old "y1 >= dy - 5" top-edge-only rule.)
TILE_REGION_MIN_FRACTION = 0.50

# ------------------- optional: downsampled global-context pass ----------------
# An EXTRA single pass over the whole image, downscaled by GLOBAL_DOWNSCALE, run
# through the SAME exemplar-strip + SAM3 pipeline as a tile (no cropping). Its
# detections are rescaled back to full resolution and pooled together with the
# tiled detections BEFORE NMS -- so the fixed NMS/confidence operating point and
# every evaluation cell downstream need no special-casing at all.
ADD_GLOBAL_CONTEXT_PASS = True    # kept ON, same default as this notebook's own code
GLOBAL_DOWNSCALE = 2              # shrink factor for the global pass image


# ------------------------------ evaluation ------------------------------------
EVAL_IOU_THRESHOLD = 0.50     # IoU needed for a prediction to count as a TP
PROMPT_IGNORE_IOU  = 0.50     # held_out mode: an unmatched prediction whose best IoU
                              # with a PROMPT GT box is >= this value is IGNORED
                              # (neither TP nor FP). Same value as EVAL_IOU_THRESHOLD
                              # so a single IoU threshold governs the whole protocol.

# ------------------------------- NMS -----------------------------------------
# Fixed from the start (no offline confidence x NMS sweep, no "best" selection).
# Applied offline to the stored pre-NMS detections (CELL 21).
NMS_IOU_THRESHOLD = 0.40

EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
# held_out : the GT instances that were used as visual prompts are IGNORED, and so
#            are the predictions that fall on them. Answers "how well does SAM3 find
#            the REMAINING Rumex plants after being shown a few examples?".

# --------------------------- qualitative plot ---------------------------------
PLOT_EVALUATION_MODE = "all_gt"   # image-level AP50 of this mode selects the image
PLOT_MIN_GT_BOXES    = 7          # the plotted image must have at least this many GT boxes
PLOT_MAX_DISPLAY_DIM = 2048       # display-only downscale (the full image is 8192 px wide)
PLOT_SHOW_SCORES     = True       # write the confidence next to each predicted box

# ------------------------------ safety net -------------------------------------
# CLUSTER CHANGE: the notebook's RAM guard (MEM_STOP_THRESHOLD_PCT = 70, checked with
# psutil in CELL 19) is removed on purpose. A shard that runs out of memory simply
# fails; resubmitting resumes from the shard manifests.


# =============================================================================
# CELL 4 - OUTPUT FOLDERS (manifest columns)
# =============================================================================
# Manifest of finished inference runs -> used for crash-safe resuming.
# CLUSTER CHANGE: + archive, flight, source_class_id (so every CSV can be split
# per archive / per flight).
MANIFEST_COLUMNS = [
    "experiment_name", "image_ID", "anchor_idx", "Prompt_ID", "Prompt_Type",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt", "n_detections_pre_nms", "n_tiles", "used_global_pass",
    "image_width", "image_height", "npz_file", "inference_seconds",
]

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run "
    "'./k_exemplars_with_tiling_extra_path_run_sam3.sh download' on a LOGIN "
    "node first, which installs it into $PYEXTRA."
)


# =============================================================================
#  CLI  -  every notebook CELL 3 parameter, with the notebook value as default
# =============================================================================

def default_dataset_root() -> Path:
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "overney" / "dataset"
    return Path(__file__).resolve().parents[2] / ".." / "02_data" / "dataset"


def str2bool(value) -> bool:
    """'true'/'false' (also 1/0, yes/no) from the run script -> bool; anything else is an error."""
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got '{value}'")


def optional_float(value):
    """'none' / '' -> None, otherwise a float (MAX_STRIP_HEIGHT_FRACTION)."""
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return None
    return float(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="k_exemplars_with_tiling_extra_path - SAM3 Rumex detection, one run per "
                    "(image, anchor): the anchor + N_EXEMPLARS-1 seeded random GT plants of "
                    "the image are cropped into an exemplar strip above every tile (tiling "
                    "ON) and used as positive box prompts, plus an extra downsampled "
                    "global-context pass merged before NMS (inference + offline "
                    "evaluation). Cluster port of k_exemplars_with_tiling_extra_path.ipynb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------- paths -------------------------------------
    g = p.add_argument_group("paths")
    g.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                   help="Folder holding the archive folders (AGS_Multi_Rumex, AgsSpringRumex).")
    g.add_argument("--output-dir", type=Path, required=True,
                   help="RESULTS_ROOT: raw_detections/, metrics/, confusion_matrices/, plots/.")
    g.add_argument("--archives", nargs="*", default=list(ARCHIVES.keys()),
                   help="Subset of archives to run on. Default: both.")

    # ------------------------ experiment identity ---------------------------
    g = p.add_argument_group("experiment identity (CELL 3)")
    g.add_argument("--experiment-name", default=EXPERIMENT_NAME,
                   help="EXPERIMENT_NAME. Written into every CSV row, every NPZ and into "
                        "the deterministic exemplar / strip-background seeds.")
    g.add_argument("--n-exemplars", type=int, default=N_EXEMPLARS,
                   help="N_EXEMPLARS (k): 3 = multiple visual prompts, 1 = single visual "
                        "prompt. PROMPT_TYPE follows from it ('multiple' / 'single').")

    # ------------------------------ tiling ----------------------------------
    g = p.add_argument_group("tiling (CELL 3 / CELL 11)")
    g.add_argument("--use-tiling", type=str2bool, default=USE_TILING,
                   help="USE_TILING: true -> overlapping tiles, false -> whole image as ONE tile.")
    g.add_argument("--tile-size", type=int, default=TILE_SIZE, help="TILE_SIZE")
    g.add_argument("--overlap", type=int, default=OVERLAP, help="OVERLAP")
    g.add_argument("--cache-tiles-in-memory", type=str2bool, default=CACHE_TILES_IN_MEMORY,
                   help="CACHE_TILES_IN_MEMORY: false -> crop tiles on demand (less RAM).")

    # --------------------------- SAM3 inference -----------------------------
    g = p.add_argument_group("sam3 inference (CELL 3 / CELL 5 / CELL 15)")
    g.add_argument("--model-id", default="facebook/sam3",
                   help="HF repo id OR a local snapshot directory.")
    g.add_argument("--confidence-threshold", "--threshold", "--operating-confidence",
                   dest="confidence_threshold", type=float, default=CONFIDENCE_THRESHOLD,
                   help="CONFIDENCE_THRESHOLD, fixed from the start. Used (a) inside SAM3 "
                        "post-processing (once per image x anchor x tile and once per "
                        "global pass) and (b) as the operating point of precision / recall / "
                        "F1 / IoU1 / IoU2. --threshold and --operating-confidence are "
                        "aliases of the same single value, exactly like the notebook.")
    g.add_argument("--mask-threshold", type=float, default=MASK_THRESHOLD,
                   help="MASK_THRESHOLD (masks are only used transiently, never stored).")
    g.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="BATCH_SIZE: composed tiles per forward pass.")
    g.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16",
                   help="Model dtype. The notebook used float16 on a T4 (USE_FP16); "
                        "bfloat16 is the cluster default (GH200/H100).")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")

    # ------------------------ exemplar strip layout -------------------------
    g = p.add_argument_group("exemplar strip (CELL 3 / CELL 12 / CELL 13)")
    g.add_argument("--strip-margin", type=int, default=STRIP_MARGIN, help="STRIP_MARGIN")
    g.add_argument("--feather-width", type=int, default=FEATHER_WIDTH, help="FEATHER_WIDTH")
    g.add_argument("--background-blur-radius", type=float, default=BACKGROUND_BLUR_RADIUS,
                   help="BACKGROUND_BLUR_RADIUS")
    g.add_argument("--max-strip-height-fraction", type=optional_float,
                   default=MAX_STRIP_HEIGHT_FRACTION,
                   help="MAX_STRIP_HEIGHT_FRACTION ('none' = disabled, as in the notebook).")

    # --------------------------- filters ------------------------------------
    g = p.add_argument_group("filters (CELL 3 / CELL 14)")
    g.add_argument("--min-fill-ratio", type=float, default=MIN_FILL_RATIO, help="MIN_FILL_RATIO")
    g.add_argument("--max-area-fraction", type=float, default=MAX_AREA_FRACTION,
                   help="MAX_AREA_FRACTION")
    g.add_argument("--edge-margin", type=int, default=EDGE_MARGIN, help="EDGE_MARGIN")
    g.add_argument("--tile-region-min-fraction", type=float, default=TILE_REGION_MIN_FRACTION,
                   help="TILE_REGION_MIN_FRACTION: minimum fraction of a predicted box's "
                        "AREA that must lie inside the real tile region.")

    # ----------------- optional: downsampled global-context pass --------------
    g = p.add_argument_group("extra path: global-context pass (CELL 3 / CELL 16)")
    g.add_argument("--add-global-context-pass", type=str2bool, default=ADD_GLOBAL_CONTEXT_PASS,
                   help="ADD_GLOBAL_CONTEXT_PASS: true -> one extra SAM3 pass per anchor run "
                        "over the whole image downscaled by --global-downscale.")
    g.add_argument("--global-downscale", type=int, default=GLOBAL_DOWNSCALE,
                   help="GLOBAL_DOWNSCALE: shrink factor for the global pass image.")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 21 / CELL 22)")
    g.add_argument("--eval-iou-threshold", type=float, default=EVAL_IOU_THRESHOLD,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")
    g.add_argument("--prompt-ignore-iou", type=float, default=PROMPT_IGNORE_IOU,
                   help="PROMPT_IGNORE_IOU: held_out mode ignore rule.")
    g.add_argument("--nms-iou-threshold", type=float, default=NMS_IOU_THRESHOLD,
                   help="NMS_IOU_THRESHOLD, applied OFFLINE in PHASE 2 (fixed, no sweep).")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot (CELL 3 / CELL 32)")
    g.add_argument("--plot-evaluation-mode", default=PLOT_EVALUATION_MODE,
                   choices=EVALUATION_MODES,
                   help="PLOT_EVALUATION_MODE: the image-level AP50 of this mode selects "
                        "the image that gets plotted.")
    g.add_argument("--plot-min-gt-boxes", type=int, default=PLOT_MIN_GT_BOXES,
                   help="PLOT_MIN_GT_BOXES: the plotted image must have at least this many "
                        "GT boxes (with the notebook's fallback if none has).")
    g.add_argument("--plot-max-display-dim", type=int, default=PLOT_MAX_DISPLAY_DIM,
                   help="PLOT_MAX_DISPLAY_DIM: display-only downscale of the figure.")
    g.add_argument("--no-plot-scores", dest="plot_show_scores", action="store_false",
                   default=PLOT_SHOW_SCORES,
                   help="PLOT_SHOW_SCORES = False: do not write the confidence next to each "
                        "predicted box.")
    g.add_argument("--no-plots", action="store_true",
                   help="Skip CELL 32 entirely (it is the only PHASE 2 step that reopens "
                        "the original images).")

    # ---------------------------- runtime -----------------------------------
    g = p.add_argument_group("runtime")
    g.add_argument("--num-shards", type=int, default=1,
                   help="Split the image list across this many concurrent processes.")
    g.add_argument("--shard-index", type=int, default=0, help="0-based shard of this process.")
    g.add_argument("--limit-images", type=int, default=0,
                   help="Debug: process at most this many images (0 = no limit).")
    g.add_argument("--max-anchors-per-image", type=int, default=0,
                   help="Debug: at most this many anchors per image (0 = every GT box, "
                        "as in the notebook).")
    g.add_argument("--no-resume", action="store_true",
                   help="Ignore the existing manifests and recompute every run.")

    # ----------------------------- modes ------------------------------------
    g = p.add_argument_group("modes")
    g.add_argument("--dry-run", action="store_true",
                   help="Discover the dataset, print the run plan and exit. No model, no GPU.")
    g.add_argument("--evaluate-only", action="store_true",
                   help="PHASE 2 only: rebuild every metric from the cached NPZ files.")
    g.add_argument("--no-evaluate", action="store_true",
                   help="PHASE 1 only: do not run the evaluation after inference.")

    args = p.parse_args(argv)

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.n_exemplars < 1:
        p.error("--n-exemplars must be >= 1")
    if args.use_tiling and args.overlap >= args.tile_size:
        p.error("--overlap must be smaller than --tile-size")
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")
    if args.global_downscale < 1:
        p.error("--global-downscale must be >= 1")

    # PROMPT_TYPE (CELL 3) follows from N_EXEMPLARS, exactly like the notebook.
    args.prompt_type = "multiple" if args.n_exemplars > 1 else "single"
    return args


# =============================================================================
# CELL 4 - OUTPUT FOLDERS
# =============================================================================
# results_new1/<EXPERIMENT_NAME>/          (cluster: <output-dir>/)
#   raw_detections/      pre-NMS detections (NPZ, one file per image x anchor)
#                        + one runs_manifest_<exp>_shard<i>.csv per GPU shard
#   metrics/             run / image / experiment / dataset level CSVs
#                        (+ experiment_summary_per_archive.csv)
#   confusion_matrices/  CSV + PNG for all_gt and held_out
#   plots/               qualitative GT-vs-prediction figure (CELL 32)
#                        (cluster: one per archive + one over ALL images)
# =============================================================================

@dataclass
class Paths:
    results_root: Path
    raw_detections: Path
    metrics: Path
    confusion_matrices: Path
    plots: Path


def build_paths(output_dir: Path) -> Paths:
    results_root = output_dir
    paths = Paths(
        results_root=results_root,
        raw_detections=results_root / "raw_detections",
        metrics=results_root / "metrics",
        confusion_matrices=results_root / "confusion_matrices",
        plots=results_root / "plots",
    )
    for d in (paths.results_root, paths.raw_detections, paths.metrics,
              paths.confusion_matrices, paths.plots):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def manifest_path(paths: Paths, experiment_name: str, shard_index: int) -> Path:
    """
    One manifest PER SHARD. The notebook had a single runs_manifest.csv, which is
    not safe when four processes append to it at the same time; the resume step
    simply reads all of them back (see load_done_runs).
    """
    safe_name = experiment_name.replace("/", "-")
    return paths.raw_detections / f"runs_manifest_{safe_name}_shard{shard_index}.csv"


# =============================================================================
# CELL 6 - STABLE REPRODUCIBILITY HELPERS
# =============================================================================

def stable_seed(*parts) -> int:
    """
    Deterministic 32-bit seed from any set of values.

    Input : any number of values (strings / ints) that identify the run,
            e.g. stable_seed(EXPERIMENT_NAME, image_id, anchor_idx)
    Output: int in [0, 2**32) - identical in every Python process, forever.
    """
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


def select_exemplar_indices(n_gt: int,
                            anchor_idx: int,
                            n_exemplars: int,
                            image_id: str,
                            experiment_name: str = EXPERIMENT_NAME) -> list:
    """
    Choose which GT instances of ONE image are used as visual prompts.

    Input : n_gt        - number of GT boxes in the image
            anchor_idx  - index of the GT box this run is "about" (always a prompt)
            n_exemplars - how many prompts in total (1 or 3)
            image_id    - "<folder>/<image name>"
    Output: list of GT indices, ANCHOR FIRST, then the randomly sampled others.

    The anchor is always included; the remaining (n_exemplars - 1) slots are
    filled by sampling without replacement from the other GT boxes of the SAME
    image. If the image does not contain enough other boxes, fewer prompts are
    used (no duplication, no crash).
    """
    seed = stable_seed(experiment_name, image_id, anchor_idx)
    rng = np.random.default_rng(seed)

    others = [i for i in range(n_gt) if i != anchor_idx]
    n_needed = min(n_exemplars - 1, len(others))
    if n_needed > 0:
        chosen = [int(i) for i in rng.choice(others, size=n_needed, replace=False)]
    else:
        chosen = []
    return [int(anchor_idx)] + chosen


def format_prompt_id(exemplar_indices: list) -> str:
    """
    Human-readable id of a prompt set.
      single   -> "5"
      multiple -> "5+12+3"  (the ANCHOR is always the first number)
    """
    return "+".join(str(int(i)) for i in exemplar_indices)


# =============================================================================
# CELL 7 - DATASET AND YOLO ANNOTATION HELPERS
# =============================================================================
# Unchanged logic from the original notebook (it worked); only wrapped into
# functions and given comments.
#
# CLUSTER CHANGE: the notebook walked ONE images folder (IMAGES_ROOT) and looked
# the label up with find_label_path() in LABELS_ROOT, with a single RUMEX_CLASS_ID.
# On the cluster both archives are pooled into one dataset, each with its own class
# id, so discover_images() walks <dataset-root>/<archive>/images for every archive,
# looks the label up with the same find_label_path() in that archive's
# annotations_yolo folder (plus a one-time label index as a fallback), and every
# image carries its archive, flight and class id in an ImageRecord.
#   image_ID = "<archive>/<flight>/<image name without extension>"
#   (the notebook's "<folder>/<name>" with the archive in front)
# load_yolo_boxes / find_label_path / safe_filename / safe_crop are the notebook code.
# =============================================================================

@dataclass(frozen=True)
class ImageRecord:
    """One image plus everything needed to evaluate it."""
    archive: str        # AGS_Multi_Rumex | AgsSpringRumex
    flight: str         # e.g. 20220518_Eschikon ("" if images/ has no sub-folder)
    image_id: str       # "<archive>/<flight>/<stem>"  - unique across both archives
    image_path: Path
    label_path: Path
    class_id: int       # the Rumex class id INSIDE this archive's YOLO files


def load_yolo_boxes(label_path, img_width, img_height, class_id):
    """
    Read a YOLO .txt annotation file and convert it to pixel corner boxes.

    Input : label_path            - YOLO txt file
            img_width, img_height - size of the ORIGINAL image in pixels
            class_id              - keep only this class (0 = Rumex in AGS_Multi_Rumex,
                                    2 = Rumex in AgsSpringRumex)
    Output: np.ndarray (N, 4) float32, boxes as [x1, y1, x2, y2] in pixels.

    YOLO stores normalised (class, x_center, y_center, width, height).
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue                      # skip empty lines
            if int(parts[0]) != class_id:
                continue                      # keep only the requested class
            xc, yc, bw, bh = map(float, parts[1:5])
            xc, yc = xc * img_width, yc * img_height
            bw, bh = bw * img_width, bh * img_height
            boxes.append([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def find_label_path(image_filename_no_ext, folder_name, labels_root):
    """
    Locate the YOLO txt belonging to an image, supporting both a mirrored folder
    structure (labels/<folder>/<name>.txt) and a flat one (labels/<name>.txt).
    Returns the path or None.
    (cluster: LABELS_ROOT = <dataset-root>/<archive>/annotations_yolo, passed in)
    """
    mirrored = os.path.join(labels_root, folder_name, image_filename_no_ext + ".txt")
    flat = os.path.join(labels_root, image_filename_no_ext + ".txt")
    if os.path.exists(mirrored):
        return mirrored
    if os.path.exists(flat):
        return flat
    return None


def _index_flat_labels(annotations_root: Path) -> dict:
    """
    Scan all annotation files of one archive once -> {image stem: label file}.
    Fallback used when find_label_path finds neither the mirrored nor the flat
    file (some cluster archives keep their labels in other sub-folders).
    """
    index: dict = {}
    duplicates: list = []
    if not annotations_root.is_dir():
        return index
    for label_path in sorted(annotations_root.rglob("*.txt")):
        if label_path.name in NON_LABEL_FILES:
            continue
        stem = label_path.stem
        if stem in index:
            duplicates.append(stem)
            continue
        index[stem] = label_path
    if duplicates:
        print(f"  WARNING: {len(duplicates)} duplicate label basenames in {annotations_root} "
              f"(first kept). Examples: {duplicates[:5]}")
    return index


def discover_images(dataset_root: Path, archives: Iterable[str]) -> List[ImageRecord]:
    """
    Walk <dataset_root>/<archive>/images (the notebook's IMAGES_ROOT) for every
    chosen archive and collect every image together with its label file
    (find_label_path first, then the one-time label index).
    Output: list of ImageRecord, sorted by image_ID (a deterministic global order,
            so the round-robin shard assignment is identical in every process and
            after every restart). Images without a label file are skipped.
    """
    records: List[ImageRecord] = []
    missing: List[str] = []

    present = {d.name for d in dataset_root.iterdir() if d.is_dir()} if dataset_root.is_dir() else set()
    ignored_present = sorted(present.intersection(IGNORED_ARCHIVES))
    if ignored_present:
        print(f"Ignoring archives (by design): {', '.join(ignored_present)}")

    for archive in archives:
        if archive not in ARCHIVES:
            raise ValueError(f"Unknown archive '{archive}'. Known: {sorted(ARCHIVES)}")
        class_id = ARCHIVES[archive]
        images_root = dataset_root / archive / "images"
        annotations_root = dataset_root / archive / "annotations_yolo"

        if not images_root.is_dir():
            print(f"  WARNING: {images_root} does not exist -- archive '{archive}' skipped.")
            continue

        label_index = _index_flat_labels(annotations_root)
        n_before = len(records)

        for image_path in sorted(images_root.rglob("*")):
            if not image_path.is_file() or not image_path.name.lower().endswith(VALID_IMAGE_EXTENSIONS):
                continue
            rel = image_path.relative_to(images_root)
            flight = rel.parts[0] if len(rel.parts) > 1 else ""
            name_no_ext = image_path.stem
            label_path = find_label_path(name_no_ext, flight, str(annotations_root))
            if label_path is None:
                label_path = label_index.get(name_no_ext)
            image_id = f"{archive}/{flight}/{name_no_ext}" if flight else f"{archive}/{name_no_ext}"
            if label_path is None:
                missing.append(image_id)
                continue
            records.append(ImageRecord(archive, flight, image_id, image_path, Path(label_path), class_id))

        n_flights = len({r.flight for r in records[n_before:]})
        print(f"  {archive:<22} class_id={class_id}  images={len(records) - n_before:<6} "
              f"flights={n_flights:<4} labels_indexed={len(label_index)}")

    print(f"Discovered {len(records) + len(missing)} images in "
          f"{len(set((r.archive, r.flight) for r in records))} folders.")
    print(f"With labels: {len(records)} | without labels (skipped): {len(missing)}")
    if missing:
        print("  first missing:", missing[:5])

    records.sort(key=lambda r: r.image_id)
    return records


def safe_filename(image_id: str) -> str:
    """'archive/flight/name' -> 'archive__flight__name' so it can be used inside a file name."""
    return image_id.replace("/", "__").replace(os.sep, "__")


def safe_crop(image, box, min_size=2):
    """
    Crop an exemplar from the full image, clamped to the image borders and to a
    minimum size. Guards against degenerate/out-of-range YOLO boxes, which would
    otherwise produce a 0-pixel crop and crash the strip composition.
    """
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, image.width - min_size))
    y1 = max(0, min(y1, image.height - min_size))
    x2 = min(image.width, max(x2, x1 + min_size))
    y2 = min(image.height, max(y2, y1 + min_size))
    return image.crop((x1, y1, x2, y2))


# =============================================================================
# CELL 9 - IoU
# =============================================================================

def compute_iou_matrix(boxes1, boxes2):
    """
    Pairwise IoU between two sets of [x1, y1, x2, y2] boxes.

    Input : boxes1 (N, 4), boxes2 (M, 4)
    Output: (N, M) matrix, entry [i, j] = IoU(boxes1[i], boxes2[j])
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])   # left
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])   # top
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])   # right
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])   # bottom

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)

# How much of this predicted bounding box lies inside a certain tile or region?
# how many pixels of a bounding box overlap with another rectangular region
# only intersection area not oc=ver union
def intersection_area(box, region):
    """Plain intersection AREA (not IoU) between one box and one region."""
    x1 = max(box[0], region[0]); y1 = max(box[1], region[1]) # top and left of inter
    x2 = min(box[2], region[2]); y2 = min(box[3], region[3]) # bottom and riht of inter
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) # area of inter = inter width x inter height


# =============================================================================
# CELL 10 -  ONE-TO-ONE MATCHING
# =============================================================================

#   sort predictions by confidence, high -> low
#   for each prediction:
#       look ONLY at GT boxes that are still unmatched
#       take the unmatched GT with the highest IoU
#       if that IoU >= evaluation IoU threshold -> match, else leave unmatched
#   one GT can be matched by at most one prediction, and vice versa.
#
# This single function is used for TP / FP / FN / precision / recall / F1 /
# IoU1 / IoU2 / confusion matrices, so the whole thesis uses one definition.
# =============================================================================

def match_one_to_one(pred_boxes, pred_scores, gt_boxes, iou_threshold=EVAL_IOU_THRESHOLD):
    """
    Input : pred_boxes  (P, 4), pred_scores (P,), gt_boxes (G, 4), iou_threshold
    Output: dict with
        'pred_match_gt' (P,) int   - matched GT index per prediction, -1 if unmatched
        'gt_match_pred' (G,) int   - matched prediction index per GT,  -1 if unmatched
        'pred_iou'      (P,) float - IoU of the accepted match, 0.0 if unmatched
        'matched_ious'  list       - IoU values of all accepted matches
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)

    pred_match_gt = np.full(n_pred, -1, dtype=np.int64)
    gt_match_pred = np.full(n_gt, -1, dtype=np.int64)
    pred_iou = np.zeros(n_pred, dtype=np.float32) # stores the IoU of the accepted match for every prediction

    if n_pred == 0 or n_gt == 0:
        return {"pred_match_gt": pred_match_gt, "gt_match_pred": gt_match_pred,
                "pred_iou": pred_iou, "matched_ious": []}

    iou = compute_iou_matrix(pred_boxes, gt_boxes)
    gt_free = np.ones(n_gt, dtype=bool) # Track which GTs are still available

    # 'stable' keeps the original order for equal scores -> fully deterministic.
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32), kind="stable") # Sort predictions by confidence

    matched_ious = []
    for p in order:
        if not gt_free.any():
            break                                   # nothing left to match, every GT already has a prediction
        # Consider ONLY currently unmatched GT boxes (this is the fix).
        candidate_ious = np.where(gt_free, iou[p], -1.0)
        g = int(np.argmax(candidate_ious))  # Select the best FREE GT
        if candidate_ious[g] >= iou_threshold:
            gt_free[g] = False
            pred_match_gt[p] = g
            gt_match_pred[g] = p
            pred_iou[p] = candidate_ious[g]
            matched_ious.append(float(candidate_ious[g]))

    return {"pred_match_gt": pred_match_gt, "gt_match_pred": gt_match_pred,
            "pred_iou": pred_iou, "matched_ious": matched_ious}


def safe_f1(precision, recall):
    """F1 = 2PR/(P+R) with a safe zero denominator (returns 0.0)."""
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


# =============================================================================
# CELL 11 - TILES: GENERATED ONCE PER IMAGE, REUSED BY EVERY ANCHOR
# =============================================================================

def tile_bboxes(img_w, img_h, tile_size=TILE_SIZE, overlap=OVERLAP):
    """
    Sliding-window tiles covering the whole image with overlap, so a plant lying
    on a tile border is fully visible in at least one window.
    Output: list of (x1, y1, x2, y2) in ORIGINAL image coordinates.
    """
    step = max(1, tile_size - overlap)
    tiles = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
    return list(dict.fromkeys(tiles))   # remove duplicates, keep order


def build_tile_cache(image, use_tiling=USE_TILING,
                     tile_size=TILE_SIZE, overlap=OVERLAP,
                     cache_in_memory=CACHE_TILES_IN_MEMORY):
    """
    Build the tile list for ONE already-open image.

    Input : image - PIL image of the full UAV photo
    Output: list of dicts, one per tile:
            {'tile_id', 'x1', 'y1', 'x2', 'y2', 'image' (PIL or None)}
            'image' is None when cache_in_memory=False (cropped on demand).

    If use_tiling is False the whole image is returned as a single "tile", which
    keeps the rest of the pipeline identical for the no-tiling experiment.
    """
    if use_tiling:
        coords = tile_bboxes(image.width, image.height, tile_size, overlap)
    else:
        coords = [(0, 0, image.width, image.height)]

    cache = []
    for tid, (x1, y1, x2, y2) in enumerate(coords):
        cache.append({
            "tile_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "image": image.crop((x1, y1, x2, y2)) if cache_in_memory else None,
        })
    return cache


def get_tile_image(tile, full_image):
    """Return the tile's PIL image, cropping on demand if it was not cached."""
    if tile["image"] is not None:
        return tile["image"]
    return full_image.crop((tile["x1"], tile["y1"], tile["x2"], tile["y2"]))


# =============================================================================
# CELL 12 - LOCAL BACKGROUND PATCH 
# =============================================================================
# WHY THE STRIP HAS A TEXTURED BACKGROUND
# The exemplar crops are pasted into a strip above the tile. A flat white strip
# would create a hard artificial edge that the image encoder's self-attention can
# latch onto, and the exemplar feature would describe "a plant in a void" instead
# of "a plant in grass". Sampling real texture from the same tile keeps colour,
# brightness and grain consistent with the current lighting conditions.
#

#   - sample from a RANDOM offset inside the tile, deliberately skipping the top
#     band that would sit next to its own copy
#   - the offset comes from a deterministic RNG (stable_seed) -> reproducible
#   - apply a light Gaussian blur so leftover structure does not look like a
#     second, sharp copy of real plants
# =============================================================================

def get_local_background_patch(tile_img, patch_w, patch_h, rng,
                               blur_radius=BACKGROUND_BLUR_RADIUS):
    """
    Input : tile_img        - PIL image of the current tile
            patch_w/patch_h - required size of the strip background
            rng             - np.random.Generator (deterministically seeded)
    Output: PIL image of size (patch_w, patch_h) with local-looking texture.
    """
    tw, th = tile_img.size
    crop_w = min(patch_w, tw)
    crop_h = min(patch_h, th)

    max_x0 = tw - crop_w
    max_y0 = th - crop_h
    # Skip the first patch_h rows: they are the ones that end up directly below
    # the strip, so re-using them would create the adjacent duplication.
    min_y0 = min(patch_h, max_y0)

    x0 = int(rng.integers(0, max_x0 + 1)) if max_x0 > 0 else 0
    y0 = int(rng.integers(min_y0, max_y0 + 1)) if max_y0 > min_y0 else max_y0

    patch = tile_img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    if patch.size != (patch_w, patch_h):
        patch = patch.resize((patch_w, patch_h), Image.BILINEAR)
    if blur_radius and blur_radius > 0:
        patch = patch.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return patch


def make_feather_mask(size, feather_width=FEATHER_WIDTH):
    """
    Soft alpha mask so a pasted exemplar crop blends into the strip background
    instead of showing a hard rectangular border. Unchanged from the original.

    Input : size (w, h) of the crop, feather_width in px
    Output: PIL 'L' image used as the paste mask
    """
    w, h = size
    mask = np.full((h, w), 255.0, dtype=np.float32)     # start fully opaque
    effective = min(feather_width, h // 2, w // 2)
    if effective >= 1:
        for i in range(effective):
            alpha = 255.0 * (i + 1) / effective
            mask[i, :] = np.minimum(mask[i, :], alpha)                   # top
            mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)   # bottom
            mask[:, i] = np.minimum(mask[:, i], alpha)                   # left
            mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)   # right
    return Image.fromarray(mask.astype(np.uint8), mode="L")


# =============================================================================
# CELL 13 - COMPOSE  (exemplar strip on top + real tile below)  
# =============================================================================

#   canvas width is ALWAYS exactly the tile width, so nothing can remain unpainted
#   next to the tile. If the exemplars do not fit in that width, ALL crops are
#   scaled down by ONE common factor:
#       scale = available_width / total_crop_width
#   Using a single common factor preserves each crop's aspect ratio AND the
#   relative size differences between the exemplars. The UAV tile itself is never
#   resized or distorted. Works for 1 and for 3 exemplars.
# =============================================================================

def compose_tile_with_exemplars(tile_img, crop_images, rng,
                                margin=STRIP_MARGIN,
                                feather_width=FEATHER_WIDTH,
                                max_strip_height_fraction=MAX_STRIP_HEIGHT_FRACTION,
                                background_blur_radius=BACKGROUND_BLUR_RADIUS):
    """
    Input : tile_img    - PIL image of the tile
            crop_images - list of PIL exemplar crops (1 or 3)
            rng         - deterministic np.random.Generator for the background
    Output: composed  - PIL image actually sent to SAM3
            crop_boxes- list of [x1,y1,x2,y2] of each exemplar in COMPOSED coords
                        (these are the positive visual prompts)
            offset    - (dx, dy) where the real tile starts inside 'composed'
    """
    n = len(crop_images)
    canvas_w = tile_img.width                       # never wider than the tile
    available_w = canvas_w - margin * (n + 1)       # space left for the crops
    total_crop_w = sum(c.width for c in crop_images)

    # --- 1) shrink the exemplars (aspect ratio preserved) if they do not fit ---
    scale = 1.0
    if total_crop_w > 0 and available_w > 0 and total_crop_w > available_w:
        scale = available_w / float(total_crop_w)

    # --- 2) optional additional height clamp (disabled by default) -------------
    if max_strip_height_fraction is not None:
        max_crop_h = max(1.0, max_strip_height_fraction * tile_img.height - 2 * margin)
        tallest = max(c.height for c in crop_images)
        if tallest * scale > max_crop_h:
            scale = min(scale, max_crop_h / float(tallest))

    if scale < 1.0:
        crop_images = [
            c.resize((max(1, int(round(c.width * scale))),
                      max(1, int(round(c.height * scale)))), Image.BILINEAR)
            for c in crop_images
        ]

    strip_h = max(c.height for c in crop_images) + 2 * margin
    canvas_h = strip_h + tile_img.height

    # --- 3) paint the whole strip with local texture (no unpainted pixels) -----
    composed = Image.new("RGB", (canvas_w, canvas_h))
    # (cluster: the blur radius is passed through, so --background-blur-radius is honoured)
    composed.paste(get_local_background_patch(tile_img, canvas_w, strip_h, rng,
                                              background_blur_radius), (0, 0))

    # --- 4) paste the real tile below the strip; it fills the full canvas width -
    offset = (0, strip_h)
    composed.paste(tile_img, offset)

    # --- 5) paste the exemplars side by side, with feathered edges -------------
    crop_boxes = []
    cursor_x = margin
    for crop in crop_images:
        composed.paste(crop, (cursor_x, margin), make_feather_mask(crop.size, feather_width))
        crop_boxes.append([cursor_x, margin, cursor_x + crop.width, margin + crop.height])
        cursor_x += crop.width + margin

    return composed, crop_boxes, offset


# =============================================================================
# CELL 14 - TARGET-REGION FILTER (bug fixed) + PLAUSIBILITY FILTER
# =============================================================================
# THE BUG IN keep_only_target_region_detections
# The old rule was "keep the box only if y1 >= dy - 5", i.e. it looked ONLY at the
# top edge of the box. A real Rumex plant sitting at the very top of a tile whose
# predicted box leaks a few dozen pixels into the exemplar strip was deleted, even
# though almost all of the box was inside the tile.
#
# THE FIX - explicit geometric criterion
# The composed image contains two regions:
#     strip region : y in [0, dy)
#     tile  region : y in [dy, dy + tile_h)
# A detection is kept if at least TILE_REGION_MIN_FRACTION (= 0.50) of its AREA
# lies inside the TILE region, i.e. the box belongs to whichever region holds the
# majority of it. Kept boxes are then CLIPPED to the tile region and shifted into
# tile coordinates. No other heuristic is applied here.
# =============================================================================

def keep_only_target_region_detections(boxes, scores, fill_ratios,
                                       offset, tile_w, tile_h,
                                       min_fraction_inside=TILE_REGION_MIN_FRACTION):
    """
    Input : boxes (N,4) in COMPOSED coordinates, scores (N,), fill_ratios (N,)
            offset (dx, dy) = where the tile starts inside the composed image
            tile_w, tile_h  = size of the real tile
    Output: boxes in TILE coordinates, scores, fill_ratios (all filtered)
    """
    dx, dy = offset
    tile_region = (dx, dy, dx + tile_w, dy + tile_h)

    kept_boxes, kept_scores, kept_fills = [], [], []
    for box, score, fill in zip(boxes, scores, fill_ratios):
        x1, y1, x2, y2 = [float(v) for v in box]
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if box_area <= 0:
            continue
        # fraction of the predicted box that lies inside the real tile
        fraction_inside = intersection_area((x1, y1, x2, y2), tile_region) / box_area
        if fraction_inside < min_fraction_inside:
            continue                                   # belongs to the strip
        # clip to the tile region, then convert composed -> tile coordinates
        cx1 = min(max(x1, tile_region[0]), tile_region[2]) - dx
        cy1 = min(max(y1, tile_region[1]), tile_region[3]) - dy
        cx2 = min(max(x2, tile_region[0]), tile_region[2]) - dx
        cy2 = min(max(y2, tile_region[1]), tile_region[3]) - dy
        kept_boxes.append([cx1, cy1, cx2, cy2])
        kept_scores.append(float(score))
        kept_fills.append(float(fill))
    return kept_boxes, kept_scores, kept_fills


def filter_implausible_boxes(boxes, scores, fill_ratios, tile_w, tile_h,
                             min_fill_ratio=MIN_FILL_RATIO,
                             max_area_fraction=MAX_AREA_FRACTION,
                             edge_margin=EDGE_MARGIN):
    """
    Remove detections that cannot be a single Rumex plant. SAME thresholds and
    SAME logic as the original notebook; the only change is that the mask-fill
    ratio is received as a PRE-COMPUTED number instead of a stored mask, because
    masks must never be kept (KEEP_MASKS = False).

    Input : boxes in TILE coordinates, scores, fill_ratios, tile size
    Output: the surviving boxes / scores / fill_ratios
    """
    kept_boxes, kept_scores, kept_fills = [], [], []
    tile_area = float(tile_w * tile_h)
    for box, score, fill in zip(boxes, scores, fill_ratios):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= edge_margin or bh <= edge_margin:          # tiny boxes
            continue
        if (bw * bh) / tile_area > max_area_fraction:       # absurdly large boxes
            continue
        if fill < min_fill_ratio:                           # hollow boxes
            continue
        kept_boxes.append(box)
        kept_scores.append(score)
        kept_fills.append(fill)
    return kept_boxes, kept_scores, kept_fills


# =============================================================================
# CELL 5 - SAM3 MODEL + PROCESSOR
# =============================================================================
# With USE_FP16 the weights are loaded in half precision, which roughly halves
# GPU memory and speeds up inference on a T4. Inference additionally runs inside
# torch.inference_mode() + torch.autocast (see CELL 15).
#
# CELL 15 - SAM3 BATCH INFERENCE  (FP16, no masks kept)
# Same batched, FP16, no-masks-kept inference as E02_2
# =============================================================================
# CLUSTER CHANGES (numerics only, the pipeline is identical):
#  * the model is loaded in --dtype (bfloat16 by default) and autocast runs in that
#    dtype; the notebook used float16 on a Colab T4 (USE_FP16 = True).
#  * the notebook used device_map="auto" (accelerate). Here each shard process
#    sees exactly ONE GPU via CUDA_VISIBLE_DEVICES, so the model is placed
#    explicitly with .to(device) - simpler and it cannot silently offload.
#  * sam3_infer_batch is a method of Sam3Runner, which owns the model and the
#    processor (the notebook kept them as the globals sam3_model / sam3_processor
#    / MODEL_DTYPE from CELL 5); run_global_context_pass (CELL 16) and
#    run_anchor_over_tiles (CELL 17) receive the runner and the command-line
#    settings instead of reading the CELL 3 globals.
#  * torch is imported lazily, so the dry run and PHASE 2 work without it.
# =============================================================================

def _to_numpy(x):
    """Torch tensor (any device/dtype) or numpy array -> float32 numpy array."""
    import torch
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _fill_ratios_from_masks(boxes_np, masks, binarise_at=0.5):
    """
    Fraction of each predicted box that is actually covered by its mask - the
    single number the plausibility filter needs. Only the small box region is
    materialised; the mask itself is dropped by the caller immediately after.
    """
    import torch
    n = len(boxes_np)
    fills = np.zeros(n, dtype=np.float32)
    if masks is None or n == 0:
        return fills
    for i in range(n):
        m = masks[i]
        h, w = int(m.shape[-2]), int(m.shape[-1])
        x1, y1, x2, y2 = boxes_np[i]
        x1c, y1c = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        x2c, y2c = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
        if x2c <= x1c or y2c <= y1c:
            continue
        region = m[..., y1c:y2c, x1c:x2c]
        if torch.is_tensor(region):
            if region.dtype == torch.bool:
                fills[i] = float(region.float().mean())
            else:
                fills[i] = float((region > binarise_at).float().mean())
        else:
            region = np.asarray(region)
            fills[i] = float((region > binarise_at).mean()) if region.size else 0.0
    return fills


class Sam3Runner:
    """CELL 5: owns the SAM3 model + processor. CELL 15: one batched forward pass."""

    def __init__(self, model_id: str, device: Optional[str], dtype: str,
                 mask_threshold: float):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        if self.device == "cpu":
            self.model_dtype = torch.float32     # half precision is pointless on CPU
        self.mask_threshold = mask_threshold

        print(f"Loading SAM3 from '{model_id}' onto {self.device} ({dtype}) ...")
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=self.model_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

        print("SAM3 loaded.")
        print("  model device:", next(self.model.parameters()).device)
        print("  model dtype :", next(self.model.parameters()).dtype)

    def sam3_infer_batch(self, composed_images, crop_boxes_batch,
                         threshold=CONFIDENCE_THRESHOLD):
        """
        Run SAM3 on a batch of composed images (exemplar strip + tile).

        Output: list (same length as composed_images) of (boxes, scores, fill_ratios),
                all numpy, boxes in COMPOSED-image coordinates, every detection with
                score >= threshold. Masks are NOT returned.
        """
        torch = self.torch
        inputs = self.processor(
            images=list(composed_images),
            input_boxes=[[[float(v) for v in b] for b in boxes] for boxes in crop_boxes_batch],
            input_boxes_labels=[[1] * len(boxes) for boxes in crop_boxes_batch],  # 1 = positive
            return_tensors="pt",
        ).to(self.device)

        # match the pixel tensor dtype to the (possibly half precision) model weights
        if self.model_dtype != torch.float32 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)

        per_image = []
        with torch.inference_mode():
            if self.model_dtype != torch.float32 and self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=self.model_dtype):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            # Post-process at CONFIDENCE_THRESHOLD (0.30, fixed)...
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )

            for i, res in enumerate(results):
                boxes = _to_numpy(res["boxes"]).reshape(-1, 4)
                scores = _to_numpy(res["scores"]).reshape(-1)
                fills = _fill_ratios_from_masks(boxes, res.get("masks", None))
                if "masks" in res:
                    res["masks"] = None

                per_image.append((boxes, scores, fills))

        del inputs, outputs, results
        return per_image


# =============================================================================
# CELL 16 - OPTIONAL GLOBAL-CONTEXT PASS (opt-in, ADD_GLOBAL_CONTEXT_PASS)
# =============================================================================
# An EXTRA single pass over the whole image, downscaled by GLOBAL_DOWNSCALE,
# run through the exact same exemplar-strip + SAM3 + filter pipeline as a tile
# (it IS treated as one big "tile" -- no cropping). Its detections are rescaled
# back to full-resolution coordinates and returned so the caller (CELL 17) can
# simply append them to the tiled pre-NMS detection pool. Because they join the
# SAME pool that gets NMS'd and evaluated offline, nothing downstream needs any
# special-casing for this pass.
# CLUSTER CHANGE: + 'runner' (CELL 5/15) and 'args' (the CELL 3 strip / filter
# settings from the command line) as parameters.
# =============================================================================

def run_global_context_pass(runner, image, exemplar_crops, run_seed, args,
                            downscale=GLOBAL_DOWNSCALE,
                            threshold=CONFIDENCE_THRESHOLD):
    """
    Input : runner         - Sam3Runner (CELL 5 model + processor)
            image          - the open, full-resolution PIL image
            exemplar_crops - list of PIL exemplar crops (same ones used for tiles)
            run_seed       - deterministic seed for the strip background sampling
            args           - the CELL 3 settings (strip layout, filters)
    Output: dict with 'boxes' (N,4), 'scores' (N,), 'fill_ratio' (N,) in
            ORIGINAL-IMAGE coordinates. Empty arrays if nothing is detected.
    """
    small_w = max(1, image.width // downscale)
    small_h = max(1, image.height // downscale)
    small_img = image.resize((small_w, small_h), Image.BILINEAR)

    rng = np.random.default_rng(stable_seed(run_seed, "global_pass_bg"))
    composed, crop_boxes, offset = compose_tile_with_exemplars(
        small_img, exemplar_crops, rng,
        margin=args.strip_margin,
        feather_width=args.feather_width,
        max_strip_height_fraction=args.max_strip_height_fraction,
        background_blur_radius=args.background_blur_radius)

    (boxes, scores, fills), = runner.sam3_infer_batch([composed], [crop_boxes], threshold)

    boxes, scores, fills = keep_only_target_region_detections(
        boxes, scores, fills, offset, small_img.width, small_img.height,
        min_fraction_inside=args.tile_region_min_fraction)
    boxes, scores, fills = filter_implausible_boxes(
        boxes, scores, fills, small_img.width, small_img.height,
        min_fill_ratio=args.min_fill_ratio,
        max_area_fraction=args.max_area_fraction,
        edge_margin=args.edge_margin)

    # small_img coords -> original image coords
    scale = image.width / small_img.width
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) * scale

    composed.close()
    small_img.close()

    return {
        "boxes": boxes.reshape(-1, 4),
        "scores": np.asarray(scores, dtype=np.float32).reshape(-1),
        "fill_ratio": np.asarray(fills, dtype=np.float32).reshape(-1),
    }


# =============================================================================
# CELL 17 - RUN ONE ANCHOR OVER ALL TILES (+ optional global-context pass)
# =============================================================================
# Runs SAM3 for ONE anchor/exemplar configuration over ALL cached tiles (same
# as E02_2), and -- if ADD_GLOBAL_CONTEXT_PASS is on -- ALSO runs the extra
# downsampled whole-image pass from CELL 16 and appends its detections to the
# SAME pre-NMS pool, tagged with tile_id=-1 so they're identifiable later
# (e.g. for diagnostics) without needing a separate storage schema.
# CLUSTER CHANGE: + 'runner' and 'args' as parameters; batch_size, threshold,
# add_global_pass and the global downscale come from the command line.
# =============================================================================

def run_anchor_over_tiles(runner, tile_cache, full_image, exemplar_crops, run_seed, args,
                          batch_size=BATCH_SIZE, threshold=CONFIDENCE_THRESHOLD,
                          add_global_pass=ADD_GLOBAL_CONTEXT_PASS):
    """
    Output: dict of numpy arrays, all in ORIGINAL-IMAGE coordinates:
            boxes (N,4), scores (N,), fill_ratio (N,), tile_id (N,),
            tile_boxes (N,4)  <- the tile each detection came from
                                 (tile_id == -1, tile_boxes == whole image
                                 extent, for detections from the global pass)
    These are the PRE-NMS detections (score >= CONFIDENCE_THRESHOLD, no NMS)
    that get written to disk for the offline evaluation.
    """
    all_boxes, all_scores, all_fills, all_tids, all_tboxes = [], [], [], [], []

    for start in range(0, len(tile_cache), batch_size):
        batch_tiles = tile_cache[start:start + batch_size]

        composed_images, crop_boxes_batch, offsets, sizes = [], [], [], []
        for tile in batch_tiles:
            tile_img = get_tile_image(tile, full_image)
            rng = np.random.default_rng(stable_seed(run_seed, "tile", tile["tile_id"]))
            composed, crop_boxes, offset = compose_tile_with_exemplars(
                tile_img, exemplar_crops, rng,
                margin=args.strip_margin,
                feather_width=args.feather_width,
                max_strip_height_fraction=args.max_strip_height_fraction,
                background_blur_radius=args.background_blur_radius)
            composed_images.append(composed)
            crop_boxes_batch.append(crop_boxes)
            offsets.append(offset)
            sizes.append((tile_img.width, tile_img.height))

        batch_results = runner.sam3_infer_batch(composed_images, crop_boxes_batch, threshold)

        for tile, (boxes, scores, fills), offset, (tw, th) in zip(
                batch_tiles, batch_results, offsets, sizes):
            boxes, scores, fills = keep_only_target_region_detections(
                boxes, scores, fills, offset, tw, th,
                min_fraction_inside=args.tile_region_min_fraction)
            boxes, scores, fills = filter_implausible_boxes(
                boxes, scores, fills, tw, th,
                min_fill_ratio=args.min_fill_ratio,
                max_area_fraction=args.max_area_fraction,
                edge_margin=args.edge_margin)
            for b, s, f in zip(boxes, scores, fills):
                all_boxes.append([b[0] + tile["x1"], b[1] + tile["y1"],
                                  b[2] + tile["x1"], b[3] + tile["y1"]])
                all_scores.append(float(s))
                all_fills.append(float(f))
                all_tids.append(int(tile["tile_id"]))
                all_tboxes.append([tile["x1"], tile["y1"], tile["x2"], tile["y2"]])

        for ct in composed_images:
            ct.close()
        del composed_images, crop_boxes_batch, batch_results

    used_global_pass = False
    if add_global_pass:
        global_det = run_global_context_pass(runner, full_image, exemplar_crops, run_seed, args,
                                             downscale=args.global_downscale,
                                             threshold=threshold)
        n_global = len(global_det["scores"])
        used_global_pass = n_global > 0
        for i in range(n_global):
            b = global_det["boxes"][i]
            all_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            all_scores.append(float(global_det["scores"][i]))
            all_fills.append(float(global_det["fill_ratio"][i]))
            all_tids.append(-1)  # sentinel: global-context-pass detection
            all_tboxes.append([0, 0, full_image.width, full_image.height])

    return {
        "boxes": np.array(all_boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.array(all_scores, dtype=np.float32).reshape(-1),
        "fill_ratio": np.array(all_fills, dtype=np.float32).reshape(-1),
        "tile_id": np.array(all_tids, dtype=np.int32).reshape(-1),
        "tile_boxes": np.array(all_tboxes, dtype=np.int32).reshape(-1, 4),
    }, used_global_pass


# =============================================================================
# CELL 18 - PRE-NMS DETECTION STORAGE
# =============================================================================
# WHAT IS SAVED AND WHY
# For every run (= one image x one anchor) we store the detections AFTER
#   SAM3 inference at 0.30 -> target-region filtering -> plausibility filtering
#   -> conversion to original-image coordinates
# but BEFORE NMS.
#
# The offline evaluation (CELL 20 onwards) therefore never needs SAM3 again.
# Masks are never stored.
# NPZ is used because it is compact and loads fast; the metric tables are CSV.
# CLUSTER CHANGE: the NPZ also stores archive, flight and source_class_id, and the
# functions take the raw_detections folder / experiment name as parameters
# (the notebook read RAW_DETECTIONS_DIR / EXPERIMENT_NAME from CELL 3 / CELL 4).
# =============================================================================

def run_npz_path(raw_detections_dir: Path, image_id, anchor_idx) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one run."""
    return raw_detections_dir / f"{safe_filename(image_id)}__anchor{int(anchor_idx):03d}.npz"


def save_run_detections(raw_detections_dir: Path, experiment_name: str,
                        image_id, anchor_idx, detections, gt_boxes,
                        prompt_indices, image_size,
                        archive: str, flight: str, class_id: int) -> Path:
    """
    Write one run's pre-NMS detections to NPZ. The file is self-contained: it also
    stores the GT boxes and the prompt indices, so the whole offline evaluation
    can run without re-opening images or label files.
    """
    path = run_npz_path(raw_detections_dir, image_id, anchor_idx)
    np.savez_compressed(
        path,
        experiment_name=np.array(experiment_name),
        image_ID=np.array(image_id),
        anchor_idx=np.array(int(anchor_idx)),
        prompt_indices=np.array(prompt_indices, dtype=np.int32),
        image_width=np.array(int(image_size[0])),
        image_height=np.array(int(image_size[1])),
        archive=np.array(archive),
        flight=np.array(flight),
        source_class_id=np.array(int(class_id)),
        gt_boxes=gt_boxes.astype(np.float32),
        boxes=detections["boxes"].astype(np.float32),        # x1,y1,x2,y2 (original img)
        scores=detections["scores"].astype(np.float32),      # confidence >= 0.30
        fill_ratio=detections["fill_ratio"].astype(np.float32),
        tile_id=detections["tile_id"].astype(np.int32),      # which tile produced it
        tile_boxes=detections["tile_boxes"].astype(np.int32),# that tile's extent
    )
    return path


def load_run_detections(path) -> dict:
    """Read one run NPZ back into a plain python dict."""
    with np.load(path, allow_pickle=False) as z:
        run = {
            "image_ID": str(z["image_ID"]),
            "anchor_idx": int(z["anchor_idx"]),
            "prompt_indices": z["prompt_indices"].astype(int),
            "image_width": int(z["image_width"]),
            "image_height": int(z["image_height"]),
            "gt_boxes": z["gt_boxes"].reshape(-1, 4),
            "boxes": z["boxes"].reshape(-1, 4),
            "scores": z["scores"].reshape(-1),
            "fill_ratio": z["fill_ratio"].reshape(-1),
            "tile_id": z["tile_id"].reshape(-1),
            "tile_boxes": z["tile_boxes"].reshape(-1, 4),
        }
        # cluster additions (read INSIDE the with-block, before the file closes)
        run["archive"] = str(z["archive"]) if "archive" in z.files else ""
        run["flight"] = str(z["flight"]) if "flight" in z.files else ""
    return run


# =============================================================================
# CELL 21 - OFFLINE NMS
# =============================================================================
# Because the same plant is visible in several overlapping tiles, it can be
# detected several times. NMS keeps the highest-scoring box of each overlapping
# group. The NMS IoU threshold is fixed (NMS_IOU_THRESHOLD = 0.40) and applied
# offline, so the raw pre-NMS detections stay untouched on disk.
#
# 'Provenance' = we also record WHICH detection suppressed which. That lets the
# false-positive diagnostics (CELL 29) count how many duplicates came from a
# DIFFERENT tile (cross-tile duplication) versus the same tile.
# =============================================================================

def nms_with_provenance(boxes, scores, iou_threshold):
    """
    Input : boxes (N,4), scores (N,), iou_threshold
    Output: keep (list of kept indices, highest score first)
            suppressed (list of (suppressed_index, suppressor_index))
    A detection is suppressed when its IoU with an already kept, higher-scoring
    detection is GREATER than the threshold (same convention as the original code).
    """
    n = len(boxes)
    if n == 0:
        return [], []
    order = list(np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable"))
    keep, suppressed = [], []
    while order:
        i = int(order[0])
        keep.append(i)
        rest = np.array(order[1:], dtype=int)
        if rest.size == 0:
            break
        ious = compute_iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        for s in rest[ious > iou_threshold]:
            suppressed.append((int(s), i))
        order = list(rest[ious <= iou_threshold])
    return keep, suppressed


def apply_nms_to_run(run, iou_threshold):
    """
    Apply NMS to one run's pre-NMS detections.

    Output: dict with 'boxes', 'scores', 'tile_id', 'tile_boxes' of the surviving
            detections SORTED BY SCORE (high -> low), plus the duplicate-source
            counters used later by the diagnostics.
    Sorting by score means that applying an operating confidence threshold later
    is just a prefix selection.
    """
    keep, suppressed = nms_with_provenance(run["boxes"], run["scores"], iou_threshold)
    keep = np.array(keep, dtype=int)

    cross_tile_suppressed, same_tile_suppressed = 0, 0
    for s, k in suppressed:
        if run["tile_id"][s] != run["tile_id"][k]:
            cross_tile_suppressed += 1
        else:
            same_tile_suppressed += 1

    return {
        "boxes": run["boxes"][keep].reshape(-1, 4),
        "scores": run["scores"][keep].reshape(-1),
        "tile_id": run["tile_id"][keep].reshape(-1),
        "tile_boxes": run["tile_boxes"][keep].reshape(-1, 4),
        "n_pre_nms": int(len(run["scores"])),
        "n_suppressed_cross_tile": int(cross_tile_suppressed),
        "n_suppressed_same_tile": int(same_tile_suppressed),
    }


# =============================================================================
# CELL 22 - EVALUATION CORE: all_gt AND held_out
# =============================================================================
# all_gt   : classical evaluation, every GT box of the image counts.
#
# held_out : the GT instances that were shown to SAM3 as visual prompts are
#            REMOVED from the GT set, and predictions that fall on those prompt
#            plants are IGNORED (they are neither TP nor FP, they simply do not
#            exist for this evaluation). It answers: "after being shown a few
#            examples, how well does SAM3 find the REMAINING Rumex plants?"
#
# ORDER OF OPERATIONS (important, this is the agreed protocol):
#   1. match predictions to the evaluated (non-prompt) GT with the corrected
#      one-to-one matcher at EVAL_IOU_THRESHOLD = 0.50
#   2. every STILL UNMATCHED prediction whose best IoU with a PROMPT GT box is
#      >= PROMPT_IGNORE_IOU (0.50) becomes IGNORED
#   3. whatever is still unmatched is a false positive
#   Prompt GT boxes themselves are never counted as false negatives.
# =============================================================================

STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2


def split_gt_for_mode(gt_boxes, prompt_indices, mode):
    """
    Input : all GT boxes of the image, the indices used as visual prompts, mode
    Output: (evaluated_gt_boxes, prompt_gt_boxes)
            all_gt   -> (all boxes, empty)
            held_out -> (non-prompt boxes, prompt boxes)
    """
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    if mode == "all_gt":
        return gt_boxes, np.zeros((0, 4), dtype=np.float32)
    is_prompt = np.zeros(len(gt_boxes), dtype=bool)
    prompt_indices = np.asarray(prompt_indices, dtype=int)
    if len(prompt_indices):
        is_prompt[prompt_indices] = True
    return gt_boxes[~is_prompt], gt_boxes[is_prompt]


def evaluate_run_predictions(pred_boxes, pred_scores, eval_gt_boxes, prompt_gt_boxes,
                             eval_iou=EVAL_IOU_THRESHOLD, ignore_iou=PROMPT_IGNORE_IOU):
    """
    Evaluate ONE prediction set against ONE GT set.

    Output dict:
      status      (P,) int  - STATUS_TP / STATUS_FP / STATUS_IGNORED per prediction
      TP, FP, FN, n_ignored, n_eval_gt, n_pred
      precision, recall, F1
      IoU1 - mean IoU of the MATCHED prediction/GT pairs only
             ("when it finds a plant, how well is it localised?")
      IoU2 - sum of matched IoUs divided by the number of evaluated GT boxes
             ("localisation quality over ALL plants, missed ones count as 0")
      valid_for_macro - False when there is no GT left to evaluate (held_out runs
             in which every plant of the image was used as a prompt)
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    eval_gt_boxes = np.asarray(eval_gt_boxes, dtype=np.float32).reshape(-1, 4)
    prompt_gt_boxes = np.asarray(prompt_gt_boxes, dtype=np.float32).reshape(-1, 4)

    n_pred, n_eval_gt = len(pred_boxes), len(eval_gt_boxes)
    status = np.full(n_pred, STATUS_FP, dtype=np.int8) # start eveything as FP then valid matches TP then leftover predictions on prompt as ignored and everything still FP

    # step 1 - corrected one-to-one matching against the evaluated GT
    match = match_one_to_one(pred_boxes, pred_scores, eval_gt_boxes, eval_iou)
    status[match["pred_match_gt"] >= 0] = STATUS_TP # Mark matched predictions as TP

    # step 2 - ignore the leftovers that sit on a PROMPT plant
    # Therefore prompt-ignore logic applies only to predictions that failed to match evaluated GT
    if n_pred and len(prompt_gt_boxes):
        leftover = np.where(status == STATUS_FP)[0] # Find predictions that are still FP (unmatched preds)
        if len(leftover):
            best_prompt_iou = compute_iou_matrix(pred_boxes[leftover], prompt_gt_boxes).max(axis=1)
            status[leftover[best_prompt_iou >= ignore_iou]] = STATUS_IGNORED

    tp = int((status == STATUS_TP).sum())
    fp = int((status == STATUS_FP).sum())          # step 3 - the rest are FP
    n_ignored = int((status == STATUS_IGNORED).sum())
    fn = int(n_eval_gt - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_eval_gt if n_eval_gt > 0 else 0.0
    f1 = safe_f1(precision, recall)
    matched_ious = match["matched_ious"]
    iou1 = float(np.mean(matched_ious)) if matched_ious else 0.0
    iou2 = float(np.sum(matched_ious) / n_eval_gt) if n_eval_gt > 0 else 0.0

    return {
        "status": status, "pred_match_gt": match["pred_match_gt"],
        "TP": tp, "FP": fp, "FN": fn, "n_ignored": n_ignored,
        "n_eval_gt": n_eval_gt, "n_pred": n_pred,
        "precision": float(precision), "recall": float(recall), "F1": float(f1),
        "IoU1": iou1, "IoU2": iou2,
        "valid_for_macro": bool(n_eval_gt > 0),
    }


# =============================================================================
# CELL 23 - AP50 AND AP50:95  (supervision.metrics.MeanAveragePrecision)
# =============================================================================
# WHY AP IS COMPUTED DIFFERENTLY FROM PRECISION / RECALL / F1
#
#   AP is an area under the precision-recall curve. That curve is produced by
#   walking through ALL detections ordered by confidence. Truncating the
#   detection list at an operating threshold would simply cut the tail off the
#   curve and report a smaller area - which says nothing about model quality.
#   Therefore AP always uses ALL saved predictions with score >= 0.30
#   (CONFIDENCE_THRESHOLD), after NMS.
#
#   Precision / recall / F1 / IoU1 / IoU2 describe ONE operating point: they
#   answer "if I deploy the detector with confidence >= c, what happens?".
#   Those use only the detections that survive the operating threshold.
#   In this experiment the operating threshold IS the inference threshold (0.30),
#   so both use exactly the same predictions - they still answer different
#   questions (ranking quality vs. deployed behaviour).
# =============================================================================

# just a helper function that converts NumPy boxes into the format expected by Supervision
def make_detections(boxes, scores=None):
    import supervision as sv
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    if scores is None:
        return sv.Detections(xyxy=boxes, class_id=np.zeros(n, dtype=int))
    return sv.Detections(xyxy=boxes,
                         confidence=np.asarray(scores, dtype=np.float32).reshape(-1),
                         class_id=np.zeros(n, dtype=int))


def compute_ap(pred_list, gt_list):
    """
    Input : two equally long lists of supervision Detections (predictions / GT).
            One entry = one evaluation episode (one run).
    Output: (AP50, AP50_95). NaN when there is no GT at all to evaluate.
    Passing several episodes at once gives the POOLED (dataset-level) AP, in which
    the detections of all episodes are ranked together in one PR curve.
    """
    from supervision.metrics import MeanAveragePrecision
    if len(gt_list) == 0 or sum(len(g) for g in gt_list) == 0: # Check whether there is any GT
        return float("nan"), float("nan")
    try:
        # Create the Supervision metric (metric calculator, provides all pred and GTs, calculates the AP results)
        # Supervision exposes map50 for IoU=0.50 and map50_95 for IoUs 0.50:0.95
        result = MeanAveragePrecision().update(pred_list, gt_list).compute()
        ap50, ap5095 = float(result.map50), float(result.map50_95)
        # supervision returns -1.0 when a metric is undefined -> report NaN instead,
        # so it is excluded from means instead of dragging them down.
        return (ap50 if ap50 >= 0 else float("nan"),
                ap5095 if ap5095 >= 0 else float("nan"))
    except Exception as e:
        print("   (AP computation failed:", e, ")")
        return float("nan"), float("nan")


# Its job is not to calculate AP yet, Its job is to prepare: pred and GT for one run so they can later be given to compute_ap(...)
# nms_run: contains your detections after NMS
def ap_inputs_for_run(nms_run, eval_gt, prompt_gt,
                      eval_iou=EVAL_IOU_THRESHOLD, ignore_iou=PROMPT_IGNORE_IOU):
    """
    Build the (prediction, GT) episode used for AP of ONE run.
    All post-NMS predictions with score >= CONFIDENCE_THRESHOLD are used;
    in held_out mode the predictions that were IGNORED (they belong to prompt
    plants) are removed first, exactly like in the operating-point evaluation.
    """
    ev = evaluate_run_predictions(nms_run["boxes"], nms_run["scores"], eval_gt, prompt_gt,
                                  eval_iou, ignore_iou) # marks each pred as TP,FP,IGNORED, it cares only about the ignored predictions for AP calculation bcz in held_out mode , predictions corresponding to prompt rumex must be removed before AP
    keep = ev["status"] != STATUS_IGNORED  # Remove ignored predictions
    # Convert remaining predictions to Supervision format
    return (make_detections(nms_run["boxes"][keep], nms_run["scores"][keep]),
            make_detections(eval_gt))


# =============================================================================
# CELL 24 - OPERATING-POINT EVALUATION
# =============================================================================
# Choose one confidence threshold, remove predictions below it, then send the
# remaining predictions to CELL 22 for evaluation. Gives Precision, Recall,
# F1, IoU1, and IoU2 at one operating threshold.
# =============================================================================

def evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, confidence_threshold,
                                eval_iou=EVAL_IOU_THRESHOLD, ignore_iou=PROMPT_IGNORE_IOU):
    """
    Input : nms_run  - output of apply_nms_to_run (sorted by score, high -> low) => contains the detections after NMS            eval_gt / prompt_gt - from split_gt_for_mode
            confidence_threshold - the operating point being tested
    Output: the dict of evaluate_run_predictions for the thresholded predictions.
    """
    keep = nms_run["scores"] >= confidence_threshold # boolean mask where True entries are selected and the False entries are excluded
    return evaluate_run_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                    eval_gt, prompt_gt, eval_iou, ignore_iou)


# =============================================================================
# CELL 27 - IMAGE-LEVEL METRICS (the metric columns; the loop is in run_evaluation)
# =============================================================================

METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2"]


# =============================================================================
# PLOTTING HELPERS  (notebook CELL 30 plot_confusion_matrix, CELL 32 _draw_boxes)
# =============================================================================
# Colours (notebook CELL 32):
#   yellow = GT Rumex boxes
#   red    = predictions
#   lime   = exemplar (prompt) boxes of the shown run, dashed
# CLUSTER CHANGE: matplotlib is imported lazily with the headless 'Agg' backend
# (run_evaluation passes 'plt' in) and figures are saved, never shown.
# =============================================================================

def plot_confusion_matrix(tp, fp, fn, title, png_path, plt):
    """2x2 detection confusion matrix; the background/background cell stays empty."""
    matrix = np.array([[tp, fn], [fp, np.nan]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(np.nan_to_num(matrix, nan=0.0), cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted\nRumex", "Predicted\nBackground"])
    ax.set_yticks([0, 1], ["Actual\nRumex", "Actual\nBackground"])
    labels = [[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", "n/a\n(no true\nnegatives)"]]
    vmax = np.nanmax(matrix) if np.nanmax(matrix) > 0 else 1.0
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            colour = "white" if (not np.isnan(value) and value > 0.5 * vmax) else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    color=colour, fontsize=11)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)                     # headless: saved, never shown


def _draw_boxes(ax, boxes, color, linewidth=1.5, linestyle="-", labels=None, fontsize=6):
    """Draw [x1,y1,x2,y2] boxes (display coordinates) on a matplotlib axis."""
    import matplotlib.patches as patches
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=color, linewidth=linewidth,
                                       linestyle=linestyle))
        if labels is not None:
            ax.text(x1, y1 - 2, labels[k], color="white", fontsize=fontsize,
                    va="bottom", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.8, alpha=0.85))


# =============================================================================
#  RESUME SUPPORT  (CELL 19, adapted to several shard manifests)
# =============================================================================

def _manifest_files(paths: Paths, experiment_name: str) -> List[Path]:
    safe_name = experiment_name.replace("/", "-")
    return sorted(paths.raw_detections.glob(f"runs_manifest_{safe_name}_shard*.csv"))


def load_done_runs(paths: Paths, experiment_name: str) -> set:
    """
    Read every shard manifest and return the set of (image_ID, anchor_idx) runs that
    are already finished for this EXPERIMENT_NAME (the notebook's done_runs).
    Every shard reads ALL manifests, so a resubmission after the walltime never
    repeats work, even if the shard assignment changed because NUM_GPUS was
    different.
    """
    done: set = set()
    for csv_path in _manifest_files(paths, experiment_name):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    try:
                        done.add((str(row["image_ID"]), int(row["anchor_idx"])))
                    except (KeyError, TypeError, ValueError):
                        continue            # ignore a half-written trailing row
        except OSError:
            continue
    return done


# =============================================================================
# CELL 19 - MAIN GPU INFERENCE LOOP  - PHASE 1
# =============================================================================
# Same structure as E02_2: open image once, build tiles once, loop anchors,
# save PRE-NMS detections. NO NMS and NO metric computation here - all done
# offline in the following cells. Resumable via
# runs_manifest.csv.
#
# CLUSTER CHANGE: this runs once per GPU shard, over this shard's images only,
# with one manifest per shard. The notebook's RAM guard (MEM_STOP_THRESHOLD_PCT,
# psutil) and its "[MEM] RSS" print are removed.
# =============================================================================

def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    import torch

    EXPERIMENT_NAME = args.experiment_name
    PROMPT_TYPE = args.prompt_type
    N_EXEMPLARS = args.n_exemplars

    done_runs = set()
    if not args.no_resume:
        done_runs = load_done_runs(paths, EXPERIMENT_NAME)
        print(f"Resuming: {len(done_runs)} runs already finished for {EXPERIMENT_NAME}.")

    manifest_csv = manifest_path(paths, EXPERIMENT_NAME, args.shard_index)
    manifest_exists = manifest_csv.exists() and manifest_csv.stat().st_size > 0
    manifest_file = open(manifest_csv, "a", newline="")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
    if not manifest_exists:
        manifest_writer.writeheader()
        manifest_file.flush()

    # CELL 5 - SAM3 MODEL + PROCESSOR
    runner = Sam3Runner(args.model_id, args.device, args.dtype, args.mask_threshold)
    device = runner.device

    start_time = time.time()
    n_new_runs = 0
    image_times = []
    n_total_images = len(records)

    try:
        for img_idx, rec in enumerate(records, start=1):
            image_path, label_path, image_id = rec.image_path, rec.label_path, rec.image_id
            image_t0 = time.time()

            # ---------------- open the original image exactly once --------------------
            image = Image.open(image_path).convert("RGB")
            img_w, img_h = image.size
            gt_boxes = load_yolo_boxes(label_path, img_w, img_h, rec.class_id)
            n_gt = len(gt_boxes)

            if n_gt == 0:
                print(f"[{EXPERIMENT_NAME}] ({img_idx}/{n_total_images}) {image_id}: 0 GT boxes, skipped.")
                image.close(); del image; gc.collect()
                continue

            # CLUSTER: --max-anchors-per-image (debug) limits the anchors; 0 = every GT box
            n_anchors = n_gt if args.max_anchors_per_image <= 0 else min(n_gt, args.max_anchors_per_image)

            if all((image_id, a) in done_runs for a in range(n_anchors)):
                print(f"[{EXPERIMENT_NAME}] ({img_idx}/{n_total_images}) {image_id}: all "
                      f"{n_anchors} anchors already done, skipped.")
                image.close(); del image; gc.collect()
                continue

            # ---------------- build the tiles exactly once ----------------------------
            tile_cache = build_tile_cache(image, args.use_tiling, args.tile_size,
                                          args.overlap, args.cache_tiles_in_memory)
            n_tiles = len(tile_cache)

            for anchor_idx in range(n_anchors):            # every GT box is an anchor once
                if (image_id, anchor_idx) in done_runs:
                    continue

                run_t0 = time.time()

                # deterministic prompt selection (SHA-256 based, see CELL 6)
                exemplar_indices = select_exemplar_indices(n_gt, anchor_idx, N_EXEMPLARS, image_id,
                                                           EXPERIMENT_NAME)
                prompt_id = format_prompt_id(exemplar_indices)
                exemplar_crops = [safe_crop(image, gt_boxes[i]) for i in exemplar_indices]

                run_seed = stable_seed(EXPERIMENT_NAME, image_id, anchor_idx, "strip_bg")
                detections, used_global_pass = run_anchor_over_tiles(
                    runner, tile_cache, image, exemplar_crops, run_seed, args,
                    batch_size=args.batch_size, threshold=args.confidence_threshold,
                    add_global_pass=args.add_global_context_pass)

                npz_path = save_run_detections(paths.raw_detections, EXPERIMENT_NAME,
                                               image_id, anchor_idx, detections,
                                               gt_boxes, exemplar_indices, (img_w, img_h),
                                               rec.archive, rec.flight, rec.class_id)

                run_seconds = time.time() - run_t0
                manifest_writer.writerow({
                    "experiment_name": EXPERIMENT_NAME,
                    "image_ID": image_id,
                    "anchor_idx": anchor_idx,
                    "Prompt_ID": prompt_id,
                    "Prompt_Type": PROMPT_TYPE,
                    "archive": rec.archive,
                    "flight": rec.flight,
                    "source_class_id": rec.class_id,
                    "n_gt": n_gt,
                    "n_prompt_gt": len(exemplar_indices),
                    "n_detections_pre_nms": int(len(detections["scores"])),
                    "n_tiles": n_tiles,
                    "used_global_pass": used_global_pass,
                    "image_width": img_w,
                    "image_height": img_h,
                    "npz_file": npz_path.name,
                    "inference_seconds": round(run_seconds, 2),
                })
                manifest_file.flush()                  # this run is on disk -> resumable
                n_new_runs += 1

                print(f"  [{EXPERIMENT_NAME}] shard{args.shard_index} run #{n_new_runs} | {image_id} | "
                      f"anchor={anchor_idx} ({anchor_idx + 1}/{n_anchors}) | prompt={prompt_id} | "
                      f"tiles={n_tiles} | global_pass={used_global_pass} | "
                      f"pre-NMS detections={len(detections['scores'])} | {run_seconds:.1f}s")

                del detections, exemplar_crops
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

            # ---------------- release the image and its tile cache --------------------
            for t in tile_cache:
                t["image"] = None
            del tile_cache
            image.close()
            del image
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            image_elapsed = time.time() - image_t0
            image_times.append(image_elapsed)
            avg_per_image = float(np.mean(image_times))
            eta = (n_total_images - img_idx) * avg_per_image
            print(f"[{EXPERIMENT_NAME}] shard{args.shard_index} ({img_idx}/{n_total_images}) {image_id} done | "
                  f"{n_gt} GT box(es) | {image_elapsed:.1f}s | avg/image={avg_per_image:.1f}s | "
                  f"ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")
    finally:
        manifest_file.close()                      # also closed if the loop crashes

    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {EXPERIMENT_NAME} (shard {args.shard_index}): {n_new_runs} new runs.")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Pre-NMS detections in: {paths.raw_detections}")


# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================

def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    """
    Opens every image of this shard (header only) and its label file, and counts
    what PHASE 1 will do with the REAL image sizes: the anchor runs (one per GT
    box), the tiles of every image (tile_bboxes, CELL 11) and the SAM3 forward
    passes per anchor run (ceil(tiles / BATCH_SIZE), + 1 for the global pass).
    """
    print("\n--- DRY RUN: counting the work without loading SAM3 ---")
    n_with_gt, n_without_gt, total_runs = 0, 0, 0
    total_tiles, total_passes = 0, 0
    per_archive: dict = {}
    for rec in my_records:
        with Image.open(rec.image_path) as im:
            w, h = im.size
        n_gt = len(load_yolo_boxes(rec.label_path, w, h, rec.class_id))
        if n_gt == 0:
            n_without_gt += 1
            continue
        n_with_gt += 1
        n_anchors = n_gt if args.max_anchors_per_image <= 0 else min(n_gt, args.max_anchors_per_image)
        n_tiles = len(tile_bboxes(w, h, args.tile_size, args.overlap)) if args.use_tiling else 1
        passes_per_run = math.ceil(n_tiles / args.batch_size) + (1 if args.add_global_context_pass else 0)
        total_runs += n_anchors
        total_tiles += n_tiles
        total_passes += n_anchors * passes_per_run
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + n_anchors
    print(f"  this shard: {len(my_records)} image(s) -> {n_with_gt} with GT boxes, "
          f"{n_without_gt} without (skipped)")
    print(f"  anchor runs (one per GT box): {total_runs}  per archive: {per_archive}")
    print(f"  prompts per run: {args.n_exemplars} ({args.prompt_type}) - anchor first, "
          f"then seeded random other GT boxes")
    if n_with_gt:
        print(f"  tiles per image: {total_tiles / n_with_gt:.1f} on average "
              f"(tiling={args.use_tiling}, tile={args.tile_size}, overlap={args.overlap})")
    print(f"  global-context pass: {args.add_global_context_pass} "
          f"(downscale={args.global_downscale}) -> "
          f"{'+1' if args.add_global_context_pass else '+0'} forward pass per run")
    print(f"  SAM3 forward passes (batch={args.batch_size}): {total_passes}")
    print(f"  NPZ files that will be written by this shard: {total_runs} (one per image x anchor)")


# =============================================================================
# CELL 20 - LOAD CACHED PRE-NMS DETECTIONS   (start of PHASE 2)
# =============================================================================
# From here on SAM3 is never touched again. Everything below works on the NPZ
# files written in CELL 19, so you can restart the runtime, free the GPU and
# still redo the complete evaluation in a few minutes.
# To run the evaluation only: run CELLS 1-4 and 6-18 (skip CELL 5 and CELL 19),
# then continue from here.
# On the cluster: run the script with --evaluate-only (the run script's
# 'evaluate' mode), which needs no GPU and never loads SAM3.
# =============================================================================

def load_runs(paths: Paths, experiment_name: str):
    import pandas as pd

    manifest_files = _manifest_files(paths, experiment_name)
    if not manifest_files:
        print(f"No manifest found in {paths.raw_detections} for {experiment_name}.")
        return []

    frames = []
    for path in manifest_files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"  WARNING: could not read {path.name}: {exc}")
    if not frames:
        print("All manifests unreadable.")
        return []

    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[manifest["experiment_name"] == experiment_name].copy()
    manifest = manifest.drop_duplicates(subset=["image_ID", "anchor_idx"], keep="last")
    # CLUSTER CHANGE: the shard manifests are concatenated in shard order, so sort
    # by (image_ID, anchor_idx) -> the same order as the notebook's single manifest,
    # and the pooled AP (CELL 29) does not depend on the shard layout.
    manifest = manifest.sort_values(["image_ID", "anchor_idx"], kind="stable")

    runs = []
    for _, row in manifest.iterrows():
        path = paths.raw_detections / str(row["npz_file"])
        if not path.exists():
            print("MISSING npz (skipped):", path)
            continue
        run = load_run_detections(path)
        run["Prompt_ID"] = str(row["Prompt_ID"])
        run["Prompt_Type"] = str(row["Prompt_Type"])
        run["archive"] = run.get("archive") or str(row.get("archive", ""))
        run["flight"] = run.get("flight") or ("" if pd.isna(row.get("flight", "")) else str(row.get("flight", "")))
        runs.append(run)

    print(f"Loaded {len(runs)} runs "
          f"({manifest['image_ID'].nunique()} images) for {experiment_name}.")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes over all runs:", int(sum(len(r['gt_boxes']) for r in runs)))
    if runs:
        print("Runs per archive:",
              {k: int(v) for k, v in pd.Series([r["archive"] for r in runs]).value_counts().sort_index().items()})
    return runs


# =============================================================================
# CELL 32 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right)
# =============================================================================
# IMAGE SELECTION (image-level AP50, mode = PLOT_EVALUATION_MODE):
#   1. keep only images that have at least PLOT_MIN_GT_BOXES (7) GT boxes
#      and take the one with the highest image-level AP50_mean
#   2. if NO image has 7 GT boxes: keep the images with the HIGHEST number of GT
#      boxes and take the one among them with the highest AP50_mean
#   ties are broken by F1_mean, then by the number of GT boxes.
#
# RUN SHOWN ON THE RIGHT:
#   the image-level AP50 is a mean over all anchors, but a prediction figure shows
#   ONE run -> the anchor of that image with the highest run-level AP50 (ties: F1)
#   is displayed. Its predictions are the post-NMS boxes (tiles + global-context
#   pass, merged by NMS) with score >= CONFIDENCE_THRESHOLD; the N_EXEMPLARS
#   exemplar (prompt) boxes of that run are drawn dashed at their real location
#   in the image.
#
# The image is downscaled for DISPLAY only (PLOT_MAX_DISPLAY_DIM).
#
# CLUSTER CHANGE: the notebook produced ONE figure, because it ran on one
# archive. Both archives are pooled here, so the selection is repeated once per
# archive AND once over ALL images - otherwise a single global winner would hide
# one of the two datasets completely. The file name carries the scope:
#   best_image_<scope>_<image>_anchor<NNN>_<mode>.png   (scope = archive name or ALL)
# =============================================================================

def select_image_for_plot(image_level_df, run_level_df, mode=PLOT_EVALUATION_MODE,
                          min_gt=PLOT_MIN_GT_BOXES, archive=None):
    """
    Output: (image_level_row, selection_rule_text) or (None, reason)
    'archive' restricts the candidates to one archive; None = all images.
    """
    img_df = image_level_df[image_level_df["evaluation_mode"] == mode].copy()
    if archive is not None:
        img_df = img_df[img_df["archive"] == archive]
    n_gt_per_image = (run_level_df[run_level_df["evaluation_mode"] == mode]
                      .groupby("image_ID")["n_gt_total"].first())
    img_df["n_gt"] = img_df["image_ID"].map(n_gt_per_image)
    img_df = img_df[img_df["AP50_mean"].notna() & img_df["n_gt"].notna()]
    if img_df.empty:
        return None, "no image with a valid AP50"
    img_df["n_gt"] = img_df["n_gt"].astype(int)

    candidates = img_df[img_df["n_gt"] >= min_gt]
    if len(candidates):
        rule = (f"highest image-level AP50 among the {len(candidates)} images "
                f"with >= {min_gt} GT boxes")
    else:
        max_gt = int(img_df["n_gt"].max())
        candidates = img_df[img_df["n_gt"] == max_gt]
        rule = (f"no image has >= {min_gt} GT boxes -> highest image-level AP50 among "
                f"the {len(candidates)} image(s) with the most GT boxes ({max_gt})")
    if archive is not None:
        rule += f" [archive={archive}]"

    ranked = candidates.sort_values(["AP50_mean", "F1_mean", "n_gt"],
                                    ascending=False, na_position="last")
    print("Top candidates:")
    print(ranked[["image_ID", "n_gt", "AP50_mean", "F1_mean",
                  "precision_mean", "recall_mean"]].head(5).to_string(index=False))
    return ranked.iloc[0], rule


def plot_gt_vs_predictions(args, paths: Paths, runs, run_level_df, image_level_df,
                           image_paths, plt, scope_label="ALL", archive=None):
    """CELL 32 for one scope (one archive, or ALL images)."""
    from matplotlib.lines import Line2D

    mode = args.plot_evaluation_mode
    EXPERIMENT_NAME = args.experiment_name
    N_EXEMPLARS = args.n_exemplars
    TILE_SIZE = args.tile_size
    OVERLAP = args.overlap
    ADD_GLOBAL_CONTEXT_PASS = args.add_global_context_pass
    GLOBAL_DOWNSCALE = args.global_downscale
    CONFIDENCE_THRESHOLD = args.confidence_threshold
    NMS_IOU_THRESHOLD = args.nms_iou_threshold
    MASK_THRESHOLD = args.mask_threshold
    PLOT_MAX_DISPLAY_DIM = args.plot_max_display_dim
    PLOT_SHOW_SCORES = args.plot_show_scores

    img_row, rule = select_image_for_plot(image_level_df, run_level_df, mode,
                                          args.plot_min_gt_boxes, archive)
    if img_row is None:
        print(f"Nothing to plot for {scope_label}:", rule)
        return
    plot_image_id = img_row["image_ID"]

    # ---- best anchor run of that image ---------------------------------------
    img_runs = run_level_df[(run_level_df["evaluation_mode"] == mode) &
                            (run_level_df["image_ID"] == plot_image_id)]
    run_row = img_runs.sort_values(["AP50", "F1"], ascending=False,
                                   na_position="last").iloc[0]
    anchor = int(run_row["anchor_idx"])
    run = next(r for r in runs
               if r["image_ID"] == plot_image_id and r["anchor_idx"] == anchor)

    nms_run = apply_nms_to_run(run, NMS_IOU_THRESHOLD)
    keep = nms_run["scores"] >= CONFIDENCE_THRESHOLD
    pred_boxes, pred_scores = nms_run["boxes"][keep], nms_run["scores"][keep]
    gt_boxes_plot = run["gt_boxes"]
    exemplar_boxes = gt_boxes_plot[np.asarray(run["prompt_indices"], dtype=int)]

    # ---- open the image, downscale for display only ---------------------------
    if plot_image_id not in image_paths:
        print("Image file not found for", plot_image_id)
        return
    with Image.open(image_paths[plot_image_id]) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(1.0, PLOT_MAX_DISPLAY_DIM / max(w, h))
        disp_w, disp_h = max(1, int(w * s)), max(1, int(h * s))
        display = np.asarray(im.resize((disp_w, disp_h), Image.BILINEAR))
    to_disp = np.array([disp_w / w, disp_h / h, disp_w / w, disp_h / h], dtype=np.float32)

    # ---- figure: GT (left) | predictions (right) -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax in axes:
        ax.imshow(display)
        ax.axis("off")

    _draw_boxes(axes[0], gt_boxes_plot * to_disp, color="yellow", linewidth=1.5)
    axes[0].set_title(f"Ground truth: {len(gt_boxes_plot)} Rumex boxes", fontsize=12)

    score_labels = [f"{v:.2f}" for v in pred_scores] if PLOT_SHOW_SCORES else None
    _draw_boxes(axes[1], pred_boxes * to_disp, color="red", linewidth=1.5,
                labels=score_labels)
    _draw_boxes(axes[1], exemplar_boxes * to_disp, color="lime", linewidth=2.5,
                linestyle="--")
    axes[1].set_title(
        f"Predictions: {len(pred_boxes)} boxes | anchor = {anchor} | "
        f"exemplars = {run['Prompt_ID']}\n"
        f"run AP50={run_row['AP50']:.3f}  P={run_row['precision']:.3f}  "
        f"R={run_row['recall']:.3f}  F1={run_row['F1']:.3f}  "
        f"TP={int(run_row['TP'])} FP={int(run_row['FP'])} FN={int(run_row['FN'])}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color="yellow", lw=2, label="ground truth"),
        Line2D([0], [0], color="red", lw=2, label="prediction"),
        Line2D([0], [0], color="lime", lw=2.5, linestyle="--",
               label=f"exemplar prompts ({len(exemplar_boxes)})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=11,
               frameon=False)
    fig.suptitle(
        f"{EXPERIMENT_NAME} | {plot_image_id} | mode={mode} | scope={scope_label} | "
        f"image-level AP50_mean={img_row['AP50_mean']:.3f} "
        f"(over {int(img_row['n_runs_valid_for_macro'])} anchors)\n"
        f"{N_EXEMPLARS} exemplars | tile={TILE_SIZE}px, overlap={OVERLAP}px, "
        f"global pass={ADD_GLOBAL_CONTEXT_PASS} "
        f"(x1/{GLOBAL_DOWNSCALE}) | conf={CONFIDENCE_THRESHOLD:.2f}, "
        f"NMS IoU={NMS_IOU_THRESHOLD:.2f}, mask thr={MASK_THRESHOLD:.2f}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    png_path = paths.plots / (f"best_image_{scope_label}_{safe_filename(plot_image_id)}"
                              f"_anchor{anchor:03d}_{mode}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

    print(f"\n[{scope_label}] Selected image : {plot_image_id} ({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Shown anchor   : {anchor} (best run-level AP50 of this image), "
          f"exemplars = {run['Prompt_ID']}")
    print(f"[{scope_label}] Figure saved   : {png_path}")


def make_qualitative_plots(args, paths: Paths, runs, run_level_df, image_level_df, plt) -> None:
    """CELL 32 for every scope: one figure per archive plus one over ALL images."""
    dataset_root = args.dataset_root.expanduser().resolve()
    try:
        records = discover_images(dataset_root, args.archives)
    except Exception as exc:
        print(f"WARNING: could not discover the images for the plots ({exc}).")
        print("         Every CSV and confusion matrix is still written.")
        return
    image_paths = {r.image_id: r.image_path for r in records}
    if not image_paths:
        print("WARNING: no image found under --dataset-root; the qualitative plots are skipped.")
        return

    archives_present = sorted(str(a) for a in image_level_df["archive"].dropna().unique() if str(a))
    for archive in archives_present:
        plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df, image_paths, plt,
                               scope_label=archive, archive=archive)
    plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df, image_paths, plt,
                           scope_label="ALL", archive=None)


# =============================================================================
#  PHASE 2 - THE WHOLE OFFLINE EVALUATION (notebook CELL 20 ... CELL 32)
# =============================================================================
#  The operating point is fixed and identical to the SAM3 inference threshold:
#  confidence 0.30, NMS IoU 0.40. No sweep is performed.
#  CLUSTER CHANGE: every CSV carries the columns archive and flight (run- and
#  image-level tables: the image's own values; pooled tables: "ALL"), and an
#  extra experiment_summary_per_archive.csv repeats CELL 28 / 29 / 30 per archive.
# =============================================================================

def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2: notebook CELL 20 ... CELL 32, in one process, no GPU."""
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")             # headless node: figures are saved, never shown
        import matplotlib.pyplot as plt
        _HAS_MPL = True
    except Exception as exc:
        plt = None
        _HAS_MPL = False
        print(f"WARNING: matplotlib unavailable ({exc}).")
        print("         Every CSV is still written; the PNGs are not.")

    # notebook CELL 3 names, taken from the command line
    EXPERIMENT_NAME = args.experiment_name
    PROMPT_TYPE = args.prompt_type
    N_EXEMPLARS = args.n_exemplars
    USE_TILING = args.use_tiling
    TILE_SIZE = args.tile_size
    OVERLAP = args.overlap
    ADD_GLOBAL_CONTEXT_PASS = args.add_global_context_pass
    GLOBAL_DOWNSCALE = args.global_downscale
    CONFIDENCE_THRESHOLD = args.confidence_threshold
    MASK_THRESHOLD = args.mask_threshold
    NMS_IOU_THRESHOLD = args.nms_iou_threshold
    EVAL_IOU_THRESHOLD = args.eval_iou_threshold
    PROMPT_IGNORE_IOU = args.prompt_ignore_iou
    RESULTS_ROOT = str(paths.results_root)
    METRICS_DIR = str(paths.metrics)
    CONFUSION_MATRIX_DIR = str(paths.confusion_matrices)

    print("=" * 92)
    print(f" PHASE 2 - OFFLINE EVALUATION | experiment={EXPERIMENT_NAME}")
    print("=" * 92)
    print("Operating configuration (fixed from the start, no sweep):")
    print(f"  Confidence threshold = {CONFIDENCE_THRESHOLD:.2f}")
    print(f"  NMS IoU threshold    = {NMS_IOU_THRESHOLD:.2f}")

    # =========================================================================
    # CELL 20 - LOAD CACHED PRE-NMS DETECTIONS
    # =========================================================================
    print("\n--- CELL 20: load cached pre-NMS detections ---")
    runs = load_runs(paths, EXPERIMENT_NAME)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    # CELL 26 - RUN-LEVEL METRICS  (one run = one image x one anchor/prompt set)
    # =========================================================================
    # Everything is evaluated at the fixed configuration from CELL 3.
    #   AP50 / AP50_95 : all post-NMS predictions >= 0.30, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : only predictions >= CONFIDENCE_THRESHOLD
    #
    # Special case (held_out with no evaluable GT, i.e. every plant of the image was
    # used as a prompt): the metrics are written as NaN and valid_for_macro = False so
    # they are excluded from every mean/std, but TP/FN = 0 and the real FP count are
    # kept, because such a run can still produce false positives that must show up in
    # the pooled counts and in the confusion matrix.
    # =========================================================================
    print("\n--- CELL 26: run-level metrics ---")
    run_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, NMS_IOU_THRESHOLD)
        for mode in EVALUATION_MODES:
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)

            # ---- AP: every prediction >= 0.30 after NMS (ignored ones removed) ----
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt,
                                             EVAL_IOU_THRESHOLD, PROMPT_IGNORE_IOU)
            ap50, ap5095 = compute_ap([p_det], [g_det])

            # ---- operating point -------------------------------------------------
            ev = evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, CONFIDENCE_THRESHOLD,
                                             EVAL_IOU_THRESHOLD, PROMPT_IGNORE_IOU)
            valid = ev["valid_for_macro"]
            nan = float("nan")

            run_rows.append({
                "experiment_name": EXPERIMENT_NAME,
                "image_ID": run["image_ID"],
                "archive": run["archive"],
                "flight": run["flight"],
                "anchor_idx": run["anchor_idx"],
                "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"],
                "evaluation_mode": mode,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "nms_iou_threshold": NMS_IOU_THRESHOLD,
                "n_gt_total": int(len(run["gt_boxes"])),
                "n_prompt_gt": int(len(run["prompt_indices"])) if mode == "held_out" else 0,
                "n_eval_gt": ev["n_eval_gt"],
                "n_predictions": ev["n_pred"],
                "n_ignored_predictions": ev["n_ignored"],
                "AP50": ap50 if valid else nan,
                "AP50_95": ap5095 if valid else nan,
                "precision": ev["precision"] if valid else nan,
                "recall": ev["recall"] if valid else nan,
                "F1": ev["F1"] if valid else nan,
                "IoU1": ev["IoU1"] if valid else nan,
                "IoU2": ev["IoU2"] if valid else nan,
                "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"],
                "valid_for_macro": valid,
            })

    run_level_df = pd.DataFrame(run_rows)
    RUN_LEVEL_CSV = os.path.join(METRICS_DIR, "run_level_metrics.csv")
    run_level_df.to_csv(RUN_LEVEL_CSV, index=False)

    print(f"Run-level metrics: {len(run_level_df)} rows -> {RUN_LEVEL_CSV}")
    for mode in EVALUATION_MODES:
        sub = run_level_df[run_level_df["evaluation_mode"] == mode]
        print(f"  {mode:9s}: {len(sub)} runs, "
              f"{int(sub['valid_for_macro'].sum())} valid for macro averaging, "
              f"F1_mean={sub['F1'].mean():.4f}")

    # =========================================================================
    # CELL 27 - IMAGE-LEVEL METRICS
    # =========================================================================
    # All anchor runs of the same image are averaged into ONE value per image and per
    # evaluation mode. The std here is the spread BETWEEN the different anchor/prompt
    # selections of the SAME image, i.e. "how sensitive is the result to which plant
    # was used as the visual prompt?".
    # NaN rows (held_out runs with no evaluable GT) are ignored by pandas mean/std.
    # std is NaN when an image has only one valid run - that is expected.
    # =========================================================================
    print("\n--- CELL 27: image-level metrics ---")
    image_rows = []
    for (image_id, mode), grp in run_level_df.groupby(["image_ID", "evaluation_mode"]):
        row = {
            "experiment_name": EXPERIMENT_NAME,
            "image_ID": image_id,
            "archive": grp["archive"].iloc[0],
            "flight": grp["flight"].iloc[0],
            "evaluation_mode": mode,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "n_runs_total": int(len(grp)),
            "n_runs_valid_for_macro": int(grp["valid_for_macro"].sum()),
            "TP_sum": int(grp["TP"].sum()),
            "FP_sum": int(grp["FP"].sum()),
            "FN_sum": int(grp["FN"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = grp[col].mean()      # NaNs skipped automatically
            row[f"{col}_std"] = grp[col].std()        # sample std (ddof=1)
        image_rows.append(row)

    image_level_df = pd.DataFrame(image_rows).sort_values(
        ["evaluation_mode", "image_ID"]).reset_index(drop=True)
    IMAGE_LEVEL_CSV = os.path.join(METRICS_DIR, "image_level_metrics.csv")
    image_level_df.to_csv(IMAGE_LEVEL_CSV, index=False)

    print(f"Image-level metrics: {len(image_level_df)} rows -> {IMAGE_LEVEL_CSV}")
    print(image_level_df.groupby("evaluation_mode")[
        ["AP50_mean", "precision_mean", "recall_mean", "F1_mean", "IoU1_mean", "IoU2_mean"]
    ].mean().to_string())

    # =========================================================================
    # CELL 28 - EXPERIMENT-LEVEL SUMMARY
    # =========================================================================
    # Computed from the IMAGE-LEVEL values, not from the raw run rows, so that every
    # UAV image contributes exactly the same weight regardless of how many GT boxes
    # (and therefore how many anchor runs) it contains.
    # The std here is the variation BETWEEN UAV images.
    # =========================================================================
    print("\n--- CELL 28: experiment-level summary ---")

    def summary_row(sub, mode, archive_label):
        """One CELL 28 row for the image-level rows 'sub' of one mode (and one scope)."""
        row = {
            "experiment_name": EXPERIMENT_NAME,
            "archive": archive_label,
            "flight": "ALL",
            "evaluation_mode": mode,
            "prompt_type": PROMPT_TYPE,
            "n_exemplars": N_EXEMPLARS,
            "use_tiling": USE_TILING,
            "tile_size": TILE_SIZE,
            "overlap": OVERLAP,
            "add_global_context_pass": ADD_GLOBAL_CONTEXT_PASS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "eval_iou_threshold": EVAL_IOU_THRESHOLD,
            "n_images": int(sub["image_ID"].nunique()),
            "n_runs": int(sub["n_runs_total"].sum()),
            "n_runs_valid_for_macro": int(sub["n_runs_valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub[f"{col}_mean"].mean()
            row[f"{col}_std"] = sub[f"{col}_mean"].std()   # spread between images
        return row

    summary_rows = []
    for mode in EVALUATION_MODES:
        sub = image_level_df[image_level_df["evaluation_mode"] == mode]
        summary_rows.append(summary_row(sub, mode, "ALL"))

    experiment_summary_df = pd.DataFrame(summary_rows)
    EXPERIMENT_SUMMARY_CSV = os.path.join(METRICS_DIR, "experiment_summary.csv")
    experiment_summary_df.to_csv(EXPERIMENT_SUMMARY_CSV, index=False)

    print(f"Experiment summary -> {EXPERIMENT_SUMMARY_CSV}\n")
    print(experiment_summary_df.to_string(index=False))

    # =========================================================================
    # CELL 29 - POOLED DATASET AP50 / AP50:95
    # =========================================================================
    # CELL 26 computes AP separately for each run.
    # This cell gives all runs to the AP evaluator together and computes one
    # overall AP for the experiment (one result row per evaluation mode).
    #
    # This is NOT the mean of the image-level AP values. All runs are handed to
    # supervision as evaluation EPISODES at once, so every detection of the whole
    # dataset is ranked in ONE precision-recall curve.
    #
    # NOTE for the thesis text: because every GT box of an image becomes an anchor
    # once, the same UAV image appears in several episodes (once per prompt set).
    # The pooled AP is therefore computed over "pooled evaluation episodes", not over
    # unique images - it measures the ranking quality of the whole experiment.
    # =========================================================================
    print("\n--- CELL 29: pooled dataset AP ---")

    # At IoU 0.50, each detection is judged as TP or FP against the GT of its own episode
    # Then, conceptually, the confidence-ranked detection sequence across the experiment is: ... As detections
    # are accumulated in confidence order, overall precision and recall change, producing the experiment-level PR curve
    def pooled_ap(run_list, mode):
        """CELL 29 for one list of runs: (n_images, AP50, AP50_95)."""
        pred_list, gt_list, images_used = [], [], set()
        for run in run_list:
            nms_run = apply_nms_to_run(run, NMS_IOU_THRESHOLD)
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt,
                                             EVAL_IOU_THRESHOLD, PROMPT_IGNORE_IOU)  # Prepare this run's AP episode
            pred_list.append(p_det)
            gt_list.append(g_det)
            images_used.add(run["image_ID"])
        ap50, ap5095 = compute_ap(pred_list, gt_list)  # After ALL runs, calculate AP (after the for run in runs loop)
        del pred_list, gt_list
        gc.collect()
        return len(images_used), ap50, ap5095

    dataset_rows = []
    for mode in EVALUATION_MODES:
        n_images, ap50, ap5095 = pooled_ap(runs, mode)
        dataset_rows.append({
            "experiment_name": EXPERIMENT_NAME,
            "archive": "ALL",
            "flight": "ALL",
            "evaluation_mode": mode,
            "n_images": n_images,
            "n_runs": len(runs),
            "confidence_used_for_AP": CONFIDENCE_THRESHOLD,   # AP uses every prediction >= 0.30
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "dataset_AP50": ap50,
            "dataset_AP50_95": ap5095,
        })

    dataset_ap_df = pd.DataFrame(dataset_rows)
    DATASET_AP_CSV = os.path.join(METRICS_DIR, "dataset_ap_metrics.csv")
    dataset_ap_df.to_csv(DATASET_AP_CSV, index=False)

    print(f"Dataset pooled AP -> {DATASET_AP_CSV}\n")
    print(dataset_ap_df.to_string(index=False))
    print("\nFor comparison, the MEAN of the image-level AP50 values "
          "(a different quantity):")
    print(image_level_df.groupby("evaluation_mode")["AP50_mean"].mean().to_string())

    # =========================================================================
    # CELL 30 - DATASET-LEVEL CONFUSION MATRICES
    # =========================================================================
    # One class (Rumex) plus a background row/column:
    #     Actual Rumex      -> Predicted Rumex      = TP
    #     Actual Rumex      -> Predicted Background = FN  (missed plants)
    #     Actual Background -> Predicted Rumex      = FP  (spurious detections)
    #     Actual Background -> Predicted Background = not defined for detection
    #                                                 (there are no true negatives)
    # Counts are pooled over every run at the fixed configuration. In held_out mode
    # the prompt plants and the detections that were ignored do not appear anywhere.
    # =========================================================================
    print("\n--- CELL 30: confusion matrices ---")
    confusion_summary = []
    for mode in EVALUATION_MODES:
        sub = run_level_df[run_level_df["evaluation_mode"] == mode]
        tp, fp, fn = int(sub["TP"].sum()), int(sub["FP"].sum()), int(sub["FN"].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = safe_f1(precision, recall)

        cm_df = pd.DataFrame(
            [[tp, fn], [fp, np.nan]],
            index=["actual_rumex", "actual_background"],
            columns=["predicted_rumex", "predicted_background"],
        )
        csv_path = os.path.join(CONFUSION_MATRIX_DIR, f"confusion_matrix_{mode}.csv")
        cm_df.to_csv(csv_path)

        if _HAS_MPL:
            png_path = os.path.join(CONFUSION_MATRIX_DIR, f"confusion_matrix_{mode}.png")
            plot_confusion_matrix(
                tp, fp, fn,
                f"{EXPERIMENT_NAME} - {mode}\nconf={CONFIDENCE_THRESHOLD:.2f}, "
                f"NMS IoU={NMS_IOU_THRESHOLD:.2f}, eval IoU={EVAL_IOU_THRESHOLD:.2f}",
                png_path, plt)

        confusion_summary.append({
            "experiment_name": EXPERIMENT_NAME, "archive": "ALL", "flight": "ALL",
            "evaluation_mode": mode,
            "TP": tp, "FP": fp, "FN": fn,
            "precision_micro": precision, "recall_micro": recall, "F1_micro": f1,
            "confidence_threshold": CONFIDENCE_THRESHOLD, "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "eval_iou_threshold": EVAL_IOU_THRESHOLD,
        })
        print(f"{mode:9s}: TP={tp}  FP={fp}  FN={fn}  "
              f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

    confusion_summary_df = pd.DataFrame(confusion_summary)
    confusion_summary_df.to_csv(
        os.path.join(CONFUSION_MATRIX_DIR, "confusion_matrix_summary.csv"), index=False)
    print("\nConfusion matrices saved to:", CONFUSION_MATRIX_DIR)

    # =========================================================================
    # PER-ARCHIVE SUMMARY  (cluster addition: CELL 28 + CELL 29 + CELL 30 per archive)
    # =========================================================================
    # The notebook ran on AGS_Multi_Rumex only. Here both archives are pooled, so
    # the same experiment-level numbers are also reported separately per archive:
    # the CELL 28 means / stds, the CELL 29 pooled AP over that archive's runs
    # and the CELL 30 pooled confusion counts.
    # =========================================================================
    print("\n--- per-archive summary (CELL 28 + 29 + 30 per archive) ---")
    per_archive_rows = []
    archives_present = sorted(str(a) for a in image_level_df["archive"].dropna().unique())
    for archive in archives_present:
        archive_runs = [r for r in runs if r["archive"] == archive]
        for mode in EVALUATION_MODES:
            sub = image_level_df[(image_level_df["evaluation_mode"] == mode) &
                                 (image_level_df["archive"] == archive)]
            row = summary_row(sub, mode, archive)
            n_images, ap50, ap5095 = pooled_ap(archive_runs, mode)
            row["dataset_AP50"] = ap50
            row["dataset_AP50_95"] = ap5095
            run_sub = run_level_df[(run_level_df["evaluation_mode"] == mode) &
                                   (run_level_df["archive"] == archive)]
            tp, fp, fn = int(run_sub["TP"].sum()), int(run_sub["FP"].sum()), int(run_sub["FN"].sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            row.update({"TP": tp, "FP": fp, "FN": fn, "precision_micro": precision,
                        "recall_micro": recall, "F1_micro": safe_f1(precision, recall)})
            per_archive_rows.append(row)
    per_archive_df = pd.DataFrame(per_archive_rows)
    PER_ARCHIVE_CSV = os.path.join(METRICS_DIR, "experiment_summary_per_archive.csv")
    per_archive_df.to_csv(PER_ARCHIVE_CSV, index=False)
    print(f"Per-archive summary -> {PER_ARCHIVE_CSV}\n")
    if not per_archive_df.empty:
        print(per_archive_df[["archive", "evaluation_mode", "n_images", "n_runs", "AP50_mean",
                              "F1_mean", "dataset_AP50", "TP", "FP", "FN"]].to_string(index=False))

    # =========================================================================
    # CELL 32 - QUALITATIVE PLOT (one per archive + one over ALL images)
    # =========================================================================
    if args.no_plots:
        print("\n--- CELL 32: qualitative figures skipped (--no-plots) ---")
    elif not _HAS_MPL:
        print("\n--- CELL 32: qualitative figures skipped (matplotlib unavailable) ---")
    else:
        print("\n--- CELL 32: qualitative GT-vs-prediction figures ---")
        make_qualitative_plots(args, paths, runs, run_level_df, image_level_df, plt)

    # =========================================================================
    # CELL 31 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {EXPERIMENT_NAME} - FINAL SUMMARY")
    print("=" * 78)
    print(f"Prompts per run          : {N_EXEMPLARS} ({PROMPT_TYPE})")
    print(f"Tiling                   : {USE_TILING}  (tile={TILE_SIZE}px, overlap={OVERLAP}px)")
    print(f"Confidence threshold     : {CONFIDENCE_THRESHOLD:.2f} (fixed; SAM3 run once per image x anchor x tile)")
    print(f"Mask threshold           : {MASK_THRESHOLD:.2f}")
    print(f"Global context pass      : {ADD_GLOBAL_CONTEXT_PASS} (downscale={GLOBAL_DOWNSCALE})")
    print(f"NMS IoU threshold        : {NMS_IOU_THRESHOLD:.2f} (fixed)")
    print(f"Evaluation IoU           : {EVAL_IOU_THRESHOLD:.2f}")
    print(f"Runs / images            : {len(runs)} runs over "
          f"{run_level_df['image_ID'].nunique()} images")
    print(f"Archives                 : {', '.join(archives_present)}")
    print("-" * 78)
    print("EXPERIMENT-LEVEL RESULTS (mean over images, std between images)")
    show = ["evaluation_mode", "AP50_mean", "AP50_std", "AP50_95_mean", "precision_mean",
            "recall_mean", "F1_mean", "F1_std", "IoU1_mean", "IoU2_mean"]
    print(experiment_summary_df[show].to_string(index=False))
    print("-" * 78)
    print("POOLED DATASET AP")
    print(dataset_ap_df[["evaluation_mode", "dataset_AP50", "dataset_AP50_95"]].to_string(index=False))
    print("-" * 78)
    print("POOLED CONFUSION COUNTS")
    print(confusion_summary_df[["evaluation_mode", "TP", "FP", "FN",
                                "precision_micro", "recall_micro", "F1_micro"]].to_string(index=False))
    if not per_archive_df.empty:
        print("-" * 78)
        print("PER ARCHIVE")
        print(per_archive_df[["archive", "evaluation_mode", "n_images", "AP50_mean",
                              "F1_mean", "dataset_AP50"]].to_string(index=False))
    print("=" * 78)

    print("\nFiles written under", RESULTS_ROOT)
    for root, dirs, files in os.walk(RESULTS_ROOT):
        dirs.sort()
        depth = root.replace(RESULTS_ROOT, "").count(os.sep)
        print("  " * depth + os.path.basename(root) + "/")
        if os.path.basename(root) == "raw_detections":
            npz_files = [f for f in files if f.endswith(".npz")]
            for f in sorted(files):
                if not f.endswith(".npz"):
                    print("  " * (depth + 1) + f)
            print("  " * (depth + 1) + f"[{len(npz_files)} run NPZ files]")
        else:
            for f in sorted(files):
                print("  " * (depth + 1) + f)


# =============================================================================
#  MAIN
# =============================================================================

def _check_supervision() -> bool:
    try:
        import supervision                                    # noqa: F401
        from supervision.metrics import MeanAveragePrecision  # noqa: F401
        return True
    except Exception:
        return False


def _check_pandas() -> bool:
    """pandas is REQUIRED by PHASE 2 (every metric table is a DataFrame)."""
    try:
        import pandas                                         # noqa: F401
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    paths = build_paths(output_dir)
    exp = args.experiment_name

    # ---------------- PHASE 2 only -------------------------------------------
    if args.evaluate_only:
        if not _check_supervision():
            print("\nERROR: " + SUPERVISION_HINT)
            sys.exit(2)
        if not _check_pandas():
            print("\nERROR: pandas is required for the evaluation. See 'Problem B' in "
                  "k_exemplars_with_tiling_extra_path_HOW_TO_RUN.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" K_EXEMPLARS_WITH_TILING_EXTRA_PATH SAM3 PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f"Configuration loaded for experiment '{exp}'")
    print(f"  dataset root   : {dataset_root}")
    print(f"  results folder : {paths.results_root}")
    print(f"  archives       : {', '.join(args.archives)}")
    print(f"  prompts        : {args.n_exemplars} ({args.prompt_type})")
    print(f"  tiling         : {args.use_tiling} (tile={args.tile_size}, overlap={args.overlap})")
    print(f"  confidence     : {args.confidence_threshold} (fixed, single inference pass per tile)")
    print(f"  mask threshold : {args.mask_threshold}")
    print(f"  NMS IoU        : {args.nms_iou_threshold} (fixed)")
    print(f"  batch / dtype  : {args.batch_size} / {args.dtype}")
    print(f"  masks stored   : {KEEP_MASKS}")
    print(f"  global pass    : {args.add_global_context_pass} (downscale={args.global_downscale})")
    print(f"  supervision    : {'available' if has_supervision else 'MISSING'}")
    print("=" * 92)

    # ---------------- dataset (CELL 7) ----------------------------------------
    print("Discovering images ...")
    records = discover_images(dataset_root, args.archives)
    if not records:
        print("No images with labels found -- check --dataset-root. Aborting.")
        sys.exit(1)
    if args.limit_images > 0:
        records = records[: args.limit_images]

    # Round-robin sharding over the (deterministically sorted) image list.
    my_records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_index]
    print(f"Total images: {len(records)} | this shard: {len(my_records)}")

    # ---------------- config snapshot (written once, by shard 0) --------------
    if args.shard_index == 0:
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        config["archives_class_ids"] = {a: ARCHIVES[a] for a in args.archives}
        config["n_images_total"] = len(records)
        config["supervision_available"] = has_supervision
        config["evaluation_modes"] = EVALUATION_MODES
        config["keep_masks"] = KEEP_MASKS
        safe_name = exp.replace("/", "-")
        (paths.results_root / f"run_config_{safe_name}.json").write_text(json.dumps(config, indent=2))

    # ---------------- dry run -------------------------------------------------
    if args.dry_run:
        dry_run(args, records, my_records)
        return

    if not has_supervision:
        print("\nERROR: " + SUPERVISION_HINT)
        sys.exit(2)

    # ---------------- PHASE 1 -------------------------------------------------
    run_inference(args, paths, my_records)

    # ---------------- PHASE 2 -------------------------------------------------
    if not args.no_evaluate:
        run_evaluation(args, paths)


if __name__ == "__main__":
    main()