#!/usr/bin/env python3
# =============================================================================
#  k_diverse_exemplars_tiling_infer_yoloe.py
# =============================================================================
#  YOLOE version of the SAM3 cluster script k_diverse_exemplars_tiling_infer_sam3.py
#  (Colab notebook k_diverse_exemplars_tiling.ipynb).
#
#  This experiment is the DIVERSITY pipeline of k_diverse_exemplars_no_tiling,
#  but YOLOE now sees OVERLAPPING TILES at NATIVE resolution instead of the
#  downscaled whole image: K_EXEMPLARS = 3 and USE_TILING = True
#  (TILE_SIZE = 1000, OVERLAP = 150).
#
#  ONCE PER IMAGE (full resolution, before any tiling, notebook CELL 14) -
#  IDENTICAL TO THE SAM3 VERSION:
#      crop every GT Rumex   GT bbox -> +10% context -> square padding -> 224x224
#      -> DINOv2 CLS embedding, L2-normalised
#      -> pairwise cosine distance  d(i,j) = 1 - cos(z_i, z_j)
#      -> diversity selection  S* = argmax_{|S|=3} min_{i,j in S} d(i,j)
#      -> 3 GT plants = the exemplar prompts of this image (D1, D2, D3)
#
#  THE VISUAL PROMPT - WHAT IS DIFFERENT FROM SAM3, AND WHY
#      THERE IS NO EXEMPLAR STRIP. YOLOE takes its exemplars from SEPARATE
#      reference images: one TILE_SIZE x TILE_SIZE window of the full image is
#      cut around each of D1 / D2 / D3 (exemplars inside the same window share
#      it), each window is encoded into a visual prompt embedding (VPE), the VPEs
#      are averaged + L2-normalised and installed with set_classes() ONCE PER
#      IMAGE ("per_exemplar_vpe", as in k_size_exemplars_tiling_yoloe). Every
#      tile is then a plain model.predict(tile) call - the tile is left untouched.
#      No strip -> no strip background, no strip/tile region rule, no random
#      ingredient at all. Only the plausibility filter stays.
#      YOLOE runs its own NMS inside the predictor. It is set very permissive
#      (PREDICT_NMS_IOU = 0.90) so that the OFFLINE NMS of PHASE 2 decides.
#
#  There is NO MAX_DIM here: the whole-image downscale of the no-tiling
#  experiment is exactly what tiling replaces, which is the point of the
#  comparison between the two notebooks.
#
#  The selection is fully DETERMINISTIC - no anchors, no random exemplar
#  sampling, no seeds - so there is exactly ONE run per image and a single
#  per-image metric table (notebook CELL 24) replaces the run-level /
#  image-level pair of the anchor experiments.
#
#  The notebook is a TWO-PHASE pipeline and this file keeps that separation:
#
#    PHASE 1 - INFERENCE   (notebook CELL 17, GPU)
#        for every image:
#            crop + embed every GT box with DINOv2         (CELL 14)
#            select the 3 most DIVERSE GT boxes            (CELL 14)
#            build the overlapping tiles once              (CELL 10)
#            build the reference windows around D1/D2/D3   (CELL 11 / CELL 12)
#            encode + install the visual prompt (VPE)      (CELL 15)
#            run YOLOE over all tiles in batches at CONFIDENCE_THRESHOLD=0.30
#                                                          (CELL 15)
#            plausibility filter                           (CELL 13)
#            save the PRE-NMS detections, the tile provenance, the reference
#            windows and the DINOv2 distance matrix to NPZ (CELL 16)
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 18 ... CELL 30, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40 (with
#        cross-tile / same-tile provenance), evaluate in both modes
#        (all_gt / held_out) at the operating point CONFIDENCE_THRESHOLD = 0.30,
#        and write the per-image / experiment / pooled-AP / size-group CSVs, the
#        confusion matrices (CSV + PNG) and the qualitative figures.
#        Runs in ONE process, after every shard has finished. It never touches
#        YOLOE nor DINOv2, so it can be repeated as often as you like from the
#        cached NPZs.
#
#
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
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# UAV orthomosaics are very large; disable PIL's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None


# =============================================================================
#  STATIC CONFIGURATION  (dataset layout - from the cluster scripts)
# =============================================================================
# The notebook pointed at ONE folder (AGS_Multi_Rumex) with RUMEX_CLASS_ID = 0.
# On the cluster the two archives live side by side and use DIFFERENT class ids
# inside their YOLO files, so the class id is a property of the archive, not a
# global constant.
ARCHIVES: dict[str, int] = {
    "AGS_Multi_Rumex": 0,
    "AgsSpringRumex": 2,
}

IGNORED_ARCHIVES = ("AGS_Multiple_Fields", "AGS_Multiple_Fields_Embeddings")

VALID_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

NON_LABEL_FILES = {"darknet.labels", "classes.txt", "obj.names"}

# ---- notebook CELL 4: manifest of finished inference runs (resume support) ---
# area_D1/D2/D3 are the areas of the three SELECTED plants (a RESULT of the
# diversity selection, not its criterion) and are logged so that the size profile
# of diversity prompts stays comparable with the S/M/L experiment.
# archive / flight / source_class_id are cluster additions.
MANIFEST_COLUMNS = [
    "experiment_name", "image_ID", "Prompt_ID", "Prompt_Type",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt",
    "area_D1_px2", "area_D2_px2", "area_D3_px2", "median_gt_area_px2",
    "min_pairwise_distance", "mean_pairwise_distance",
    "gt_mean_distance", "search_mode",
    "n_tiles", "n_reference_windows", "n_detections_pre_nms",
    "image_width", "image_height",
    "npz_file", "embedding_seconds", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
# held_out : the 3 GT instances used as prompts (D1, D2, D3) are IGNORED, and so
#            are the predictions that fall on them. Answers "how well does YOLOE
#            find the REMAINING Rumex plants after being shown three visually
#            DIFFERENT examples?".

# ---- notebook CELL 24 ------------------------------------------------------
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2"]

# ---- notebook CELL 20 ------------------------------------------------------
STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2

# ---- notebook CELL 3: DINOv2 crop geometry ---------------------------------
CROP_MODE = "context_pad_square"
CROP_PAD_FILL = (124, 116, 104)   # padding colour of the square canvas = ImageNet
                                  # mean (neutral grey after DINOv2 normalisation).
                                  # "edge" replicates the border pixels instead.

# ---- notebook CELL 3: diversity selection ----------------------------------
DIVERSITY_METRIC = "cosine"       # distance = 1 - cosine similarity
DIVERSITY_OBJECTIVE = "max_min"   # maximise the SMALLEST pairwise distance in S

# ---- notebook CELL 3: prompt type ------------------------------------------
PROMPT_TYPE = "diversity_dinov2_tiling"

# ---- the YOLOE prompt encoding (as in k_size_exemplars_tiling_yoloe) ---------
# One reference window per exemplar (shared when they fall in the same window),
# one VPE per window, VPEs averaged + L2-normalised.
PROMPT_MODE = "per_exemplar_vpe"

# ---- notebook CELL 23: plotting colours ------------------------------------
COLOR_GT, COLOR_PRED = "yellow", "red"
ROLE_COLORS = {"D1": "cyan", "D2": "lime", "D3": "orange"}
ROLE_NAMES = {"D1": "diverse #1", "D2": "diverse #2", "D3": "diverse #3"}

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './k_diverse_exemplars_tiling_run_yoloe.sh download' "
    "on a LOGIN node first, which installs it into $PYEXTRA."
)


# =============================================================================
#  CLI  -  every notebook CELL 3 parameter, with the notebook value as default
# =============================================================================

def default_dataset_root() -> Path:
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "overney" / "dataset"
    return Path(__file__).resolve().parents[2] / ".." / "02_data" / "dataset"


def default_weights() -> str:
    """
    YOLOE_WEIGHTS. On the cluster the checkpoint is pre-fetched by the download
    mode into $SCRATCH/yoloe_weights (compute nodes are offline). Elsewhere the
    bare name lets Ultralytics download it on first use.
    """
    scratch = os.getenv("SCRATCH")
    if scratch:
        return str(Path(scratch) / "yoloe_weights" / "yoloe-11l-seg.pt")
    return "yoloe-11l-seg.pt"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="k_diverse_exemplars_tiling_yoloe - YOLOE prompted with K GT boxes chosen "
                    "by max-min DINOv2 embedding diversity (one reference window per "
                    "exemplar, averaged visual prompt embedding), tiling ON (inference + "
                    "offline evaluation). YOLOE version of the SAM3 notebook "
                    "k_diverse_exemplars_tiling.ipynb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------- paths -------------------------------------
    g = p.add_argument_group("paths")
    g.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                   help="Folder holding the archive folders (AGS_Multi_Rumex, AgsSpringRumex). "
                        "PHASE 2 uses it as well, for the qualitative figures (CELL 30).")
    g.add_argument("--output-dir", type=Path, required=True,
                   help="RESULTS_ROOT: raw_detections/, metrics/, confusion_matrices/, plots/.")
    g.add_argument("--archives", nargs="*", default=list(ARCHIVES.keys()),
                   help="Subset of archives to run on. Default: both.")

    # ------------------------ experiment identity ---------------------------
    g = p.add_argument_group("experiment identity (CELL 3)")
    g.add_argument("--experiment-name", default="k_diverse_exemplars_tiling_yoloe",
                   help="EXPERIMENT_NAME. Written into every CSV row and every NPZ.")
    g.add_argument("--k-exemplars", type=int, default=3,
                   help="K_EXEMPLARS = N_EXEMPLARS. The number of GT boxes encoded into the "
                        "YOLOE visual prompt (D1, D2, D3). An image with fewer GT boxes uses "
                        "all of them, exactly like the notebook.")

    tiling = g.add_mutually_exclusive_group()
    tiling.add_argument("--tiling", dest="use_tiling", action="store_true", default=True,
                        help="USE_TILING = True (default): overlapping tiles at native "
                             "resolution.")
    tiling.add_argument("--no-tiling", dest="use_tiling", action="store_false",
                        help="USE_TILING = False: the whole image is one single tile (this is "
                             "NOT the no-tiling experiment, which resizes to MAX_DIM and "
                             "prompts in place).")

    # ------------------------------ tiling ----------------------------------
    g = p.add_argument_group("tiling (CELL 3 / CELL 10)")
    g.add_argument("--tile-size", type=int, default=1000, help="TILE_SIZE")
    g.add_argument("--overlap", type=int, default=150, help="OVERLAP")
    g.add_argument("--no-cache-tiles", dest="cache_tiles_in_memory", action="store_false",
                   default=True,
                   help="CACHE_TILES_IN_MEMORY = False: crop tiles on demand (less RAM).")

    # ----------------- DINOv2 embeddings of the GT crops --------------------
    g = p.add_argument_group("DINOv2 exemplar embeddings (CELL 3 / CELL 14)")
    g.add_argument("--dinov2-model-id", default="facebook/dinov2-large",
                   help="DINOV2_MODEL_ID. ViT-L/14, 1024-d CLS token. NOT gated. "
                        "HF repo id OR a local snapshot directory.")
    g.add_argument("--embedding-feature", default="cls", choices=["cls"],
                   help="EMBEDDING_FEATURE: CLS token of the last hidden layer, then "
                        "L2-normalised.")
    g.add_argument("--crop-context", type=float, default=0.10,
                   help="CROP_CONTEXT: +10%% of the box WIDTH left and right and of the box "
                        "HEIGHT top and bottom -> the context box is 1.2 x 1.2 times the GT "
                        "box. 0.0 for a tight box.")
    g.add_argument("--dinov2-input-size", type=int, default=224,
                   help="DINOV2_INPUT_SIZE: the square canvas is resized DIRECTLY to this, so "
                        "the context margin is not cropped away again.")
    g.add_argument("--dinov2-center-crop", action="store_true", default=False,
                   help="DINOV2_CENTER_CROP. The notebook switches the processor's default "
                        "'shortest edge 256 + centre-crop 224' OFF; leave this flag unset to "
                        "reproduce it.")
    g.add_argument("--dinov2-batch-size", type=int, default=16,
                   help="DINOV2_BATCH_SIZE: GT crops per DINOv2 forward pass.")
    g.add_argument("--dinov2-dtype", choices=["float32", "float16", "bfloat16"],
                   default="float32",
                   help="DINOV2_DTYPE. The notebook keeps the embeddings in fp32 because the "
                        "cosine distances are compared directly, not trained (YOLOE may still "
                        "run in half precision).")

    # ------------------ diversity-based exemplar selection ------------------
    g = p.add_argument_group("diversity selection (CELL 3 / CELL 14)")
    g.add_argument("--diversity-exact-max-gt", type=int, default=300,
                   help="DIVERSITY_EXACT_MAX_GT: up to this many GT boxes the max-min search "
                        "is EXHAUSTIVE (C(300,3) = 4.5M triples, vectorised, < 1 s); above it "
                        "a greedy farthest-point fallback is used. The dataset maxes out "
                        "around 100 GT boxes per image, so in practice it is always exact.")

    # --------------------------- YOLOE inference ----------------------------
    g = p.add_argument_group("yoloe inference (CELL 3 / CELL 5 / CELL 15)")
    g.add_argument("--weights", default=default_weights(),
                   help="YOLOE_WEIGHTS: path to yoloe-11l-seg.pt (a visual-prompt capable "
                        "checkpoint; the *-seg-pf.pt variants CANNOT take visual prompts).")
    g.add_argument("--prompt-class-name", default="rumex",
                   help="PROMPT_CLASS_NAME (cosmetic).")
    g.add_argument("--imgsz", type=int, default=1024,
                   help="IMGSZ, multiple of 32. 1024 >= TILE_SIZE keeps a tile at "
                        "essentially native resolution. Do NOT drop to 640.")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="CONFIDENCE_THRESHOLD. YOLOE is executed EXACTLY ONCE per tile at this "
                        "score. In this experiment it is ALSO the evaluation operating point "
                        "(see --operating-confidence).")
    g.add_argument("--predict-nms-iou", type=float, default=0.90,
                   help="YOLOE_PREDICT_NMS_IOU: the per-tile NMS inside the Ultralytics "
                        "predictor, permissive on purpose (the offline NMS decides).")
    g.add_argument("--max-det", type=int, default=300, help="MAX_DET per tile.")
    g.add_argument("--no-retina-masks", dest="retina_masks", action="store_false",
                   default=True,
                   help="RETINA_MASKS = False (masks at network resolution; they are "
                        "resized to the tile before the fill ratio is computed).")
    g.add_argument("--mask-binarise", type=float, default=0.50,
                   help="MASK_BINARISE. The masks are used ONLY to compute the mask-fill "
                        "ratio of the plausibility filter and are then dropped immediately "
                        "(KEEP_MASKS = False).")
    g.add_argument("--batch-size", type=int, default=4,
                   help="BATCH_SIZE: tiles per predict() call.")
    fp = g.add_mutually_exclusive_group()
    fp.add_argument("--fp16", dest="use_fp16", action="store_true", default=True,
                    help="USE_FP16 = True (default): half precision YOLOE inference on the GPU.")
    fp.add_argument("--no-fp16", dest="use_fp16", action="store_false",
                    help="USE_FP16 = False: use this if you hit a dtype error while the "
                         "visual prompt embedding is applied.")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")

    # ------------------------ reference windows -----------------------------
    g = p.add_argument_group("visual prompt (CELL 3 / CELL 11 / CELL 12)")
    g.add_argument("--reference-window-size", type=int, default=None,
                   help="REFERENCE_WINDOW_SIZE. Default: equal to --tile-size, so the "
                        "exemplar is encoded at the same pixel scale as the tiles.")

    # --------------------------- filters ------------------------------------
    g = p.add_argument_group("filters (CELL 3 / CELL 13)")
    g.add_argument("--min-fill-ratio", type=float, default=0.15, help="MIN_FILL_RATIO")
    g.add_argument("--max-area-fraction", type=float, default=0.80, help="MAX_AREA_FRACTION")
    g.add_argument("--edge-margin", type=int, default=5, help="EDGE_MARGIN")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 19 / CELL 22)")
    g.add_argument("--nms-iou-threshold", type=float, default=0.40,
                   help="NMS_IOU_THRESHOLD, applied OFFLINE in PHASE 2 (fixed, no sweep).")
    g.add_argument("--operating-confidence", type=float, default=0.30,
                   help="The operating point of precision / recall / F1 / IoU1 / IoU2. The "
                        "notebook keeps it EQUAL to --threshold (0.30), so the inference "
                        "threshold and the operating point are the same single value and no "
                        "confidence x NMS sweep is performed.")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")
    g.add_argument("--prompt-ignore-iou", type=float, default=0.50,
                   help="PROMPT_IGNORE_IOU: held_out mode ignore rule. An unmatched "
                        "prediction whose best IoU with a PROMPT GT box is >= this value is "
                        "IGNORED (neither TP nor FP).")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot (CELL 3 / CELL 30)")
    g.add_argument("--plot-evaluation-mode", default="all_gt", choices=EVALUATION_MODES,
                   help="PLOT_EVALUATION_MODE: the per-image AP50 of this mode selects the "
                        "image that gets plotted.")
    g.add_argument("--plot-min-gt-boxes", type=int, default=7,
                   help="PLOT_MIN_GT_BOXES: the plotted image must have at least this many "
                        "GT boxes (with the notebook's fallback if none has).")
    g.add_argument("--plot-max-display-dim", type=int, default=2048,
                   help="PLOT_MAX_DISPLAY_DIM: display-only downscale of the figure.")
    g.add_argument("--no-plot-scores", dest="plot_show_scores", action="store_false",
                   default=True,
                   help="PLOT_SHOW_SCORES = False: do not write the confidence next to each "
                        "predicted box.")
    g.add_argument("--no-plots", action="store_true",
                   help="Skip CELL 30 entirely (it is the only PHASE 2 step that reopens the "
                        "original images).")

    # ---------------------------- runtime -----------------------------------
    g = p.add_argument_group("runtime")
    g.add_argument("--num-shards", type=int, default=1,
                   help="Split the image list across this many concurrent processes.")
    g.add_argument("--shard-index", type=int, default=0, help="0-based shard of this process.")
    g.add_argument("--limit-images", type=int, default=0,
                   help="Debug: process at most this many images (0 = no limit).")
    g.add_argument("--no-resume", action="store_true",
                   help="Ignore the existing manifests and recompute every image.")

    # ----------------------------- modes ------------------------------------
    g = p.add_argument_group("modes")
    g.add_argument("--dry-run", action="store_true",
                   help="Discover the dataset, print the run plan and exit. No model, no GPU.")
    g.add_argument("--evaluate-only", action="store_true",
                   help="PHASE 2 only: rebuild every metric from the cached NPZ files.")
    g.add_argument("--no-evaluate", action="store_true",
                   help="PHASE 1 only: do not run the evaluation after inference.")

    args = p.parse_args(argv)

    if args.k_exemplars < 1:
        p.error("--k-exemplars must be >= 1")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.use_tiling and args.overlap >= args.tile_size:
        p.error("--overlap must be smaller than --tile-size")
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")
    if args.crop_context < 0:
        p.error("--crop-context must be >= 0")
    if args.imgsz % 32 != 0:
        p.error("--imgsz must be a multiple of 32")

    # REFERENCE_WINDOW_SIZE = TILE_SIZE (as in k_size_exemplars_tiling_yoloe)
    if args.reference_window_size is None:
        args.reference_window_size = args.tile_size
    args.model_name = Path(str(args.weights)).name

    # N_EXEMPLARS (CELL 3) - the notebook keeps the two names in sync
    args.n_exemplars = args.k_exemplars
    # PROMPT_TYPE (CELL 3)
    args.prompt_type = PROMPT_TYPE
    # CELL 3 constants that are not command-line options
    args.crop_mode = CROP_MODE
    args.crop_pad_fill = CROP_PAD_FILL
    args.diversity_metric = DIVERSITY_METRIC
    args.diversity_objective = DIVERSITY_OBJECTIVE
    args.keep_masks = False
    return args


# =============================================================================
#  CELL 4 - OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections + tile provenance + the DINOv2
#                          GxG distance matrix (NPZ, ONE file per image)
#                          + one runs_manifest_shard<i>.csv per shard
#     metrics/             per-image table, experiment summary, pooled dataset AP,
#                          size-group recall
#     confusion_matrices/  CSV + PNG for all_gt and held_out
#     plots/               qualitative GT-vs-prediction figures (CELL 30)
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
    simply reads all of them back (see load_done_images).
    """
    return paths.raw_detections / f"runs_manifest_{experiment_name}_shard{shard_index}.csv"


# =============================================================================
#  CELL 7 - DATASET AND YOLO ANNOTATION HELPERS
# =============================================================================
#  The notebook walked ONE images folder and used a single RUMEX_CLASS_ID, with a
#  mirrored-or-flat label lookup. On the cluster both archives are pooled into one
#  dataset, each with its own class id and a FLAT annotations_yolo folder, so
#  discover_images() is the archive-aware version from the cluster scripts.
#  load_yolo_boxes / safe_filename are unchanged notebook code.
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


def _index_flat_labels(annotations_root: Path) -> dict[str, Path]:
    """Scan all annotation files once -> {image stem: label file}."""
    index: dict[str, Path] = {}
    duplicates: list[str] = []
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
    Scan the chosen archives, keep every image that has a matching annotation file
    and return them in a deterministic global order (so the round-robin shard
    assignment is identical in every process and after every restart).
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
        archive_root = dataset_root / archive
        images_root = archive_root / "images"
        annotations_root = archive_root / "annotations_yolo"

        if not images_root.is_dir():
            print(f"  WARNING: {images_root} does not exist -- archive '{archive}' skipped.")
            continue

        label_index = _index_flat_labels(annotations_root)
        n_before = len(records)

        for image_path in sorted(images_root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXT:
                continue
            rel = image_path.relative_to(images_root)
            flight = rel.parts[0] if len(rel.parts) > 1 else ""
            stem = image_path.stem
            label_path = label_index.get(stem)
            image_id = f"{archive}/{flight}/{stem}" if flight else f"{archive}/{stem}"
            if label_path is None:
                missing.append(image_id)
                continue
            records.append(ImageRecord(archive, flight, image_id, image_path, label_path, class_id))

        n_flights = len({r.flight for r in records[n_before:]})
        print(f"  {archive:<22} class_id={class_id}  images={len(records) - n_before:<6} "
              f"flights={n_flights:<4} labels_indexed={len(label_index)}")

    if missing:
        print(f"  WARNING: {len(missing)} image(s) have no matching label file and were "
              f"skipped. First few: {missing[:5]}")

    records.sort(key=lambda r: r.image_id)
    return records


def load_yolo_boxes(label_path: Path, img_width: int, img_height: int,
                    class_id: int) -> np.ndarray:
    """
    Read a YOLO .txt annotation file and convert it to pixel corner boxes.

    Input : label_path            - YOLO txt file
            img_width, img_height - size of the ORIGINAL image in pixels
            class_id              - keep only this class (archive dependent)
    Output: np.ndarray (N, 4) float32, boxes as [x1, y1, x2, y2] in pixels.

    YOLO stores normalised (class, x_center, y_center, width, height).
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue                      # skip empty lines
            try:
                if int(parts[0]) != class_id:
                    continue                  # keep only the requested class
                xc, yc, bw, bh = map(float, parts[1:5])
            except (ValueError, IndexError):
                continue                      # ignore a malformed line
            xc, yc = xc * img_width, yc * img_height
            bw, bh = bw * img_width, bh * img_height
            boxes.append([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def safe_filename(image_id: str) -> str:
    """'archive/flight/name' -> 'archive__flight__name' for use inside a file name."""
    return image_id.replace("/", "__").replace(os.sep, "__")


# =============================================================================
#  CELL 8 - IoU
# =============================================================================

def compute_iou_matrix(boxes1, boxes2) -> np.ndarray:
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


def intersection_area(box, region) -> float:
    """
    Plain intersection AREA (not IoU) between one box and one region: how many
    pixels of a predicted box lie inside a given tile / region.
    """
    x1 = max(box[0], region[0]); y1 = max(box[1], region[1])   # top and left of inter
    x2 = min(box[2], region[2]); y2 = min(box[3], region[3])   # bottom and right of inter
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)               # inter width x inter height


# =============================================================================
#  CELL 9 - CORRECT ONE-TO-ONE MATCHING
# =============================================================================
# THE RULE
#   sort predictions by confidence, high -> low
#   for each prediction:
#       look ONLY at GT boxes that are still unmatched
#       take the unmatched GT with the highest IoU
#       if that IoU >= evaluation IoU threshold -> match, else leave unmatched
#   one GT can be matched by at most one prediction, and vice versa.
#
# This single function is used for TP / FP / FN / precision / recall / F1 /
# IoU1 / IoU2 / confusion matrices / size-group recall, so the whole thesis uses
# one definition.
# =============================================================================

def match_one_to_one(pred_boxes, pred_scores, gt_boxes, iou_threshold: float) -> dict:
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
    pred_iou = np.zeros(n_pred, dtype=np.float32)   # IoU of the accepted match per prediction

    if n_pred == 0 or n_gt == 0:
        return {"pred_match_gt": pred_match_gt, "gt_match_pred": gt_match_pred,
                "pred_iou": pred_iou, "matched_ious": []}

    iou = compute_iou_matrix(pred_boxes, gt_boxes)
    gt_free = np.ones(n_gt, dtype=bool)             # which GTs are still available

    # 'stable' keeps the original order for equal scores -> fully deterministic.
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32), kind="stable")

    matched_ious = []
    for p in order:
        if not gt_free.any():
            break                                   # every GT already has a prediction
        # Consider ONLY currently unmatched GT boxes.
        candidate_ious = np.where(gt_free, iou[p], -1.0)
        g = int(np.argmax(candidate_ious))          # best FREE GT
        if candidate_ious[g] >= iou_threshold:
            gt_free[g] = False
            pred_match_gt[p] = g
            gt_match_pred[g] = p
            pred_iou[p] = candidate_ious[g]
            matched_ious.append(float(candidate_ious[g]))

    return {"pred_match_gt": pred_match_gt, "gt_match_pred": gt_match_pred,
            "pred_iou": pred_iou, "matched_ious": matched_ious}


def safe_f1(precision: float, recall: float) -> float:
    """F1 = 2PR/(P+R) with a safe zero denominator (returns 0.0)."""
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0


# =============================================================================
#  CELL 10 - TILES: GENERATED ONCE PER IMAGE
# =============================================================================
# Open the image once -> build the tile list once -> reuse it for the single run
# of this image. Tiles are kept as CPU/PIL images (never on the GPU).
# =============================================================================

def tile_bboxes(img_w: int, img_h: int, tile_size: int, overlap: int
                ) -> List[Tuple[int, int, int, int]]:
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


def build_tile_cache(image: Image.Image, use_tiling: bool, tile_size: int,
                     overlap: int, cache_in_memory: bool) -> List[dict]:
    """
    Build the tile list for ONE already-open image.

    Input : image - PIL image of the full UAV photo
    Output: list of dicts, one per tile:
            {'tile_id', 'x1', 'y1', 'x2', 'y2', 'image' (PIL or None)}
            'image' is None when cache_in_memory=False (cropped on demand).

    If use_tiling is False the whole image is returned as a single "tile", which
    keeps the rest of the pipeline identical.
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


def get_tile_image(tile: dict, full_image: Image.Image) -> Image.Image:
    """Return the tile's PIL image, cropping on demand if it was not cached."""
    if tile["image"] is not None:
        return tile["image"]
    return full_image.crop((tile["x1"], tile["y1"], tile["x2"], tile["y2"]))


# =============================================================================
#  CELL 11 + CELL 12 - REFERENCE WINDOWS FOR THE VISUAL PROMPT (replaces the strip)
# =============================================================================
# THIS REPLACES THE WHOLE EXEMPLAR-STRIP MACHINERY OF THE SAM3 VERSION (CELL 11
# local background sampling, CELL 12 feathering + canvas composition, CELL 13
# strip/tile region rule). The D1/D2/D3 plants are usually NOT inside the tile
# being scanned - SAM3 solved that with the strip, YOLOE solves it by encoding
# the prompt from separate reference images, once per image.
#
# WHY NOT SIMPLY USE THE FULL IMAGE AS THE REFERENCE
#   YOLOE letterboxes whatever it is given to IMGSZ (1024). An 8192 px wide UAV
#   frame is shrunk ~8x, so a 110 px Rumex plant arrives at the encoder as a
#   ~14 px blob, while the tiles arrive at ~1:1. The model would be prompted with
#   an object at a completely different scale from the one it has to find.
#
# WHAT WE DO INSTEAD (PROMPT_MODE = "per_exemplar_vpe")
#   A REFERENCE_WINDOW_SIZE x REFERENCE_WINDOW_SIZE window of the ORIGINAL image
#   is cut around each exemplar: real, untouched UAV pixels at tile scale, with
#   the plant's natural surroundings. Exemplars that fall inside the same window
#   share it, so N exemplars need at most N windows and often fewer.
# =============================================================================

def make_reference_window(centre_xy, image: Image.Image, window_size: int):
    """
    Cut an axis-aligned window of `window_size` px centred on `centre_xy`,
    shifted so it stays fully inside the image.

    Output: (x1, y1, x2, y2) in ORIGINAL image coordinates.
    """
    cx, cy = centre_xy
    half = window_size / 2.0
    x1 = int(round(cx - half)); y1 = int(round(cy - half))
    x1 = max(0, min(x1, max(0, image.width - window_size)))
    y1 = max(0, min(y1, max(0, image.height - window_size)))
    x2 = min(image.width, x1 + window_size)
    y2 = min(image.height, y1 + window_size)
    return (x1, y1, x2, y2)


def box_inside(box, region, min_fraction: float = 1.0) -> bool:
    """True if at least `min_fraction` of `box`'s area lies inside `region`."""
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 0:
        return False
    return (intersection_area(box, region) / area) >= min_fraction


def build_reference_windows(image: Image.Image, exemplar_indices: Sequence[int],
                            all_gt_boxes: np.ndarray, window_size: int) -> List[dict]:
    """
    Greedily group the exemplars into as few reference windows as possible.

    For each exemplar that is not covered yet, a window is centred on it and
    EVERY other still-uncovered exemplar that lies fully inside that window is
    attached to the same window (YOLOE's SAVPE takes several boxes of the same
    class in one reference image).

    Output: list of dicts
        {'window'    : (x1, y1, x2, y2) in original coords,
         'image'     : PIL crop of that window,
         'gt_indices': the exemplar GT indices it carries,
         'boxes'     : their boxes in WINDOW coordinates (N, 4) float32}
    """
    remaining = list(exemplar_indices)
    windows = []
    while remaining:
        idx = remaining[0]
        bx1, by1, bx2, by2 = all_gt_boxes[idx]
        win = make_reference_window(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0),
                                    image, window_size)
        members = [i for i in remaining if box_inside(all_gt_boxes[i], win)]
        if idx not in members:                       # exemplar bigger than the window
            members = [idx]
        # original coordinates -> coordinates relative to the reference window
        local_boxes = np.array(
            [[all_gt_boxes[i][0] - win[0], all_gt_boxes[i][1] - win[1],
              all_gt_boxes[i][2] - win[0], all_gt_boxes[i][3] - win[1]]
             for i in members], dtype=np.float32)
        # clip to the window so a slightly overhanging box stays valid
        local_boxes[:, 0::2] = np.clip(local_boxes[:, 0::2], 0, win[2] - win[0])
        local_boxes[:, 1::2] = np.clip(local_boxes[:, 1::2], 0, win[3] - win[1])
        windows.append({
            "window": win,
            "image": image.crop(win),
            "gt_indices": members,
            "boxes": local_boxes,
        })
        remaining = [i for i in remaining if i not in members]
    return windows


def build_visual_prompts(boxes_in_ref) -> dict:
    """
    The dict Ultralytics expects. ALL exemplars carry class id 0 because they are
    all examples of the SAME concept (Rumex): they form one visual-prompt group.
    Class ids must be sequential from 0.
    """
    boxes = np.asarray(boxes_in_ref, dtype=np.float32).reshape(-1, 4)
    return {"bboxes": boxes, "cls": np.zeros(len(boxes), dtype=np.int64)}


# =============================================================================
#  CELL 13 - PLAUSIBILITY FILTER
# =============================================================================
# The target-region (strip) rule of the SAM3 version is GONE - there is no
# strip, so every box YOLOE returns is already a detection on real tile content,
# in tile coordinates. The plausibility filter is unchanged so that both models
# are cleaned with identical rules. The mask-fill ratio is received as a
# PRE-COMPUTED number; masks are never kept.
# =============================================================================

def filter_implausible_boxes(boxes, scores, fill_ratios, tile_w: int, tile_h: int,
                             min_fill_ratio: float, max_area_fraction: float,
                             edge_margin: int):
    """
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
        kept_boxes.append(list(box))
        kept_scores.append(float(score))
        kept_fills.append(float(fill))
    return kept_boxes, kept_scores, kept_fills


# =============================================================================
#  CELL 14 - DIVERSITY-BASED EXEMPLAR SELECTION (D1 / D2 / D3)
# =============================================================================
# This is the ONLY methodological change with respect to the size-based and the
# anchor-based tiling experiments. It runs at FULL resolution, never on a tile,
# so the selection is identical to the no-tiling notebook and completely
# independent of the tile grid.
#
# STEP 1  crop every GT Rumex of the image, at FULL resolution:
#
#           GT bbox -> +10% fixed context -> square padding -> 224x224 -> DINOv2
#
#         a) the GT box is enlarged by CROP_CONTEXT (10%) of its own width on the
#            left/right and of its own height on the top/bottom, so every plant is
#            embedded together with a little of its surrounding soil/grass, in
#            proportion to its own size;
#         b) that context box is cut out of the image and PASTED, centred, onto a
#            SQUARE canvas of side max(context_w, context_h) filled with
#            CROP_PAD_FILL. Padding (instead of widening the crop) means the extra
#            area is neutral filler rather than more of the field, so the embedding
#            cannot be driven by whatever happens to lie next to an elongated plant.
#            Parts of the context box that fall OUTSIDE the image are padded the same
#            way, so a plant at the image border stays centred in its canvas;
#         c) the square canvas is resized DIRECTLY to 224x224 (aspect ratio of the
#            context box is preserved by the padding, so nothing is stretched).
#            The DINOv2 processor's default "shortest edge 256 + centre-crop 224" is
#            switched OFF (DINOV2_CENTER_CROP = False), otherwise it would crop the
#            context margin away again.
#
# STEP 2  embed each crop with DINOv2 (CLS token of the last hidden layer),
#         then L2-normalise:  z_i = f(crop_i) / ||f(crop_i)||
#
# STEP 3  pairwise cosine distance
#             d(i,j) = 1 - cos(z_i, z_j) = 1 - <z_i, z_j>      (in [0, 2])
#
# STEP 4  diversity-based selection
#             S* = argmax_{|S| = 3}  min_{i,j in S, i != j} d(i,j)
#         "pick the three plants such that even the CLOSEST pair among the three
#          is still as different as possible" (max-min / maximin dispersion).
#         The search is EXHAUSTIVE over all C(G,3) triples, evaluated vectorised,
#         so the optimum is exact - not a greedy approximation. Triples are
#         enumerated in lexicographic order and the FIRST maximiser is taken,
#         so ties are broken by the lowest GT indices -> fully deterministic.
#         (For G > DIVERSITY_EXACT_MAX_GT a greedy farthest-point fallback keeps
#          the selection from exploding; the dataset never reaches that size.)
#
# ROLE ORDER  D1, D2, D3 = the three selected plants sorted by DECREASING mean
#         distance to the other two selected plants (ties -> lower GT index).
#         D1 is therefore the most "isolated" of the three. The order only
#         affects labels/colours and the order in which the reference windows are
#         built; the prompt set itself is order-independent (the VPEs are averaged).
#
# Fewer than 3 GT boxes: 1 box -> [D1], 2 boxes -> [D1, D2] (both used).
#
# The padded square crops are used ONLY for the embeddings. YOLOE is shown the
# SAME three plants, but through their GT boxes inside TILE_SIZE reference windows
# of the original image (CELL 11 / CELL 12) - there is no strip to paste into.
# =============================================================================

def box_areas(boxes) -> np.ndarray:
    """(N,4) [x1,y1,x2,y2] -> (N,) areas in px^2."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


# ----------------------------- STEP 1: cropping -------------------------------
def crop_gt_for_embedding(image: Image.Image, box, context: float = 0.10,
                          pad_fill=CROP_PAD_FILL, input_size: int = 224) -> Image.Image:
    """
    GT bbox -> +context -> square padding.  (The resize to input_size is done by
    the DINOv2 processor in Dinov2Embedder.embed_gt_crops.)

    image : PIL.Image at ORIGINAL resolution
    box   : [x1, y1, x2, y2] in full-resolution pixels
    Returns a SQUARE PIL.Image (RGB) with the padded context box centred in it.
    """
    img_w, img_h = image.size
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    y_h = max(1.0, y2 - y1)

    # (a) +context on every side, proportional to the box itself
    left   = int(np.floor(x1 - context * w))
    right  = int(np.ceil (x2 + context * w))
    top    = int(np.floor(y1 - context * y_h))
    bottom = int(np.ceil (y2 + context * y_h))
    crop_w, crop_h = max(1, right - left), max(1, bottom - top)

    # the part of the context box that actually exists inside the image
    ax1, ay1 = max(left, 0), max(top, 0)
    ax2, ay2 = min(right, img_w), min(bottom, img_h)
    if ax2 <= ax1 or ay2 <= ay1:                    # degenerate box outside the image
        return Image.new("RGB", (input_size, input_size),
                         CROP_PAD_FILL if pad_fill == "edge" else tuple(pad_fill))
    patch_img = image.crop((ax1, ay1, ax2, ay2))

    # (b) paste it, centred, on a square canvas -> padding, never stretching
    side = max(crop_w, crop_h)
    off_x = (side - crop_w) // 2 + (ax1 - left)     # keeps the plant centred even at
    off_y = (side - crop_h) // 2 + (ay1 - top)      # the image border

    if pad_fill == "edge":                          # replicate border pixels
        canvas = patch_img.resize((side, side), Image.NEAREST)
        canvas.paste(patch_img, (off_x, off_y))
    else:
        canvas = Image.new("RGB", (side, side), tuple(pad_fill))
        canvas.paste(patch_img, (off_x, off_y))
    return canvas


# ---------------------- STEP 2: DINOv2 embeddings -----------------------------
class Dinov2Embedder:
    """
    Owns the DINOv2 processor + model and turns the GT boxes of ONE image into
    L2-normalised CLS embeddings.

    DINOv2-large is ~1.2 GB in fp32 and stays resident on the GPU next to YOLOE
    for the whole inference loop (notebook CELL 5). It is only run once per image
    on a handful of small crops, and the cosine distances that drive the
    selection are compared, not trained, so it stays in fp32 even when YOLOE runs
    in half precision.
    """

    def __init__(self, model_id: str, device: Optional[str], dtype: str,
                 batch_size: int, input_size: int, center_crop: bool,
                 crop_context: float, embedding_feature: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        if self.device == "cpu":
            self.model_dtype = torch.float32     # half precision is pointless on CPU
        self.batch_size = batch_size
        self.input_size = input_size
        self.center_crop = center_crop
        self.crop_context = crop_context
        self.embedding_feature = embedding_feature

        print(f"Loading DINOv2 from '{model_id}' onto {self.device} ({dtype}) ...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.model_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.embed_dim = int(self.model.config.hidden_size)

        print(f"DINOv2 loaded ({model_id}).")
        print("  model device:", next(self.model.parameters()).device)
        print("  model dtype :", next(self.model.parameters()).dtype)
        print("  embedding   :", embedding_feature, f"({self.embed_dim}-d)")

    def embed_gt_crops(self, image: Image.Image, gt_boxes) -> np.ndarray:
        """
        image    : PIL.Image, ORIGINAL resolution
        gt_boxes : (G,4) full-resolution GT boxes
        Returns  : (G, D) float32 array of L2-NORMALISED DINOv2 CLS embeddings.
        Pipeline : GT bbox -> +CROP_CONTEXT -> square padding -> input_size -> CLS.
        """
        torch = self.torch
        crops = [crop_gt_for_embedding(image, b, self.crop_context, CROP_PAD_FILL,
                                       self.input_size)
                 for b in np.asarray(gt_boxes).reshape(-1, 4)]
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = crops[start:start + self.batch_size]
                # explicit input_size x input_size resize, NO centre crop
                # -> the +CROP_CONTEXT margin survives
                inputs = self.processor(
                    images=batch,
                    do_resize=True,
                    size={"height": self.input_size, "width": self.input_size},
                    do_center_crop=self.center_crop,
                    return_tensors="pt",
                ).to(self.device)
                if self.model_dtype != torch.float32:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)
                outputs = self.model(**inputs)
                if self.embedding_feature == "cls":
                    feats = outputs.last_hidden_state[:, 0, :]          # CLS token
                else:
                    raise ValueError(
                        f"embedding_feature={self.embedding_feature!r} not implemented.")
                chunks.append(feats.float().cpu().numpy())
                del inputs, outputs, feats

        z = np.concatenate(chunks, axis=0).astype(np.float32)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return z / norms                                           # ||z_i|| = 1


# ------------------- STEP 3: pairwise cosine distances ------------------------
def cosine_distance_matrix(z) -> np.ndarray:
    """z (G,D) L2-normalised -> (G,G) distances d(i,j) = 1 - <z_i, z_j>, diag = 0."""
    d = 1.0 - (np.asarray(z, dtype=np.float32) @ np.asarray(z, dtype=np.float32).T)
    d = np.clip(d, 0.0, 2.0)
    d = 0.5 * (d + d.T)                    # exact symmetry despite float round-off
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32)


# --------------- STEP 4: max-min (maximin dispersion) selection ---------------
def _maxmin_exhaustive(d, k: int):
    """Exact argmax over all C(n,k) subsets of the minimum in-subset distance."""
    n = d.shape[0]
    combos = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n), k)),
                         dtype=np.int64).reshape(-1, k)             # lexicographic order
    pair_mins = None
    for a, b in itertools.combinations(range(k), 2):
        dist_ab = d[combos[:, a], combos[:, b]]
        pair_mins = dist_ab if pair_mins is None else np.minimum(pair_mins, dist_ab)
    best = int(np.argmax(pair_mins))        # first maximiser -> lowest indices on ties
    return [int(i) for i in combos[best]], float(pair_mins[best])


def _maxmin_greedy(d, k: int):
    """Farthest-point fallback for very large G: start from the farthest pair,
    then repeatedly add the point whose minimum distance to the set is largest."""
    i, j = np.unravel_index(int(np.argmax(np.triu(d, 1))), d.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < k:
        min_to_set = d[:, chosen].min(axis=1)
        min_to_set[chosen] = -np.inf
        chosen.append(int(np.argmax(min_to_set)))
    sub = d[np.ix_(chosen, chosen)]
    min_pair = float(sub[np.triu_indices(len(chosen), 1)].min())
    return sorted(chosen), min_pair


def select_diverse_exemplars(gt_boxes, embeddings, k: int = 3,
                             exact_max_gt: int = 300) -> dict:
    """
    Input : gt_boxes   (G,4) full-resolution GT boxes of ONE image
            embeddings (G,D) L2-normalised DINOv2 embeddings of the same boxes
    Output: dict with
              'indices'   - GT indices in prompt order (D1, D2, D3)
              'roles'     - ["D1","D2","D3"] (shorter when G < 3)
              'areas'     - area of each selected box (px^2, reported only)
              'distance_matrix'        - (G,G) pairwise cosine distances
              'min_pairwise_distance'  - the maximised objective  min_{i,j in S} d(i,j)
              'mean_pairwise_distance' - mean distance inside the selected triple
              'gt_mean_distance'       - mean distance over ALL GT pairs of the image
              'search_mode'            - "exhaustive" | "greedy" | "all_gt_used"
              'median_area', 'all_areas'
    """
    areas = box_areas(gt_boxes)
    n = len(areas)
    if n == 0:
        raise ValueError("The image has no GT boxes -> no diversity-based exemplars.")

    d = cosine_distance_matrix(embeddings)
    all_pairs = d[np.triu_indices(n, 1)] if n > 1 else np.array([0.0], dtype=np.float32)

    if n <= k:                                   # 1 or 2 GT boxes -> use them all
        indices = list(range(n))
        min_pair = float(all_pairs.min()) if n > 1 else float("nan")
        search_mode = "all_gt_used"
    elif n <= exact_max_gt:
        indices, min_pair = _maxmin_exhaustive(d, k)
        search_mode = "exhaustive"
    else:
        indices, min_pair = _maxmin_greedy(d, k)
        search_mode = "greedy"

    # ---- role order: decreasing mean distance to the other selected plants ----
    if len(indices) > 1:
        sub = d[np.ix_(indices, indices)]
        mean_to_others = sub.sum(axis=1) / (len(indices) - 1)
        order = sorted(range(len(indices)),
                       key=lambda p: (-float(mean_to_others[p]), indices[p]))
        indices = [indices[p] for p in order]
        sub = d[np.ix_(indices, indices)]
        mean_pair = float(sub[np.triu_indices(len(indices), 1)].mean())
    else:
        mean_pair = float("nan")

    roles = [f"D{r + 1}" for r in range(len(indices))]

    return {
        "indices": indices,
        "roles": roles,
        "areas": areas[indices],
        "distance_matrix": d,
        "min_pairwise_distance": float(min_pair),
        "mean_pairwise_distance": float(mean_pair),
        "gt_mean_distance": float(all_pairs.mean()),
        "search_mode": search_mode,
        "median_area": float(np.median(areas)),
        "all_areas": areas,
    }


def format_diversity_prompt_id(selection: dict) -> str:
    """e.g. indices [12, 5, 40] with roles D1,D2,D3 -> 'D1:12+D2:5+D3:40'."""
    return "+".join(f"{r}:{i}" for r, i in zip(selection["roles"], selection["indices"]))


def self_test_selection(k: int = 3, input_size: int = 224) -> None:
    """
    The notebook's two CELL 14 self-tests, kept verbatim. No GPU and no model are
    needed, so they run in every mode (including --dry-run) and fail loudly if the
    selection or the crop geometry was broken by an edit.
    """
    # 5 unit vectors: 0, 1, 2 are mutually orthogonal (distance 1 to each other),
    # 3 and 4 are near-duplicates of 0 and 1. The max-min triple must be {0, 1, 2}.
    toy_z = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.9998, 0.02, 0.0, 0.0],
        [0.02, 0.9998, 0.0, 0.0],
    ], dtype=np.float32)
    toy_z /= np.linalg.norm(toy_z, axis=1, keepdims=True)
    toy_boxes = np.array([[0, 0, 10, 10]] * 5, dtype=np.float32)
    sel = select_diverse_exemplars(toy_boxes, toy_z, k=k)
    assert sorted(sel["indices"]) == [0, 1, 2], sel["indices"]
    assert abs(sel["min_pairwise_distance"] - 1.0) < 1e-4, sel["min_pairwise_distance"]

    # crop geometry
    probe = Image.new("RGB", (200, 120), (0, 0, 0))
    for box, expect in [([50, 40, 90, 60], 48),      # 40x20 box -> 48x24 context -> 48x48
                        ([0, 0, 20, 20], 24),        # touches the corner -> padded, square
                        ([190, 110, 200, 120], 12)]: # bottom-right corner
        c = crop_gt_for_embedding(probe, box, 0.10, CROP_PAD_FILL, input_size)
        assert c.size == (expect, expect), (box, c.size, expect)

    print(f"Self-test OK: diversity selection ({DIVERSITY_OBJECTIVE}, {DIVERSITY_METRIC}) "
          f"-> {format_diversity_prompt_id(sel)} "
          f"| min pairwise distance = {sel['min_pairwise_distance']:.3f} "
          f"| search = {sel['search_mode']} | crop geometry OK")


# =============================================================================
#  CELL 5 + CELL 15 - YOLOE MODEL, VISUAL PROMPT AND BATCHED TILE INFERENCE
# =============================================================================
#  * the D1/D2/D3 exemplars are encoded ONCE PER RUN (= one image) into one visual
#    prompt embedding (VPE) and installed with set_classes(); every tile is then
#    a plain model.predict(tile) call - the prompt costs nothing per tile
#  * tiles are sent in batches of BATCH_SIZE (4)
#  * imgsz=IMGSZ (1024) keeps a 1000 px tile at essentially native resolution
#  * conf=YOLOE_INFERENCE_THRESHOLD (0.30): one pass, every higher operating
#    threshold is replayed offline later
#  * iou=YOLOE_PREDICT_NMS_IOU (0.90) is deliberately permissive so the OFFLINE
#    NMS is the one that decides
#  * masks are converted into ONE number per detection (the mask-fill ratio used
#    by the plausibility filter) and then dropped (KEEP_MASKS = False here: the
#    qualitative figure draws boxes only).
#
#  Cluster change: each shard process sees exactly ONE GPU via
#  CUDA_VISIBLE_DEVICES. CUDA is initialised BEFORE Ultralytics' select_device()
#  runs, so that it cannot re-point the process at another physical GPU.
# =============================================================================

def _precision_kwargs(use_half: bool) -> dict:
    """
    The notebook passes `quantize=16` (ultralytics 8.4.x). Older ultralytics
    releases only know `half=True`; use whichever the installed version accepts.
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16 if use_half else 32}
    except Exception:
        pass
    return {"half": bool(use_half)}


def _masks_to_bool_list(result_masks, target_hw, binarise_at: float):
    """
    Ultralytics masks -> list of 2D BOOL numpy arrays in TILE coordinates.
    If the mask resolution differs from the tile (retina_masks=False), the mask
    is nearest-neighbour resized so box and mask share one coordinate system.
    """
    if result_masks is None:
        return None
    data = result_masks.data if hasattr(result_masks, "data") else result_masks
    if data is None or len(data) == 0:
        return []
    th, tw = target_hw
    out = []
    for m in data:
        m = np.squeeze(m.detach().float().cpu().numpy())
        m = m > binarise_at
        if m.shape[:2] != (th, tw):
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                         .resize((tw, th), Image.NEAREST)) > 127
        out.append(m.astype(bool))
    return out


def _fill_ratios_from_masks(boxes_np, masks) -> np.ndarray:
    """
    Fraction of each predicted box that is actually covered by its mask - the
    single number the plausibility filter needs.

    Input : boxes_np (N,4) in tile coords, masks - list of 2D bool arrays
    Output: (N,) float32 fill ratios
    """
    n = len(boxes_np)
    fills = np.zeros(n, dtype=np.float32)
    if masks is None or n == 0 or len(masks) == 0:
        return fills
    for i in range(min(n, len(masks))):
        m = masks[i]
        h, w = int(m.shape[-2]), int(m.shape[-1])
        x1, y1, x2, y2 = boxes_np[i]
        x1c, y1c = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        x2c, y2c = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
        if x2c <= x1c or y2c <= y1c:
            continue
        region = m[y1c:y2c, x1c:x2c]
        fills[i] = float(region.mean()) if region.size else 0.0
    return fills


class YoloeRunner:
    """Owns the YOLOE model, installs the visual prompt and runs batched inference."""

    def __init__(self, args):
        import torch
        from ultralytics import YOLOE

        self.torch = torch
        self.args = args
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda"):
            torch.cuda.init()                 # lock CUDA_VISIBLE_DEVICES before ultralytics
        self.use_half = bool(args.use_fp16 and self.device.startswith("cuda"))
        self.precision_kwargs = _precision_kwargs(self.use_half)

        print(f"Loading YOLOE from '{args.weights}' onto {self.device} "
              f"(half={self.use_half}, {self.precision_kwargs}) ...")
        self.model = YOLOE(args.weights)
        self.model.to(self.device)

        print("YOLOE loaded.")
        print("  weights     :", args.weights)
        print("  device      :", self.device)
        print("  half        :", self.use_half)
        print("  default vocab size:", len(self.model.names),
              "classes (replaced by the visual prompt of every run)")

    # ---------------------------- CELL 13 -----------------------------------
    def extract_vpe(self, reference_image: Image.Image, boxes_in_ref):
        """
        Run ONLY the visual-prompt encoder on one reference image.

        Input : reference_image - PIL image (a reference window)
                boxes_in_ref    - (N, 4) exemplar boxes in that image's coordinates
        Output: torch.Tensor (1, n_classes, D) on CPU, float32
        """
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        a = self.args
        vp = YOLOEVPSegPredictor(overrides=dict(
            task="segment", mode="predict", model=a.weights,
            imgsz=a.imgsz, conf=a.threshold,
            device=self.device, save=False, verbose=False,
            **self.precision_kwargs,
        ))
        vp.setup_model(self.model.model)
        # the boxes of ONE reference window form ONE visual-prompt group
        vp.set_prompts(build_visual_prompts(boxes_in_ref))
        vpe = vp.get_vpe(reference_image)
        del vp
        return vpe.detach().float().cpu()

    def install_prompt(self, reference_windows: List[dict]):
        """
        PROMPT_MODE = "per_exemplar_vpe": one VPE per reference window, averaged
        and L2-normalised, then installed on the model with set_classes().
        """
        torch = self.torch
        vpes = [self.extract_vpe(w["image"], w["boxes"]) for w in reference_windows]
        vpe_final = torch.stack(vpes, dim=0).mean(dim=0)
        vpe_final = torch.nn.functional.normalize(vpe_final, dim=-1, p=2)
        self.model.set_classes([self.args.prompt_class_name], vpe_final)
        self.model.predictor = None        # drop the VP predictor; tiles use the plain one
        return vpe_final

    # ---------------------------- CELL 15 -----------------------------------
    def infer_batch(self, tile_images: Sequence[Image.Image], threshold: float):
        """
        Run the prompted YOLOE on a batch of raw tiles.

        Input : tile_images - list of PIL images (length <= BATCH_SIZE), each a
                              plain crop of the UAV image (NO strip, NO padding)
        Output: list (same length) of (boxes, scores, fill_ratios), all numpy,
                boxes in TILE coordinates, every detection with score >= threshold.
                Masks are NOT returned.
        """
        a = self.args
        results = self.model.predict(
            list(tile_images),
            imgsz=a.imgsz,
            conf=threshold,
            iou=a.predict_nms_iou,
            max_det=a.max_det,
            retina_masks=a.retina_masks,
            device=self.device,
            verbose=False,
            **self.precision_kwargs,
        )

        per_image = []
        for res, tile_img in zip(results, tile_images):
            tw, th = tile_img.size
            if res.boxes is None or len(res.boxes) == 0:
                per_image.append((np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                                  np.zeros((0,), np.float32)))
                continue
            boxes = res.boxes.xyxy.detach().float().cpu().numpy().reshape(-1, 4)
            scores = res.boxes.conf.detach().float().cpu().numpy().reshape(-1)
            masks = _masks_to_bool_list(res.masks, (th, tw), a.mask_binarise)
            fills = _fill_ratios_from_masks(boxes, masks)
            del masks                      # no longer needed -> free immediately
            per_image.append((boxes, scores, fills))

        del results
        return per_image

# =============================================================================
#  CELL 15 (cont.) - THE D1/D2/D3 PROMPT SET OVER ALL TILES OF ONE IMAGE
# =============================================================================
# For every tile:
#     YOLOE   (prompted model, plain predict, threshold 0.30)
#     plausibility filter
#     tile coords -> ORIGINAL image coords
# There is no "compose" step and no strip/tile region remapping.
# =============================================================================

def run_image_over_tiles(runner: YoloeRunner, tile_cache: List[dict],
                          full_image: Image.Image, args) -> dict:
    """
    Run the (already prompted) YOLOE over ALL cached tiles.

    Input : tile_cache     - list of tile dicts (CELL 9), built once per image
            full_image     - the open PIL image (used only if tiles are not cached)
    Output: dict of numpy arrays, all in ORIGINAL-IMAGE coordinates:
            boxes (N,4), scores (N,), fill_ratio (N,), tile_id (N,),
            tile_boxes (N,4)  <- the tile each detection came from
            These are the PRE-NMS detections (no confidence filtering beyond 0.30,
            no offline NMS) that get written to disk for the offline evaluation.
    """
    all_boxes, all_scores, all_fills, all_tids, all_tboxes = [], [], [], [], []

    for start in range(0, len(tile_cache), args.batch_size):
        batch_tiles = tile_cache[start:start + args.batch_size]
        tile_images = [get_tile_image(t, full_image) for t in batch_tiles]

        batch_results = runner.infer_batch(tile_images, args.threshold)

        for tile, tile_img, (boxes, scores, fills) in zip(
                batch_tiles, tile_images, batch_results):
            tw, th = tile_img.size
            # 1) plausibility filter (boxes are already in tile coordinates)
            boxes, scores, fills = filter_implausible_boxes(
                boxes, scores, fills, tw, th,
                args.min_fill_ratio, args.max_area_fraction, args.edge_margin)
            # 2) tile coords -> original image coords
            for b, s, f in zip(boxes, scores, fills):
                all_boxes.append([b[0] + tile["x1"], b[1] + tile["y1"],
                                  b[2] + tile["x1"], b[3] + tile["y1"]])
                all_scores.append(float(s))
                all_fills.append(float(f))
                all_tids.append(int(tile["tile_id"]))
                all_tboxes.append([tile["x1"], tile["y1"], tile["x2"], tile["y2"]])

        del tile_images, batch_results

    return {
        "boxes": np.array(all_boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.array(all_scores, dtype=np.float32).reshape(-1),
        "fill_ratio": np.array(all_fills, dtype=np.float32).reshape(-1),
        "tile_id": np.array(all_tids, dtype=np.int32).reshape(-1),
        "tile_boxes": np.array(all_tboxes, dtype=np.int32).reshape(-1, 4),
    }


# =============================================================================
#  CELL 16 - PRE-NMS DETECTION STORAGE
# =============================================================================
# For every run (= ONE image, one D1/D2/D3 prompt set) we store the detections
#   AFTER YOLOE inference at CONFIDENCE_THRESHOLD -> plausibility filtering ->
#   conversion to original-image coordinates,
# but BEFORE NMS.
# The offline evaluation (PHASE 2) therefore never needs YOLOE again.
#
# Each detection keeps the tile it came from (tile_id / tile_boxes), so the NMS
# of CELL 19 can report how many duplicates were cross-tile. Masks are never
# stored. The DINOv2 distance matrix of the image IS stored (G x G float32, a few
# kB), so the selection can be re-analysed without re-running DINOv2.
# =============================================================================

def run_npz_path(raw_detections_dir: Path, image_id: str) -> Path:
    """Path of the NPZ holding the pre-NMS detections of ONE image."""
    return raw_detections_dir / f"{safe_filename(image_id)}.npz"


def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        detections: dict, gt_boxes: np.ndarray, selection: dict,
                        image_size: Tuple[int, int], n_tiles: int,
                        archive: str, flight: str, class_id: int,
                        reference_windows: Sequence[Tuple[int, int, int, int]]) -> Path:
    """
    Write one run's pre-NMS detections to NPZ. The file is self-contained: it also
    stores the GT boxes, the D1/D2/D3 prompt indices (in prompt order), the tile
    provenance, the reference windows and the pairwise DINOv2 distance matrix,
    so the whole offline
    evaluation and the final plots can run without re-opening label files, tiles
    or re-embedding the crops.
    """
    path = run_npz_path(raw_detections_dir, image_id)
    np.savez_compressed(
        path,
        experiment_name=np.array(experiment_name),
        image_ID=np.array(image_id),
        prompt_indices=np.array(selection["indices"], dtype=np.int32),   # order: D1, D2, D3
        prompt_roles=np.array(selection["roles"]),                       # ["D1","D2","D3"]
        prompt_areas=np.asarray(selection["areas"], dtype=np.float32),
        median_gt_area=np.array(float(selection["median_area"]), dtype=np.float32),
        distance_matrix=np.asarray(selection["distance_matrix"], dtype=np.float32),
        min_pairwise_distance=np.array(float(selection["min_pairwise_distance"]), dtype=np.float32),
        mean_pairwise_distance=np.array(float(selection["mean_pairwise_distance"]), dtype=np.float32),
        gt_mean_distance=np.array(float(selection["gt_mean_distance"]), dtype=np.float32),
        search_mode=np.array(selection["search_mode"]),
        n_tiles=np.array(int(n_tiles)),
        reference_windows=np.array(reference_windows, dtype=np.int32).reshape(-1, 4),
        image_width=np.array(int(image_size[0])),
        image_height=np.array(int(image_size[1])),
        archive=np.array(archive),
        flight=np.array(flight),
        source_class_id=np.array(int(class_id)),
        gt_boxes=gt_boxes.astype(np.float32),
        boxes=detections["boxes"].astype(np.float32),      # x1,y1,x2,y2 (original img)
        scores=detections["scores"].astype(np.float32),    # confidence >= 0.30
        fill_ratio=detections["fill_ratio"].astype(np.float32),
        tile_id=detections["tile_id"].astype(np.int32),    # which tile produced it
        tile_boxes=detections["tile_boxes"].astype(np.int32),  # that tile's extent
    )
    return path


def load_run_detections(path: Path) -> dict:
    """Read one run NPZ back into a plain python dict."""
    with np.load(path, allow_pickle=False) as z:
        run = {
            "image_ID": str(z["image_ID"]),
            "prompt_indices": z["prompt_indices"].astype(int),
            "prompt_roles": [str(r) for r in z["prompt_roles"]],
            "prompt_areas": z["prompt_areas"].reshape(-1),
            "median_gt_area": float(z["median_gt_area"]),
            "distance_matrix": z["distance_matrix"],
            "min_pairwise_distance": float(z["min_pairwise_distance"]),
            "mean_pairwise_distance": float(z["mean_pairwise_distance"]),
            "gt_mean_distance": float(z["gt_mean_distance"]),
            "search_mode": str(z["search_mode"]),
            "n_tiles": int(z["n_tiles"]),
            "image_width": int(z["image_width"]),
            "image_height": int(z["image_height"]),
            "gt_boxes": z["gt_boxes"].reshape(-1, 4),
            "boxes": z["boxes"].reshape(-1, 4),
            "scores": z["scores"].reshape(-1),
            "fill_ratio": z["fill_ratio"].reshape(-1),
            "tile_id": z["tile_id"].reshape(-1),
            "tile_boxes": z["tile_boxes"].reshape(-1, 4),
        }
        # archive / flight are cluster additions; tolerate older NPZs.
        run["archive"] = str(z["archive"]) if "archive" in z else ""
        run["flight"] = str(z["flight"]) if "flight" in z else ""
    return run


# =============================================================================
#  RESUME SUPPORT  (CELL 17, adapted to several shard manifests)
# =============================================================================

def load_done_images(paths: Paths, experiment_name: str, prompt_type: str) -> set:
    """
    Read every shard manifest and return the set of image_IDs that are already
    finished. Every shard reads ALL manifests, so a resubmission after the walltime
    never repeats work, even if the shard assignment changed because NUM_GPUS was
    different.

    The notebook's guard is kept: if a manifest holds results produced with a
    DIFFERENT prompt setting, the run is aborted instead of silently mixing them.
    """
    done: set = set()
    other_types: set = set()
    for csv_path in sorted(paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    row_type = str(row.get("Prompt_Type", ""))
                    if row_type and row_type != prompt_type:
                        other_types.add(row_type)
                        continue
                    try:
                        done.add(row["image_ID"])
                    except KeyError:
                        continue            # ignore a half-written trailing row
        except OSError:
            continue
    if other_types:
        raise RuntimeError(
            f"{paths.raw_detections} already holds results for another prompt setting "
            f"{sorted(other_types)}. Use a new EXPERIMENT_NAME (and OUTPUT_DIR) for "
            f"{prompt_type} so the runs are not mixed.")
    return done


# =============================================================================
#  CELL 17 - MAIN GPU INFERENCE LOOP  (DINOv2 -> selection -> tiles -> YOLOE)
#            [PHASE 1]
# =============================================================================
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE
#     read its GT boxes ONCE
#     crop every GT box (+10% context, square padding) and embed it with DINOv2
#     pick the 3 most DIVERSE GT boxes (max-min cosine distance, CELL 14)
#         - deterministic, so there is exactly ONE run per image
#     build the overlapping tiles ONCE (cached on CPU)
#     build the reference windows around D1 / D2 / D3 and install ONE averaged
#         visual prompt embedding (CELL 11 / CELL 12 / CELL 15)
#     run YOLOE over those tiles in batches (plain predict, threshold 0.30),
#         plausibility filter (CELL 13)
#     save the PRE-NMS detections + tile provenance + distance matrix (NPZ)
#     release the image and the tile cache
#
# Images without any Rumex GT box are skipped, exactly like in the other
# experiments, so every experiment is evaluated on the same set of images.
# NO NMS and NO metric computation happens here - that is all done offline in
# PHASE 2. The loop is resumable: finished images are listed in the shard
# manifests.
# =============================================================================

def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    import torch

    exp = args.experiment_name

    # ---- resume support ------------------------------------------------------
    done_images: set = set()
    if not args.no_resume:
        done_images = load_done_images(paths, exp, args.prompt_type)
        print(f"Resuming: {len(done_images)} image(s) already finished for {exp}; skipped.")

    # ---- this shard's manifest ----------------------------------------------
    manifest_csv = manifest_path(paths, exp, args.shard_index)
    manifest_exists = manifest_csv.exists() and manifest_csv.stat().st_size > 0
    manifest_file = open(manifest_csv, "a", newline="")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS,
                                     extrasaction="ignore")
    if not manifest_exists:
        manifest_writer.writeheader()
        manifest_file.flush()

    # ---- both models stay resident for the whole loop (CELL 5) ---------------
    embedder = Dinov2Embedder(args.dinov2_model_id, args.device, args.dinov2_dtype,
                              args.dinov2_batch_size, args.dinov2_input_size,
                              args.dinov2_center_crop, args.crop_context,
                              args.embedding_feature)
    runner = YoloeRunner(args)
    device_is_cuda = runner.device.startswith("cuda")

    start_time = time.time()
    n_new_runs = 0
    image_times: List[float] = []
    n_total_images = len(records)

    try:
        for img_idx, rec in enumerate(records, start=1):
            image_id = rec.image_id
            if image_id in done_images:
                print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id}: already done, skipped.")
                continue
            image_t0 = time.time()

            # ---------------- open the original image exactly once ----------------
            image = Image.open(rec.image_path).convert("RGB")
            img_w, img_h = image.size
            gt_boxes = load_yolo_boxes(rec.label_path, img_w, img_h, rec.class_id)
            n_gt = len(gt_boxes)

            if n_gt == 0:
                print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id}: 0 GT boxes, skipped.")
                image.close(); del image; gc.collect()
                continue

            # ------- DINOv2 embeddings + diversity-based selection (CELL 14) -------
            # full resolution, independent of the tile grid
            embed_t0 = time.time()
            gt_embeddings = embedder.embed_gt_crops(image, gt_boxes)
            selection = select_diverse_exemplars(gt_boxes, gt_embeddings,
                                                 k=args.k_exemplars,
                                                 exact_max_gt=args.diversity_exact_max_gt)
            embed_seconds = time.time() - embed_t0

            exemplar_indices = selection["indices"]
            prompt_id = format_diversity_prompt_id(selection)

            # ---------------- build the tiles exactly once -------------------------
            tile_cache = build_tile_cache(image, args.use_tiling, args.tile_size,
                                          args.overlap, args.cache_tiles_in_memory)
            n_tiles = len(tile_cache)

            # -------- reference windows + visual prompt (CELL 11 / 12 / 15) --------
            yoloe_t0 = time.time()
            reference_windows = build_reference_windows(image, exemplar_indices, gt_boxes,
                                                        args.reference_window_size)
            runner.install_prompt(reference_windows)

            # ---------------- YOLOE over all tiles (CELL 15) -----------------------
            detections = run_image_over_tiles(runner, tile_cache, image, args)
            yoloe_seconds = time.time() - yoloe_t0

            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, detections, gt_boxes, selection,
                (img_w, img_h), n_tiles, rec.archive, rec.flight, rec.class_id,
                [w["window"] for w in reference_windows])

            areas_by_role = dict(zip(selection["roles"], selection["areas"]))
            run_seconds = time.time() - image_t0
            manifest_writer.writerow({
                "experiment_name": exp,
                "image_ID": image_id,
                "Prompt_ID": prompt_id,
                "Prompt_Type": args.prompt_type,
                "archive": rec.archive,
                "flight": rec.flight,
                "source_class_id": rec.class_id,
                "n_gt": n_gt,
                "n_prompt_gt": len(exemplar_indices),
                "area_D1_px2": round(float(areas_by_role["D1"])) if "D1" in areas_by_role else "",
                "area_D2_px2": round(float(areas_by_role["D2"])) if "D2" in areas_by_role else "",
                "area_D3_px2": round(float(areas_by_role["D3"])) if "D3" in areas_by_role else "",
                "median_gt_area_px2": round(float(selection["median_area"])),
                "min_pairwise_distance": round(float(selection["min_pairwise_distance"]), 6),
                "mean_pairwise_distance": round(float(selection["mean_pairwise_distance"]), 6),
                "gt_mean_distance": round(float(selection["gt_mean_distance"]), 6),
                "search_mode": selection["search_mode"],
                "n_tiles": n_tiles,
                "n_reference_windows": len(reference_windows),
                "n_detections_pre_nms": int(len(detections["scores"])),
                "image_width": img_w,
                "image_height": img_h,
                "npz_file": npz_path.name,
                "embedding_seconds": round(embed_seconds, 2),
                "inference_seconds": round(run_seconds, 2),
            })
            manifest_file.flush()                  # this image is on disk -> resumable
            n_new_runs += 1

            if len(exemplar_indices) < args.k_exemplars:
                print(f"    NOTE: only {n_gt} GT box(es) -> {len(exemplar_indices)} prompt(s) "
                      f"({'+'.join(selection['roles'])}).")

            n_dets = int(len(detections["scores"]))
            n_windows = len(reference_windows)

            # ---------------- release the image and its tile cache -----------------
            for t in tile_cache:
                t["image"] = None
            for w in reference_windows:
                w["image"] = None
            del tile_cache, reference_windows, detections, gt_embeddings
            image.close()
            del image
            gc.collect()
            if device_is_cuda:
                torch.cuda.empty_cache()

            image_times.append(time.time() - image_t0)
            avg_per_image = float(np.mean(image_times))
            eta = (n_total_images - img_idx) * avg_per_image
            print(f"[{exp}] shard{args.shard_index} run #{n_new_runs} "
                  f"({img_idx}/{n_total_images}) {image_id} | {n_gt} GT box(es) | "
                  f"prompts={prompt_id} | min_dist={selection['min_pairwise_distance']:.3f} "
                  f"(mean over all GT pairs {selection['gt_mean_distance']:.3f}) | "
                  f"ref windows={n_windows} | tiles={n_tiles} | pre-NMS detections={n_dets} | "
                  f"DINOv2 {embed_seconds:.1f}s + YOLOE {yoloe_seconds:.1f}s = {run_seconds:.1f}s | "
                  f"avg/image={avg_per_image:.1f}s | ETA={eta / 60:.1f} min "
                  f"({eta / 3600:.2f} h)")
    finally:
        manifest_file.close()                    # also closed if the loop crashes

    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new images.")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Pre-NMS detections in: {paths.raw_detections}")


# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================

def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading YOLOE or DINOv2 ---")
    sample = my_records[:min(len(my_records), 200)]
    total_gt, per_archive, n_small = 0, {}, 0
    for rec in sample:
        with Image.open(rec.image_path) as im:
            w, h = im.size
        n_gt = len(load_yolo_boxes(rec.label_path, w, h, rec.class_id))
        total_gt += n_gt
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + n_gt
        if 0 < n_gt < args.k_exemplars:
            n_small += 1

    tiles = (len(tile_bboxes(8192, 5460, args.tile_size, args.overlap))
             if args.use_tiling else 1)
    batches = int(np.ceil(tiles / max(1, args.batch_size)))
    print(f"  sampled {len(sample)} image(s) of this shard -> {len(sample)} runs "
          f"(ONE per image: the diversity selection is deterministic)")
    print(f"  GT boxes in those images: {total_gt} ({per_archive})")
    print(f"  DINOv2 crops to embed: {total_gt} "
          f"(~{int(np.ceil(total_gt / max(1, args.dinov2_batch_size)))} forward passes "
          f"at batch size {args.dinov2_batch_size})")
    print(f"  tiles per run at 8192x5460: {tiles}  "
          f"(TILE_SIZE={args.tile_size}, OVERLAP={args.overlap}, use_tiling={args.use_tiling})")
    print(f"  => ~{len(sample) * tiles} YOLOE tile forward passes for those {len(sample)} "
          f"images (~{len(sample) * batches} batched calls at batch size {args.batch_size})")
    print(f"     (+ 1..{args.k_exemplars} reference-window VPE encodings per image, "
          f"window={args.reference_window_size}px, imgsz={args.imgsz})")
    weights = Path(str(args.weights))
    if weights.is_absolute() or weights.parent != Path("."):
        print(f"  weights file: {weights} -> {'FOUND' if weights.is_file() else 'MISSING'}")
    print(f"  max-min search: exhaustive up to {args.diversity_exact_max_gt} GT boxes "
          f"(C(G,{args.k_exemplars}) triples, vectorised)")
    print(f"  image(s) with fewer than K={args.k_exemplars} GT boxes -> fewer prompts: "
          f"{n_small}")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{len(sample)} (one per image)")


# =============================================================================
#  CELL 18 - LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
# =============================================================================
# From here on YOLOE and DINOv2 are never touched again. Everything below works on
# the NPZ files written in PHASE 1, so the complete evaluation can be redone in
# minutes on a login node or in a small CPU allocation.
# =============================================================================

def load_runs(paths: Paths, experiment_name: str, prompt_type: str):
    import pandas as pd

    manifest_files = sorted(
        paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv"))
    if not manifest_files:
        print(f"No manifest found in {paths.raw_detections} for {experiment_name}.")
        return [], None

    frames = []
    for path in manifest_files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"  WARNING: could not read {path.name}: {exc}")
    if not frames:
        print("All manifests unreadable.")
        return [], None

    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[(manifest["experiment_name"] == experiment_name) &
                        (manifest["Prompt_Type"].astype(str) == prompt_type)].copy()
    manifest = manifest.drop_duplicates(subset=["image_ID"], keep="last")
    manifest = manifest.sort_values(["image_ID"], kind="stable")

    runs = []
    n_missing = 0
    for _, row in manifest.iterrows():
        path = paths.raw_detections / str(row["npz_file"])
        if not path.exists():
            n_missing += 1
            continue
        run = load_run_detections(path)
        run["Prompt_ID"] = str(row["Prompt_ID"])
        run["Prompt_Type"] = str(row["Prompt_Type"])
        run["archive"] = run.get("archive") or str(row.get("archive", ""))
        run["flight"] = run.get("flight") or str(row.get("flight", ""))
        runs.append(run)

    if n_missing:
        print(f"  WARNING: {n_missing} manifest row(s) point at a missing NPZ (skipped).")

    print(f"Loaded {len(runs)} runs (= images) for {experiment_name} ({prompt_type}).")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes:", int(sum(len(r['gt_boxes']) for r in runs)))
    print("Tiles per image:", ", ".join(str(x) for x in sorted(set(r['n_tiles'] for r in runs))))
    return runs, manifest


# =============================================================================
#  CELL 19 - OFFLINE NMS
# =============================================================================
# Because the same plant is visible in several overlapping tiles, it can be
# detected several times. NMS keeps the highest-scoring box of each overlapping
# group. The NMS IoU threshold is fixed (NMS_IOU_THRESHOLD = 0.40) and applied
# offline, so the raw pre-NMS detections stay untouched on disk.
#
# 'Provenance' = we also record WHICH detection suppressed which. That lets the
# per-image table (CELL 24) report how many duplicates came from a DIFFERENT
# tile (cross-tile duplication) versus the same tile.
# =============================================================================

def nms_with_provenance(boxes, scores, iou_threshold: float):
    """
    Input : boxes (N,4), scores (N,), iou_threshold
    Output: keep (list of kept indices, highest score first)
            suppressed (list of (suppressed_index, suppressor_index))
    A detection is suppressed when its IoU with an already kept, higher-scoring
    detection is GREATER than the threshold.
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


def apply_nms_to_run(run: dict, iou_threshold: float) -> dict:
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
#  CELL 20 - EVALUATION CORE: all_gt AND held_out
# =============================================================================
# all_gt   : classical evaluation, every GT box of the image counts.
#
# held_out : the GT instances that were shown to YOLOE as visual prompts (D1, D2,
#            D3) are REMOVED from the GT set, and predictions that fall on those
#            prompt plants are IGNORED (neither TP nor FP). It answers: "after
#            being shown three visually DIFFERENT examples, how well does YOLOE
#            find the REMAINING Rumex plants?"
#
# ORDER OF OPERATIONS (the agreed protocol):
#   1. match predictions to the evaluated (non-prompt) GT with the corrected
#      one-to-one matcher at EVAL_IOU_THRESHOLD = 0.50
#   2. every STILL UNMATCHED prediction whose best IoU with a PROMPT GT box is
#      >= PROMPT_IGNORE_IOU (0.50) becomes IGNORED
#   3. whatever is still unmatched is a false positive
#   Prompt GT boxes themselves are never counted as false negatives.
# =============================================================================

def split_gt_for_mode(gt_boxes, prompt_indices, mode: str):
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
                             eval_iou: float, ignore_iou: float) -> dict:
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
    # start with everything as FP -> valid matches become TP -> leftovers on a
    # prompt plant become IGNORED -> the rest stays FP
    status = np.full(n_pred, STATUS_FP, dtype=np.int8)

    # step 1 - corrected one-to-one matching against the evaluated GT
    match = match_one_to_one(pred_boxes, pred_scores, eval_gt_boxes, eval_iou)
    status[match["pred_match_gt"] >= 0] = STATUS_TP

    # step 2 - ignore the unmatched leftovers that sit on a PROMPT plant
    if n_pred and len(prompt_gt_boxes):
        leftover = np.where(status == STATUS_FP)[0]
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
#  CELL 21 - AP50 AND AP50:95  (supervision.metrics.MeanAveragePrecision)
# =============================================================================
# AP is an area under the precision-recall curve, built by walking through ALL
# detections ordered by confidence. Truncating the detection list at an operating
# threshold would simply cut the tail off the curve and report a smaller area -
# which says nothing about model quality. AP therefore always uses every post-NMS
# prediction that YOLOE returned (score >= CONFIDENCE_THRESHOLD).
# Precision / recall / F1 / IoU1 / IoU2 describe ONE operating point.
# In this experiment the operating point IS the inference threshold (0.30), so
# both use exactly the same prediction set - they still answer different
# questions (ranking quality vs. deployed behaviour).
# =============================================================================

def make_detections(boxes, scores=None):
    """NumPy boxes (+ optional scores) -> supervision Detections (class 0 = Rumex)."""
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
            One entry = one evaluation episode (one run = one image).
    Output: (AP50, AP50_95). NaN when there is no GT at all to evaluate.
    Passing several episodes at once gives the POOLED (dataset-level) AP, in which
    the detections of all episodes are ranked together in one PR curve.
    """
    from supervision.metrics import MeanAveragePrecision
    if len(gt_list) == 0 or sum(len(g) for g in gt_list) == 0:
        return float("nan"), float("nan")
    try:
        result = MeanAveragePrecision().update(pred_list, gt_list).compute()
        ap50, ap5095 = float(result.map50), float(result.map50_95)
        # supervision returns -1.0 when a metric is undefined -> report NaN instead,
        # so it is excluded from means instead of dragging them down.
        return (ap50 if ap50 >= 0 else float("nan"),
                ap5095 if ap5095 >= 0 else float("nan"))
    except Exception as e:
        print("   (AP computation failed:", e, ")")
        return float("nan"), float("nan")


def ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou: float, ignore_iou: float):
    """
    Build the (prediction, GT) episode used for AP of ONE run.
    All post-NMS predictions are used; in held_out mode the predictions that were
    IGNORED (they belong to a prompt plant) are removed first, exactly like in the
    operating-point evaluation.
    """
    ev = evaluate_run_predictions(nms_run["boxes"], nms_run["scores"], eval_gt, prompt_gt,
                                  eval_iou, ignore_iou)
    keep = ev["status"] != STATUS_IGNORED
    return (make_detections(nms_run["boxes"][keep], nms_run["scores"][keep]),
            make_detections(eval_gt))


# =============================================================================
#  CELL 22 - OPERATING-POINT EVALUATION
# =============================================================================
# Keeps the predictions with score >= CONFIDENCE_THRESHOLD and evaluates them
# (precision, recall, F1, IoU1, IoU2, TP, FP, FN).
# The operating point is fixed from the start :
#     confidence threshold = CONFIDENCE_THRESHOLD (0.30)
#     NMS IoU threshold    = NMS_IOU_THRESHOLD    (0.40)
# =============================================================================

def evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, confidence_threshold: float,
                                eval_iou: float, ignore_iou: float) -> dict:
    """
    Input : nms_run  - output of apply_nms_to_run (sorted by score, high -> low)
            eval_gt / prompt_gt - from split_gt_for_mode
            confidence_threshold - the operating point
    Output: the dict of evaluate_run_predictions for the thresholded predictions.
    """
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_run_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                    eval_gt, prompt_gt, eval_iou, ignore_iou)


# =============================================================================
#  CELL 23 - PLOTTING HELPERS
# =============================================================================
# Colours used everywhere:
#   yellow = GT Rumex boxes
#   cyan   = D1 exemplar prompt (most isolated of the three in embedding space)
#   lime   = D2 exemplar prompt
#   orange = D3 exemplar prompt
#   red    = predictions
#
# matplotlib is imported lazily (inside the functions), so a container without it
# still writes every CSV - only the PNGs are lost.
# =============================================================================

def plot_confusion_matrix(tp, fp, fn, title, png_path, plt) -> None:
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
            ax.text(x1, y1 - 2, labels[k],
                    color="white" if color in ("red",) else "black",
                    fontsize=fontsize, va="bottom", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.8, alpha=0.85))


def _display_copy(image: Image.Image, max_display_dim: int):
    """Downscaled numpy copy of the image + the factor to map full-res boxes onto it."""
    w, h = image.size
    s = min(1.0, max_display_dim / max(w, h))
    disp_w, disp_h = max(1, int(w * s)), max(1, int(h * s))
    display = np.asarray(image.resize((disp_w, disp_h), Image.BILINEAR))
    to_disp = np.array([disp_w / w, disp_h / h, disp_w / w, disp_h / h], dtype=np.float32)
    return display, to_disp


# =============================================================================
#  CELL 30 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right)
# =============================================================================
# IMAGE SELECTION (per-image AP50, mode = PLOT_EVALUATION_MODE):
#   1. keep only images that have at least PLOT_MIN_GT_BOXES (7) GT boxes
#      and take the one with the highest AP50
#   2. if NO image has 7 GT boxes: keep the images with the HIGHEST number of GT
#      boxes and take the one among them with the highest AP50
#   ties are broken by F1, then by the number of GT boxes.
#
# Right panel = post-NMS predictions of that image with
# score >= CONFIDENCE_THRESHOLD, plus its D1 / D2 / D3 prompts (dashed, in the
# role colours of CELL 23). The image is downscaled for DISPLAY only
# (PLOT_MAX_DISPLAY_DIM).
#
# The notebook produced ONE figure. Both archives are pooled here, so the same
# selection is run once per archive and once over everything.
# =============================================================================

def select_image_for_plot(image_level_df, mode: str, min_gt: int,
                          archive: Optional[str] = None):
    """
    Output: (per-image row, selection_rule_text) or (None, reason)
    'archive' restricts the candidates to one archive; None = the whole dataset.
    """
    img_df = image_level_df[(image_level_df["evaluation_mode"] == mode) &
                            image_level_df["AP50"].notna()].copy()
    if archive is not None:
        img_df = img_df[img_df["archive"] == archive]
    if img_df.empty:
        return None, (f"no image with a valid AP50 for archive={archive}" if archive
                      else "no image with a valid AP50")

    candidates = img_df[img_df["n_gt"] >= min_gt]
    if len(candidates):
        rule = (f"highest AP50 among the {len(candidates)} images "
                f"with >= {min_gt} GT boxes")
    else:
        max_gt = int(img_df["n_gt"].max())
        candidates = img_df[img_df["n_gt"] == max_gt]
        rule = (f"no image has >= {min_gt} GT boxes -> highest AP50 among "
                f"the {len(candidates)} image(s) with the most GT boxes ({max_gt})")
    if archive is not None:
        rule += f" [archive={archive}]"

    ranked = candidates.sort_values(["AP50", "F1", "n_gt"], ascending=False,
                                    na_position="last")
    print("Top candidates:")
    print(ranked[["image_ID", "n_gt", "Prompt_ID", "n_predictions", "AP50", "F1",
                  "precision", "recall"]].head(5).to_string(index=False))
    return ranked.iloc[0], rule


def plot_best_image(args, paths: Paths, runs_by_image: dict, image_level_df,
                    image_paths: dict, plt, scope_label: str,
                    archive: Optional[str] = None) -> None:
    """One qualitative figure for one scope (a single archive, or the whole dataset)."""
    from matplotlib.lines import Line2D

    mode = args.plot_evaluation_mode
    exp = args.experiment_name

    row, rule = select_image_for_plot(image_level_df, mode, args.plot_min_gt_boxes, archive)
    if row is None:
        print(f"Nothing to plot for {scope_label}:", rule)
        return
    plot_image_id = row["image_ID"]
    run = runs_by_image.get(plot_image_id)
    if run is None:
        print(f"Nothing to plot for {scope_label}: no cached run for {plot_image_id}")
        return

    nms_run = apply_nms_to_run(run, args.nms_iou_threshold)
    keep = nms_run["scores"] >= args.operating_confidence
    pred_boxes, pred_scores = nms_run["boxes"][keep], nms_run["scores"][keep]
    gt_boxes_plot = run["gt_boxes"]

    # ---- open the image, downscale for display only ---------------------------
    if plot_image_id not in image_paths:
        print("Image file not found for", plot_image_id)
        return
    with Image.open(image_paths[plot_image_id]) as im:
        display, to_disp = _display_copy(im.convert("RGB"), args.plot_max_display_dim)

    # ---- figure: GT (left) | predictions (right) -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax in axes:
        ax.imshow(display)
        ax.axis("off")

    gt_d = gt_boxes_plot * to_disp
    _draw_boxes(axes[0], gt_d, COLOR_GT, linewidth=1.5)
    axes[0].set_title(f"Ground truth: {len(gt_boxes_plot)} Rumex boxes", fontsize=12)

    score_labels = [f"{v:.2f}" for v in pred_scores] if args.plot_show_scores else None
    _draw_boxes(axes[1], pred_boxes * to_disp, COLOR_PRED, linewidth=1.5, labels=score_labels)
    for role, idx in zip(run["prompt_roles"], run["prompt_indices"]):
        _draw_boxes(axes[1], gt_d[[idx]], ROLE_COLORS[role], linewidth=2.5, linestyle="--")
    axes[1].set_title(
        f"Predictions: {len(pred_boxes)} boxes | prompts {run['Prompt_ID']}\n"
        f"AP50={row['AP50']:.3f}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
        f"F1={row['F1']:.3f}  TP={int(row['TP'])} FP={int(row['FP'])} FN={int(row['FN'])}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color=COLOR_GT, lw=2, label="ground truth"),
        Line2D([0], [0], color=COLOR_PRED, lw=2, label="prediction"),
    ] + [Line2D([0], [0], color=ROLE_COLORS[r], lw=2.5, linestyle="--",
                label=f"{r} prompt ({ROLE_NAMES[r]})") for r in run["prompt_roles"]]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=11, frameon=False)
    fig.suptitle(
        f"{exp} | {plot_image_id} | mode={mode} | scope={scope_label}\n"
        f"diversity prompts {run['Prompt_ID']} | min pairwise cosine distance "
        f"{run['min_pairwise_distance']:.3f} | {run['n_tiles']} tiles "
        f"(TILE_SIZE={args.tile_size}, OVERLAP={args.overlap}) | "
        f"YOLOE {args.model_name} @ imgsz={args.imgsz} ({PROMPT_MODE}) | "
        f"conf={args.operating_confidence:.2f}, NMS IoU={args.nms_iou_threshold:.2f}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])

    png_path = paths.plots / (f"best_image_{scope_label}_{safe_filename(plot_image_id)}"
                              f"_{mode}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

    print(f"\n[{scope_label}] Selected image : {plot_image_id} "
          f"({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Prompts        : {run['Prompt_ID']}")
    print(f"[{scope_label}] Figure saved   : {png_path}")


def make_qualitative_plots(args, paths: Paths, runs_by_image: dict, image_level_df,
                           plt) -> None:
    """CELL 30 for every scope: one figure per archive plus one global figure."""
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
        plot_best_image(args, paths, runs_by_image, image_level_df, image_paths, plt,
                        scope_label=archive, archive=archive)
    plot_best_image(args, paths, runs_by_image, image_level_df, image_paths, plt,
                    scope_label="ALL", archive=None)


# =============================================================================
#  PHASE 2 - THE WHOLE OFFLINE EVALUATION (notebook CELL 18 ... CELL 30)
# =============================================================================
#  The operating point is frozen and identical to the YOLOE inference threshold:
#  confidence 0.30, NMS IoU 0.40. No sweep is performed.
# =============================================================================

def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2: notebook CELL 18 ... CELL 30, in one process, no GPU."""
    import pandas as pd

    # matplotlib is only needed for the confusion-matrix PNGs and the qualitative
    # figures. If the container does not ship it, every CSV is still written.
    try:
        import matplotlib
        matplotlib.use("Agg")             # headless node: figures are saved, never shown
        import matplotlib.pyplot as plt
        _HAS_MPL = True
    except Exception as exc:
        plt = None
        _HAS_MPL = False
        print(f"WARNING: matplotlib unavailable ({exc}).")
        print("         Confusion-matrix CSVs will still be written; the PNGs and the "
              "qualitative figures will not.")

    exp = args.experiment_name
    conf = args.operating_confidence
    nms_iou = args.nms_iou_threshold
    eval_iou = args.eval_iou_threshold
    ignore_iou = args.prompt_ignore_iou

    print("=" * 92)
    print(f" PHASE 2 - OFFLINE EVALUATION | experiment={exp}")
    print("=" * 92)
    print("Operating configuration is fixed (no sweep performed):")
    print(f"  Confidence threshold = {conf:.2f}  (== the YOLOE inference threshold)")
    print(f"  NMS IoU threshold    = {nms_iou:.2f}")
    print(f"  Evaluation IoU       = {eval_iou:.2f}")

    runs, manifest = load_runs(paths, exp, args.prompt_type)
    if not runs:
        print("Nothing to evaluate.")
        return
    runs_by_image = {r["image_ID"]: r for r in runs}
    print("Images with fewer than K prompts:",
          int(sum(1 for r in runs if len(r["prompt_indices"]) < args.k_exemplars)))

    # =========================================================================
    #  CELL 24 - PER-IMAGE METRICS  (one run = one image)
    # =========================================================================
    # The diversity choice is deterministic, so every image is evaluated exactly
    # once and this single table replaces the run-level + image-level tables of the
    # anchor experiments. Everything is evaluated at the fixed configuration.
    #   AP50 / AP50_95 : all post-NMS predictions, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : predictions >= CONFIDENCE_THRESHOLD
    #
    # Special case (held_out on an image whose every plant was used as a prompt):
    # the metrics are written as NaN and valid_for_macro = False so they are
    # excluded from every mean/std, but TP/FN = 0 and the real FP count are kept,
    # because such a run can still produce false positives that must show up in the
    # pooled counts and in the confusion matrix.
    # =========================================================================
    print("\n--- CELL 24: per-image metrics ---")
    image_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)
        gt = run["gt_boxes"]
        for mode in EVALUATION_MODES:
            eval_gt, prompt_gt = split_gt_for_mode(gt, run["prompt_indices"], mode)

            # ---- AP: every post-NMS prediction (ignored ones removed) ------------
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            ap50, ap5095 = compute_ap([p_det], [g_det])

            # ---- operating point -------------------------------------------------
            ev = evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, conf,
                                             eval_iou, ignore_iou)
            valid = ev["valid_for_macro"]
            nan = float("nan")

            image_rows.append({
                "experiment_name": exp,
                "image_ID": run["image_ID"],
                "archive": run.get("archive", ""),
                "flight": run.get("flight", ""),
                "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"],
                "evaluation_mode": mode,
                "confidence_threshold": conf,
                "nms_iou_threshold": nms_iou,
                "n_gt": len(gt),
                "n_prompt_gt": int(len(run["prompt_indices"])),
                "median_gt_area_px2": round(run["median_gt_area"]),
                "min_pairwise_distance": run["min_pairwise_distance"],
                "mean_pairwise_distance": run["mean_pairwise_distance"],
                "gt_mean_distance": run["gt_mean_distance"],
                "search_mode": run["search_mode"],
                "n_eval_gt": ev["n_eval_gt"],
                "n_tiles": run["n_tiles"],
                "n_predictions_pre_nms": nms_run["n_pre_nms"],
                "n_suppressed_cross_tile": nms_run["n_suppressed_cross_tile"],
                "n_suppressed_same_tile": nms_run["n_suppressed_same_tile"],
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

    image_level_df = pd.DataFrame(image_rows).sort_values(
        ["evaluation_mode", "image_ID"]).reset_index(drop=True)
    image_level_csv = paths.metrics / "image_level_metrics.csv"
    image_level_df.to_csv(image_level_csv, index=False)

    print(f"Per-image metrics: {len(image_level_df)} rows "
          f"({image_level_df['image_ID'].nunique()} images x {len(EVALUATION_MODES)} modes) "
          f"-> {image_level_csv}")
    for mode in EVALUATION_MODES:
        sub = image_level_df[image_level_df["evaluation_mode"] == mode]
        print(f"  {mode:9s}: {int(sub['valid_for_macro'].sum())} images valid for macro "
              f"averaging, AP50_mean={sub['AP50'].mean():.4f}, "
              f"F1_mean={sub['F1'].mean():.4f}")

    # =========================================================================
    #  CELL 25 - EXPERIMENT-LEVEL SUMMARY
    # =========================================================================
    # Mean over images and std BETWEEN images of every per-image metric, so every
    # UAV image contributes exactly the same weight regardless of how many GT boxes
    # it contains.
    # =========================================================================
    print("\n--- CELL 25: experiment-level summary ---")
    summary_rows = []
    for mode in EVALUATION_MODES:
        sub = image_level_df[image_level_df["evaluation_mode"] == mode]
        row = {
            "experiment_name": exp,
            "model": args.model_name,
            "evaluation_mode": mode,
            "prompt_type": args.prompt_type,
            "prompt_mode": PROMPT_MODE,
            "n_exemplars": args.k_exemplars,
            "embedding_model": args.dinov2_model_id,
            "embedding_feature": args.embedding_feature,
            "selection_rule": f"{args.diversity_objective}_{args.diversity_metric}",
            "crop_mode": args.crop_mode,
            "crop_context": args.crop_context,
            "use_tiling": args.use_tiling,
            "tile_size": args.tile_size,
            "tile_overlap": args.overlap,
            "imgsz": args.imgsz,
            "reference_window_size": args.reference_window_size,
            "confidence_threshold": conf,
            "nms_iou_threshold": nms_iou,
            "eval_iou_threshold": eval_iou,
            "n_images": int(sub["image_ID"].nunique()),
            "n_images_valid_for_macro": int(sub["valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()      # spread between images
        summary_rows.append(row)

    experiment_summary_df = pd.DataFrame(summary_rows)
    experiment_summary_csv = paths.metrics / "experiment_summary.csv"
    experiment_summary_df.to_csv(experiment_summary_csv, index=False)

    print(f"Experiment summary -> {experiment_summary_csv}\n")
    print(experiment_summary_df.to_string(index=False))

    # ---- per-archive version of the same table (cluster addition) -----------
    # The notebook ran on ONE archive, so a single summary row per mode was the
    # whole story. Both archives are pooled here, so the same numbers are also
    # written per archive - the pooled rows above stay exactly as the notebook
    # defines them.
    per_archive_rows = []
    for (archive, mode), sub in image_level_df.groupby(["archive", "evaluation_mode"]):
        row = {
            "experiment_name": exp,
            "archive": archive,
            "evaluation_mode": mode,
            "n_images": int(sub["image_ID"].nunique()),
            "n_images_valid_for_macro": int(sub["valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()
        per_archive_rows.append(row)
    per_archive_df = pd.DataFrame(per_archive_rows)
    per_archive_csv = paths.metrics / "experiment_summary_per_archive.csv"
    per_archive_df.to_csv(per_archive_csv, index=False)
    print(f"\nPer-archive summary -> {per_archive_csv}\n")
    if not per_archive_df.empty:
        print(per_archive_df[["archive", "evaluation_mode", "n_images",
                              "AP50_mean", "precision_mean", "recall_mean",
                              "F1_mean"]].to_string(index=False))

    # =========================================================================
    #  CELL 26 - POOLED DATASET AP50 / AP50:95
    # =========================================================================
    # This is NOT the mean of the per-image AP values. All images are handed to
    # supervision at once, so every detection of the whole dataset is ranked in ONE
    # precision-recall curve. Here one episode = one image.
    # =========================================================================
    print("\n--- CELL 26: pooled dataset AP ---")
    dataset_rows = []
    for mode in EVALUATION_MODES:
        pred_list, gt_list, images_used = [], [], set()
        for run in runs:
            nms_run = apply_nms_to_run(run, nms_iou)
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            pred_list.append(p_det)
            gt_list.append(g_det)
            images_used.add(run["image_ID"])
        ap50, ap5095 = compute_ap(pred_list, gt_list)   # one PR curve over ALL images
        dataset_rows.append({
            "experiment_name": exp,
            "evaluation_mode": mode,
            "prompt_type": args.prompt_type,
            "n_images": len(images_used),
            "confidence_used_for_AP": args.threshold,
            "nms_iou_threshold": nms_iou,
            "dataset_AP50": ap50,
            "dataset_AP50_95": ap5095,
        })
        del pred_list, gt_list
        gc.collect()

    dataset_ap_df = pd.DataFrame(dataset_rows)
    dataset_ap_csv = paths.metrics / "dataset_ap_metrics.csv"
    dataset_ap_df.to_csv(dataset_ap_csv, index=False)

    print(f"Dataset pooled AP -> {dataset_ap_csv}\n")
    print(dataset_ap_df.to_string(index=False))
    print("\nFor comparison, the MEAN of the per-image AP50 values "
          "(a different quantity):")
    print(image_level_df.groupby("evaluation_mode")["AP50"].mean().to_string())

    # =========================================================================
    #  CELL 27 - DATASET-LEVEL CONFUSION MATRICES
    # =========================================================================
    # One class (Rumex) plus a background row/column:
    #     Actual Rumex      -> Predicted Rumex      = TP
    #     Actual Rumex      -> Predicted Background = FN  (missed plants)
    #     Actual Background -> Predicted Rumex      = FP  (spurious detections)
    #     Actual Background -> Predicted Background = not defined for detection
    #                                                 (there are no true negatives)
    # Counts are pooled over every image at the fixed configuration. In held_out
    # mode the D1/D2/D3 prompt plants and the detections that were ignored do not
    # appear anywhere.
    # =========================================================================
    print("\n--- CELL 27: confusion matrices ---")
    confusion_summary = []
    for mode in EVALUATION_MODES:
        sub = image_level_df[image_level_df["evaluation_mode"] == mode]
        tp, fp, fn = int(sub["TP"].sum()), int(sub["FP"].sum()), int(sub["FN"].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = safe_f1(precision, recall)

        cm_df = pd.DataFrame(
            [[tp, fn], [fp, np.nan]],
            index=["actual_rumex", "actual_background"],
            columns=["predicted_rumex", "predicted_background"],
        )
        cm_df.to_csv(paths.confusion_matrices / f"confusion_matrix_{mode}.csv")

        if _HAS_MPL:
            plot_confusion_matrix(
                tp, fp, fn,
                f"{exp} - {mode}\ndiversity prompts + tiling | conf={conf:.2f}, "
                f"NMS IoU={nms_iou:.2f}, eval IoU={eval_iou:.2f}",
                paths.confusion_matrices / f"confusion_matrix_{mode}.png", plt)

        confusion_summary.append({
            "experiment_name": exp, "evaluation_mode": mode,
            "TP": tp, "FP": fp, "FN": fn,
            "precision_micro": precision, "recall_micro": recall, "F1_micro": f1,
            "confidence_threshold": conf, "nms_iou_threshold": nms_iou,
            "eval_iou_threshold": eval_iou,
        })
        print(f"{mode:9s}: TP={tp}  FP={fp}  FN={fn}  "
              f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

    confusion_summary_df = pd.DataFrame(confusion_summary)
    confusion_summary_df.to_csv(
        paths.confusion_matrices / "confusion_matrix_summary.csv", index=False)
    print("\nConfusion matrices saved to:", paths.confusion_matrices)

    # =========================================================================
    #  CELL 28 - RECALL PER SIZE GROUP
    # =========================================================================
    # For every image the GT boxes are split at that image's MEDIAN area:
    #   "area <= median" (the smaller half) and "area > median" (the larger half).
    # Matching uses all_gt at the fixed operating point (same matcher as everywhere).
    # The prompt plants themselves are reported separately, because YOLOE usually
    # re-detects them and they would flatter the recall of their size group.
    # NOTE: size is NOT the selection criterion in this experiment - this table
    # shows whether three VISUALLY DIVERSE prompts happen to cover both size halves
    # anyway, and is directly comparable with the same table of
    # k_size_exemplars_no_tiling.
    # =========================================================================
    print("\n--- CELL 28: recall per size group ---")
    size_rows, prompt_rows = [], []
    for run in runs:
        gt = run["gt_boxes"]
        nms_run = apply_nms_to_run(run, nms_iou)
        keep = nms_run["scores"] >= conf
        match = match_one_to_one(nms_run["boxes"][keep], nms_run["scores"][keep], gt,
                                 eval_iou)
        found = match["gt_match_pred"] >= 0

        areas = box_areas(gt)
        is_prompt = np.isin(np.arange(len(gt)), run["prompt_indices"])
        small_half = areas <= run["median_gt_area"]
        for name, grp in [("area <= median", small_half), ("area >  median", ~small_half)]:
            mask = grp & ~is_prompt                      # non-prompt GT boxes only
            size_rows.append({
                "image_ID": run["image_ID"],
                "archive": run.get("archive", ""),
                "flight": run.get("flight", ""),
                "size_group": name,
                "n_gt_non_prompt": int(mask.sum()), "found": int((found & mask).sum()),
                "recall": float((found & mask).sum() / mask.sum()) if mask.sum() else float("nan"),
            })
        for role, idx in zip(run["prompt_roles"], run["prompt_indices"]):
            prompt_rows.append({"image_ID": run["image_ID"],
                                "archive": run.get("archive", ""),
                                "flight": run.get("flight", ""),
                                "role": role,
                                "gt_index": int(idx), "area_px2": float(areas[idx]),
                                "re_detected": bool(found[idx])})

    size_group_df = pd.DataFrame(size_rows)
    prompt_hits_df = pd.DataFrame(prompt_rows)
    size_group_df.to_csv(paths.metrics / "size_group_recall.csv", index=False)
    prompt_hits_df.to_csv(paths.metrics / "prompt_plants_redetected.csv", index=False)

    pooled = (size_group_df.groupby("size_group")[["n_gt_non_prompt", "found"]].sum()
              .assign(recall_pooled=lambda d: d["found"] / d["n_gt_non_prompt"]))
    per_image_mean = (size_group_df.groupby("size_group")["recall"].mean()
                      .rename("recall_mean_over_images"))

    print("Recall of NON-PROMPT GT boxes per size group (all_gt matching):")
    print(pooled.join(per_image_mean).to_string())
    print("\nPrompt plants re-detected (share per role):")
    print(prompt_hits_df.groupby("role")["re_detected"].agg(["count", "sum", "mean"]).to_string())

    # ---- per-archive version of the same two tables (cluster addition) ------
    pooled_archive = (size_group_df.groupby(["archive", "size_group"])
                      [["n_gt_non_prompt", "found"]].sum()
                      .assign(recall_pooled=lambda d: d["found"] / d["n_gt_non_prompt"]))
    per_image_mean_archive = (size_group_df.groupby(["archive", "size_group"])["recall"]
                              .mean().rename("recall_mean_over_images"))
    print("\nSame table per archive:")
    print(pooled_archive.join(per_image_mean_archive).to_string())
    print("\nPrompt plants re-detected per archive and role:")
    print(prompt_hits_df.groupby(["archive", "role"])["re_detected"]
          .agg(["count", "sum", "mean"]).to_string())
    print("\nSaved:", paths.metrics / "size_group_recall.csv", "and",
          paths.metrics / "prompt_plants_redetected.csv")

    # =========================================================================
    #  CELL 30 - QUALITATIVE FIGURES (one per archive + one global)
    # =========================================================================
    if args.no_plots:
        print("\n--- CELL 30: qualitative figures skipped (--no-plots) ---")
    elif not _HAS_MPL:
        print("\n--- CELL 30: qualitative figures skipped (matplotlib unavailable) ---")
    else:
        print("\n--- CELL 30: qualitative GT-vs-prediction figures ---")
        make_qualitative_plots(args, paths, runs_by_image, image_level_df, plt)

    # =========================================================================
    #  CELL 29 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY")
    print("=" * 78)
    print(f"Prompts per image        : {args.k_exemplars} GT boxes selected by DINOv2 "
          f"embedding diversity (max-min cosine distance), no text, no negatives")
    print(f"Embedding model          : {args.dinov2_model_id} "
          f"({args.embedding_feature} token, L2-normalised)")
    print(f"Crops                    : {args.crop_mode} crop, context "
          f"{args.crop_context:.2f}, full resolution")
    print(f"Tiling                   : {args.use_tiling}  (TILE_SIZE={args.tile_size}, "
          f"OVERLAP={args.overlap}, {int(image_level_df['n_tiles'].mean())} tiles/image "
          f"on average)")
    print(f"Detector                 : YOLOE {args.model_name} @ imgsz={args.imgsz}, "
          f"{PROMPT_MODE} (reference windows {args.reference_window_size}px, VPEs averaged), "
          f"in-predictor NMS {args.predict_nms_iou} (permissive)")
    print(f"Confidence threshold     : {conf:.2f} (fixed; YOLOE run once per tile)")
    print(f"NMS IoU threshold        : {nms_iou:.2f} (fixed)")
    _cross = int(image_level_df.drop_duplicates("image_ID")["n_suppressed_cross_tile"].sum())
    _same = int(image_level_df.drop_duplicates("image_ID")["n_suppressed_same_tile"].sum())
    print(f"NMS suppressions         : {_cross} cross-tile + {_same} same-tile duplicates")
    print(f"Evaluation IoU           : {eval_iou:.2f}")
    print(f"Images                   : {len(runs)}")
    print(f"Archives                 : "
          f"{', '.join(sorted(str(a) for a in image_level_df['archive'].dropna().unique()))}")
    _min_d = np.array([r["min_pairwise_distance"] for r in runs], dtype=float)
    _all_d = np.array([r["gt_mean_distance"] for r in runs], dtype=float)
    print(f"Prompt diversity         : min pairwise distance mean={np.nanmean(_min_d):.3f} "
          f"(median={np.nanmedian(_min_d):.3f}), mean distance over ALL GT pairs="
          f"{np.nanmean(_all_d):.3f}")
    print("-" * 78)
    print("EXPERIMENT-LEVEL RESULTS (mean over images, std between images)")
    show = ["evaluation_mode", "AP50_mean", "AP50_std", "AP50_95_mean", "precision_mean",
            "recall_mean", "F1_mean", "F1_std", "IoU1_mean", "IoU2_mean"]
    print(experiment_summary_df[show].to_string(index=False))
    print("-" * 78)
    print("POOLED DATASET AP")
    print(dataset_ap_df[["evaluation_mode", "dataset_AP50",
                         "dataset_AP50_95"]].to_string(index=False))
    print("-" * 78)
    print("POOLED CONFUSION COUNTS")
    print(confusion_summary_df[["evaluation_mode", "TP", "FP", "FN",
                                "precision_micro", "recall_micro",
                                "F1_micro"]].to_string(index=False))
    print("=" * 78)

    print("\nFiles written under", paths.results_root)
    for root, dirs, files in os.walk(paths.results_root):
        depth = root.replace(str(paths.results_root), "").count(os.sep)
        print("  " * depth + os.path.basename(root) + "/")
        if os.path.basename(root) == "raw_detections":
            npz_files = [f for f in files if f.endswith(".npz")]
            for f in sorted(files):
                if not f.endswith(".npz"):
                    print("  " * (depth + 1) + f)
            print("  " * (depth + 1) + f"[{len(npz_files)} image NPZ files]")
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

    # The notebook's CELL 14 self-tests: no GPU, no model, so they guard every mode.
    self_test_selection(k=args.k_exemplars, input_size=args.dinov2_input_size)

    # ---------------- PHASE 2 only -------------------------------------------
    if args.evaluate_only:
        if not _check_supervision():
            print("\nERROR: " + SUPERVISION_HINT)
            sys.exit(2)
        if not _check_pandas():
            print("\nERROR: pandas is required for the evaluation. See "
                  "'Problem B' in k_diverse_exemplars_tiling_HOW_TO_RUN_yoloe.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" K_DIVERSE_EXEMPLARS_TILING YOLOE PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" k_exemplars    : {args.k_exemplars}  ({args.prompt_type} prompts)")
    print(f" selection      : {args.diversity_objective} {args.diversity_metric} diversity of "
          f"{args.dinov2_model_id} {args.embedding_feature} embeddings")
    print(f" crops          : GT box +{args.crop_context:.0%} context -> square padding "
          f"{args.crop_pad_fill} -> {args.dinov2_input_size}x{args.dinov2_input_size}, "
          f"full resolution, center_crop={args.dinov2_center_crop}")
    print(f" dinov2 dtype   : {args.dinov2_dtype}  (batch size {args.dinov2_batch_size})")
    print(f" tiling         : {args.use_tiling}  (tile={args.tile_size}, "
          f"overlap={args.overlap}, cache_tiles={args.cache_tiles_in_memory})")
    print(f" model          : {args.weights} @ imgsz={args.imgsz}  ({PROMPT_MODE})")
    print(f" ref windows    : {args.reference_window_size}px around each of D1/D2/D3 "
          f"(shared when they fall in the same window), VPEs averaged")
    print(f" yoloe threshold: {args.threshold}  (one pass per tile, batch "
          f"size {args.batch_size})")
    print(f" in-pred. NMS   : {args.predict_nms_iou}  (permissive on purpose)")
    print(f" fp16           : {args.use_fp16}  (masks stored: {args.keep_masks})")
    print(f" filters        : min_fill={args.min_fill_ratio}, "
          f"max_area_frac={args.max_area_fraction}, edge_margin={args.edge_margin}")
    print(f" operating pt   : confidence={args.operating_confidence}, "
          f"NMS IoU={args.nms_iou_threshold} (fixed, no sweep)")
    print(f" eval IoU       : {args.eval_iou_threshold}  "
          f"(prompt ignore IoU={args.prompt_ignore_iou})")
    print(f" supervision    : {'available' if has_supervision else 'MISSING'}")
    print("=" * 92)

    # ---------------- dataset -------------------------------------------------
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

    # ---------------- config snapshot ----------------------------------------
    if args.shard_index == 0:
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        config["archives_class_ids"] = {a: ARCHIVES[a] for a in args.archives}
        config["n_images_total"] = len(records)
        config["supervision_available"] = has_supervision
        config["evaluation_modes"] = EVALUATION_MODES
        config["prompt_mode"] = PROMPT_MODE
        (paths.results_root / f"run_config_{exp}.json").write_text(json.dumps(config, indent=2))

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