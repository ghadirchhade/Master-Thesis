#!/usr/bin/env python3
# =============================================================================
#  E01_2_infer_sam3.py
# =============================================================================
#  Cluster (CSCS) port of the Colab notebook  E01_2.ipynb
#
#  E01_2 is the SINGLE-EXEMPLAR tiling experiment: N_EXEMPLARS = 1, so each run
#  shows SAM3 exactly ONE example plant (the anchor itself) instead of three.
#  Everything else - tiling, the exemplar strip, the filters, the offline NMS
#  and every metric - is identical to E02_2, which is what makes the pair
#  "one prompt vs three prompts" a clean comparison. E01_2.ipynb and E02_2.ipynb
#  differ in exactly two configuration lines (EXPERIMENT_NAME and N_EXEMPLARS).
#
#  The notebook is a TWO-PHASE pipeline and this file keeps that separation:
#
#    PHASE 1 - INFERENCE   (notebook CELL 16, GPU)
#        for every image x every anchor:
#            select the exemplars deterministically  (CELL 6)
#            compose  exemplar strip + tile          (CELL 11 + 12)
#            run SAM3 once at SAM3_INFERENCE_THRESHOLD = 0.30 (CELL 14)
#            filter to the tile region + plausibility filter  (CELL 13)
#            save the PRE-NMS detections to NPZ               (CELL 15)
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 17 ... CELL 30, no GPU)
#        load the NPZ files, apply offline NMS at BEST_NMS_IOU = 0.40,
#        evaluate in both modes (all_gt / held_out) at BEST_CONFIDENCE = 0.40,
#        and write run-level / image-level / experiment-level / pooled-AP CSVs
#        plus the confusion matrices (CSV + PNG).
#        Runs in ONE process, after every shard has finished. It never touches
#        SAM3, so it can be repeated as often as you like from the cached NPZs.
#
#  WHAT WAS CHANGED VS THE NOTEBOOK (structure only, never the maths)
#    * Google Drive paths      -> $SCRATCH archive layout, BOTH datasets
#                                 (AGS_Multi_Rumex class_id 0, AgsSpringRumex
#                                 class_id 2), flat annotations_yolo index.
#    * module-level constants  -> argparse flags with the notebook values as
#                                 defaults; the function BODIES are unchanged.
#    * one runs_manifest.csv   -> one manifest per shard (concurrent writers).
#    * USE_FP16 / T4           -> --dtype, default bfloat16 for GH200/H100.
#    * matplotlib inline       -> Agg backend, figures are saved, never shown.
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
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

# UAV orthomosaics are very large; disable PIL's decompression-bomb guard.
# (notebook CELL 2)
Image.MAX_IMAGE_PIXELS = None


# =============================================================================
#  STATIC CONFIGURATION  (dataset layout - from the cluster scripts)
# =============================================================================
# The notebook pointed at ONE folder with RUMEX_CLASS_ID = 0. On the cluster the
# two archives live side by side and use DIFFERENT class ids inside their YOLO
# files, so the class id is a property of the archive, not a global constant.
ARCHIVES: dict[str, int] = {
    "AGS_Multi_Rumex": 0,
    "AgsSpringRumex": 2,
}

IGNORED_ARCHIVES = ("AGS_Multiple_Fields", "AGS_Multiple_Fields_Embeddings")

VALID_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

NON_LABEL_FILES = {"darknet.labels", "classes.txt", "obj.names"}

# ---- notebook CELL 4: manifest of finished inference runs (resume support) ---
MANIFEST_COLUMNS = [
    "experiment_name", "image_ID", "anchor_idx", "Prompt_ID", "Prompt_Type",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt", "n_detections_pre_nms", "n_tiles",
    "image_width", "image_height", "npz_file", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
# held_out : the GT instances used as visual prompts are IGNORED, and so are the
#            predictions that fall on them.

# ---- notebook CELL 25 ------------------------------------------------------
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2"]

# ---- notebook CELL 19 ------------------------------------------------------
STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './E01_2_run_sam3.sh download' on a LOGIN node first, "
    "which installs it into $PYEXTRA."
)


# =============================================================================
#  CLI  -  every notebook CELL 3 parameter, with the notebook value as default
# =============================================================================

def default_dataset_root() -> Path:
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "overney" / "dataset"
    return Path(__file__).resolve().parents[2] / ".." / "02_data" / "dataset"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E01_2 - SAM3 single-exemplar-prompted Rumex detection, tiling ON "
                    "(inference + offline evaluation), cluster port of E01_2.ipynb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------- paths -------------------------------------
    g = p.add_argument_group("paths")
    g.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                   help="Folder holding the archive folders (AGS_Multi_Rumex, AgsSpringRumex).")
    g.add_argument("--output-dir", type=Path, required=True,
                   help="RESULTS_ROOT: raw_detections/, metrics/, confusion_matrices/.")
    g.add_argument("--archives", nargs="*", default=list(ARCHIVES.keys()),
                   help="Subset of archives to run on. Default: both.")

    # ------------------------ experiment identity ---------------------------
    g = p.add_argument_group("experiment identity (CELL 3)")
    g.add_argument("--experiment-name", default="E01_2",
                   help="EXPERIMENT_NAME. Written into every CSV row, every NPZ and "
                        "into the deterministic exemplar seed.")
    g.add_argument("--n-exemplars", type=int, default=1,
                   help="N_EXEMPLARS. E01_2 uses 1 (a single visual prompt: the "
                        "anchor itself). Set 3 to reproduce E02_2's prompt scheme.")
    tiling = g.add_mutually_exclusive_group()
    tiling.add_argument("--tiling", dest="use_tiling", action="store_true", default=True,
                        help="USE_TILING = True (default): overlapping tiles.")
    tiling.add_argument("--no-tiling", dest="use_tiling", action="store_false",
                        help="USE_TILING = False: the whole image is one single tile.")

    # ------------------------------ tiling ----------------------------------
    g = p.add_argument_group("tiling (CELL 3 / CELL 10)")
    g.add_argument("--tile-size", type=int, default=1000, help="TILE_SIZE")
    g.add_argument("--overlap", type=int, default=150, help="OVERLAP")
    g.add_argument("--no-cache-tiles", dest="cache_tiles_in_memory", action="store_false",
                   default=True,
                   help="CACHE_TILES_IN_MEMORY = False: crop tiles on demand (less RAM).")

    # --------------------------- SAM3 inference -----------------------------
    g = p.add_argument_group("sam3 inference (CELL 3 / CELL 5 / CELL 14)")
    g.add_argument("--model-id", default="facebook/sam3",
                   help="HF repo id OR a local snapshot directory.")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="SAM3_INFERENCE_THRESHOLD. SAM3 is executed EXACTLY ONCE per "
                        "(image, anchor) at this score; higher operating thresholds are "
                        "applied offline afterwards.")
    g.add_argument("--mask-threshold", type=float, default=0.40,
                   help="MASK_THRESHOLD (masks are only used transiently).")
    g.add_argument("--batch-size", type=int, default=4,
                   help="BATCH_SIZE: composed tiles per forward pass.")
    g.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16",
                   help="Model dtype. The notebook used float16 on a T4; bfloat16 is the "
                        "cluster default (GH200/H100) with a negligible accuracy change.")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")

    # ------------------------ exemplar strip layout -------------------------
    g = p.add_argument_group("exemplar strip (CELL 3 / CELL 11 / CELL 12)")
    g.add_argument("--strip-margin", type=int, default=6, help="STRIP_MARGIN")
    g.add_argument("--feather-width", type=int, default=8, help="FEATHER_WIDTH")
    g.add_argument("--background-blur-radius", type=float, default=1.5,
                   help="BACKGROUND_BLUR_RADIUS")
    g.add_argument("--max-strip-height-fraction", type=float, default=None,
                   help="MAX_STRIP_HEIGHT_FRACTION. Disabled (None) in the notebook.")

    # --------------------------- filters ------------------------------------
    g = p.add_argument_group("filters (CELL 3 / CELL 13)")
    g.add_argument("--min-fill-ratio", type=float, default=0.15, help="MIN_FILL_RATIO")
    g.add_argument("--max-area-fraction", type=float, default=0.80, help="MAX_AREA_FRACTION")
    g.add_argument("--edge-margin", type=int, default=5, help="EDGE_MARGIN")
    g.add_argument("--tile-region-min-fraction", type=float, default=0.50,
                   help="TILE_REGION_MIN_FRACTION: minimum fraction of a predicted box's "
                        "AREA that must lie inside the real tile region.")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 19 / CELL 22-23)")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")
    g.add_argument("--prompt-ignore-iou", type=float, default=0.50,
                   help="PROMPT_IGNORE_IOU: held_out mode ignore rule.")
    g.add_argument("--best-confidence", type=float, default=0.40,
                   help="BEST_CONFIDENCE (frozen operating point, no sweep).")
    g.add_argument("--best-nms-iou", type=float, default=0.40,
                   help="BEST_NMS_IOU (frozen operating point, no sweep).")

    # ---------------------------- runtime -----------------------------------
    g = p.add_argument_group("runtime")
    g.add_argument("--num-shards", type=int, default=1,
                   help="Split the image list across this many concurrent processes.")
    g.add_argument("--shard-index", type=int, default=0, help="0-based shard of this process.")
    g.add_argument("--limit-images", type=int, default=0,
                   help="Debug: process at most this many images (0 = no limit).")
    g.add_argument("--max-anchors-per-image", type=int, default=0,
                   help="Debug / cost control: use at most this many GT boxes as anchors per "
                        "image (0 = every GT box becomes an anchor once, as in the notebook).")
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

    if args.n_exemplars < 1:
        p.error("--n-exemplars must be >= 1")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.use_tiling and args.overlap >= args.tile_size:
        p.error("--overlap must be smaller than --tile-size")
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")

    # PROMPT_TYPE (CELL 3)
    args.prompt_type = "multiple" if args.n_exemplars > 1 else "single"
    return args


# =============================================================================
#  CELL 4 - OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections (NPZ, one file per image x anchor)
#                          + one runs_manifest_shard<i>.csv per shard
#     metrics/             run / image / experiment / dataset level CSVs
#     confusion_matrices/  CSV + PNG for all_gt and held_out
# =============================================================================

@dataclass
class Paths:
    results_root: Path
    raw_detections: Path
    metrics: Path
    confusion_matrices: Path


def build_paths(output_dir: Path) -> Paths:
    results_root = output_dir
    paths = Paths(
        results_root=results_root,
        raw_detections=results_root / "raw_detections",
        metrics=results_root / "metrics",
        confusion_matrices=results_root / "confusion_matrices",
    )
    for d in (paths.results_root, paths.raw_detections,
              paths.metrics, paths.confusion_matrices):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def manifest_path(paths: Paths, experiment_name: str, shard_index: int) -> Path:
    """
    One manifest PER SHARD. The notebook had a single runs_manifest.csv, which is
    not safe when four processes append to it at the same time; the resume step
    simply reads all of them back (see load_done_runs).
    """
    return paths.raw_detections / f"runs_manifest_{experiment_name}_shard{shard_index}.csv"


# =============================================================================
#  CELL 6 - STABLE REPRODUCIBILITY HELPERS
# =============================================================================
# Python's built-in hash() is randomised per interpreter process (PYTHONHASHSEED),
# so the SAME image + anchor could select DIFFERENT extra exemplars in another
# session. SHA-256 is a fixed mathematical function: the same key always yields
# the same seed on every machine, every Python version, every session - which is
# exactly what is needed when the work is split over four independent processes.
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


def select_exemplar_indices(n_gt: int, anchor_idx: int, n_exemplars: int,
                            image_id: str, experiment_name: str) -> List[int]:
    """
    Choose which GT instances of ONE image are used as visual prompts.

    Input : n_gt        - number of GT boxes in the image
            anchor_idx  - index of the GT box this run is "about" (always a prompt)
            n_exemplars - how many prompts in total (1 or 3)
            image_id    - "<archive>/<flight>/<image name>"
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


def format_prompt_id(exemplar_indices: Sequence[int]) -> str:
    """
    Human-readable id of a prompt set.
      single   -> "5"
      multiple -> "5+12+3"  (the ANCHOR is always the first number)
    """
    return "+".join(str(int(i)) for i in exemplar_indices)


# =============================================================================
#  CELL 7 - DATASET AND YOLO ANNOTATION HELPERS
# =============================================================================
#  The notebook walked ONE images folder and used a single RUMEX_CLASS_ID. On the
#  cluster both archives are pooled into one dataset, each with its own class id
#  and a FLAT annotations_yolo folder, so discover_images() is the archive-aware
#  version from the cluster scripts. load_yolo_boxes / safe_crop / safe_filename
#  are unchanged notebook code.
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


def safe_crop(image: Image.Image, box, min_size: int = 2) -> Image.Image:
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
    """Plain intersection AREA (not IoU) between one box and one region."""
    x1 = max(box[0], region[0]); y1 = max(box[1], region[1])
    x2 = min(box[2], region[2]); y2 = min(box[3], region[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# =============================================================================
#  CELL 9 - CORRECT ONE-TO-ONE MATCHING
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

def match_one_to_one(pred_boxes, pred_scores, gt_boxes, iou_threshold: float) -> dict:
    """
    Input : pred_boxes (P, 4), pred_scores (P,), gt_boxes (G, 4), iou_threshold
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
    pred_iou = np.zeros(n_pred, dtype=np.float32)

    if n_pred == 0 or n_gt == 0:
        return {"pred_match_gt": pred_match_gt, "gt_match_pred": gt_match_pred,
                "pred_iou": pred_iou, "matched_ious": []}

    iou = compute_iou_matrix(pred_boxes, gt_boxes)
    gt_free = np.ones(n_gt, dtype=bool)                 # which GTs are still available

    # 'stable' keeps the original order for equal scores -> fully deterministic.
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32), kind="stable")

    matched_ious = []
    for p in order:
        if not gt_free.any():
            break                                       # every GT already has a prediction
        # Consider ONLY currently unmatched GT boxes (this is the fix).
        candidate_ious = np.where(gt_free, iou[p], -1.0)
        g = int(np.argmax(candidate_ious))              # best FREE GT
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
#  CELL 10 - TILES: GENERATED ONCE PER IMAGE, REUSED BY EVERY ANCHOR
# =============================================================================
# Open the image once -> build the tile list once -> reuse it for every anchor.
# Tiles are kept as CPU/PIL images (never on the GPU).
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


def get_tile_image(tile: dict, full_image: Image.Image) -> Image.Image:
    """Return the tile's PIL image, cropping on demand if it was not cached."""
    if tile["image"] is not None:
        return tile["image"]
    return full_image.crop((tile["x1"], tile["y1"], tile["x2"], tile["y2"]))


# =============================================================================
#  CELL 11 - LOCAL BACKGROUND PATCH
# =============================================================================
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

def get_local_background_patch(tile_img: Image.Image, patch_w: int, patch_h: int,
                               rng, blur_radius: float) -> Image.Image:
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


def make_feather_mask(size: Tuple[int, int], feather_width: int) -> Image.Image:
    """
    Soft alpha mask so a pasted exemplar crop blends into the strip background
    instead of showing a hard rectangular border.

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
#  CELL 12 - COMPOSE  (exemplar strip on top + real tile below)
# =============================================================================
#   canvas width is ALWAYS exactly the tile width, so nothing can remain
#   unpainted next to the tile. If the exemplars do not fit in that width, ALL
#   crops are scaled down by ONE common factor:
#       scale = available_width / total_crop_width
#   Using a single common factor preserves each crop's aspect ratio AND the
#   relative size differences between the exemplars. The UAV tile itself is never
#   resized or distorted. Works for 1 and for 3 exemplars.
# =============================================================================

def compose_tile_with_exemplars(tile_img: Image.Image, crop_images: Sequence[Image.Image],
                                rng, margin: int, feather_width: int,
                                background_blur_radius: float,
                                max_strip_height_fraction: Optional[float]):
    """
    Input : tile_img    - PIL image of the tile
            crop_images - list of PIL exemplar crops (1 for E01_2, 3 for E02_2)
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
#  CELL 13 - TARGET-REGION FILTER + PLAUSIBILITY FILTER
# =============================================================================
# The composed image contains two regions:
#     strip region : y in [0, dy)
#     tile  region : y in [dy, dy + tile_h)
# A detection is kept if at least TILE_REGION_MIN_FRACTION (= 0.50) of its AREA
# lies inside the TILE region, i.e. the box belongs to whichever region holds the
# majority of it. Kept boxes are then CLIPPED to the tile region and shifted into
# tile coordinates. No other heuristic is applied here.
# =============================================================================

def keep_only_target_region_detections(boxes, scores, fill_ratios, offset,
                                       tile_w: int, tile_h: int,
                                       min_fraction_inside: float):
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


def filter_implausible_boxes(boxes, scores, fill_ratios, tile_w: int, tile_h: int,
                             min_fill_ratio: float, max_area_fraction: float,
                             edge_margin: int):
    """
    Remove detections that cannot be a single Rumex plant. The mask-fill ratio is
    received as a PRE-COMPUTED number instead of a stored mask, because masks are
    never kept (KEEP_MASKS = False in the notebook).

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
#  CELL 5 + CELL 14 - SAM3 MODEL, PROCESSOR AND BATCHED INFERENCE
# =============================================================================
#  * tiles are sent in batches of BATCH_SIZE (4) instead of one at a time
#  * torch.inference_mode() + autocast instead of torch.no_grad()
#  * masks are converted into ONE number per detection (the mask-fill ratio used
#    by the plausibility filter) and then deleted immediately. Nothing mask-shaped
#    ever leaves this class -> no GPU/RAM/disk cost.
#  * SAM3 is called ONCE per (image, anchor) at SAM3_INFERENCE_THRESHOLD = 0.30;
#    all higher operating thresholds are applied later, offline.
#
#  Cluster change: the notebook used device_map="auto" (accelerate). Here each
#  shard process sees exactly ONE GPU via CUDA_VISIBLE_DEVICES, so the model is
#  placed explicitly with .to(device) - simpler and it cannot silently offload.
# =============================================================================

def _to_numpy(x) -> np.ndarray:
    """Torch tensor (any device/dtype) or numpy array -> float32 numpy array."""
    import torch
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _fill_ratios_from_masks(boxes_np, masks, binarise_at: float = 0.5) -> np.ndarray:
    """
    Fraction of each predicted box that is actually covered by its mask - the
    single number the plausibility filter needs. Only the small box region is
    materialised; the mask itself is dropped by the caller immediately after.

    Input : boxes_np (N,4) in composed coords,
            masks    (N,H,W) torch tensor OR list of (H,W) masks
    Output: (N,) float32 fill ratios
    """
    import torch
    n = len(boxes_np)
    fills = np.zeros(n, dtype=np.float32)
    if masks is None or n == 0:
        return fills
    for i in range(n):
        m = masks[i]                        # works for a tensor and for a list
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
    """Owns the model + processor and performs one batched forward pass."""

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

    def infer_batch(self, composed_images: Sequence[Image.Image],
                    crop_boxes_batch: Sequence[Sequence[Sequence[float]]],
                    threshold: float):
        """
        Run SAM3 on a batch of composed images (exemplar strip + tile).

        Input : composed_images  - list of PIL images (length <= BATCH_SIZE)
                crop_boxes_batch - list (same length) of lists of exemplar boxes,
                                   each [x1,y1,x2,y2] in that composed image
        Output: list (same length) of (boxes, scores, fill_ratios), all numpy,
                boxes in COMPOSED-image coordinates, every detection with
                score >= threshold. Masks are NOT returned.
        """
        torch = self.torch
        inputs = self.processor(
            images=list(composed_images),
            input_boxes=[[[float(v) for v in b] for b in boxes] for boxes in crop_boxes_batch],
            input_boxes_labels=[[1] * len(boxes) for boxes in crop_boxes_batch],  # 1 = positive
            return_tensors="pt",
        ).to(self.device)

        # match the pixel tensor dtype to the (possibly half precision) weights
        if self.model_dtype != torch.float32 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)

        per_image = []
        with torch.inference_mode():
            if self.model_dtype != torch.float32 and self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=self.model_dtype):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            # target_sizes -> predictions are mapped back to the composed-image size
            # (SAM3 internally resizes the input to its own resolution).
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )

            for res in results:
                boxes = _to_numpy(res["boxes"]).reshape(-1, 4)
                scores = _to_numpy(res["scores"]).reshape(-1)
                fills = _fill_ratios_from_masks(boxes, res.get("masks", None))
                # masks are no longer needed -> free them immediately
                if "masks" in res:
                    res["masks"] = None
                per_image.append((boxes, scores, fills))

        del inputs, outputs, results
        return per_image


def run_anchor_over_tiles(runner: Sam3Runner, tile_cache: List[dict],
                          full_image: Image.Image, exemplar_crops: Sequence[Image.Image],
                          run_seed: int, args) -> dict:
    """
    Run SAM3 for ONE anchor/exemplar configuration over ALL cached tiles.

    Input : tile_cache     - list of tile dicts (CELL 10), built once per image
            full_image     - the open PIL image (used only if tiles are not cached)
            exemplar_crops - list of PIL crops used as visual prompts
            run_seed       - deterministic seed for the strip background sampling
    Output: dict of numpy arrays, all in ORIGINAL-IMAGE coordinates:
            boxes (N,4), scores (N,), fill_ratio (N,), tile_id (N,),
            tile_boxes (N,4)  <- the tile each detection came from
            These are the PRE-NMS detections (no confidence filtering beyond 0.30,
            no NMS) that get written to disk for the offline evaluation.
    """
    all_boxes, all_scores, all_fills, all_tids, all_tboxes = [], [], [], [], []

    for start in range(0, len(tile_cache), args.batch_size):
        batch_tiles = tile_cache[start:start + args.batch_size]

        composed_images, crop_boxes_batch, offsets, sizes = [], [], [], []
        for tile in batch_tiles:
            tile_img = get_tile_image(tile, full_image)
            # deterministic RNG per (run, tile) -> identical strip background
            # every time this run is repeated
            rng = np.random.default_rng(stable_seed(run_seed, "tile", tile["tile_id"]))
            composed, crop_boxes, offset = compose_tile_with_exemplars(
                tile_img, exemplar_crops, rng,
                margin=args.strip_margin,
                feather_width=args.feather_width,
                background_blur_radius=args.background_blur_radius,
                max_strip_height_fraction=args.max_strip_height_fraction)
            composed_images.append(composed)
            crop_boxes_batch.append(crop_boxes)
            offsets.append(offset)
            sizes.append((tile_img.width, tile_img.height))

        batch_results = runner.infer_batch(composed_images, crop_boxes_batch, args.threshold)

        for tile, (boxes, scores, fills), offset, (tw, th) in zip(
                batch_tiles, batch_results, offsets, sizes):
            # 1) keep only detections that belong to the real tile, remap to tile coords
            boxes, scores, fills = keep_only_target_region_detections(
                boxes, scores, fills, offset, tw, th, args.tile_region_min_fraction)
            # 2) plausibility filter (unchanged thresholds)
            boxes, scores, fills = filter_implausible_boxes(
                boxes, scores, fills, tw, th,
                args.min_fill_ratio, args.max_area_fraction, args.edge_margin)
            # 3) tile coords -> original image coords
            for b, s, f in zip(boxes, scores, fills):
                all_boxes.append([b[0] + tile["x1"], b[1] + tile["y1"],
                                  b[2] + tile["x1"], b[3] + tile["y1"]])
                all_scores.append(float(s))
                all_fills.append(float(f))
                all_tids.append(int(tile["tile_id"]))
                all_tboxes.append([tile["x1"], tile["y1"], tile["x2"], tile["y2"]])

        del composed_images, crop_boxes_batch, batch_results

    return {
        "boxes": np.array(all_boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.array(all_scores, dtype=np.float32).reshape(-1),
        "fill_ratio": np.array(all_fills, dtype=np.float32).reshape(-1),
        "tile_id": np.array(all_tids, dtype=np.int32).reshape(-1),
        "tile_boxes": np.array(all_tboxes, dtype=np.int32).reshape(-1, 4),
    }


# =============================================================================
#  CELL 15 - PRE-NMS DETECTION STORAGE
# =============================================================================
# For every run (= one image x one anchor) we store the detections AFTER
#   SAM3 inference at 0.30 -> target-region filtering -> plausibility filtering
#   -> conversion to original-image coordinates
# but BEFORE
#   any operating confidence threshold and BEFORE NMS.
#
# That is exactly the state needed to replay any (confidence, NMS IoU) pair
# offline without ever running SAM3 again. Masks are never stored.
# =============================================================================

def run_npz_path(raw_detections_dir: Path, image_id: str, anchor_idx: int) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one run."""
    return raw_detections_dir / f"{safe_filename(image_id)}__anchor{int(anchor_idx):03d}.npz"


def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        anchor_idx: int, detections: dict, gt_boxes: np.ndarray,
                        prompt_indices: Sequence[int], image_size: Tuple[int, int],
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
        boxes=detections["boxes"].astype(np.float32),         # x1,y1,x2,y2 (original img)
        scores=detections["scores"].astype(np.float32),       # confidence >= 0.30
        fill_ratio=detections["fill_ratio"].astype(np.float32),
        tile_id=detections["tile_id"].astype(np.int32),       # which tile produced it
        tile_boxes=detections["tile_boxes"].astype(np.int32), # that tile's extent
    )
    return path


def load_run_detections(path: Path) -> dict:
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
        # archive / flight are cluster additions; tolerate NPZs without them.
        run["archive"] = str(z["archive"]) if "archive" in z else ""
        run["flight"] = str(z["flight"]) if "flight" in z else ""
    return run


# =============================================================================
#  RESUME SUPPORT  (CELL 16, adapted to several shard manifests)
# =============================================================================

def load_done_runs(paths: Paths, experiment_name: str) -> set:
    """
    Read every shard manifest and return {(image_ID, anchor_idx)} of the runs that
    are already finished. Every shard reads ALL manifests, so a resubmission after
    the walltime never repeats work, even if the shard assignment changed because
    --num-gpus was different.
    """
    done: set = set()
    for csv_path in sorted(paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    try:
                        done.add((row["image_ID"], int(row["anchor_idx"])))
                    except (KeyError, ValueError, TypeError):
                        continue            # ignore a half-written trailing row
        except OSError:
            continue
    return done


# =============================================================================
#  CELL 16 - MAIN GPU INFERENCE LOOP  (PHASE 1)
# =============================================================================
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE
#     read its GT boxes ONCE
#     build the overlapping tiles ONCE (cached on CPU)
#     FOR EACH anchor:
#         select the exemplars deterministically
#         crop them from the already-open image
#         run SAM3 over the cached tiles in batches (threshold 0.30)
#         save the PRE-NMS detections (NPZ)
#     release the image and the tile cache
#
# NO NMS and NO metric computation happens here - that is all done offline in
# PHASE 2. The loop is resumable: finished runs are listed in the shard manifests.
# =============================================================================

def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    import torch

    exp = args.experiment_name

    # ---- resume support ------------------------------------------------------
    done_runs: set = set()
    if not args.no_resume:
        done_runs = load_done_runs(paths, exp)
        print(f"Resuming: {len(done_runs)} run(s) already finished for {exp}; skipped.")

    # ---- this shard's manifest ----------------------------------------------
    manifest_csv = manifest_path(paths, exp, args.shard_index)
    manifest_exists = manifest_csv.exists() and manifest_csv.stat().st_size > 0
    manifest_file = open(manifest_csv, "a", newline="")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS,
                                     extrasaction="ignore")
    if not manifest_exists:
        manifest_writer.writeheader()
        manifest_file.flush()

    runner = Sam3Runner(args.model_id, args.device, args.dtype, args.mask_threshold)
    device_is_cuda = runner.device.startswith("cuda")

    start_time = time.time()
    n_new_runs = 0
    image_times: List[float] = []
    n_total_images = len(records)

    for img_idx, rec in enumerate(records, start=1):
        image_t0 = time.time()
        image_id = rec.image_id

        # ---------------- open the original image exactly once ----------------
        image = Image.open(rec.image_path).convert("RGB")
        img_w, img_h = image.size
        gt_boxes = load_yolo_boxes(rec.label_path, img_w, img_h, rec.class_id)
        n_gt = len(gt_boxes)

        if n_gt == 0:
            print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id}: 0 GT boxes, skipped.")
            image.close(); del image; gc.collect()
            continue

        # every GT box is an anchor once, unless --max-anchors-per-image caps it
        n_anchors = n_gt if args.max_anchors_per_image <= 0 else min(n_gt, args.max_anchors_per_image)

        # skip the whole image if every anchor is already done
        if all((image_id, a) in done_runs for a in range(n_anchors)):
            print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id}: all "
                  f"{n_anchors} anchors already done, skipped.")
            image.close(); del image; gc.collect()
            continue

        # ---------------- build the tiles exactly once ------------------------
        tile_cache = build_tile_cache(image, args.use_tiling, args.tile_size,
                                      args.overlap, args.cache_tiles_in_memory)
        n_tiles = len(tile_cache)

        for anchor_idx in range(n_anchors):
            if (image_id, anchor_idx) in done_runs:
                continue
            run_t0 = time.time()

            # deterministic prompt selection (SHA-256 based, see CELL 6)
            exemplar_indices = select_exemplar_indices(n_gt, anchor_idx, args.n_exemplars,
                                                       image_id, exp)
            prompt_id = format_prompt_id(exemplar_indices)
            exemplar_crops = [safe_crop(image, gt_boxes[i]) for i in exemplar_indices]

            run_seed = stable_seed(exp, image_id, anchor_idx, "strip_bg")
            detections = run_anchor_over_tiles(runner, tile_cache, image,
                                               exemplar_crops, run_seed, args)

            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, anchor_idx, detections, gt_boxes,
                exemplar_indices, (img_w, img_h), rec.archive, rec.flight, rec.class_id)

            run_seconds = time.time() - run_t0
            manifest_writer.writerow({
                "experiment_name": exp,
                "image_ID": image_id,
                "anchor_idx": anchor_idx,
                "Prompt_ID": prompt_id,
                "Prompt_Type": args.prompt_type,
                "archive": rec.archive,
                "flight": rec.flight,
                "source_class_id": rec.class_id,
                "n_gt": n_gt,
                "n_prompt_gt": len(exemplar_indices),
                "n_detections_pre_nms": int(len(detections["scores"])),
                "n_tiles": n_tiles,
                "image_width": img_w,
                "image_height": img_h,
                "npz_file": npz_path.name,
                "inference_seconds": round(run_seconds, 2),
            })
            manifest_file.flush()
            n_new_runs += 1

            print(f"  [{exp}] shard{args.shard_index} run #{n_new_runs} | {image_id} | "
                  f"anchor={anchor_idx} ({anchor_idx + 1}/{n_anchors}) | prompt={prompt_id} | "
                  f"tiles={n_tiles} | pre-NMS detections={len(detections['scores'])} | "
                  f"{run_seconds:.1f}s")

            del detections, exemplar_crops
            gc.collect()
            if device_is_cuda:
                torch.cuda.empty_cache()

        # ---------------- release the image and its tile cache ----------------
        for t in tile_cache:
            t["image"] = None
        del tile_cache
        image.close()
        del image
        gc.collect()
        if device_is_cuda:
            torch.cuda.empty_cache()

        image_elapsed = time.time() - image_t0
        image_times.append(image_elapsed)
        avg_per_image = float(np.mean(image_times))
        eta = (n_total_images - img_idx) * avg_per_image
        print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id} done | "
              f"{n_gt} GT box(es) | {image_elapsed:.1f}s | avg/image={avg_per_image:.1f}s | "
              f"ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")

    manifest_file.close()
    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new runs.")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Pre-NMS detections in: {paths.raw_detections}")


# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================

def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading SAM3 ---")
    sample = my_records[:min(len(my_records), 200)]
    total_anchors, per_archive = 0, {}
    for rec in sample:
        with Image.open(rec.image_path) as im:
            w, h = im.size
        n_gt = len(load_yolo_boxes(rec.label_path, w, h, rec.class_id))
        if args.max_anchors_per_image > 0:
            n_gt = min(n_gt, args.max_anchors_per_image)
        total_anchors += n_gt
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + n_gt
    tiles = len(tile_bboxes(8192, 5460, args.tile_size, args.overlap)) if args.use_tiling else 1
    print(f"  sampled {len(sample)} image(s) of this shard -> {total_anchors} anchor runs "
          f"({per_archive})")
    print(f"  tiles per anchor run at 8192x5460: {tiles}")
    print(f"  => ~{total_anchors * tiles} SAM3 forward passes for those {len(sample)} images")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{total_anchors} "
          f"(one per image x anchor)")


# =============================================================================
#  CELL 17 - LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
# =============================================================================
# From here on SAM3 is never touched again. Everything below works on the NPZ
# files written in PHASE 1, so the complete evaluation can be redone in minutes
# on a login node or in a small CPU allocation.
# =============================================================================

def load_runs(paths: Paths, experiment_name: str):
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
    manifest = manifest[manifest["experiment_name"] == experiment_name].copy()
    manifest = manifest.drop_duplicates(subset=["image_ID", "anchor_idx"], keep="last")
    manifest = manifest.sort_values(["image_ID", "anchor_idx"], kind="stable")

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

    print(f"Loaded {len(runs)} runs "
          f"({manifest['image_ID'].nunique()} images) for {experiment_name}.")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes over all runs:", int(sum(len(r['gt_boxes']) for r in runs)))
    return runs, manifest


# =============================================================================
#  CELL 18 - OFFLINE NMS
# =============================================================================
# Because the same plant is visible in several overlapping tiles, it can be
# detected several times. NMS keeps the highest-scoring box of each overlapping
# group. The NMS IoU threshold is applied offline, so the raw pre-NMS detections
# stay untouched on disk and every value can be replayed.
#
# 'Provenance' = we also record WHICH detection suppressed which, so the
# duplicates coming from a DIFFERENT tile can be counted separately.
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
#  CELL 19 - EVALUATION CORE: all_gt AND held_out
# =============================================================================
# ORDER OF OPERATIONS (this is the agreed protocol):
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
    # start everything as FP, then valid matches become TP, then the leftovers
    # sitting on a prompt plant become IGNORED; whatever remains stays FP.
    status = np.full(n_pred, STATUS_FP, dtype=np.int8)

    # step 1 - corrected one-to-one matching against the evaluated GT
    match = match_one_to_one(pred_boxes, pred_scores, eval_gt_boxes, eval_iou)
    status[match["pred_match_gt"] >= 0] = STATUS_TP

    # step 2 - ignore the leftovers that sit on a PROMPT plant
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
#  CELL 20 - AP50 AND AP50:95  (supervision.metrics.MeanAveragePrecision)
# =============================================================================
#   AP is an area under the precision-recall curve. That curve is produced by
#   walking through ALL detections ordered by confidence, so AP always uses ALL
#   saved predictions with score >= 0.30 (the SAM3 inference threshold), after
#   the selected NMS.
#
#   Precision / recall / F1 / IoU1 / IoU2 describe ONE operating point.
#   Both numbers can appear in the same row - they answer different questions.
# =============================================================================

def make_detections(boxes, scores=None):
    """Convert NumPy boxes into the format expected by supervision."""
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
    All post-NMS predictions with score >= SAM3_INFERENCE_THRESHOLD are used;
    in held_out mode the predictions that were IGNORED (they belong to prompt
    plants) are removed first, exactly like in the operating-point evaluation.
    """
    ev = evaluate_run_predictions(nms_run["boxes"], nms_run["scores"],
                                  eval_gt, prompt_gt, eval_iou, ignore_iou)
    keep = ev["status"] != STATUS_IGNORED
    return (make_detections(nms_run["boxes"][keep], nms_run["scores"][keep]),
            make_detections(eval_gt))


# =============================================================================
#  CELL 21 - OPERATING POINT
# =============================================================================
# Choose one confidence threshold, remove the predictions below it, then send the
# remaining predictions to the CELL 19 evaluator. This is what produces
# Precision, Recall, F1, IoU1 and IoU2 at one operating threshold.
# =============================================================================

def evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, confidence_threshold: float,
                                eval_iou: float, ignore_iou: float) -> dict:
    """
    Input : nms_run - output of apply_nms_to_run (sorted by score, high -> low)
            eval_gt / prompt_gt  - from split_gt_for_mode
            confidence_threshold - the frozen operating point
    Output: the dict of evaluate_run_predictions for the thresholded predictions.
    """
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_run_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                    eval_gt, prompt_gt, eval_iou, ignore_iou)


# =============================================================================
#  CELL 22/23 - OPERATING CONFIGURATION (FROZEN, NO OFFLINE SWEEP)
# =============================================================================
#  BEST_CONFIDENCE and BEST_NMS_IOU are fixed (both 0.40) and used directly by
#  every step below. No confidence x NMS sweep is performed.
# =============================================================================

def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2: notebook CELL 17 ... CELL 30, in one process, no GPU."""
    import pandas as pd

    # matplotlib is only needed for the confusion-matrix PNGs. If the container
    # does not ship it, every CSV is still written and only the PNGs are skipped.
    try:
        import matplotlib
        matplotlib.use("Agg")             # headless node: figures are saved, never shown
        import matplotlib.pyplot as plt
        _HAS_MPL = True
    except Exception as exc:
        plt = None
        _HAS_MPL = False
        print(f"WARNING: matplotlib unavailable ({exc}).")
        print("         Confusion-matrix CSVs will still be written; the PNGs will not.")

    exp = args.experiment_name
    best_conf = args.best_confidence
    best_nms = args.best_nms_iou
    eval_iou = args.eval_iou_threshold
    ignore_iou = args.prompt_ignore_iou

    print("=" * 92)
    print(f" PHASE 2 - OFFLINE EVALUATION | experiment={exp}")
    print("=" * 92)
    print("Operating configuration is frozen (no sweep performed):")
    print(f"  Confidence threshold = {best_conf:.2f}")
    print(f"  NMS IoU threshold    = {best_nms:.2f}")
    print(f"  Evaluation IoU       = {eval_iou:.2f}")

    runs, manifest = load_runs(paths, exp)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  CELL 24 - RUN-LEVEL METRICS  (one run = one image x one anchor/prompt set)
    # =========================================================================
    #   AP50 / AP50_95 : all post-NMS predictions >= 0.30, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : only predictions >= BEST_CONFIDENCE
    #
    # Special case (held_out with no evaluable GT, i.e. every plant of the image
    # was used as a prompt): the metrics are written as NaN and valid_for_macro is
    # False so they are excluded from every mean/std, but TP/FN = 0 and the real FP
    # count are kept, because such a run can still produce false positives that
    # must show up in the pooled counts and in the confusion matrix.
    # =========================================================================
    print("\n--- CELL 24: run-level metrics ---")
    run_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, best_nms)
        for mode in EVALUATION_MODES:
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)

            # ---- AP: every prediction >= 0.30 after NMS (ignored ones removed) ----
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            ap50, ap5095 = compute_ap([p_det], [g_det])

            # ---- operating point -------------------------------------------------
            ev = evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, best_conf,
                                             eval_iou, ignore_iou)
            valid = ev["valid_for_macro"]
            nan = float("nan")

            run_rows.append({
                "experiment_name": exp,
                "image_ID": run["image_ID"],
                "archive": run.get("archive", ""),
                "flight": run.get("flight", ""),
                "anchor_idx": run["anchor_idx"],
                "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"],
                "evaluation_mode": mode,
                "confidence_threshold": best_conf,
                "nms_iou_threshold": best_nms,
                "n_gt_total": int(len(run["gt_boxes"])),
                "n_prompt_gt": int(len(run["prompt_indices"])) if mode == "held_out" else 0,
                "n_eval_gt": ev["n_eval_gt"],
                "n_predictions": ev["n_pred"],
                "n_ignored_predictions": ev["n_ignored"],
                "n_pre_nms": nms_run["n_pre_nms"],
                "n_suppressed_cross_tile": nms_run["n_suppressed_cross_tile"],
                "n_suppressed_same_tile": nms_run["n_suppressed_same_tile"],
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
    run_level_csv = paths.metrics / "run_level_metrics.csv"
    run_level_df.to_csv(run_level_csv, index=False)

    print(f"Run-level metrics: {len(run_level_df)} rows -> {run_level_csv}")
    for mode in EVALUATION_MODES:
        sub = run_level_df[run_level_df["evaluation_mode"] == mode]
        print(f"  {mode:9s}: {len(sub)} runs, "
              f"{int(sub['valid_for_macro'].sum())} valid for macro averaging, "
              f"F1_mean={sub['F1'].mean():.4f}")

    # =========================================================================
    #  CELL 25 - IMAGE-LEVEL METRICS
    # =========================================================================
    # All anchor runs of the same image are averaged into ONE value per image and
    # per evaluation mode. The std here is the spread BETWEEN the different
    # anchor/prompt selections of the SAME image, i.e. "how sensitive is the
    # result to which plant was used as the visual prompt?".
    # NaN rows (held_out runs with no evaluable GT) are ignored by pandas mean/std.
    # std is NaN when an image has only one valid run - that is expected.
    # =========================================================================
    print("\n--- CELL 25: image-level metrics ---")
    image_rows = []
    for (image_id, mode), grp in run_level_df.groupby(["image_ID", "evaluation_mode"]):
        row = {
            "experiment_name": exp,
            "image_ID": image_id,
            "archive": grp["archive"].iloc[0],
            "flight": grp["flight"].iloc[0],
            "evaluation_mode": mode,
            "confidence_threshold": best_conf,
            "nms_iou_threshold": best_nms,
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
    image_level_csv = paths.metrics / "image_level_metrics.csv"
    image_level_df.to_csv(image_level_csv, index=False)

    print(f"Image-level metrics: {len(image_level_df)} rows -> {image_level_csv}")
    print(image_level_df.groupby("evaluation_mode")[
        ["AP50_mean", "precision_mean", "recall_mean", "F1_mean", "IoU1_mean", "IoU2_mean"]
    ].mean().to_string())

    # =========================================================================
    #  CELL 26 - EXPERIMENT-LEVEL SUMMARY
    # =========================================================================
    # Computed from the IMAGE-LEVEL values, not from the raw run rows, so that
    # every UAV image contributes exactly the same weight regardless of how many
    # GT boxes (and therefore how many anchor runs) it contains.
    # The std here is the variation BETWEEN UAV images.
    # =========================================================================
    print("\n--- CELL 26: experiment-level summary ---")
    summary_rows = []
    for mode in EVALUATION_MODES:
        sub = image_level_df[image_level_df["evaluation_mode"] == mode]
        row = {
            "experiment_name": exp,
            "evaluation_mode": mode,
            "prompt_type": args.prompt_type,
            "n_exemplars": args.n_exemplars,
            "use_tiling": args.use_tiling,
            "confidence_threshold": best_conf,
            "nms_iou_threshold": best_nms,
            "eval_iou_threshold": eval_iou,
            "n_images": int(sub["image_ID"].nunique()),
            "n_runs": int(sub["n_runs_total"].sum()),
            "n_runs_valid_for_macro": int(sub["n_runs_valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub[f"{col}_mean"].mean()
            row[f"{col}_std"] = sub[f"{col}_mean"].std()   # spread between images
        summary_rows.append(row)

    experiment_summary_df = pd.DataFrame(summary_rows)
    experiment_summary_csv = paths.metrics / "experiment_summary.csv"
    experiment_summary_df.to_csv(experiment_summary_csv, index=False)

    print(f"Experiment summary -> {experiment_summary_csv}\n")
    print(experiment_summary_df.to_string(index=False))

    # =========================================================================
    #  CELL 27 - POOLED DATASET AP50 / AP50:95
    # =========================================================================
    # This is NOT the mean of the image-level AP values. All runs are handed to
    # supervision as evaluation EPISODES at once, so every detection of the whole
    # dataset is ranked in ONE precision-recall curve.
    #
    # NOTE for the thesis text: because every GT box of an image becomes an anchor
    # once, the same UAV image appears in several episodes (once per prompt set).
    # The pooled AP is therefore computed over "pooled evaluation episodes", not
    # over unique images - it measures the ranking quality of the whole experiment.
    # =========================================================================
    print("\n--- CELL 27: pooled dataset AP ---")
    dataset_rows = []
    for mode in EVALUATION_MODES:
        pred_list, gt_list, images_used = [], [], set()
        for run in runs:
            nms_run = apply_nms_to_run(run, best_nms)
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            pred_list.append(p_det)
            gt_list.append(g_det)
            images_used.add(run["image_ID"])
        ap50, ap5095 = compute_ap(pred_list, gt_list)
        dataset_rows.append({
            "experiment_name": exp,
            "evaluation_mode": mode,
            "n_images": len(images_used),
            "n_runs": len(runs),
            "confidence_used_for_AP": args.threshold,     # AP always uses >= 0.30
            "nms_iou_threshold": best_nms,
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
    print("\nFor comparison, the MEAN of the image-level AP50 values (a different quantity):")
    print(image_level_df.groupby("evaluation_mode")["AP50_mean"].mean().to_string())

    # =========================================================================
    #  CELL 28 - DATASET-LEVEL CONFUSION MATRICES
    # =========================================================================
    #     Actual Rumex      -> Predicted Rumex      = TP
    #     Actual Rumex      -> Predicted Background = FN  (missed plants)
    #     Actual Background -> Predicted Rumex      = FP  (spurious detections)
    #     Actual Background -> Predicted Background = not defined for detection
    #                                                 (there are no true negatives)
    # Counts are pooled over every run at the frozen configuration. In held_out
    # mode the prompt plants and the ignored detections do not appear anywhere.
    # =========================================================================
    print("\n--- CELL 28: confusion matrices ---")

    def plot_confusion_matrix(tp, fp, fn, title, png_path):
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
        cm_df.to_csv(paths.confusion_matrices / f"confusion_matrix_{mode}.csv")

        if _HAS_MPL:
            plot_confusion_matrix(
                tp, fp, fn,
                f"{exp} - {mode}\nconf={best_conf:.2f}, "
                f"NMS IoU={best_nms:.2f}, eval IoU={eval_iou:.2f}",
                paths.confusion_matrices / f"confusion_matrix_{mode}.png")

        confusion_summary.append({
            "experiment_name": exp, "evaluation_mode": mode,
            "TP": tp, "FP": fp, "FN": fn,
            "precision_micro": precision, "recall_micro": recall, "F1_micro": f1,
            "confidence_threshold": best_conf, "nms_iou_threshold": best_nms,
            "eval_iou_threshold": eval_iou,
        })
        print(f"{mode:9s}: TP={tp}  FP={fp}  FN={fn}  "
              f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

    confusion_summary_df = pd.DataFrame(confusion_summary)
    confusion_summary_df.to_csv(
        paths.confusion_matrices / "confusion_matrix_summary.csv", index=False)
    print("\nConfusion matrices saved to:", paths.confusion_matrices)

    # =========================================================================
    #  CELL 30 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY")
    print("=" * 78)
    print(f"Prompts per run          : {args.n_exemplars} ({args.prompt_type})")
    print(f"Tiling                   : {args.use_tiling}  (tile={args.tile_size}px, "
          f"overlap={args.overlap}px)")
    print(f"SAM3 inference threshold : {args.threshold} (executed once per image x anchor)")
    print(f"Selected operating point : confidence={best_conf:.2f}, NMS IoU={best_nms:.2f}")
    print(f"Evaluation IoU           : {eval_iou:.2f}")
    print(f"Runs / images            : {len(runs)} runs over "
          f"{run_level_df['image_ID'].nunique()} images")
    print(f"Archives                 : "
          f"{', '.join(sorted(str(a) for a in run_level_df['archive'].dropna().unique()))}")
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
            print("\nERROR: pandas is required for the evaluation. See "
                  "'Problem B' in E01_2_HOW_TO_RUN.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" E01_2 SAM3 SINGLE-EXEMPLAR PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" n_exemplars    : {args.n_exemplars}  ({args.prompt_type} prompt)")
    print(f" tiling         : {args.use_tiling}  (tile={args.tile_size}, overlap={args.overlap})")
    print(f" sam3 threshold : {args.threshold}  (single inference pass per image x anchor)")
    print(f" mask threshold : {args.mask_threshold}")
    print(f" batch / dtype  : {args.batch_size} / {args.dtype}")
    print(f" operating pt   : confidence={args.best_confidence}, "
          f"NMS IoU={args.best_nms_iou} (frozen, no sweep)")
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
