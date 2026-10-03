#!/usr/bin/env python3
# =============================================================================
#  text_only_tiling_infer_countGDpp.py
# =============================================================================
#
#  This is the TEXT-ONLY TILING experiment with CountGD++ (CVPR 2026): the SAM3
#  notebook text_only_with_tiling, with CountGD++ as the detector.
#
#  Every UAV image is split into overlapping tiles (TILE_SIZE = 1000, OVERLAP =
#  150) and every RAW tile is sent to CountGD++ with ONLY the text prompt
#  TEXT_PROMPT = " rumex obtusifolius " (caption "<text> . "): no exemplar box,
#  no exemplar image of its own, nothing pasted onto the tile. CountGD++ still
#  runs its exemplar branch on an exemplar image; exactly like the official
#  test_dataset.py / app.py for text-only counting, the tile itself is passed as
#  that image with an EMPTY box list, so no exemplar token is created and the
#  prediction depends on the text only.
#
#  There are no visual prompts, so there are no anchors and no prompt plants:
#  ONE run = ONE image x the text prompt, evaluated against ALL GT boxes (all_gt
#  only; held_out does not exist). The GT boxes are used ONLY for evaluation and
#  the figures, never as model input.
#
#  The pipeline is TWO-PHASE:
#
#    PHASE 1 - INFERENCE   (notebook CELL 9 + CELL 10 + CELL 12, GPU)
#        for every image (ONE run per image):
#            build the overlapping tiles ONCE                (CELL 9)
#            run CountGD++ on every tile with the text, ONE
#              tile per forward pass, at CONFIDENCE_THRESHOLD = 0.30
#            clip to the tile + plausibility filter (the CountGD++ filter of
#              the other experiments; the SAM3 notebook has none)
#            save the PRE-NMS detections to NPZ               (CELL 11)
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 13 ... CELL 23, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40,
#        evaluate (all_gt) at CONFIDENCE_THRESHOLD = 0.30, and write the
#        per-image / experiment-level / pooled-AP CSVs, the confusion matrix
#        (CSV + PNG) and the qualitative GT-vs-prediction figures (CELL 23).
#        Runs in ONE process, after every shard has finished. It never touches
#        CountGD++, so it can be repeated as often as you like from the NPZs.
#
#  MODES
#    (default)          sharded inference, then the evaluation
#    --dry-run          discover the dataset, print the cost, exit. No GPU.
#    --no-evaluate      inference only (used by the per-GPU shard processes)
#    --evaluate-only    PHASE 2 only (used once after all shards finished)
# =============================================================================

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import glob
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

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

# ---- manifest of finished inference runs (resume support) --------------------
MANIFEST_COLUMNS = [
    "experiment_name", "image_ID", "Prompt_Type", "text_prompt",
    "archive", "flight", "source_class_id",
    "n_gt", "n_gt_larger_than_overlap", "n_tiles", "n_detections_pre_nms",
    "image_width", "image_height", "npz_file", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
PROMPT_TYPE = "text"
N_EXEMPLARS = 0               # no visual prompts
EVALUATION_MODE = "all_gt"    # every GT box of the image is evaluated
                              # (held_out does not exist: there are no prompt plants)

# ---- notebook CELL 18 ------------------------------------------------------
# count_abs_error = |predicted count - GT count| at the operating point (added
# like in the other CountGD++ experiments, because CountGD++ is a counting model).
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2",
                  "count_abs_error"]

# ---- notebook CELL 15 ------------------------------------------------------
STATUS_FP, STATUS_TP = 0, 1

# ---- CountGD++ post-processing ---------------------------------------------
DOT_TOKEN_ID = 1012   # BERT id of "." - separates the positive prompt from negatives

# ---- CountGD++ input normalisation -----------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # the repo's own normalisation
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './text_only_tiling_run_countGDpp.sh download' on a "
    "LOGIN node, then './text_only_tiling_run_countGDpp.sh build' inside the "
    "container, which installs it into $COUNTGD_PYEXTRA."
)

COUNTGD_HINT = (
    "Run './text_only_tiling_run_countGDpp.sh download' on a LOGIN node (repo + "
    "checkpoint + BERT), then './text_only_tiling_run_countGDpp.sh build' inside the "
    "container (packages + BERT folder + CUDA op). See text_only_tiling_HOW_TO_RUN_countGDpp.md."
)


# =============================================================================
#  CLI  -  every notebook CELL 3 parameter, with the notebook value as default
# =============================================================================

def default_dataset_root() -> Path:
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "overney" / "dataset"
    return Path(__file__).resolve().parents[2] / ".." / "02_data" / "dataset"


def default_countgd_repo() -> Path:
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "CountGDPlusPlus"
    return Path(__file__).resolve().parent / "CountGDPlusPlus"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="text_only_tiling_countGDpp - CountGD++ Rumex detection prompted with TEXT ONLY, "
                    "tiling ON (inference + offline evaluation).",
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
    g.add_argument("--countgd-repo", type=Path, default=default_countgd_repo(),
                   help="Clone of github.com/niki-amini-naieni/CountGDPlusPlus. Holds cfg_app.py, "
                        "checkpoints/bert-base-uncased and the compiled deformable-attention op.")
    g.add_argument("--checkpoint", type=Path, default=None,
                   help="countgd_plusplus.pth. Default: <countgd-repo>/checkpoints/countgd_plusplus.pth")

    # ------------------------ experiment identity ---------------------------
    g = p.add_argument_group("experiment identity (CELL 3)")
    g.add_argument("--experiment-name", default="text_only_tiling_countGDpp",
                   help="EXPERIMENT_NAME. Written into every CSV row and every NPZ.")
    g.add_argument("--text-prompt", default=" rumex obtusifolius ",
                   help="TEXT_PROMPT: the ONLY prompt (caption '<text> . '). Same text as the "
                        "exemplar + text CountGD++ experiments.")
    tiling = g.add_mutually_exclusive_group()
    tiling.add_argument("--tiling", dest="use_tiling", action="store_true", default=True,
                        help="USE_TILING = True (default): overlapping tiles.")
    tiling.add_argument("--no-tiling", dest="use_tiling", action="store_false",
                        help="USE_TILING = False: the whole image is one single tile.")

    # ------------------------------ tiling ----------------------------------
    g = p.add_argument_group("tiling (CELL 3 / CELL 9)")
    g.add_argument("--tile-size", type=int, default=1000, help="TILE_SIZE")
    g.add_argument("--overlap", type=int, default=150,
                   help="OVERLAP. Plants whose width AND height are <= OVERLAP lie completely "
                        "inside at least one tile; the larger ones are counted per image "
                        "(n_gt_larger_than_overlap).")
    g.add_argument("--no-cache-tiles", dest="cache_tiles_in_memory", action="store_false",
                   default=True,
                   help="CACHE_TILES_IN_MEMORY = False: crop tiles on demand (less RAM).")

    # ------------------------- CountGD++ inference --------------------------
    g = p.add_argument_group("countgd++ inference (CELL 3 / CELL 10)")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="CONFIDENCE_THRESHOLD used INSIDE the CountGD++ post-processing "
                        "(replaces the repo default 0.23). CountGD++ is executed EXACTLY ONCE "
                        "per image x tile at this score.")
    g.add_argument("--batch-size", type=int, default=1,
                   help="BATCH_SIZE. Fixed at 1, like every CountGD++ experiment: one tile per "
                        "forward pass.")
    g.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="float32 (CountGD++ is released and evaluated in fp32). float16 = "
                        "autocast, not validated.")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")
    g.add_argument("--model-short-side", type=int, default=800,
                   help="MODEL_SHORT_SIDE: official resize, short side -> 800 px.")
    g.add_argument("--model-max-size", type=int, default=1333,
                   help="MODEL_MAX_SIZE: official resize, long side capped at 1333 px.")
    g.add_argument("--seed", type=int, default=42,
                   help="SEED: same seed as the official inference scripts.")

    # --------------------------- filters ------------------------------------
    g = p.add_argument_group("plausibility filter (kept from the other CountGD++ experiments)")
    g.add_argument("--max-area-fraction", type=float, default=0.80, help="MAX_AREA_FRACTION")
    g.add_argument("--edge-margin", type=int, default=5, help="EDGE_MARGIN")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 15 / CELL 17)")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")
    g.add_argument("--nms-iou-threshold", type=float, default=0.40,
                   help="NMS_IOU_THRESHOLD, applied OFFLINE in PHASE 2 (fixed, no sweep).")
    g.add_argument("--operating-confidence", type=float, default=0.30,
                   help="The operating point of precision / recall / F1 / IoU1 / IoU2 / "
                        "count_abs_error. The notebook keeps it EQUAL to --threshold (0.30), so "
                        "the inference threshold and the operating point are the same single "
                        "value and no confidence x NMS sweep is performed.")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot (CELL 23)")
    g.add_argument("--plot-min-gt-boxes", type=int, default=7,
                   help="PLOT_MIN_GT_BOXES: the plotted image must have at least this many "
                        "GT boxes (with a fallback if none has).")
    g.add_argument("--plot-max-display-dim", type=int, default=2048,
                   help="PLOT_MAX_DISPLAY_DIM: display-only downscale of the figure.")
    g.add_argument("--no-plot-scores", dest="plot_show_scores", action="store_false",
                   default=True,
                   help="PLOT_SHOW_SCORES = False: do not write the confidence next to each "
                        "predicted box.")
    g.add_argument("--no-plots", action="store_true",
                   help="Skip the qualitative figures entirely (the only PHASE 2 step that "
                        "reopens the original images).")

    # ---------------------------- runtime -----------------------------------
    g = p.add_argument_group("runtime")
    g.add_argument("--num-shards", type=int, default=1,
                   help="Split the image list across this many concurrent processes.")
    g.add_argument("--shard-index", type=int, default=0, help="0-based shard of this process.")
    g.add_argument("--limit-images", type=int, default=0,
                   help="Debug: process at most this many images (0 = no limit).")
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

    if not args.text_prompt.strip():
        p.error("--text-prompt is empty: the text is the ONLY prompt of this experiment.")
    if args.batch_size != 1:
        p.error("--batch-size must be 1: one tile per forward pass, like every CountGD++ "
                "experiment.")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.use_tiling and args.overlap >= args.tile_size:
        p.error("--overlap must be smaller than --tile-size")

    if args.checkpoint is None:
        args.checkpoint = args.countgd_repo / "checkpoints" / "countgd_plusplus.pth"

    args.prompt_type = PROMPT_TYPE
    args.n_exemplars = N_EXEMPLARS
    return args


# =============================================================================
#  OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections (NPZ, one file per image)
#                          + one runs_manifest_<exp>_shard<i>.csv per shard
#     metrics/             image / experiment / dataset level CSVs
#     confusion_matrices/  CSV + PNG (all_gt)
#     plots/               qualitative GT-vs-prediction figures
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
    One manifest PER SHARD, so that four processes never append to the same file
    at the same time; the resume step simply reads all of them back
    (see load_done_images).
    """
    return paths.raw_detections / f"runs_manifest_{experiment_name}_shard{shard_index}.csv"


# =============================================================================
#  CELL 6 - DATASET AND YOLO ANNOTATION HELPERS
# =============================================================================
#  The notebook opened ONE image of ONE folder with a single RUMEX_CLASS_ID. On
#  the cluster both archives are pooled into one dataset, each with its own class
#  id and a FLAT annotations_yolo folder, so discover_images() is the
#  archive-aware version from the cluster scripts. load_yolo_boxes is unchanged
#  notebook code.
# =============================================================================

@dataclass(frozen=True)
class ImageRecord:
    """One image plus everything needed to evaluate it."""
    archive: str        # AGS_Multi_Rumex | AgsSpringRumex
    flight: str         # e.g. 20230426_Wallenwil ("" if images/ has no sub-folder)
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
#  CELL 7 - IoU
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


# =============================================================================
#  CELL 8 - CORRECT ONE-TO-ONE MATCHING
# =============================================================================
#   sort predictions by confidence, high -> low
#   for each prediction: take the still-UNMATCHED GT with the highest IoU;
#   match if that IoU >= EVAL_IOU_THRESHOLD. One GT <-> at most one prediction.
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
        # Consider ONLY currently unmatched GT boxes.
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
#  CELL 9 - TILES: GENERATED ONCE PER IMAGE
# =============================================================================
# Open the image once -> build the tile list once -> run the text prompt over it.
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
#  CELL 10 - PLAUSIBILITY FILTER
# =============================================================================
# The SAM3 text-only notebook uses NO plausibility filter. Here the filter of
# all the other CountGD++ experiments is KEPT, so that the CountGD++ experiments
# differ only in their prompt: tiny boxes (<= EDGE_MARGIN px) and boxes covering
# > MAX_AREA_FRACTION (80 %) of the tile are removed. There is no exemplar strip,
# so no target-region filter, and no mask-fill criterion (CountGD++ outputs boxes
# only). The boxes are first clipped to the tile: CountGD++ boxes can extend past
# the tile border.
# =============================================================================

def clip_boxes_to_tile(boxes, tile_w: int, tile_h: int) -> np.ndarray:
    """Clip predicted boxes (tile coordinates) to the tile borders."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, tile_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, tile_h)
    return boxes


def filter_implausible_boxes(boxes, scores, tile_w: int, tile_h: int,
                             max_area_fraction: float, edge_margin: int):
    """
    Remove detections that cannot be a single Rumex plant (same thresholds as the
    SAM3 experiments, without the mask-fill criterion).

    Input : boxes (N,4) in TILE coordinates, scores (N,), tile size
    Output: the surviving boxes (K,4) and scores (K,)
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (bw > edge_margin) & (bh > edge_margin)                      # tiny boxes
    keep &= (bw * bh) / float(tile_w * tile_h) <= max_area_fraction     # huge boxes
    return boxes[keep], scores[keep]


# =============================================================================
#  CELL 10 - COUNTGD++ INPUT: THE TILE (+ the text prompt)
# =============================================================================
# The tile is resized the way the official code does (short side 800, long side
# <= 1333): a 1000 x 1000 tile -> 800 x 800. The predicted boxes are normalised
# to the resized tile, so they are mapped back with the ORIGINAL tile size.
# =============================================================================

def countgd_resize_hw(w: int, h: int, short_side: int,
                      max_size: Optional[int]) -> Tuple[int, int]:
    """
    Output size (new_h, new_w) of the official resize (get_size_with_aspect_ratio
    in datasets/transforms_app.py): short side -> short_side, long side <= max_size.
    """
    size = short_side
    if max_size is not None:
        mn, mx = float(min(w, h)), float(max(w, h))
        if mx / mn * size > max_size:
            size = int(round(max_size * mn / mx))
    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def cxcywh_norm_to_xyxy(boxes, w: float, h: float) -> np.ndarray:
    """Normalised (cx, cy, bw, bh) -> pixel [x1, y1, x2, y2] in a w x h image."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


# =============================================================================
#  CELL 5 + CELL 10 - COUNTGD++ MODEL AND TEXT-ONLY TILE INFERENCE
# =============================================================================
#  The model is built exactly like build_model_and_transforms() in the official
#  test_dataset.py: cfg_app.py (Swin-B backbone, 900 queries, BERT text encoder)
#  + countgd_plusplus.pth loaded with strict=False (as in the official code).
#
#  One forward pass per tile (BATCH_SIZE = 1). Post-processing = the official
#  get_boxes_from_prediction():
#    stage 1: keep queries whose best POSITIVE-token probability > threshold
#             (threshold = CONFIDENCE_THRESHOLD = 0.30 instead of the default 0.23)
#    stage 2: keep queries more similar to the positive than to any negative
#             prompt (there is no negative prompt here, so stage 2 keeps all)
#    score  : the highest token probability of the kept query
#
#  Cluster change: on Colab the notebook chdir'ed into the repo in CELL 1. Here
#  the repo location comes from --countgd-repo, and each shard process sees
#  exactly ONE GPU via CUDA_VISIBLE_DEVICES.
# =============================================================================

def countgd_op_build_dirs(repo_dir: Path) -> List[str]:
    """build/lib* folders of the compiled MultiScaleDeformableAttention op."""
    ops_dir = repo_dir / "models" / "GroundingDINO" / "ops"
    return sorted(glob.glob(str(ops_dir / "build" / "lib*")))


class CountGDRunner:
    """Owns CountGD++ and its input normalisation; performs one forward pass per tile."""

    def __init__(self, args):
        import torch

        self.torch = torch
        self.args = args
        repo_dir = args.countgd_repo.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()

        # ---- make the repo importable (notebook CELL 1, last lines) ----------
        # The compiled op must be on sys.path BEFORE the repo is imported: the
        # repo runs `import MultiScaleDeformableAttention as _C` at import time.
        for lib_dir in countgd_op_build_dirs(repo_dir):
            if lib_dir not in sys.path:
                sys.path.insert(0, lib_dir)
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        # The repo uses relative paths (cfg_app.py, checkpoints/bert-base-uncased),
        # so we work from inside it, exactly like the notebook. Every path used by
        # THIS script is absolute, so changing the working directory is harmless.
        os.chdir(repo_dir)

        from util.slconfig import SLConfig                       # config loader of the repo
        from util.misc import nested_tensor_from_tensor_list     # tensor -> NestedTensor (+ mask)
        import datasets.transforms_app as T                      # the repo's own transforms

        self.nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = args.dtype == "float16" and self.device.startswith("cuda")

        # fixed seeds, as in the official inference script
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # ---- model arguments = config file + device ---------------------------
        cfg = SLConfig.fromfile("cfg_app.py")
        cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
        model_args = argparse.Namespace(device=self.device)
        for k, v in cfg._cfg_dict.to_dict().items():
            setattr(model_args, k, v)

        # ---- deformable attention: compiled CUDA op, or pure-PyTorch fallback ----
        # The repo calls the compiled op whenever the tensors are on the GPU and only
        # uses its pure-PyTorch implementation on the CPU. If the op was not compiled
        # ('run_countGDpp.sh build'), that GPU call crashes with "NameError: name '_C'
        # is not defined". In that case we route the GPU call to the pure-PyTorch
        # implementation of the repo (multi_scale_deformable_attn_pytorch): same
        # computation, runs on the GPU, slower.
        import models.GroundingDINO.ms_deform_attn as msda
        if hasattr(msda, "_C"):
            self.deform_attn_impl = "compiled CUDA op"
        else:
            class _PyTorchDeformAttn:
                """Drop-in replacement of MultiScaleDeformableAttnFunction (inference only)."""
                @staticmethod
                def apply(value, spatial_shapes, level_start_index, sampling_locations,
                          attention_weights, im2col_step):
                    return msda.multi_scale_deformable_attn_pytorch(
                        value, spatial_shapes, sampling_locations, attention_weights)
            msda.MultiScaleDeformableAttnFunction = _PyTorchDeformAttn   # looked up at call time
            self.deform_attn_impl = "pure-PyTorch fallback (slower, same computation)"

        from models.GroundingDINO import groundingdino_app

        print(f"Loading CountGD++ from '{checkpoint}' onto {self.device} ({args.dtype}) ...")
        model, _, _ = groundingdino_app.build_groundingdino(model_args)

        # weights_only=False: since PyTorch 2.6 torch.load defaults to weights_only=True,
        # which refuses this checkpoint because it also stores the training arguments
        # (an argparse.Namespace). The official code was written for torch < 2.6, where
        # False was the default. Safe here: the file is the official CountGD++ release.
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)["model"]
        load_info = model.load_state_dict(state, strict=False)
        del state
        self.model = model.to(self.device).eval()

        # The repo's own normalisation (ImageNet mean / std). Resizing is done by us
        # (the official short side 800 / long side 1333 rule).
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        print("CountGD++ ready.")
        print("  model device    :", next(self.model.parameters()).device)
        print("  deformable attn :", self.deform_attn_impl)
        print("  fp16 autocast   :", self.use_fp16)
        print("  missing keys    :", len(load_info.missing_keys),
              "| unexpected keys:", len(load_info.unexpected_keys))

    # ---------------------------- CELL 10 ------------------------------------
    def prepare_tile_tensor(self, tile_img: Image.Image):
        """
        Resize + normalise one tile.
        Output: tensor (3, H', W') and the scale factor (W'/W, H'/H) that was applied.
        """
        a = self.args
        new_h, new_w = countgd_resize_hw(tile_img.width, tile_img.height,
                                         a.model_short_side, a.model_max_size)
        resized = tile_img.resize((new_w, new_h), Image.BILINEAR)
        tensor, _ = self.normalize(resized, None)
        return tensor, (new_w / tile_img.width, new_h / tile_img.height)

    # ------------------------- post-processing --------------------------------
    @staticmethod
    def postprocess(model_output, threshold: float):
        """
        Official 2-stage filtering of CountGD++ (test_dataset.py), threshold exposed.
        Output: boxes (K,4) normalised cx,cy,w,h in [0,1] and scores (K,), numpy.
        """
        input_ids = model_output["input_ids"][0]
        logits = model_output["pred_logits"].sigmoid()[0]     # (900, n_tokens)
        boxes = model_output["pred_boxes"][0]                 # (900, 4)

        # the first "." closes the positive prompt (the text)
        split_idx = int((input_ids == DOT_TOKEN_ID).nonzero()[0])
        pos_logits = logits[:, :split_idx + 1]
        neg_logits = logits[:, split_idx + 1:]

        keep = pos_logits.max(dim=-1).values > threshold                    # stage 1
        boxes, logits = boxes[keep], logits[keep]
        pos_logits, neg_logits = pos_logits[keep], neg_logits[keep]
        if neg_logits.shape[1] > 0:                                         # stage 2
            keep = pos_logits.max(dim=-1).values > neg_logits.max(dim=-1).values
            boxes, logits = boxes[keep], logits[keep]
        scores = logits.max(dim=-1).values
        return boxes.float().cpu().numpy(), scores.float().cpu().numpy()

    def infer_tile(self, tile_img: Image.Image, tile_t=None):
        """
        Run CountGD++ on ONE tile with the TEXT prompt only.
        Input : tile_img - PIL tile (original resolution)
        Output: boxes (N,4) in TILE pixel coordinates, scores (N,)

        CountGD++'s forward always runs its exemplar branch on an exemplar image.
        For text-only counting the official test_dataset.py / app.py pass the
        INPUT IMAGE ITSELF as that image, with an EMPTY box list
        ({"image": image, "points": []}): with zero boxes no exemplar token is
        created, so the result depends on the text only. The same is done here.
        """
        torch = self.torch
        if tile_t is None:
            tile_t, _ = self.prepare_tile_tensor(tile_img)

        caption = self.args.text_prompt + " . "   # official format: "<positive text> . "
        no_exemplar_boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)

        autocast = (torch.autocast("cuda", dtype=torch.float16) if self.use_fp16
                    else contextlib.nullcontext())
        with torch.inference_mode():
            with autocast:
                out = self.model(
                    self.nested_tensor_from_tensor_list([tile_t.to(self.device)]),  # image to count in
                    self.nested_tensor_from_tensor_list([tile_t.to(self.device)]),  # exemplar image = the tile (unused)
                    [no_exemplar_boxes],                                            # NO exemplar box
                    [],                                                             # no negative exemplar images
                    [],                                                             # no negative exemplar boxes
                    captions=[caption],
                )
            boxes_n, scores = self.postprocess(out, self.args.threshold)
        del out
        # pred_boxes are normalised to the (unpadded) resized tile; the resize keeps
        # the aspect ratio, so multiplying by the ORIGINAL tile size maps them back.
        return cxcywh_norm_to_xyxy(boxes_n, tile_img.width, tile_img.height), scores


def run_text_prompt_over_tiles(runner: CountGDRunner, tile_cache: List[dict],
                               full_image: Image.Image, args) -> dict:
    """
    Run CountGD++ with the text prompt over ALL cached tiles of ONE image
    (notebook CELL 10, run_text_prompt_over_tiles).

    Input : tile_cache - list of tile dicts (CELL 9), built once per image
            full_image - the open PIL image (used only if tiles are not cached)
    Output: dict of numpy arrays, all in ORIGINAL-IMAGE coordinates:
            boxes (N,4), scores (N,), tile_id (N,), tile_boxes (N,4)
            These are the PRE-NMS detections (no confidence filtering beyond 0.30,
            no NMS) that get written to disk for the offline evaluation.
    """
    all_boxes, all_scores, all_tids, all_tboxes = [], [], [], []

    for tile in tile_cache:
        tile_img = get_tile_image(tile, full_image)
        tw, th = tile_img.size
        tile_t, _ = runner.prepare_tile_tensor(tile_img)

        boxes, scores = runner.infer_tile(tile_img, tile_t=tile_t)
        # 1) clip to the tile, 2) plausibility filter (CountGD++ experiments' thresholds)
        boxes = clip_boxes_to_tile(boxes, tw, th)
        boxes, scores = filter_implausible_boxes(boxes, scores, tw, th,
                                                 args.max_area_fraction, args.edge_margin)
        # 3) tile coords -> original image coords
        for b, s in zip(boxes, scores):
            all_boxes.append([b[0] + tile["x1"], b[1] + tile["y1"],
                              b[2] + tile["x1"], b[3] + tile["y1"]])
            all_scores.append(float(s))
            all_tids.append(int(tile["tile_id"]))
            all_tboxes.append([tile["x1"], tile["y1"], tile["x2"], tile["y2"]])

    return {
        "boxes": np.array(all_boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.array(all_scores, dtype=np.float32).reshape(-1),
        "tile_id": np.array(all_tids, dtype=np.int32).reshape(-1),
        "tile_boxes": np.array(all_tboxes, dtype=np.int32).reshape(-1, 4),
    }


# =============================================================================
#  CELL 11 - PRE-NMS DETECTION STORAGE
# =============================================================================
# For every run (= one image x the text prompt) we store the detections AFTER
#   CountGD++ inference at 0.30 -> clipping + plausibility filtering
#   -> conversion to original-image coordinates
# but BEFORE
#   any operating confidence threshold and BEFORE NMS.
#
# That is exactly the state needed to replay any (confidence, NMS IoU) pair
# offline without ever running CountGD++ again.
# =============================================================================

def run_npz_path(raw_detections_dir: Path, image_id: str) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one image (= one run)."""
    return raw_detections_dir / f"{safe_filename(image_id)}.npz"


def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        detections: dict, gt_boxes: np.ndarray, text_prompt: str,
                        image_size: Tuple[int, int], archive: str, flight: str,
                        class_id: int) -> Path:
    """
    Write one image's pre-NMS detections to NPZ. The file is self-contained: it
    also stores the GT boxes, so the whole offline evaluation can run without
    re-opening label files.
    """
    path = run_npz_path(raw_detections_dir, image_id)
    np.savez_compressed(
        path,
        experiment_name=np.array(experiment_name),
        image_ID=np.array(image_id),
        text_prompt=np.array(text_prompt),
        image_width=np.array(int(image_size[0])),
        image_height=np.array(int(image_size[1])),
        archive=np.array(archive),
        flight=np.array(flight),
        source_class_id=np.array(int(class_id)),
        gt_boxes=gt_boxes.astype(np.float32),
        boxes=detections["boxes"].astype(np.float32),         # x1,y1,x2,y2 (original img)
        scores=detections["scores"].astype(np.float32),       # confidence >= 0.30
        tile_id=detections["tile_id"].astype(np.int32),       # which tile produced it
        tile_boxes=detections["tile_boxes"].astype(np.int32), # that tile's extent
    )
    return path


def load_run_detections(path: Path) -> dict:
    """Read one image NPZ back into a plain python dict."""
    with np.load(path, allow_pickle=False) as z:
        run = {
            "image_ID": str(z["image_ID"]),
            "text_prompt": str(z["text_prompt"]),
            "image_width": int(z["image_width"]),
            "image_height": int(z["image_height"]),
            "gt_boxes": z["gt_boxes"].reshape(-1, 4),
            "boxes": z["boxes"].reshape(-1, 4),
            "scores": z["scores"].reshape(-1),
            "tile_id": z["tile_id"].reshape(-1),
            "tile_boxes": z["tile_boxes"].reshape(-1, 4),
            "archive": str(z["archive"]),
            "flight": str(z["flight"]),
        }
    return run


# =============================================================================
#  RESUME SUPPORT  (several shard manifests)
# =============================================================================

def load_done_images(paths: Paths, experiment_name: str, text_prompt: str) -> set:
    """
    Read every shard manifest and return {image_ID} of the images that are already
    finished (one run per image). Every shard reads ALL manifests, so a
    resubmission after the walltime never repeats work, even if the shard
    assignment changed because NUM_GPUS was different.

    Refuses to continue when the manifests already hold runs of ANOTHER text
    prompt (notebook CELL 12): two prompts must never be mixed in one results
    folder.
    """
    done: set = set()
    other_prompts: set = set()
    for csv_path in sorted(paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    # only complete rows (last column present) are checked, so a row
                    # cut short by the walltime cannot trigger a false alarm
                    if row.get("inference_seconds") and row.get("text_prompt") != text_prompt:
                        other_prompts.add(row.get("text_prompt"))
                    if row.get("image_ID"):
                        done.add(row["image_ID"])
        except OSError:
            continue
    if other_prompts:
        raise RuntimeError(
            f"{paths.raw_detections} already holds results for another text prompt "
            f"{sorted(other_prompts)}. Use a new EXPERIMENT_NAME and OUTPUT_DIR for "
            f"{text_prompt!r} (or delete the old results folder) so the runs are not mixed.")
    return done


# =============================================================================
#  CELL 12 - MAIN GPU INFERENCE LOOP  (PHASE 1)
# =============================================================================
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE
#     read its GT boxes ONCE (evaluation only, never fed to CountGD++)
#     build the overlapping tiles ONCE (cached on CPU)
#     run CountGD++ over the cached tiles with the text, one tile per pass
#     save the PRE-NMS detections (NPZ)
#     release the image and the tile cache
#
# Images without any Rumex GT box are skipped, exactly like in the exemplar
# experiments, so every experiment is evaluated on the same set of images. NO NMS
# and NO metric computation happens here - that is all done offline in PHASE 2.
# The loop is resumable: finished images are listed in the shard manifests.
# =============================================================================

def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    exp = args.experiment_name

    # ---- resume support (+ the text-prompt guard, checked even with --no-resume) ----
    done_images: set = load_done_images(paths, exp, args.text_prompt)
    if args.no_resume:
        done_images = set()
    else:
        print(f"Resuming: {len(done_images)} image(s) already finished for {exp} "
              f"(text {args.text_prompt!r}); skipped.")

    # ---- this shard's manifest ----------------------------------------------
    manifest_csv = manifest_path(paths, exp, args.shard_index)
    manifest_exists = manifest_csv.exists() and manifest_csv.stat().st_size > 0
    manifest_file = open(manifest_csv, "a", newline="")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS,
                                     extrasaction="ignore")
    if not manifest_exists:
        manifest_writer.writeheader()
        manifest_file.flush()

    try:
        runner = CountGDRunner(args)
        torch = runner.torch
        device_is_cuda = runner.device.startswith("cuda")

        start_time = time.time()
        n_new_runs = 0
        image_times: List[float] = []
        n_total_images = len(records)

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

            gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
            n_gt_large = int(((gt_w > args.overlap) | (gt_h > args.overlap)).sum())

            # ---------------- tiles: built exactly once ----------------------------
            tile_cache = build_tile_cache(image, args.use_tiling, args.tile_size,
                                          args.overlap, args.cache_tiles_in_memory)
            n_tiles = len(tile_cache)

            detections = run_text_prompt_over_tiles(runner, tile_cache, image, args)
            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, detections, gt_boxes, args.text_prompt,
                (img_w, img_h), rec.archive, rec.flight, rec.class_id)

            n_dets = int(len(detections["scores"]))
            n_tiles_with_dets = int(len(set(detections["tile_id"].tolist())))
            run_seconds = time.time() - image_t0
            manifest_writer.writerow({
                "experiment_name": exp,
                "image_ID": image_id,
                "Prompt_Type": args.prompt_type,
                "text_prompt": args.text_prompt,
                "archive": rec.archive,
                "flight": rec.flight,
                "source_class_id": rec.class_id,
                "n_gt": n_gt,
                "n_gt_larger_than_overlap": n_gt_large,
                "n_tiles": n_tiles,
                "n_detections_pre_nms": n_dets,
                "image_width": img_w,
                "image_height": img_h,
                "npz_file": npz_path.name,
                "inference_seconds": round(run_seconds, 2),
            })
            manifest_file.flush()                  # this image is on disk -> resumable
            n_new_runs += 1

            # ---------------- release the image and its tile cache ----------------
            for t in tile_cache:
                t["image"] = None
            del tile_cache, detections
            image.close()
            del image
            gc.collect()
            if device_is_cuda:
                torch.cuda.empty_cache()

            image_times.append(time.time() - image_t0)
            avg_per_image = float(np.mean(image_times))
            eta = (n_total_images - img_idx) * avg_per_image
            print(f"[{exp}] shard{args.shard_index} ({img_idx}/{n_total_images}) {image_id} | "
                  f"{n_gt} GT box(es) ({n_gt_large} > {args.overlap}px) | "
                  f"pre-NMS detections={n_dets} from {n_tiles_with_dets}/{n_tiles} tiles | "
                  f"{run_seconds:.1f}s | avg/image={avg_per_image:.1f}s | "
                  f"ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")
    finally:
        manifest_file.close()                      # also closed if the loop crashes

    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new images.")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Pre-NMS detections in: {paths.raw_detections}")


# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================

def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading CountGD++ ---")
    sample = my_records[:min(len(my_records), 200)]
    n_runs, n_gt_total, n_gt_large, per_archive = 0, 0, 0, {}
    for rec in sample:
        with Image.open(rec.image_path) as im:
            w, h = im.size
        gt = load_yolo_boxes(rec.label_path, w, h, rec.class_id)
        if len(gt) == 0:
            continue
        n_runs += 1                                     # ONE run per image
        n_gt_total += len(gt)
        n_gt_large += int((((gt[:, 2] - gt[:, 0]) > args.overlap) |
                           ((gt[:, 3] - gt[:, 1]) > args.overlap)).sum())
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + 1
    tiles = len(tile_bboxes(8192, 5460, args.tile_size, args.overlap)) if args.use_tiling else 1
    print(f"  sampled {len(sample)} image(s) of this shard -> {n_runs} runs, ONE per image "
          f"({per_archive})")
    print(f"  GT boxes: {n_gt_total} ({n_gt_large} larger than the {args.overlap} px overlap)")
    print(f"  tiles per run at 8192x5460: {tiles}")
    print(f"  => ~{n_runs * tiles} CountGD++ forward passes (1 tile each) for those "
          f"{len(sample)} images")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{n_runs} (one per image)")


# =============================================================================
#  CELL 13 - LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
# =============================================================================
# From here on CountGD++ is never touched again. Everything below works on the
# NPZ files written in PHASE 1, so the complete evaluation can be redone in
# minutes on a login node or in a small CPU allocation.
# =============================================================================

def load_runs(paths: Paths, experiment_name: str, text_prompt: str):
    import pandas as pd

    manifest_files = sorted(
        paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv"))
    if not manifest_files:
        print(f"No manifest found in {paths.raw_detections} for {experiment_name}.")
        return [], None

    frames = []
    for path in manifest_files:
        try:
            # text_prompt read as a plain string, surrounding spaces kept
            frames.append(pd.read_csv(path, dtype={"text_prompt": str}, keep_default_na=False))
        except Exception as exc:
            print(f"  WARNING: could not read {path.name}: {exc}")
    if not frames:
        print("All manifests unreadable.")
        return [], None

    manifest = pd.concat(frames, ignore_index=True)
    # only the runs of THIS experiment and THIS text prompt (notebook CELL 13)
    manifest = manifest[(manifest["experiment_name"] == experiment_name) &
                        (manifest["text_prompt"].astype(str) == text_prompt)].copy()
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
        run["Prompt_Type"] = str(row["Prompt_Type"])
        run["n_gt_larger_than_overlap"] = int(row["n_gt_larger_than_overlap"])
        run["n_tiles"] = int(row["n_tiles"])
        runs.append(run)

    if n_missing:
        print(f"  WARNING: {n_missing} manifest row(s) point at a missing NPZ (skipped).")

    print(f"Loaded {len(runs)} runs (= images) for {experiment_name}, "
          f"text prompt {text_prompt!r}.")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes:", int(sum(len(r['gt_boxes']) for r in runs)))
    return runs, manifest


# =============================================================================
#  CELL 14 - OFFLINE NMS (with tile provenance)
# =============================================================================
# The same plant is visible in several overlapping tiles, so it can be detected
# several times. NMS keeps the highest-scoring box of each overlapping group and
# records whether the removed duplicate came from another tile or the same tile.
# The NMS IoU threshold is applied offline, so the raw pre-NMS detections stay
# untouched on disk and every value can be replayed.
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
#  CELL 15 - EVALUATION CORE (all_gt)
# =============================================================================
# Text-only prompting uses no GT box as input, so every GT box of the image is
# evaluated (all_gt). The held_out mode of the exemplar experiments does not
# apply here: there is no prompt plant to remove.
#
#   1. match predictions to ALL GT boxes with the corrected one-to-one matcher
#      at EVAL_IOU_THRESHOLD = 0.50
#   2. matched predictions are TP, unmatched predictions are FP,
#      unmatched GT boxes are FN
# =============================================================================

def evaluate_predictions(pred_boxes, pred_scores, gt_boxes, eval_iou: float) -> dict:
    """
    Evaluate ONE prediction set against ALL GT boxes of the image.

    Output dict:
      status      (P,) int  - STATUS_TP / STATUS_FP per prediction
      TP, FP, FN, n_gt, n_pred
      precision, recall, F1
      IoU1 - mean IoU of the MATCHED prediction/GT pairs only
             ("when it finds a plant, how well is it localised?")
      IoU2 - sum of matched IoUs divided by the number of GT boxes
             ("localisation quality over ALL plants, missed ones count as 0")
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    status = np.full(n_pred, STATUS_FP, dtype=np.int8)   # everything FP ...

    match = match_one_to_one(pred_boxes, pred_scores, gt_boxes, eval_iou)
    status[match["pred_match_gt"] >= 0] = STATUS_TP      # ... until it is matched

    tp = int((status == STATUS_TP).sum())
    fp = int((status == STATUS_FP).sum())
    fn = int(n_gt - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    matched_ious = match["matched_ious"]
    return {
        "status": status, "pred_match_gt": match["pred_match_gt"],
        "gt_match_pred": match["gt_match_pred"],
        "TP": tp, "FP": fp, "FN": fn, "n_gt": n_gt, "n_pred": n_pred,
        "precision": float(precision), "recall": float(recall),
        "F1": safe_f1(precision, recall),
        "IoU1": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "IoU2": float(np.sum(matched_ious) / n_gt) if n_gt > 0 else 0.0,
    }


# =============================================================================
#  CELL 16 + CELL 17 - AP50 AND AP50:95 + OPERATING POINT
# =============================================================================
# AP uses ALL post-NMS predictions (>= 0.30, the inference threshold), ranked by
# confidence. Precision / recall / F1 / IoU1 / IoU2 / count_abs_error describe
# ONE operating point: only predictions >= CONFIDENCE_THRESHOLD (identical set
# here, since inference already used 0.30).
# =============================================================================

def make_detections(boxes, scores=None):
    """numpy boxes (+ scores) -> supervision Detections (single class 0)."""
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


def evaluate_at_operating_point(nms_run, gt_boxes, confidence_threshold: float,
                                eval_iou: float) -> dict:
    """Precision / recall / F1 / IoU1 / IoU2 for predictions >= confidence_threshold."""
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                gt_boxes, eval_iou)


# =============================================================================
#  CELL 23 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right)
# =============================================================================
#   yellow = GT boxes | red = predictions
#
# IMAGE SELECTION (per-image AP50):
#   1. keep only images that have at least PLOT_MIN_GT_BOXES (7) GT boxes
#      and take the one with the highest AP50
#   2. if NO image has 7 GT boxes: keep the images with the HIGHEST number of GT
#      boxes and take the one among them with the highest AP50
#   ties are broken by F1, then by the number of GT boxes.
#
# Right panel = post-NMS predictions of that image (merged over all tiles) with
# score >= CONFIDENCE_THRESHOLD, i.e. exactly the set evaluated in CELL 18.
# One figure per archive + one global figure. The image is downscaled for
# DISPLAY only (PLOT_MAX_DISPLAY_DIM).
# =============================================================================

COLOR_GT, COLOR_PRED = "yellow", "red"


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


def _display_copy(image: Image.Image, max_display_dim: int):
    """Downscaled numpy copy of the image + factor mapping full-res boxes onto it."""
    w, h = image.size
    s = min(1.0, max_display_dim / max(w, h))
    disp_w, disp_h = max(1, int(w * s)), max(1, int(h * s))
    display = np.asarray(image.resize((disp_w, disp_h), Image.BILINEAR))
    return display, np.array([disp_w / w, disp_h / h, disp_w / w, disp_h / h], dtype=np.float32)


def select_image_for_plot(image_level_df, min_gt: int, archive: Optional[str] = None):
    """
    Output: (per-image row, selection_rule_text) or (None, reason)
    'archive' restricts the candidates to one archive; None = the whole dataset.
    """
    img_df = image_level_df[image_level_df["AP50"].notna()].copy()
    if archive is not None:
        img_df = img_df[img_df["archive"] == archive]
    if img_df.empty:
        return None, (f"no image with a valid AP50" +
                      (f" for archive={archive}" if archive is not None else ""))

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
    print(ranked[["image_ID", "n_gt", "n_predictions", "AP50", "F1",
                  "precision", "recall"]].head(5).to_string(index=False))
    return ranked.iloc[0], rule


def plot_gt_vs_predictions(args, paths: Paths, runs_by_image: dict, image_level_df,
                           image_paths: dict, plt, scope_label: str,
                           archive: Optional[str] = None) -> None:
    """One qualitative figure for one scope."""
    from matplotlib.lines import Line2D

    exp = args.experiment_name
    row, rule = select_image_for_plot(image_level_df, args.plot_min_gt_boxes, archive)
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

    _draw_boxes(axes[0], gt_boxes_plot * to_disp, COLOR_GT, linewidth=1.5)
    axes[0].set_title(f"Ground truth: {len(gt_boxes_plot)} Rumex boxes", fontsize=12)

    score_labels = [f"{v:.2f}" for v in pred_scores] if args.plot_show_scores else None
    _draw_boxes(axes[1], pred_boxes * to_disp, COLOR_PRED, linewidth=1.5, labels=score_labels)
    axes[1].set_title(
        f"Predictions: {len(pred_boxes)} boxes | text prompt = {run['text_prompt']!r}\n"
        f"AP50={row['AP50']:.3f}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
        f"F1={row['F1']:.3f}  TP={int(row['TP'])} FP={int(row['FP'])} FN={int(row['FN'])}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color=COLOR_GT, lw=2, label="ground truth"),
        Line2D([0], [0], color=COLOR_PRED, lw=2, label="prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=11,
               frameon=False)
    fig.suptitle(
        f"{exp} | {plot_image_id} | CountGD++ text only | mode={EVALUATION_MODE} | "
        f"scope={scope_label}\n"
        f"tile={args.tile_size}px, overlap={args.overlap}px | "
        f"conf={args.operating_confidence:.2f}, NMS IoU={args.nms_iou_threshold:.2f}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    png_path = (paths.plots /
                f"best_image_{scope_label}_{safe_filename(plot_image_id)}_{EVALUATION_MODE}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

    print(f"\n[{scope_label}] Selected image : {plot_image_id} "
          f"({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Figure saved   : {png_path}")


def make_qualitative_plots(args, paths: Paths, runs, image_level_df, plt) -> None:
    """One figure per archive plus one global figure."""
    dataset_root = args.dataset_root.expanduser().resolve()
    try:
        records = discover_images(dataset_root, args.archives)
    except Exception as exc:
        print(f"WARNING: could not discover the images for the plots ({exc}).")
        print("         Every CSV and the confusion matrix are still written.")
        return
    image_paths = {r.image_id: r.image_path for r in records}
    if not image_paths:
        print("WARNING: no image found under --dataset-root; the qualitative plots are skipped.")
        return

    runs_by_image = {r["image_ID"]: r for r in runs}
    archives_present = sorted(str(a) for a in image_level_df["archive"].dropna().unique() if str(a))
    for archive in archives_present:
        plot_gt_vs_predictions(args, paths, runs_by_image, image_level_df,
                               image_paths, plt, scope_label=archive, archive=archive)
    plot_gt_vs_predictions(args, paths, runs_by_image, image_level_df,
                           image_paths, plt, scope_label="ALL", archive=None)


# =============================================================================
#  PHASE 2 - OFFLINE EVALUATION
# =============================================================================

def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2: in one process, no GPU, CountGD++ is never loaded."""
    import pandas as pd

    # matplotlib is only needed for the PNGs. If the container does not ship it,
    # every CSV is still written and only the PNGs are skipped.
    try:
        import matplotlib
        matplotlib.use("Agg")             # headless node: figures are saved, never shown
        import matplotlib.pyplot as plt
        _HAS_MPL = True
    except Exception as exc:
        plt = None
        _HAS_MPL = False
        print(f"WARNING: matplotlib unavailable ({exc}).")
        print("         The confusion-matrix CSV will still be written; the PNGs will not.")

    exp = args.experiment_name
    conf = args.operating_confidence
    nms_iou = args.nms_iou_threshold
    eval_iou = args.eval_iou_threshold
    mode = EVALUATION_MODE

    print("=" * 92)
    print(f" PHASE 2 - OFFLINE EVALUATION | experiment={exp}")
    print("=" * 92)
    print("Operating configuration is fixed (no sweep performed):")
    print(f"  Confidence threshold = {conf:.2f}  (== the CountGD++ inference threshold)")
    print(f"  NMS IoU threshold    = {nms_iou:.2f}")
    print(f"  Evaluation IoU       = {eval_iou:.2f}  (mode {mode})")

    runs, manifest = load_runs(paths, exp, args.text_prompt)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  CELL 18 - PER-IMAGE METRICS  (one run = one image)
    # =========================================================================
    # Text-only prompting has no anchors, so every image is evaluated exactly once
    # and this single table replaces the run-level + image-level tables of the
    # exemplar experiments. Everything is evaluated at the fixed configuration.
    #   AP50 / AP50_95 : all post-NMS predictions >= 0.30, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : only predictions >= CONFIDENCE_THRESHOLD
    #   count_abs_error : |predictions - GT boxes| at the operating point
    # =========================================================================
    print("\n--- per-image metrics ---")
    image_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)

        # ---- AP: every post-NMS prediction ---------------------------------------
        ap50, ap5095 = compute_ap([make_detections(nms_run["boxes"], nms_run["scores"])],
                                  [make_detections(run["gt_boxes"])])

        # ---- operating point ------------------------------------------------------
        ev = evaluate_at_operating_point(nms_run, run["gt_boxes"], conf, eval_iou)

        image_rows.append({
            "experiment_name": exp,
            "image_ID": run["image_ID"],
            "archive": run["archive"],
            "flight": run["flight"],
            "Prompt_Type": run["Prompt_Type"],
            "text_prompt": run["text_prompt"],
            "evaluation_mode": mode,
            "confidence_threshold": conf,
            "nms_iou_threshold": nms_iou,
            "n_gt": ev["n_gt"],
            "n_gt_larger_than_overlap": run["n_gt_larger_than_overlap"],
            "n_tiles": run["n_tiles"],
            "n_predictions_pre_nms": nms_run["n_pre_nms"],
            "n_suppressed_cross_tile": nms_run["n_suppressed_cross_tile"],
            "n_suppressed_same_tile": nms_run["n_suppressed_same_tile"],
            "n_predictions": ev["n_pred"],
            "AP50": ap50,
            "AP50_95": ap5095,
            "precision": ev["precision"],
            "recall": ev["recall"],
            "F1": ev["F1"],
            "IoU1": ev["IoU1"],
            "IoU2": ev["IoU2"],
            "count_abs_error": abs(ev["n_pred"] - ev["n_gt"]),
            "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"],
        })

    image_level_df = pd.DataFrame(image_rows).sort_values("image_ID").reset_index(drop=True)
    image_level_csv = paths.metrics / "image_level_metrics.csv"
    image_level_df.to_csv(image_level_csv, index=False)

    print(f"Per-image metrics: {len(image_level_df)} images -> {image_level_csv}")
    print(f"Images with at least one prediction: "
          f"{int((image_level_df['n_predictions'] > 0).sum())}/{len(image_level_df)}")
    print(image_level_df[METRIC_COLUMNS].mean().to_string())

    # =========================================================================
    #  CELL 19 - EXPERIMENT-LEVEL SUMMARY
    # =========================================================================
    # Mean over images and std BETWEEN images of every per-image metric, so every
    # UAV image contributes exactly the same weight regardless of how many GT
    # boxes it contains.
    # =========================================================================
    print("\n--- experiment-level summary ---")
    summary_row = {
        "experiment_name": exp,
        "evaluation_mode": mode,
        "prompt_type": args.prompt_type,
        "text_prompt": args.text_prompt,
        "n_exemplars": N_EXEMPLARS,
        "use_tiling": args.use_tiling,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "confidence_threshold": conf,
        "nms_iou_threshold": nms_iou,
        "eval_iou_threshold": eval_iou,
        "n_images": int(image_level_df["image_ID"].nunique()),
        "n_images_with_predictions": int((image_level_df["n_predictions"] > 0).sum()),
    }
    for col in METRIC_COLUMNS:
        summary_row[f"{col}_mean"] = image_level_df[col].mean()
        summary_row[f"{col}_std"] = image_level_df[col].std()   # spread between images

    experiment_summary_df = pd.DataFrame([summary_row])
    experiment_summary_csv = paths.metrics / "experiment_summary.csv"
    experiment_summary_df.to_csv(experiment_summary_csv, index=False)

    print(f"Experiment summary -> {experiment_summary_csv}\n")
    print(experiment_summary_df.T.to_string(header=False))

    # ---- per-archive version of the same table ------------------------------
    # Both archives are pooled here, so the same numbers are also written per
    # archive - the pooled row above stays exactly as defined.
    per_archive_rows = []
    for archive, sub_df in image_level_df.groupby("archive"):
        row = {
            "experiment_name": exp,
            "archive": archive,
            "evaluation_mode": mode,
            "n_images": int(sub_df["image_ID"].nunique()),
            "n_images_with_predictions": int((sub_df["n_predictions"] > 0).sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub_df[col].mean()
            row[f"{col}_std"] = sub_df[col].std()
        per_archive_rows.append(row)
    per_archive_df = pd.DataFrame(per_archive_rows)
    per_archive_csv = paths.metrics / "experiment_summary_per_archive.csv"
    per_archive_df.to_csv(per_archive_csv, index=False)
    print(f"\nPer-archive summary -> {per_archive_csv}\n")
    if not per_archive_df.empty:
        print(per_archive_df[["archive", "n_images", "AP50_mean", "precision_mean",
                              "recall_mean", "F1_mean",
                              "count_abs_error_mean"]].to_string(index=False))

    # =========================================================================
    #  CELL 20 - POOLED DATASET AP50 / AP50:95
    # =========================================================================
    # This is NOT the mean of the per-image AP values. All images are handed to
    # supervision at once, so every detection of the whole dataset is ranked in
    # ONE precision-recall curve. Here one episode = one image.
    # =========================================================================
    print("\n--- pooled dataset AP ---")
    pred_list, gt_list = [], []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)
        pred_list.append(make_detections(nms_run["boxes"], nms_run["scores"]))
        gt_list.append(make_detections(run["gt_boxes"]))
    ap50, ap5095 = compute_ap(pred_list, gt_list)   # one PR curve over ALL images
    del pred_list, gt_list
    gc.collect()

    dataset_ap_df = pd.DataFrame([{
        "experiment_name": exp,
        "evaluation_mode": mode,
        "text_prompt": args.text_prompt,
        "n_images": len(runs),
        "confidence_used_for_AP": args.threshold,     # AP always uses >= 0.30
        "nms_iou_threshold": nms_iou,
        "dataset_AP50": ap50,
        "dataset_AP50_95": ap5095,
    }])
    dataset_ap_csv = paths.metrics / "dataset_ap_metrics.csv"
    dataset_ap_df.to_csv(dataset_ap_csv, index=False)

    print(f"Dataset pooled AP -> {dataset_ap_csv}\n")
    print(dataset_ap_df.to_string(index=False))
    print(f"\nFor comparison, the MEAN of the per-image AP50 values "
          f"(a different quantity): {image_level_df['AP50'].mean():.4f}")

    # =========================================================================
    #  CELL 21 - DATASET-LEVEL CONFUSION MATRIX
    # =========================================================================
    #     Actual Rumex      -> Predicted Rumex      = TP
    #     Actual Rumex      -> Predicted Background = FN  (missed plants)
    #     Actual Background -> Predicted Rumex      = FP  (spurious detections)
    #     Actual Background -> Predicted Background = not defined for detection
    #                                                 (there are no true negatives)
    # Counts are pooled over every image at the fixed configuration.
    # =========================================================================
    print("\n--- confusion matrix ---")

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

    tp = int(image_level_df["TP"].sum())
    fp = int(image_level_df["FP"].sum())
    fn = int(image_level_df["FN"].sum())
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
            f"{exp} - {mode}\nprompt={args.text_prompt!r} | conf={conf:.2f}, "
            f"NMS IoU={nms_iou:.2f}, eval IoU={eval_iou:.2f}",
            paths.confusion_matrices / f"confusion_matrix_{mode}.png")

    confusion_summary_df = pd.DataFrame([{
        "experiment_name": exp, "evaluation_mode": mode,
        "TP": tp, "FP": fp, "FN": fn,
        "precision_micro": precision, "recall_micro": recall, "F1_micro": f1,
        "confidence_threshold": conf, "nms_iou_threshold": nms_iou,
        "eval_iou_threshold": eval_iou,
    }])
    confusion_summary_df.to_csv(
        paths.confusion_matrices / "confusion_matrix_summary.csv", index=False)
    print(f"{mode}: TP={tp}  FP={fp}  FN={fn}  "
          f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")
    print("\nConfusion matrix saved to:", paths.confusion_matrices)

    # =========================================================================
    #  CELL 23 - QUALITATIVE FIGURES (one per archive + one global)
    # =========================================================================
    if args.no_plots:
        print("\n--- qualitative figures skipped (--no-plots) ---")
    elif not _HAS_MPL:
        print("\n--- qualitative figures skipped (matplotlib unavailable) ---")
    else:
        print("\n--- qualitative GT-vs-prediction figures ---")
        make_qualitative_plots(args, paths, runs, image_level_df, plt)

    # =========================================================================
    #  CELL 22 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY (CountGD++)")
    print("=" * 78)
    print(f"Prompt                   : text only, {args.text_prompt!r} (caption "
          f"{args.text_prompt + ' . '!r}), no exemplar")
    print(f"Tiling                   : {args.use_tiling}  (tile={args.tile_size}px, "
          f"overlap={args.overlap}px)")
    print(f"Model input              : short side {args.model_short_side}px "
          f"(max {args.model_max_size}px), one tile per forward pass")
    print(f"CountGD++ threshold      : {args.threshold} (executed once per image x tile; "
          f"CountGD++ default is 0.23)")
    print(f"Plausibility filter      : max area fraction {args.max_area_fraction}, edge margin "
          f"{args.edge_margin}px (as in the other CountGD++ experiments)")
    print(f"Operating point          : confidence={conf:.2f} (== the inference "
          f"threshold), NMS IoU={nms_iou:.2f} (both fixed)")
    print(f"Evaluation               : {mode}, IoU >= {eval_iou:.2f}")
    print(f"Images                   : {len(runs)} "
          f"({summary_row['n_images_with_predictions']} with at least one prediction)")
    print(f"Archives                 : "
          f"{', '.join(sorted(str(a) for a in image_level_df['archive'].dropna().unique()))}")
    print("-" * 78)
    print("EXPERIMENT-LEVEL RESULTS (mean over images, std between images)")
    show = ["AP50_mean", "AP50_std", "AP50_95_mean", "precision_mean",
            "recall_mean", "F1_mean", "F1_std", "IoU1_mean", "IoU2_mean",
            "count_abs_error_mean"]
    print(experiment_summary_df[show].to_string(index=False))
    print("-" * 78)
    print("POOLED DATASET AP")
    print(dataset_ap_df[["dataset_AP50", "dataset_AP50_95"]].to_string(index=False))
    print("-" * 78)
    print("POOLED CONFUSION COUNTS")
    print(confusion_summary_df[["TP", "FP", "FN", "precision_micro",
                                "recall_micro", "F1_micro"]].to_string(index=False))
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


def git_commit(repo_dir: Path) -> str:
    """HEAD commit of the CountGD++ clone (recorded in run_config for the thesis appendix)."""
    try:
        git_dir = repo_dir / ".git"
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1].strip()
        if (git_dir / ref).exists():
            return (git_dir / ref).read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except Exception:
        pass
    return "unknown"


def main() -> None:
    args = parse_args()

    # Every path is made absolute HERE: CountGDRunner later chdir's into the repo.
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.countgd_repo = args.countgd_repo.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    paths = build_paths(args.output_dir)
    exp = args.experiment_name

    # ---------------- PHASE 2 only -------------------------------------------
    if args.evaluate_only:
        if not _check_supervision():
            print("\nERROR: " + SUPERVISION_HINT)
            sys.exit(2)
        if not _check_pandas():
            print("\nERROR: pandas is required for the evaluation. See "
                  "'Problem B' in text_only_tiling_HOW_TO_RUN_countGDpp.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" TEXT_ONLY_TILING COUNTGD++ PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {args.dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" countgd_repo   : {args.countgd_repo}")
    print(f" checkpoint     : {args.checkpoint}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" prompt         : TEXT ONLY {args.text_prompt!r} (no exemplar, "
          f"{args.n_exemplars} visual prompts)")
    print(f" tiling         : {args.use_tiling}  (tile={args.tile_size}, overlap={args.overlap})")
    print(f" model input    : short side {args.model_short_side}px (max {args.model_max_size}px)")
    print(f" threshold      : {args.threshold}  (single inference pass per image x tile)")
    print(f" batch / dtype  : {args.batch_size} / {args.dtype}")
    print(f" operating pt   : confidence={args.operating_confidence}, "
          f"NMS IoU={args.nms_iou_threshold} (fixed, no sweep)")
    print(f" filter         : max area fraction {args.max_area_fraction}, "
          f"edge margin {args.edge_margin}px")
    print(f" eval IoU       : {args.eval_iou_threshold}  (mode {EVALUATION_MODE})")
    print(f" supervision    : {'available' if has_supervision else 'MISSING'}")
    print("=" * 92)

    # ---------------- text-prompt guard ---------------------------------------
    # Refuse a different text prompt BEFORE anything is written (config
    # snapshot, manifest, NPZ), so an existing experiment is never touched by a
    # mixed run.
    load_done_images(paths, exp, args.text_prompt)

    # ---------------- dataset -------------------------------------------------
    print("Discovering images ...")
    records = discover_images(args.dataset_root, args.archives)
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
        config["model"] = "CountGD++"
        config["countgd_repo_commit"] = git_commit(args.countgd_repo)
        config["archives_class_ids"] = {a: ARCHIVES[a] for a in args.archives}
        config["n_images_total"] = len(records)
        config["supervision_available"] = has_supervision
        config["evaluation_mode"] = EVALUATION_MODE
        config["countgd_exemplar_input"] = ("text only: the tile itself as exemplar image "
                                            "with an EMPTY box list (official text-only usage)")
        (paths.results_root / f"run_config_{exp}.json").write_text(json.dumps(config, indent=2))

    # ---------------- dry run -------------------------------------------------
    if args.dry_run:
        dry_run(args, records, my_records)
        return

    if not has_supervision:
        print("\nERROR: " + SUPERVISION_HINT)
        sys.exit(2)
    if not (args.countgd_repo / "cfg_app.py").exists():
        print(f"\nERROR: no CountGD++ repository at {args.countgd_repo}. " + COUNTGD_HINT)
        sys.exit(2)
    if not args.checkpoint.exists():
        print(f"\nERROR: checkpoint not found: {args.checkpoint}. " + COUNTGD_HINT)
        sys.exit(2)
    if not (args.countgd_repo / "checkpoints" / "bert-base-uncased").is_dir():
        print(f"\nERROR: {args.countgd_repo}/checkpoints/bert-base-uncased is missing. "
              + COUNTGD_HINT)
        sys.exit(2)

    # ---------------- PHASE 1 -------------------------------------------------
    run_inference(args, paths, my_records)

    # ---------------- PHASE 2 -------------------------------------------------
    if not args.no_evaluate:
        run_evaluation(args, paths)


if __name__ == "__main__":
    main()
