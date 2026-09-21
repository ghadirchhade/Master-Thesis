#!/usr/bin/env python3
# =============================================================================
#  pos_neg_exemplars_text_tiling_extra_path_infer_sam3.py
# =============================================================================
#  Cluster (CSCS) port of the Colab notebook pos_neg_exemplars_text_tiling_extra_path.ipynb
#
#  This experiment is the TILED, POSITIVE + NEGATIVE + TEXT variant with an
#  EXTRA DOWNSAMPLED GLOBAL-CONTEXT PASS:
#  USE_TILING = True. The UAV image is split into overlapping tiles
#  (TILE_SIZE=1536, OVERLAP=384). For each (image, anchor), an exemplar strip
#  is composed above each tile (K_POSITIVES=2 positives, then J_NEGATIVES=3
#  negatives, with local background and feathering). The tile + strip is sent
#  to SAM3 in batches of BATCH_SIZE=4 together with the text prompt
#  TEXT_PROMPT="Rumex obtusifolius".
#  Detections are filtered by target-region (>=50% area in tile) and
#  plausibility (fill ratio, max area, edge margin), then mapped back to full
#  image coords.
#
#  EXTRA PATH (ADD_GLOBAL_CONTEXT_PASS = True):
#  An additional single pass over the whole image, downscaled by GLOBAL_DOWNSCALE=2,
#  is run through the SAME exemplar-strip + SAM3 pipeline. Its detections are
#  rescaled back to full resolution and merged with the tiled detections BEFORE NMS.
#
#  NEGATIVE EXEMPLARS (notebook CELL 8), generated automatically for every run:
#      GT Rumex boxes
#        -> candidate regions (sizes drawn from this image's own GT boxes)
#        -> remove candidates overlapping Rumex (0 overlap, gap >= NEG_MIN_GAP_PX)
#        -> prefer candidates close to Rumex (gap <= NEG_MAX_GAP_FACTOR x median
#           plant size), to avoid picking bare soil
#        -> select J_NEGATIVES spatially diverse ones (farthest-point sampling)
#        -> crop them from the full image -> exemplar strip -> SAM3
#  NOTE: the generator uses ALL GT boxes of the image (that is what guarantees
#  zero overlap), i.e. label information of the evaluated plants.
#
#  The notebook is a TWO-PHASE pipeline and this file keeps that separation:
#
#    PHASE 1 - INFERENCE   (notebook CELL 19, GPU)
#        for every image x every anchor:
#            select the positives deterministically     (CELL 6)
#            generate the negatives deterministically   (CELL 8)
#            build the overlapping tiles                (CELL 11)
#            run SAM3 over the tiles in batches at CONFIDENCE_THRESHOLD=0.30 with
#                positive boxes + negative boxes + text prompt (CELL 15)
#            run the extra downsampled global-context pass (CELL 16)
#            filter by target-region and plausibility   (CELL 14)
#            save the PRE-NMS detections + prompts to NPZ (CELL 18)
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#        A lightweight RAM guard (MEM_STOP_THRESHOLD_PCT) stops the job cleanly
#        if host memory gets too high, allowing safe resumption.
#
#    PHASE 2 - EVALUATION  (notebook CELL 20 ... CELL 32, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40
#        (with cross-tile provenance), evaluate in both modes (all_gt / held_out)
#        at the operating point CONFIDENCE_THRESHOLD = 0.30, and write run-level /
#        image-level / experiment-level / pooled-AP CSVs, the confusion matrices
#        (CSV + PNG) and the qualitative GT-vs-prediction figures.
#        Runs in ONE process, after every shard has finished. It never touches
#        SAM3, so it can be repeated as often as you like from the cached NPZs.
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
Image.MAX_IMAGE_PIXELS = None

# =============================================================================
#  STATIC CONFIGURATION  (dataset layout - from the cluster scripts)
# =============================================================================
# The notebook pointed at ONE folder (AGS_Multi_Rumex) with RUMEX_CLASS_ID = 0.
# On the cluster the two archives live side by side and use DIFFERENT class ids
# inside their YOLO files, so the class id is a property of the archive.
ARCHIVES: dict[str, int] = {
    "AGS_Multi_Rumex": 0,
    "AgsSpringRumex": 2,
}
IGNORED_ARCHIVES = ("AGS_Multiple_Fields", "AGS_Multiple_Fields_Embeddings")
VALID_IMAGE_EXT = (".jpg", ".jpeg", ".png")
NON_LABEL_FILES = {"darknet.labels", "classes.txt", "obj.names"}

# ---- notebook CELL 4: manifest of finished inference runs (resume support) ---
MANIFEST_COLUMNS = [
    "experiment_name", "image_ID", "anchor_idx", "Prompt_ID", "Prompt_Type", "text_prompt",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt", "n_negatives", "neg_max_gap_px", "n_detections_pre_nms", "n_tiles",
    "used_global_pass",
    "image_width", "image_height", "npz_file", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
# held_out : the K_POSITIVES GT instances used as positive prompts are IGNORED, and
#            so are the predictions that fall on them. Answers "how well does SAM3
#            find the REMAINING Rumex plants after being shown K examples?".
#            (Negatives are not GT boxes, so they do not change the evaluation.)

# ---- notebook CELL 27 ------------------------------------------------------
# exit code of a shard that stopped early because of the RAM guard (PHASE 1)
RAM_STOP_EXIT_CODE = 3

METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2"]

# ---- notebook CELL 22 ------------------------------------------------------
STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './pos_neg_exemplars_text_tiling_extra_path_run_sam3.sh download' "
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

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="pos_neg_exemplars_text_tiling_extra_path - SAM3 prompted with K positive GT "
                    "boxes + J generated negative boxes + text, tiling ON + extra global-context pass "
                    "(inference + offline evaluation). Cluster port of "
                    "pos_neg_exemplars_text_tiling_extra_path.ipynb.",
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
    g.add_argument("--experiment-name", default="pos_neg_exemplars_text_tiling_extra_path",
                   help="EXPERIMENT_NAME.")
    g.add_argument("--k-positives", type=int, default=2, help="K_POSITIVES")
    g.add_argument("--j-negatives", type=int, default=3, help="J_NEGATIVES")
    g.add_argument("--text-prompt", default="Rumex obtusifolius", help="TEXT_PROMPT")

    # ------------------------------- tiling ---------------------------------
    g = p.add_argument_group("tiling (CELL 3 / CELL 11)")
    g.add_argument("--use-tiling", type=lambda x: str(x).lower() == "true", default=True,
                   help="USE_TILING")
    g.add_argument("--tile-size", type=int, default=1536, help="TILE_SIZE")
    g.add_argument("--overlap", type=int, default=384, help="OVERLAP")
    g.add_argument("--cache-tiles-in-memory", type=lambda x: str(x).lower() == "true", default=True,
                   help="CACHE_TILES_IN_MEMORY")

    # ----------------------- negative exemplar generator --------------------
    g = p.add_argument_group("negative exemplar generator (CELL 3 / CELL 8)")
    g.add_argument("--neg-num-candidates", type=int, default=2000, help="NEG_NUM_CANDIDATES")
    g.add_argument("--neg-min-gap-px", type=float, default=10, help="NEG_MIN_GAP_PX")
    g.add_argument("--neg-max-gap-factor", type=float, default=1.0, help="NEG_MAX_GAP_FACTOR")

    # --------------------------- SAM3 inference -----------------------------
    g = p.add_argument_group("sam3 inference (CELL 3 / CELL 5 / CELL 15)")
    g.add_argument("--model-id", default="facebook/sam3", help="HF repo id OR local snapshot.")
    g.add_argument("--threshold", type=float, default=0.30, help="CONFIDENCE_THRESHOLD")
    g.add_argument("--mask-threshold", type=float, default=0.40, help="MASK_THRESHOLD")
    g.add_argument("--batch-size", type=int, default=4, help="BATCH_SIZE")
    g.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16",
                   help="Model dtype. Cluster default is bfloat16.")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")

    # ------------------------- exemplar strip layout ------------------------
    g = p.add_argument_group("exemplar strip layout (CELL 3 / CELL 12 / CELL 13)")
    g.add_argument("--strip-margin", type=int, default=6, help="STRIP_MARGIN")
    g.add_argument("--feather-width", type=int, default=8, help="FEATHER_WIDTH")
    g.add_argument("--background-blur-radius", type=float, default=1.5, help="BACKGROUND_BLUR_RADIUS")

    # ------------------- plausibility filter --------------------------------
    g = p.add_argument_group("plausibility filter (CELL 3 / CELL 14)")
    g.add_argument("--min-fill-ratio", type=float, default=0.15, help="MIN_FILL_RATIO")
    g.add_argument("--max-area-fraction", type=float, default=0.80, help="MAX_AREA_FRACTION")
    g.add_argument("--edge-margin", type=int, default=5, help="EDGE_MARGIN")
    g.add_argument("--tile-region-min-fraction", type=float, default=0.50, help="TILE_REGION_MIN_FRACTION")

    # ------------------- optional: downsampled global-context pass ----------
    g = p.add_argument_group("extra path (CELL 3 / CELL 16)")
    g.add_argument("--add-global-context-pass", type=lambda x: str(x).lower() == "true", default=True,
                   help="ADD_GLOBAL_CONTEXT_PASS")
    g.add_argument("--global-downscale", type=int, default=2, help="GLOBAL_DOWNSCALE")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 22 / CELL 24)")
    g.add_argument("--nms-iou-threshold", type=float, default=0.40, help="NMS_IOU_THRESHOLD")
    g.add_argument("--operating-confidence", type=float, default=0.30, help="OPERATING_CONFIDENCE")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50, help="EVAL_IOU_THRESHOLD")
    g.add_argument("--prompt-ignore-iou", type=float, default=0.50, help="PROMPT_IGNORE_IOU")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot (CELL 3 / CELL 32)")
    g.add_argument("--plot-evaluation-mode", default="all_gt", choices=EVALUATION_MODES)
    g.add_argument("--plot-min-gt-boxes", type=int, default=7, help="PLOT_MIN_GT_BOXES")
    g.add_argument("--plot-max-display-dim", type=int, default=2048, help="PLOT_MAX_DISPLAY_DIM")
    g.add_argument("--no-plot-scores", dest="plot_show_scores", action="store_false", default=True)
    g.add_argument("--no-plots", action="store_true", help="Skip CELL 32 entirely.")

    # ---------------------------- runtime -----------------------------------
    g = p.add_argument_group("runtime")
    g.add_argument("--num-shards", type=int, default=1)
    g.add_argument("--shard-index", type=int, default=0)
    g.add_argument("--limit-images", type=int, default=0)
    g.add_argument("--max-anchors-per-image", type=int, default=0)
    g.add_argument("--no-resume", action="store_true")
    g.add_argument("--mem-stop-threshold-pct", type=int, default=70, help="MEM_STOP_THRESHOLD_PCT")

    # ----------------------------- modes ------------------------------------
    g = p.add_argument_group("modes")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--evaluate-only", action="store_true")
    g.add_argument("--no-evaluate", action="store_true")

    args = p.parse_args(argv)

    if args.k_positives < 1: p.error("--k-positives must be >= 1")
    if args.j_negatives < 0: p.error("--j-negatives must be >= 0")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")

    args.n_exemplars = args.k_positives
    args.prompt_type = f"{args.k_positives}pos+{args.j_negatives}neg+text"
    args.experiment_dir_name = args.experiment_name.replace("/", "-")
    args.keep_masks = False

    return args

# =============================================================================
#  CELL 4 - OUTPUT FOLDERS
# =============================================================================
@dataclass
class Paths:
    results_root: Path
    raw_detections: Path
    metrics: Path
    confusion_matrices: Path
    plots: Path

def build_paths(output_dir: Path) -> Paths:
    paths = Paths(
        results_root=output_dir,
        raw_detections=output_dir / "raw_detections",
        metrics=output_dir / "metrics",
        confusion_matrices=output_dir / "confusion_matrices",
        plots=output_dir / "plots",
    )
    for d in (paths.results_root, paths.raw_detections, paths.metrics,
              paths.confusion_matrices, paths.plots):
        d.mkdir(parents=True, exist_ok=True)
    return paths

def manifest_path(paths: Paths, experiment_dir_name: str, shard_index: int) -> Path:
    return paths.raw_detections / f"runs_manifest_{experiment_dir_name}_shard{shard_index}.csv"

# =============================================================================
#  CELL 6 - STABLE REPRODUCIBILITY HELPERS
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
    Choose which GT instances of ONE image are used as POSITIVE visual prompts.
    The anchor is always included; the remaining (n_exemplars - 1) slots are
    filled by sampling without replacement from the other GT boxes of the SAME image.
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
    """Human-readable id of a positive prompt set."""
    return "+".join(str(int(i)) for i in exemplar_indices)

# =============================================================================
#  CELL 7 - DATASET AND YOLO ANNOTATION HELPERS
# =============================================================================
@dataclass(frozen=True)
class ImageRecord:
    archive: str
    flight: str
    image_id: str
    image_path: Path
    label_path: Path
    class_id: int

def _index_flat_labels(annotations_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not annotations_root.is_dir(): return index
    for label_path in sorted(annotations_root.rglob("*.txt")):
        if label_path.name in NON_LABEL_FILES: continue
        stem = label_path.stem
        if stem in index: continue
        index[stem] = label_path
    return index

def discover_images(dataset_root: Path, archives: Iterable[str]) -> List[ImageRecord]:
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
        if not images_root.is_dir(): continue

        label_index = _index_flat_labels(annotations_root)
        n_before = len(records)

        for image_path in sorted(images_root.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXT: continue
            rel = image_path.relative_to(images_root)
            flight = rel.parts[0] if len(rel.parts) > 1 else ""
            stem = image_path.stem
            label_path = label_index.get(stem)
            image_id = f"{archive}/{flight}/{stem}" if flight else f"{archive}/{stem}"
            if label_path is None:
                missing.append(image_id)
                continue
            records.append(ImageRecord(archive, flight, image_id, image_path, label_path, class_id))

    if missing:
        print(f"  WARNING: {len(missing)} image(s) have no matching label file and were skipped.")
    
    records.sort(key=lambda r: r.image_id)
    return records

def load_yolo_boxes(label_path: Path, img_width: int, img_height: int, class_id: int) -> np.ndarray:
    """Read a YOLO .txt annotation file and convert it to pixel corner boxes."""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            try:
                if int(parts[0]) != class_id: continue
                xc, yc, bw, bh = map(float, parts[1:5])
            except (ValueError, IndexError): continue
            xc, yc = xc * img_width, yc * img_height
            bw, bh = bw * img_width, bh * img_height
            boxes.append([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)

def safe_filename(image_id: str) -> str:
    """'folder/name' -> 'folder__name' so it can be used inside a file name."""
    return image_id.replace("/", "__").replace(os.sep, "__")

def safe_crop(image, box, min_size=2):
    """
    Crop an exemplar from the full image, clamped to the image borders and to a
    minimum size. Guards against degenerate/out-of-range YOLO boxes.
    """
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, image.width - min_size))
    y1 = max(0, min(y1, image.height - min_size))
    x2 = min(image.width, max(x2, x1 + min_size))
    y2 = min(image.height, max(y2, y1 + min_size))
    return image.crop((x1, y1, x2, y2))

# =============================================================================
#  CELL 9 - IoU
# =============================================================================
def compute_iou_matrix(boxes1, boxes2) -> np.ndarray:
    """Pairwise IoU between two sets of [x1, y1, x2, y2] boxes."""
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

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
#  CELL 10 - CORRECT ONE-TO-ONE MATCHING
# =============================================================================
def match_one_to_one(pred_boxes, pred_scores, gt_boxes, iou_threshold: float) -> dict:
    """
    Correct one-to-one matching. Sort predictions by confidence, high -> low.
    For each prediction, look ONLY at GT boxes that are still unmatched.
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
    gt_free = np.ones(n_gt, dtype=bool)
    order = np.argsort(-np.asarray(pred_scores, dtype=np.float32), kind="stable")

    matched_ious = []
    for p in order:
        if not gt_free.any(): break
        candidate_ious = np.where(gt_free, iou[p], -1.0)
        g = int(np.argmax(candidate_ious))
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
#  CELL 11 - TILES: GENERATED ONCE PER IMAGE, REUSED BY EVERY ANCHOR
# =============================================================================
def tile_bboxes(img_w, img_h, tile_size, overlap) -> list:
    """Sliding-window tiles covering the whole image with overlap."""
    step = max(1, tile_size - overlap)
    tiles = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
    return list(dict.fromkeys(tiles))

def build_tile_cache(image, use_tiling, tile_size, overlap, cache_in_memory) -> list:
    """Build the tile list for ONE already-open image."""
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
    if tile["image"] is not None: return tile["image"]
    return full_image.crop((tile["x1"], tile["y1"], tile["x2"], tile["y2"]))

# =============================================================================
#  CELL 12 - LOCAL BACKGROUND PATCH
# =============================================================================
def get_local_background_patch(tile_img, patch_w, patch_h, rng, blur_radius):
    """
    Sample a random offset inside the tile, deliberately skipping the top band
    that would sit next to its own copy. Apply a light Gaussian blur.
    """
    tw, th = tile_img.size
    crop_w = min(patch_w, tw)
    crop_h = min(patch_h, th)

    max_x0 = tw - crop_w
    max_y0 = th - crop_h
    min_y0 = min(patch_h, max_y0)

    x0 = int(rng.integers(0, max_x0 + 1)) if max_x0 > 0 else 0
    y0 = int(rng.integers(min_y0, max_y0 + 1)) if max_y0 > min_y0 else max_y0

    patch = tile_img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    if patch.size != (patch_w, patch_h):
        patch = patch.resize((patch_w, patch_h), Image.BILINEAR)
    if blur_radius and blur_radius > 0:
        patch = patch.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return patch

def make_feather_mask(size, feather_width):
    """Soft alpha mask so a pasted exemplar crop blends into the strip background."""
    w, h = size
    mask = np.full((h, w), 255.0, dtype=np.float32)
    effective = min(feather_width, h // 2, w // 2)
    if effective >= 1:
        for i in range(effective):
            alpha = 255.0 * (i + 1) / effective
            mask[i, :] = np.minimum(mask[i, :], alpha)
            mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)
            mask[:, i] = np.minimum(mask[:, i], alpha)
            mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)
    return Image.fromarray(mask.astype(np.uint8), mode="L")

# =============================================================================
#  CELL 13 - COMPOSE (exemplar strip on top + real tile below)
# =============================================================================
def compose_tile_with_exemplars(tile_img, crop_images, rng, margin, feather_width, blur_radius):
    """
    Canvas width is ALWAYS exactly the tile width. If the exemplars do not fit
    in that width, ALL crops are scaled down by ONE common factor.
    """
    n = len(crop_images)
    canvas_w = tile_img.width
    available_w = canvas_w - margin * (n + 1)
    total_crop_w = sum(c.width for c in crop_images)

    scale = 1.0
    if total_crop_w > 0 and available_w > 0 and total_crop_w > available_w:
        scale = available_w / float(total_crop_w)

    if scale < 1.0:
        crop_images = [
            c.resize((max(1, int(round(c.width * scale))),
                      max(1, int(round(c.height * scale)))), Image.BILINEAR)
            for c in crop_images
        ]

    strip_h = max(c.height for c in crop_images) + 2 * margin
    canvas_h = strip_h + tile_img.height

    composed = Image.new("RGB", (canvas_w, canvas_h))
    composed.paste(get_local_background_patch(tile_img, canvas_w, strip_h, rng, blur_radius), (0, 0))

    offset = (0, strip_h)
    composed.paste(tile_img, offset)

    crop_boxes = []
    cursor_x = margin
    for crop in crop_images:
        composed.paste(crop, (cursor_x, margin), make_feather_mask(crop.size, feather_width))
        crop_boxes.append([cursor_x, margin, cursor_x + crop.width, margin + crop.height])
        cursor_x += crop.width + margin

    return composed, crop_boxes, offset

# =============================================================================
#  CELL 14 - TARGET-REGION FILTER + PLAUSIBILITY FILTER
# =============================================================================
def keep_only_target_region_detections(boxes, scores, fill_ratios, offset, tile_w, tile_h, min_fraction_inside):
    """
    A detection is kept if at least min_fraction_inside of its AREA lies inside
    the TILE region. Kept boxes are then CLIPPED to the tile region and shifted
    into tile coordinates.
    """
    dx, dy = offset
    tile_region = (dx, dy, dx + tile_w, dy + tile_h)

    kept_boxes, kept_scores, kept_fills = [], [], []
    for box, score, fill in zip(boxes, scores, fill_ratios):
        x1, y1, x2, y2 = [float(v) for v in box]
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if box_area <= 0: continue

        fraction_inside = intersection_area((x1, y1, x2, y2), tile_region) / box_area
        if fraction_inside < min_fraction_inside: continue

        cx1 = min(max(x1, tile_region[0]), tile_region[2]) - dx
        cy1 = min(max(y1, tile_region[1]), tile_region[3]) - dy
        cx2 = min(max(x2, tile_region[0]), tile_region[2]) - dx
        cy2 = min(max(y2, tile_region[1]), tile_region[3]) - dy

        kept_boxes.append([cx1, cy1, cx2, cy2])
        kept_scores.append(float(score))
        kept_fills.append(float(fill))
    return kept_boxes, kept_scores, kept_fills

def filter_implausible_boxes(boxes, scores, fill_ratios, tile_w, tile_h, min_fill_ratio, max_area_fraction, edge_margin):
    """Remove detections that cannot be a single Rumex plant."""
    kept_boxes, kept_scores, kept_fills = [], [], []
    tile_area = float(tile_w * tile_h)
    for box, score, fill in zip(boxes, scores, fill_ratios):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= edge_margin or bh <= edge_margin: continue
        if (bw * bh) / tile_area > max_area_fraction: continue
        if fill < min_fill_ratio: continue
        kept_boxes.append(box)
        kept_scores.append(score)
        kept_fills.append(fill)
    return kept_boxes, kept_scores, kept_fills

# =============================================================================
#  CELL 8 - NEGATIVE EXEMPLAR GENERATOR
# =============================================================================
def box_gap_matrix(boxes_a, boxes_b) -> np.ndarray:
    """Euclidean gap between axis-aligned boxes."""
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)[None, :, :]
    dx = np.maximum(0.0, np.maximum(b[..., 0] - a[..., 2], a[..., 0] - b[..., 2]))
    dy = np.maximum(0.0, np.maximum(b[..., 1] - a[..., 3], a[..., 1] - b[..., 3]))
    return np.sqrt(dx ** 2 + dy ** 2)

def intersection_area_matrix(boxes_a, boxes_b) -> np.ndarray:
    """(N,M) intersection area in px^2 between two sets of [x1,y1,x2,y2] boxes."""
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    return iw * ih

def generate_negative_exemplars(gt_boxes, img_w, img_h, n_negatives, seed, n_candidates, min_gap, max_gap_factor):
    """
    Generate negative exemplars: plant-sized candidates, zero overlap with Rumex,
    close to Rumex, spatially diverse.
    """
    gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    if len(gt) == 0:
        raise ValueError("The image has no Rumex GT boxes -> no negatives can be placed near Rumex.")
    rng = np.random.default_rng(seed)

    gt_w = gt[:, 2] - gt[:, 0]
    gt_h = gt[:, 3] - gt[:, 1]
    size_idx = rng.integers(0, len(gt), n_candidates)
    cw = np.minimum(gt_w[size_idx], img_w - 1)
    ch = np.minimum(gt_h[size_idx], img_h - 1)
    cx1 = rng.uniform(0, img_w - cw)
    cy1 = rng.uniform(0, img_h - ch)
    cand = np.stack([cx1, cy1, cx1 + cw, cy1 + ch], axis=1).astype(np.float32)

    gap_to_rumex = box_gap_matrix(cand, gt).min(axis=1)
    overlap_to_rumex = intersection_area_matrix(cand, gt).max(axis=1)
    valid = (overlap_to_rumex == 0) & (gap_to_rumex >= min_gap)

    median_plant = float(np.median(np.maximum(gt_w, gt_h)))
    factor = max_gap_factor
    for _ in range(4):
        max_gap = factor * median_plant
        close = valid & (gap_to_rumex <= max_gap)
        if close.sum() >= n_negatives: break
        factor *= 2.0

    pool = np.where(close)[0]
    if len(pool) < n_negatives: pool = np.where(valid)[0]
    pool = pool[np.argsort(gap_to_rumex[pool], kind="stable")]

    centres = np.stack([(cand[:, 0] + cand[:, 2]) / 2, (cand[:, 1] + cand[:, 3]) / 2], axis=1)
    selected = []
    if len(pool): selected.append(int(pool[0]))
    while len(selected) < n_negatives and len(selected) < len(pool):
        sel = np.array(selected)
        d = np.linalg.norm(centres[pool][:, None, :] - centres[sel][None, :, :], axis=2).min(axis=1)
        clash = intersection_area_matrix(cand[pool], cand[sel]).max(axis=1) > 0
        d[np.isin(pool, sel) | clash] = -1.0
        best = int(np.argmax(d))
        if d[best] < 0: break
        selected.append(int(pool[best]))

    boxes = cand[selected].reshape(-1, 4)
    assert (intersection_area_matrix(boxes, gt) == 0).all()

    return {
        "boxes": boxes, "gaps": gap_to_rumex[selected],
        "pool_boxes": cand[pool].reshape(-1, 4),
        "stats": {"n_candidates": int(n_candidates), "n_no_overlap": int(valid.sum()),
                  "n_close": int(len(pool)), "median_plant_px": median_plant,
                  "max_gap_px": float(max_gap), "n_selected": int(len(selected))},
    }

# =============================================================================
#  CELL 15 - SAM3 BATCH INFERENCE
# =============================================================================
def _to_numpy(x) -> np.ndarray:
    """Torch tensor (any device/dtype) or numpy array -> float32 numpy array."""
    import torch
    if torch.is_tensor(x):
        if x.dtype in (torch.bfloat16, torch.float16): x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def _fill_ratios_from_masks(boxes_np, masks, binarise_at=0.5):
    """Fraction of each predicted box that is actually covered by its mask."""
    import torch
    n = len(boxes_np)
    fills = np.zeros(n, dtype=np.float32)
    if masks is None or n == 0: return fills

    for i in range(n):
        m = masks[i]
        h, w = int(m.shape[-2]), int(m.shape[-1])
        x1, y1, x2, y2 = boxes_np[i]
        x1c, y1c = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        x2c, y2c = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
        if x2c <= x1c or y2c <= y1c: continue

        region = m[..., y1c:y2c, x1c:x2c]
        if torch.is_tensor(region):
            fills[i] = float((region > binarise_at).float().mean()) if region.dtype != torch.bool else float(region.float().mean())
        else:
            region = np.asarray(region)
            fills[i] = float((region > binarise_at).mean()) if region.size else 0.0
    return fills

def _labels_for(boxes, box_labels):
    """Label list for one composed image: 1 = positive prompt, 0 = negative prompt."""
    if box_labels is None: return [1] * len(boxes)
    return [int(v) for v in box_labels]

class Sam3Runner:
    def __init__(self, model_id: str, device: Optional[str], dtype: str, mask_threshold: float):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
        if self.device == "cpu": self.model_dtype = torch.float32
        self.mask_threshold = mask_threshold

        print(f"Loading SAM3 from '{model_id}' onto {self.device} ({dtype}) ...")
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=self.model_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        print("SAM3 loaded.")

    def infer_batch(self, composed_images, crop_boxes_batch, threshold, text_prompt, box_labels):
        """
        Run SAM3 on a batch of composed images, with the exemplar boxes AND text_prompt.
        """
        torch = self.torch
        inputs = self.processor(
            images=composed_images,
            text=[text_prompt] * len(composed_images),
            input_boxes=[[[float(v) for v in b] for b in boxes] for boxes in crop_boxes_batch],
            input_boxes_labels=[_labels_for(boxes, box_labels) for boxes in crop_boxes_batch],
            return_tensors="pt",
        ).to(self.device)

        if self.model_dtype != torch.float32 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)

        per_image = []
        with torch.inference_mode():
            if self.model_dtype != torch.float32 and self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=self.model_dtype):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )

            for i, res in enumerate(results):
                boxes = _to_numpy(res["boxes"]).reshape(-1, 4)
                scores = _to_numpy(res["scores"]).reshape(-1)
                fills = _fill_ratios_from_masks(boxes, res.get("masks", None))
                if "masks" in res: res["masks"] = None
                per_image.append((boxes, scores, fills))

        del inputs, outputs, results
        return per_image

# =============================================================================
#  CELL 16 - OPTIONAL GLOBAL-CONTEXT PASS
# =============================================================================
def run_global_context_pass(runner, image, exemplar_crops, run_seed,
                            downscale, threshold, text_prompt, box_labels, args):
    """
    An EXTRA single pass over the whole image, downscaled by GLOBAL_DOWNSCALE,
    run through the SAME exemplar-strip + SAM3 pipeline. Its detections are
    rescaled back to full-resolution coordinates.
    """
    small_w = max(1, image.width // downscale)
    small_h = max(1, image.height // downscale)
    small_img = image.resize((small_w, small_h), Image.BILINEAR)

    rng = np.random.default_rng(stable_seed(run_seed, "global_pass_bg"))
    composed, crop_boxes, offset = compose_tile_with_exemplars(
        small_img, exemplar_crops, rng, args.strip_margin, args.feather_width,
        args.background_blur_radius)

    (boxes, scores, fills), = runner.infer_batch(
        [composed], [crop_boxes], threshold, text_prompt, box_labels)

    boxes, scores, fills = keep_only_target_region_detections(
        boxes, scores, fills, offset, small_img.width, small_img.height,
        args.tile_region_min_fraction)
    boxes, scores, fills = filter_implausible_boxes(
        boxes, scores, fills, small_img.width, small_img.height,
        args.min_fill_ratio, args.max_area_fraction, args.edge_margin)

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
#  CELL 17 - RUN ONE ANCHOR OVER ALL TILES (+ optional global-context pass)
# =============================================================================
def run_anchor_over_tiles(runner, tile_cache, full_image, exemplar_crops, run_seed,
                          batch_size, threshold, text_prompt, box_labels, args):
    """
    Runs SAM3 for ONE anchor over ALL cached tiles, and if ADD_GLOBAL_CONTEXT_PASS
    is on, ALSO runs the extra downsampled whole-image pass and appends its
    detections to the SAME pre-NMS pool, tagged with tile_id=-1.
    """
    all_boxes, all_scores, all_fills, all_tids, all_tboxes = [], [], [], [], []

    for start in range(0, len(tile_cache), batch_size):
        batch_tiles = tile_cache[start:start + batch_size]

        composed_images, crop_boxes_batch, offsets, sizes = [], [], [], []
        for tile in batch_tiles:
            tile_img = get_tile_image(tile, full_image)
            rng = np.random.default_rng(stable_seed(run_seed, "tile", tile["tile_id"]))
            composed, crop_boxes, offset = compose_tile_with_exemplars(
                tile_img, exemplar_crops, rng, args.strip_margin, args.feather_width,
                args.background_blur_radius)
            composed_images.append(composed)
            crop_boxes_batch.append(crop_boxes)
            offsets.append(offset)
            sizes.append((tile_img.width, tile_img.height))

        batch_results = runner.infer_batch(composed_images, crop_boxes_batch, threshold, text_prompt, box_labels)

        for tile, (boxes, scores, fills), offset, (tw, th) in zip(batch_tiles, batch_results, offsets, sizes):
            boxes, scores, fills = keep_only_target_region_detections(
                boxes, scores, fills, offset, tw, th, args.tile_region_min_fraction)
            boxes, scores, fills = filter_implausible_boxes(
                boxes, scores, fills, tw, th, args.min_fill_ratio, args.max_area_fraction, args.edge_margin)
            for b, s, f in zip(boxes, scores, fills):
                all_boxes.append([b[0] + tile["x1"], b[1] + tile["y1"], b[2] + tile["x1"], b[3] + tile["y1"]])
                all_scores.append(float(s))
                all_fills.append(float(f))
                all_tids.append(int(tile["tile_id"]))
                all_tboxes.append([tile["x1"], tile["y1"], tile["x2"], tile["y2"]])

        del composed_images, crop_boxes_batch, batch_results

    used_global_pass = False
    if args.add_global_context_pass:
        global_det = run_global_context_pass(
            runner, full_image, exemplar_crops, run_seed,
            args.global_downscale, threshold, text_prompt, box_labels, args)
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
#  CELL 18 - PRE-NMS DETECTION STORAGE
# =============================================================================
def run_npz_path(raw_detections_dir: Path, image_id: str, anchor_idx: int) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one run."""
    return raw_detections_dir / f"{safe_filename(image_id)}__anchor{int(anchor_idx):03d}.npz"

def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str, anchor_idx: int,
                        detections, gt_boxes, prompt_indices, neg_boxes, image_size,
                        archive: str, flight: str, class_id: int, text_prompt: str) -> Path:
    """Write one run's pre-NMS detections to NPZ."""
    path = run_npz_path(raw_detections_dir, image_id, anchor_idx)
    np.savez_compressed(
        path,
        experiment_name=np.array(experiment_name), image_ID=np.array(image_id),
        anchor_idx=np.array(int(anchor_idx)), text_prompt=np.array(text_prompt),
        prompt_indices=np.array(prompt_indices, dtype=np.int32),
        neg_boxes=np.asarray(neg_boxes, dtype=np.float32).reshape(-1, 4),
        image_width=np.array(int(image_size[0])), image_height=np.array(int(image_size[1])),
        archive=np.array(archive), flight=np.array(flight), source_class_id=np.array(int(class_id)),
        gt_boxes=gt_boxes.astype(np.float32),
        boxes=detections["boxes"].astype(np.float32),
        scores=detections["scores"].astype(np.float32),
        fill_ratio=detections["fill_ratio"].astype(np.float32),
        tile_id=detections["tile_id"].astype(np.int32),
        tile_boxes=detections["tile_boxes"].astype(np.int32),
    )
    return path

def load_run_detections(path: Path) -> dict:
    """Read one run NPZ back into a plain python dict."""
    with np.load(path, allow_pickle=False) as z:
        run = {
            "image_ID": str(z["image_ID"]), "anchor_idx": int(z["anchor_idx"]),
            "text_prompt": str(z["text_prompt"]),
            "prompt_indices": z["prompt_indices"].astype(int),
            "neg_boxes": z["neg_boxes"].reshape(-1, 4),
            "image_width": int(z["image_width"]), "image_height": int(z["image_height"]),
            "gt_boxes": z["gt_boxes"].reshape(-1, 4),
            "boxes": z["boxes"].reshape(-1, 4), "scores": z["scores"].reshape(-1),
            "fill_ratio": z["fill_ratio"].reshape(-1),
            "tile_id": z["tile_id"].reshape(-1), "tile_boxes": z["tile_boxes"].reshape(-1, 4),
        }
        run["archive"] = str(z["archive"]) if "archive" in z else ""
        run["flight"] = str(z["flight"]) if "flight" in z else ""
        return run

# =============================================================================
#  RESUME SUPPORT
# =============================================================================
def load_done_runs(paths: Paths, args) -> set:
    exp = args.experiment_name
    done: set = set()
    other_types: set = set()
    other_prompts: set = set()

    for csv_path in sorted(paths.raw_detections.glob(f"runs_manifest_{args.experiment_dir_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != exp: continue
                    row_type = str(row.get("Prompt_Type", ""))
                    if row_type and row_type != args.prompt_type: other_types.add(row_type)
                    row_text = str(row.get("text_prompt", ""))
                    if row_text and row_text != args.text_prompt: other_prompts.add(row_text)
                    try: done.add((row["image_ID"], int(row["anchor_idx"])))
                    except (KeyError, ValueError, TypeError): continue
        except OSError: continue

    if other_prompts or other_types:
        raise RuntimeError(
            f"{paths.raw_detections} already holds results for another prompt setting "
            f"(text {sorted(other_prompts) or [args.text_prompt]}, "
            f"type {sorted(other_types) or [args.prompt_type]}). "
            f"Use a new EXPERIMENT_NAME and OUTPUT_DIR for '{args.text_prompt}' ({args.prompt_type}) "
            "(or delete the old results folder) so the runs are not mixed.")
    return done

# =============================================================================
#  CELL 19 - MAIN GPU INFERENCE LOOP (PHASE 1)
# =============================================================================
def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    import torch
    # psutil is only needed for the RAM guard of PHASE 1. It is imported here (not
    # at the top of the file) so that --dry-run and PHASE 2 never depend on it; if
    # the container does not ship it, the guard is switched off with a warning
    # instead of crashing the whole job.
    try:
        import psutil
    except ImportError:
        psutil = None
        print("WARNING: psutil is not installed -> the RAM guard (--mem-stop-threshold-pct) "
              "is DISABLED for this shard. Run './pos_neg_exemplars_text_tiling_extra_path_run_sam3.sh "
              "download' on a login node to install it into $PYEXTRA.")
    stopped_for_ram = False

    exp = args.experiment_name
    done_runs = set() if args.no_resume else load_done_runs(paths, args)
    if done_runs: print(f"Resuming: {len(done_runs)} run(s) already finished for {exp}; skipped.")

    manifest_csv = manifest_path(paths, args.experiment_dir_name, args.shard_index)
    manifest_exists = manifest_csv.exists() and manifest_csv.stat().st_size > 0
    manifest_file = open(manifest_csv, "a", newline="")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
    if not manifest_exists: manifest_writer.writeheader(); manifest_file.flush()

    runner = Sam3Runner(args.model_id, args.device, args.dtype, args.mask_threshold)
    device_is_cuda = runner.device.startswith("cuda")

    start_time = time.time()
    n_new_runs = 0
    image_times: List[float] = []
    n_total_images = len(records)

    try:
        for img_idx, rec in enumerate(records, start=1):
            # --- RAM guard (notebook CELL 19) ---
            mem_pct = psutil.virtual_memory().percent if psutil is not None else 0.0
            if mem_pct > args.mem_stop_threshold_pct:
                print(f"RAM at {mem_pct:.0f}% (threshold {args.mem_stop_threshold_pct}%) -- "
                      f"stopping cleanly before {rec.image_id} to avoid a hard crash. "
                      f"Rerun to resume from where you left off.")
                stopped_for_ram = True
                break

            image_t0 = time.time()
            image_id = rec.image_id
            image = Image.open(rec.image_path).convert("RGB")
            img_w, img_h = image.size
            gt_boxes = load_yolo_boxes(rec.label_path, img_w, img_h, rec.class_id)
            n_gt = len(gt_boxes)

            if n_gt == 0:
                print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id}: 0 GT boxes, skipped.")
                image.close(); del image; gc.collect(); continue

            n_anchors = n_gt if args.max_anchors_per_image <= 0 else min(n_gt, args.max_anchors_per_image)
            if all((image_id, a) in done_runs for a in range(n_anchors)):
                print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id}: all anchors already done, skipped.")
                image.close(); del image; gc.collect(); continue

            tile_cache = build_tile_cache(image, args.use_tiling, args.tile_size, args.overlap, args.cache_tiles_in_memory)
            n_tiles = len(tile_cache)

            for anchor_idx in range(n_anchors):
                if (image_id, anchor_idx) in done_runs: continue

                # --- RAM guard mid-image ---
                mem_pct = psutil.virtual_memory().percent if psutil is not None else 0.0
                if mem_pct > args.mem_stop_threshold_pct:
                    print(f"RAM at {mem_pct:.0f}% mid-image at {image_id} anchor={anchor_idx} -- "
                          f"stopping cleanly. Completed anchors are already saved; rerun to resume.")
                    manifest_file.close()
                    for t in tile_cache: t["image"] = None
                    del tile_cache
                    image.close(); del image; gc.collect()
                    print(f"STOPPED: shard {args.shard_index} hit the RAM threshold -- NOT all images "
                          f"were processed. Resubmit the job to resume.")
                    sys.exit(RAM_STOP_EXIT_CODE)

                run_t0 = time.time()

                exemplar_indices = select_exemplar_indices(n_gt, anchor_idx, args.n_exemplars, image_id, exp)
                prompt_id = format_prompt_id(exemplar_indices)
                neg = generate_negative_exemplars(
                    gt_boxes, img_w, img_h, args.j_negatives,
                    seed=stable_seed(exp, image_id, anchor_idx, "negatives"),
                    n_candidates=args.neg_num_candidates, min_gap=args.neg_min_gap_px,
                    max_gap_factor=args.neg_max_gap_factor)

                pos_crops = [safe_crop(image, gt_boxes[i]) for i in exemplar_indices]
                neg_crops = [safe_crop(image, b) for b in neg["boxes"]]
                exemplar_crops = pos_crops + neg_crops
                box_labels = [1] * len(pos_crops) + [0] * len(neg_crops)

                run_seed = stable_seed(exp, image_id, anchor_idx, "strip_bg")
                detections, used_global_pass = run_anchor_over_tiles(
                    runner, tile_cache, image, exemplar_crops, run_seed,
                    args.batch_size, args.threshold, args.text_prompt, box_labels, args)

                npz_path = save_run_detections(
                    paths.raw_detections, exp, image_id, anchor_idx, detections,
                    gt_boxes, exemplar_indices, neg["boxes"], (img_w, img_h),
                    rec.archive, rec.flight, rec.class_id, args.text_prompt)

                run_seconds = time.time() - run_t0
                manifest_writer.writerow({
                    "experiment_name": exp, "image_ID": image_id, "anchor_idx": anchor_idx,
                    "Prompt_ID": prompt_id, "Prompt_Type": args.prompt_type, "text_prompt": args.text_prompt,
                    "archive": rec.archive, "flight": rec.flight, "source_class_id": rec.class_id,
                    "n_gt": n_gt, "n_prompt_gt": len(exemplar_indices),
                    "n_negatives": int(len(neg["boxes"])),
                    "neg_max_gap_px": round(neg["stats"]["max_gap_px"], 1),
                    "n_detections_pre_nms": int(len(detections["scores"])), "n_tiles": n_tiles,
                    "used_global_pass": used_global_pass,
                    "image_width": img_w, "image_height": img_h,
                    "npz_file": npz_path.name, "inference_seconds": round(run_seconds, 2),
                })
                manifest_file.flush()
                n_new_runs += 1

                print(f"  [{exp}] shard{args.shard_index} run #{n_new_runs} | {image_id} | "
                      f"anchor={anchor_idx} ({anchor_idx + 1}/{n_anchors}) | "
                      f"prompt=POS {prompt_id} + {len(neg['boxes'])} NEG + text | "
                      f"tiles={n_tiles} | global_pass={used_global_pass} | "
                      f"pre-NMS detections={len(detections['scores'])} | {run_seconds:.1f}s")

                if len(neg["boxes"]) < args.j_negatives:
                    print(f"    WARNING: only {len(neg['boxes'])} of {args.j_negatives} negatives could be placed.")

                del detections, exemplar_crops, pos_crops, neg_crops, neg
                gc.collect()
                if device_is_cuda: torch.cuda.empty_cache()

            for t in tile_cache: t["image"] = None
            del tile_cache
            image.close(); del image; gc.collect()
            if device_is_cuda: torch.cuda.empty_cache()

            image_elapsed = time.time() - image_t0
            image_times.append(image_elapsed)
            avg_per_image = float(np.mean(image_times))
            eta = (n_total_images - img_idx) * avg_per_image
            rss_gb = psutil.Process().memory_info().rss / 1e9 if psutil is not None else float("nan")
            print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id} done | {n_gt} GT box(es) | "
                  f"{image_elapsed:.1f}s | avg/image={avg_per_image:.1f}s | ETA={eta / 60:.1f} min | "
                  f"[MEM] RSS={rss_gb:.2f} GB")

    finally:
        manifest_file.close()

    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new runs.")
    print(f"Total time: {total_elapsed / 60:.1f} min")
    if stopped_for_ram:
        # Exit NON-ZERO so run_sam3.sh reports this shard as FAILED (and SLURM sends
        # the FAIL mail) instead of "All shards completed" on a partial result.
        print(f"STOPPED: shard {args.shard_index} hit the RAM threshold -- NOT all images "
              f"were processed. Resubmit the job to resume.")
        sys.exit(RAM_STOP_EXIT_CODE)

# =============================================================================
#  DRY RUN
# =============================================================================
def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading SAM3 ---")
    sample = my_records[:min(len(my_records), 200)]
    total_anchors, n_short_negatives, n_checked = 0, 0, 0
    total_tiles, total_passes, per_archive = 0, 0, {}

    for rec in sample:
        with Image.open(rec.image_path) as im:
            w, h = im.size
        gt = load_yolo_boxes(rec.label_path, w, h, rec.class_id)
        n_gt = len(gt)
        n_anchors = n_gt if args.max_anchors_per_image <= 0 else min(n_gt, args.max_anchors_per_image)
        n_tiles = len(tile_bboxes(w, h, args.tile_size, args.overlap)) if args.use_tiling else 1
        # per anchor run: ceil(n_tiles / batch) batched tile passes + 1 global pass
        passes_per_run = -(-n_tiles // max(1, args.batch_size)) + (1 if args.add_global_context_pass else 0)
        total_anchors += n_anchors
        total_tiles += n_tiles
        total_passes += n_anchors * passes_per_run
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + n_anchors
        if n_gt > 0:
            neg = generate_negative_exemplars(gt, w, h, args.j_negatives,
                                              seed=stable_seed(args.experiment_name, rec.image_id, 0, "negatives"),
                                              n_candidates=args.neg_num_candidates, min_gap=args.neg_min_gap_px,
                                              max_gap_factor=args.neg_max_gap_factor)
            n_checked += 1
            if len(neg["boxes"]) < args.j_negatives: n_short_negatives += 1

    n_img = max(1, len(sample))
    tiles_per_image = total_tiles / n_img
    print(f"  sampled {len(sample)} image(s) of this shard -> {total_anchors} anchor runs ({per_archive})")
    print(f"  prompts per run: {args.k_positives} positive + {args.j_negatives} negative + text ({args.prompt_type})")
    print(f"  negative generator probed on {n_checked} image(s): {n_short_negatives} could not place all {args.j_negatives} negatives")
    print(f"  tiles per image: ~{tiles_per_image:.0f} ({args.tile_size}px, {args.overlap}px overlap)")
    print(f"  forward passes per anchor run: ~{-(-round(tiles_per_image) // max(1, args.batch_size))} "
          f"(batched, {args.batch_size} tiles per pass)"
          + (f" + 1 global-context pass (downscale={args.global_downscale})"
             if args.add_global_context_pass else " (global-context pass OFF)"))
    print(f"  => ~{total_passes} SAM3 forward passes for those {len(sample)} images")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{total_anchors} (one per image x anchor)")

# =============================================================================
#  CELL 20 - LOAD CACHED PRE-NMS DETECTIONS
# =============================================================================
def load_runs(paths: Paths, args):
    import pandas as pd

    exp = args.experiment_name
    manifest_files = sorted(paths.raw_detections.glob(f"runs_manifest_{args.experiment_dir_name}_shard*.csv"))
    if not manifest_files: return [], None

    frames = [pd.read_csv(p) for p in manifest_files if p.exists()]
    if not frames: return [], None

    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[(manifest["experiment_name"] == exp) &
                        (manifest["text_prompt"].astype(str) == args.text_prompt) &
                        (manifest["Prompt_Type"].astype(str) == args.prompt_type)].copy()
    manifest = manifest.drop_duplicates(subset=["image_ID", "anchor_idx"], keep="last")
    manifest = manifest.sort_values(["image_ID", "anchor_idx"], kind="stable")

    runs = []
    for _, row in manifest.iterrows():
        path = paths.raw_detections / str(row["npz_file"])
        if not path.exists(): continue
        run = load_run_detections(path)
        run["Prompt_ID"] = str(row["Prompt_ID"])
        run["Prompt_Type"] = str(row["Prompt_Type"])
        run["archive"] = run.get("archive") or str(row.get("archive", ""))
        run["flight"] = run.get("flight") or str(row.get("flight", ""))
        runs.append(run)

    print(f"Loaded {len(runs)} runs ({manifest['image_ID'].nunique()} images) for {exp}.")
    return runs, manifest

# =============================================================================
#  CELL 21 - OFFLINE NMS (with cross-tile provenance)
# =============================================================================
def nms_with_provenance(boxes, scores, iou_threshold):
    n = len(boxes)
    if n == 0: return [], []
    order = list(np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable"))
    keep, suppressed = [], []
    while order:
        i = int(order[0]); keep.append(i)
        rest = np.array(order[1:], dtype=int)
        if rest.size == 0: break
        ious = compute_iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        for s in rest[ious > iou_threshold]: suppressed.append((int(s), i))
        order = list(rest[ious <= iou_threshold])
    return keep, suppressed

def apply_nms_to_run(run, iou_threshold):
    keep, suppressed = nms_with_provenance(run["boxes"], run["scores"], iou_threshold)
    keep = np.array(keep, dtype=int)

    cross_tile_suppressed, same_tile_suppressed = 0, 0
    for s, k in suppressed:
        if run["tile_id"][s] != run["tile_id"][k]: cross_tile_suppressed += 1
        else: same_tile_suppressed += 1

    return {
        "boxes": run["boxes"][keep].reshape(-1, 4), "scores": run["scores"][keep].reshape(-1),
        "tile_id": run["tile_id"][keep].reshape(-1), "tile_boxes": run["tile_boxes"][keep].reshape(-1, 4),
        "n_pre_nms": int(len(run["scores"])),
        "n_suppressed_cross_tile": int(cross_tile_suppressed),
        "n_suppressed_same_tile": int(same_tile_suppressed),
    }

# =============================================================================
#  CELL 22 - EVALUATION CORE
# =============================================================================
def split_gt_for_mode(gt_boxes, prompt_indices, mode: str):
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    if mode == "all_gt": return gt_boxes, np.zeros((0, 4), dtype=np.float32)
    is_prompt = np.zeros(len(gt_boxes), dtype=bool)
    prompt_indices = np.asarray(prompt_indices, dtype=int)
    if len(prompt_indices): is_prompt[prompt_indices] = True
    return gt_boxes[~is_prompt], gt_boxes[is_prompt]

def evaluate_run_predictions(pred_boxes, pred_scores, eval_gt_boxes, prompt_gt_boxes, eval_iou, ignore_iou):
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    eval_gt_boxes = np.asarray(eval_gt_boxes, dtype=np.float32).reshape(-1, 4)
    prompt_gt_boxes = np.asarray(prompt_gt_boxes, dtype=np.float32).reshape(-1, 4)

    n_pred, n_eval_gt = len(pred_boxes), len(eval_gt_boxes)
    status = np.full(n_pred, STATUS_FP, dtype=np.int8)

    match = match_one_to_one(pred_boxes, pred_scores, eval_gt_boxes, eval_iou)
    status[match["pred_match_gt"] >= 0] = STATUS_TP

    if n_pred and len(prompt_gt_boxes):
        leftover = np.where(status == STATUS_FP)[0]
        if len(leftover):
            best_prompt_iou = compute_iou_matrix(pred_boxes[leftover], prompt_gt_boxes).max(axis=1)
            status[leftover[best_prompt_iou >= ignore_iou]] = STATUS_IGNORED

    tp = int((status == STATUS_TP).sum())
    fp = int((status == STATUS_FP).sum())
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
        "IoU1": iou1, "IoU2": iou2, "valid_for_macro": bool(n_eval_gt > 0),
    }

# =============================================================================
#  CELL 23 - AP50 AND AP50:95
# =============================================================================
def make_detections(boxes, scores=None):
    import supervision as sv
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    if scores is None: return sv.Detections(xyxy=boxes, class_id=np.zeros(n, dtype=int))
    return sv.Detections(xyxy=boxes, confidence=np.asarray(scores, dtype=np.float32).reshape(-1), class_id=np.zeros(n, dtype=int))

def compute_ap(pred_list, gt_list):
    from supervision.metrics import MeanAveragePrecision
    if len(gt_list) == 0 or sum(len(g) for g in gt_list) == 0: return float("nan"), float("nan")
    try:
        result = MeanAveragePrecision().update(pred_list, gt_list).compute()
        ap50, ap5095 = float(result.map50), float(result.map50_95)
        return (ap50 if ap50 >= 0 else float("nan"), ap5095 if ap5095 >= 0 else float("nan"))
    except Exception as e:
        print("   (AP computation failed:", e, ")")
        return float("nan"), float("nan")

def ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou):
    ev = evaluate_run_predictions(nms_run["boxes"], nms_run["scores"], eval_gt, prompt_gt, eval_iou, ignore_iou)
    keep = ev["status"] != STATUS_IGNORED
    return (make_detections(nms_run["boxes"][keep], nms_run["scores"][keep]), make_detections(eval_gt))

# =============================================================================
#  CELL 24 - OPERATING-POINT EVALUATION
# =============================================================================
def evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, confidence_threshold, eval_iou, ignore_iou):
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_run_predictions(nms_run["boxes"][keep], nms_run["scores"][keep], eval_gt, prompt_gt, eval_iou, ignore_iou)

# =============================================================================
#  CELL 30 - CONFUSION-MATRIX FIGURE
# =============================================================================
def plot_confusion_matrix(tp, fp, fn, title, png_path, plt):
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
            ax.text(j, i, labels[i][j], ha="center", va="center", color=colour, fontsize=11)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

# =============================================================================
#  CELL 32 - QUALITATIVE PLOT
# =============================================================================
def _draw_boxes(ax, boxes, color, linewidth=1.5, linestyle="-", labels=None, fontsize=6):
    import matplotlib.patches as patches
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle))
        if labels is not None:
            ax.text(x1, y1 - 2, labels[k], color="white", fontsize=fontsize, va="bottom", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.8, alpha=0.85))

def select_image_for_plot(run_level_df, image_level_df, mode, min_gt, archive=None):
    img_df = image_level_df[image_level_df["evaluation_mode"] == mode].copy()
    runs_df = run_level_df[run_level_df["evaluation_mode"] == mode]
    if archive is not None:
        img_df = img_df[img_df["archive"] == archive]
        runs_df = runs_df[runs_df["archive"] == archive]
    if img_df.empty: return None, f"no image for archive={archive}"

    n_gt_per_image = runs_df.groupby("image_ID")["n_gt_total"].first()
    img_df["n_gt"] = img_df["image_ID"].map(n_gt_per_image)
    img_df = img_df[img_df["AP50_mean"].notna() & img_df["n_gt"].notna()]
    if img_df.empty: return None, "no image with a valid AP50"
    img_df["n_gt"] = img_df["n_gt"].astype(int)

    candidates = img_df[img_df["n_gt"] >= min_gt]
    if len(candidates):
        rule = f"highest image-level AP50 among the {len(candidates)} images with >= {min_gt} GT boxes"
    else:
        max_gt = int(img_df["n_gt"].max())
        candidates = img_df[img_df["n_gt"] == max_gt]
        rule = f"no image has >= {min_gt} GT boxes -> highest image-level AP50 among the {len(candidates)} image(s) with the most GT boxes ({max_gt})"
    if archive is not None: rule += f" [archive={archive}]"

    ranked = candidates.sort_values(["AP50_mean", "F1_mean", "n_gt"], ascending=False, na_position="last")
    return ranked.iloc[0], rule

def plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df, image_paths, plt, scope_label, archive=None):
    from matplotlib.lines import Line2D
    mode = args.plot_evaluation_mode
    img_row, rule = select_image_for_plot(run_level_df, image_level_df, mode, args.plot_min_gt_boxes, archive)
    if img_row is None: return

    plot_image_id = img_row["image_ID"]
    img_runs = run_level_df[(run_level_df["evaluation_mode"] == mode) & (run_level_df["image_ID"] == plot_image_id)]
    run_row = img_runs.sort_values(["AP50", "F1"], ascending=False, na_position="last").iloc[0]
    anchor = int(run_row["anchor_idx"])
    run = next((r for r in runs if r["image_ID"] == plot_image_id and r["anchor_idx"] == anchor), None)
    if run is None: return

    nms_run = apply_nms_to_run(run, args.nms_iou_threshold)
    keep = nms_run["scores"] >= args.operating_confidence
    pred_boxes, pred_scores = nms_run["boxes"][keep], nms_run["scores"][keep]
    gt_boxes_plot = run["gt_boxes"]
    exemplar_boxes = gt_boxes_plot[np.asarray(run["prompt_indices"], dtype=int)]
    neg_boxes_plot = run["neg_boxes"]

    if plot_image_id not in image_paths: return
    with Image.open(image_paths[plot_image_id]) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(1.0, args.plot_max_display_dim / max(w, h))
        disp_w, disp_h = max(1, int(w * s)), max(1, int(h * s))
        display = np.asarray(im.resize((disp_w, disp_h), Image.BILINEAR))
    to_disp = np.array([disp_w / w, disp_h / h, disp_w / w, disp_h / h], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax in axes: ax.imshow(display); ax.axis("off")

    _draw_boxes(axes[0], gt_boxes_plot * to_disp, color="yellow", linewidth=1.5)
    axes[0].set_title(f"Ground truth: {len(gt_boxes_plot)} Rumex boxes", fontsize=12)

    score_labels = [f"{v:.2f}" for v in pred_scores] if args.plot_show_scores else None
    _draw_boxes(axes[1], pred_boxes * to_disp, color="red", linewidth=1.5, labels=score_labels)
    _draw_boxes(axes[1], exemplar_boxes * to_disp, color="lime", linewidth=2.5, linestyle="--")
    _draw_boxes(axes[1], neg_boxes_plot * to_disp, color="magenta", linewidth=2.5, linestyle="--")

    axes[1].set_title(
        f"Predictions: {len(pred_boxes)} boxes | anchor = {anchor} | "
        f"POS {run['Prompt_ID']} + {len(neg_boxes_plot)} NEG + text '{run['text_prompt']}'\n"
        f"run AP50={run_row['AP50']:.3f}  P={run_row['precision']:.3f}  R={run_row['recall']:.3f}  F1={run_row['F1']:.3f}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color="yellow", lw=2, label="ground truth"),
        Line2D([0], [0], color="red", lw=2, label="prediction"),
        Line2D([0], [0], color="lime", lw=2.5, linestyle="--", label=f"positive prompts ({len(exemplar_boxes)})"),
        Line2D([0], [0], color="magenta", lw=2.5, linestyle="--", label=f"negative prompts ({len(neg_boxes_plot)})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=11, frameon=False)

    fig.suptitle(f"{args.experiment_name} | {plot_image_id} | mode={mode} | scope={scope_label}\n"
                 f"{args.k_positives} pos + {args.j_negatives} neg + text | tile={args.tile_size}px, overlap={args.overlap}px\n"
                 f"global pass={args.add_global_context_pass} (x1/{args.global_downscale})\n"
                 f"selection: {rule}", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    png_path = paths.plots / f"best_image_{scope_label}_{safe_filename(plot_image_id)}_anchor{anchor:03d}_{mode}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def make_qualitative_plots(args, paths, runs, run_level_df, image_level_df, plt):
    records = discover_images(args.dataset_root, args.archives)
    image_paths = {r.image_id: r.image_path for r in records}
    if not image_paths: return

    archives_present = sorted(str(a) for a in run_level_df["archive"].dropna().unique() if str(a))
    for archive in archives_present:
        plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df, image_paths, plt, scope_label=archive, archive=archive)
    plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df, image_paths, plt, scope_label="ALL", archive=None)

# =============================================================================
#  PHASE 2 - THE WHOLE OFFLINE EVALUATION
# =============================================================================
def run_evaluation(args, paths: Paths) -> None:
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _HAS_MPL = True
    except Exception:
        plt = None; _HAS_MPL = False

    exp = args.experiment_name
    conf = args.operating_confidence
    nms_iou = args.nms_iou_threshold
    eval_iou = args.eval_iou_threshold
    ignore_iou = args.prompt_ignore_iou

    print("=" * 92)
    print(f" PHASE 2 - OFFLINE EVALUATION | experiment={exp}")
    print("=" * 92)

    runs, manifest = load_runs(paths, args)
    if not runs: print("Nothing to evaluate."); return

    # CELL 26 - RUN-LEVEL METRICS
    print("\n--- CELL 26: run-level metrics ---")
    run_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)
        for mode in EVALUATION_MODES:
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            ap50, ap5095 = compute_ap([p_det], [g_det])
            ev = evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, conf, eval_iou, ignore_iou)
            valid = ev["valid_for_macro"]
            nan = float("nan")

            run_rows.append({
                "experiment_name": exp, "image_ID": run["image_ID"],
                "archive": run.get("archive", ""), "flight": run.get("flight", ""),
                "anchor_idx": run["anchor_idx"], "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"], "text_prompt": run["text_prompt"],
                "n_negatives": int(len(run["neg_boxes"])), "evaluation_mode": mode,
                "confidence_threshold": conf, "nms_iou_threshold": nms_iou,
                "n_gt_total": int(len(run["gt_boxes"])),
                "n_prompt_gt": int(len(run["prompt_indices"])) if mode == "held_out" else 0,
                "n_eval_gt": ev["n_eval_gt"], "n_predictions": ev["n_pred"],
                "n_ignored_predictions": ev["n_ignored"],
                "AP50": ap50 if valid else nan, "AP50_95": ap5095 if valid else nan,
                "precision": ev["precision"] if valid else nan, "recall": ev["recall"] if valid else nan,
                "F1": ev["F1"] if valid else nan, "IoU1": ev["IoU1"] if valid else nan, "IoU2": ev["IoU2"] if valid else nan,
                "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"], "valid_for_macro": valid,
            })

    run_level_df = pd.DataFrame(run_rows)
    run_level_df.to_csv(paths.metrics / "run_level_metrics.csv", index=False)

    # CELL 27 - IMAGE-LEVEL METRICS
    print("\n--- CELL 27: image-level metrics ---")
    image_rows = []
    for (image_id, mode), grp in run_level_df.groupby(["image_ID", "evaluation_mode"]):
        row = {
            "experiment_name": exp, "image_ID": image_id,
            "archive": grp["archive"].iloc[0], "flight": grp["flight"].iloc[0],
            "evaluation_mode": mode, "confidence_threshold": conf, "nms_iou_threshold": nms_iou,
            "n_runs_total": int(len(grp)), "n_runs_valid_for_macro": int(grp["valid_for_macro"].sum()),
            "TP_sum": int(grp["TP"].sum()), "FP_sum": int(grp["FP"].sum()), "FN_sum": int(grp["FN"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std()
        image_rows.append(row)

    image_level_df = pd.DataFrame(image_rows).sort_values(["evaluation_mode", "image_ID"]).reset_index(drop=True)
    image_level_df.to_csv(paths.metrics / "image_level_metrics.csv", index=False)

    # CELL 28 - EXPERIMENT-LEVEL SUMMARY
    print("\n--- CELL 28: experiment-level summary ---")
    summary_rows = []
    for mode in EVALUATION_MODES:
        sub = image_level_df[image_level_df["evaluation_mode"] == mode]
        row = {
            "experiment_name": exp, "evaluation_mode": mode, "prompt_type": args.prompt_type,
            "text_prompt": args.text_prompt, "n_exemplars": args.n_exemplars, "n_negatives": args.j_negatives,
            "use_tiling": args.use_tiling, "tile_size": args.tile_size, "overlap": args.overlap,
            "add_global_context_pass": args.add_global_context_pass,
            "confidence_threshold": conf, "nms_iou_threshold": nms_iou,
            "eval_iou_threshold": eval_iou, "n_images": int(sub["image_ID"].nunique()),
            "n_runs": int(sub["n_runs_total"].sum()), "n_runs_valid_for_macro": int(sub["n_runs_valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub[f"{col}_mean"].mean()
            row[f"{col}_std"] = sub[f"{col}_mean"].std()
        summary_rows.append(row)

    experiment_summary_df = pd.DataFrame(summary_rows)
    experiment_summary_df.to_csv(paths.metrics / "experiment_summary.csv", index=False)

    # Per-archive version
    per_archive_rows = []
    for (archive, mode), sub in image_level_df.groupby(["archive", "evaluation_mode"]):
        row = {"experiment_name": exp, "archive": archive, "evaluation_mode": mode,
               "n_images": int(sub["image_ID"].nunique()), "n_runs": int(sub["n_runs_total"].sum()),
               "n_runs_valid_for_macro": int(sub["n_runs_valid_for_macro"].sum())}
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub[f"{col}_mean"].mean()
            row[f"{col}_std"] = sub[f"{col}_mean"].std()
        per_archive_rows.append(row)
    pd.DataFrame(per_archive_rows).to_csv(paths.metrics / "experiment_summary_per_archive.csv", index=False)

    # CELL 29 - POOLED DATASET AP
    print("\n--- CELL 29: pooled dataset AP ---")
    dataset_rows = []
    for mode in EVALUATION_MODES:
        pred_list, gt_list, images_used = [], [], set()
        for run in runs:
            nms_run = apply_nms_to_run(run, nms_iou)
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            pred_list.append(p_det); gt_list.append(g_det); images_used.add(run["image_ID"])
        ap50, ap5095 = compute_ap(pred_list, gt_list)
        dataset_rows.append({
            "experiment_name": exp, "evaluation_mode": mode, "n_images": len(images_used), "n_runs": len(runs),
            "confidence_used_for_AP": args.threshold, "nms_iou_threshold": nms_iou,
            "dataset_AP50": ap50, "dataset_AP50_95": ap5095,
        })
        pd.DataFrame([dataset_rows[-1]]).to_csv(paths.metrics / "dataset_ap_metrics.csv", index=False, mode='a' if mode != EVALUATION_MODES[0] else 'w', header=(mode == EVALUATION_MODES[0]))
    
    # Rewrite dataset AP cleanly to avoid append issues
    pd.DataFrame(dataset_rows).to_csv(paths.metrics / "dataset_ap_metrics.csv", index=False)

    # CELL 30 - CONFUSION MATRICES
    print("\n--- CELL 30: confusion matrices ---")
    confusion_summary = []
    for mode in EVALUATION_MODES:
        sub = run_level_df[run_level_df["evaluation_mode"] == mode]
        tp, fp, fn = int(sub["TP"].sum()), int(sub["FP"].sum()), int(sub["FN"].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = safe_f1(precision, recall)

        pd.DataFrame([[tp, fn], [fp, np.nan]], index=["actual_rumex", "actual_background"],
                     columns=["predicted_rumex", "predicted_background"]).to_csv(paths.confusion_matrices / f"confusion_matrix_{mode}.csv")

        if _HAS_MPL:
            plot_confusion_matrix(tp, fp, fn, f"{exp} - {mode}\nconf={conf:.2f}, NMS IoU={nms_iou:.2f}",
                                  paths.confusion_matrices / f"confusion_matrix_{mode}.png", plt)

        confusion_summary.append({
            "experiment_name": exp, "evaluation_mode": mode, "TP": tp, "FP": fp, "FN": fn,
            "precision_micro": precision, "recall_micro": recall, "F1_micro": f1,
            "confidence_threshold": conf, "nms_iou_threshold": nms_iou, "eval_iou_threshold": eval_iou,
        })
    pd.DataFrame(confusion_summary).to_csv(paths.confusion_matrices / "confusion_matrix_summary.csv", index=False)

    # CELL 32 - QUALITATIVE FIGURES
    if not args.no_plots and _HAS_MPL:
        print("\n--- CELL 32: qualitative GT-vs-prediction figures ---")
        make_qualitative_plots(args, paths, runs, run_level_df, image_level_df, plt)

    print("\n" + "=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY")
    print("=" * 78)
    print(f"Prompts per run          : {args.k_positives} positive + {args.j_negatives} negative crops "
          f"(one strip) + text ({args.prompt_type})")
    print(f"Text prompt              : '{args.text_prompt}'")
    print(f"Tiling                   : {args.use_tiling}  (tile={args.tile_size}px, overlap={args.overlap}px)")
    print(f"Global context pass      : {args.add_global_context_pass} (downscale={args.global_downscale})")
    print(f"Confidence threshold     : {conf:.2f} (fixed)")
    print(f"NMS IoU threshold        : {nms_iou:.2f} (fixed)")
    print("-" * 78)
    print("EXPERIMENT-LEVEL RESULTS (mean over images, std between images)")
    show = ["evaluation_mode", "AP50_mean", "AP50_std", "AP50_95_mean", "precision_mean",
            "recall_mean", "F1_mean", "F1_std", "IoU1_mean", "IoU2_mean"]
    print(experiment_summary_df[show].to_string(index=False))
    print("=" * 78)

# =============================================================================
#  MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    paths = build_paths(output_dir)

    if args.evaluate_only:
        run_evaluation(args, paths)
        return

    print("=" * 92)
    print(f" POS_NEG_EXEMPLARS_TEXT_TILING_EXTRA_PATH SAM3 PIPELINE | experiment={args.experiment_name} | shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)

    records = discover_images(dataset_root, args.archives)
    if not records: print("No images with labels found -- check --dataset-root. Aborting."); sys.exit(1)

    if args.limit_images > 0: records = records[: args.limit_images]
    my_records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_index]
    
    print(f"Total images: {len(records)} | this shard: {len(my_records)}")
    for arch in args.archives:
        sub = [r for r in records if r.archive == arch]
        print(f"  {arch}: {len(sub)} images, class_id={ARCHIVES[arch]}")

    if args.dry_run:
        dry_run(args, records, my_records)
        return

    run_inference(args, paths, my_records)

    if not args.no_evaluate:
        run_evaluation(args, paths)

if __name__ == "__main__":
    main()