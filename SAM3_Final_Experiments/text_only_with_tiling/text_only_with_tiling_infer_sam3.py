#!/usr/bin/env python3
# =============================================================================
#  text_only_with_tiling_infer_sam3.py
# =============================================================================
#  Cluster (CSCS) port of the Colab notebook  text_only_with_tiling.ipynb
#
#  This experiment is the TILED, TEXT-ONLY variant: N_EXEMPLARS = 0
#  and USE_TILING = True. There are no visual prompts, no exemplar strip and
#  no plausibility filter. Every UAV image is split into overlapping tiles
#  (TILE_SIZE=1000, OVERLAP=150), and every RAW tile is sent to SAM3 in batches
#  of BATCH_SIZE=4 with ONLY a text phrase (e.g. "Rumex obtusifolius"). Tile
#  detections are shifted back to original-image coordinates, and duplicates
#  from overlapping tiles are merged by offline NMS.
#
#  The notebook is a TWO-PHASE pipeline and this file keeps that separation:
#
#    PHASE 1 - INFERENCE   (notebook CELL 12, GPU)
#        for every image:
#            build the tile list once                       (CELL 9)
#            run SAM3 over all tiles in batches at 0.30     (CELL 10)
#            shift boxes back to full resolution            (CELL 10)
#            save the PRE-NMS detections to NPZ             (CELL 11)
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 13 ... CELL 23, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40
#        (tracking cross-tile vs same-tile suppression), evaluate in all_gt
#        mode at the operating point CONFIDENCE_THRESHOLD = 0.30, and write
#        image-level / experiment-level / pooled-AP CSVs, the confusion
#        matrices (CSV + PNG) and the qualitative GT-vs-prediction figures.
#        Runs in ONE process, after every shard has finished. It never touches
#        SAM3, so it can be repeated as often as you like from the cached NPZs.
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
import json
import os
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
MANIFEST_COLUMNS = [
    "experiment_name", "image_ID", "Prompt_Type", "text_prompt",
    "archive", "flight", "source_class_id",
    "n_gt", "n_gt_larger_than_overlap", "n_tiles", "n_detections_pre_nms",
    "image_width", "image_height", "npz_file", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
EVALUATION_MODES = ["all_gt"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
#            Text-only prompting uses no GT box as input, so there are no prompt
#            plants to hold out. The held_out mode of the exemplar experiments
#            does not apply here.

# ---- notebook CELL 18 ------------------------------------------------------
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2"]

# ---- notebook CELL 15 ------------------------------------------------------
STATUS_FP, STATUS_TP = 0, 1

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './text_only_with_tiling_run_sam3.sh download' on a "
    "LOGIN node first, which installs it into $PYEXTRA."
)

# =============================================================================
#  CLI  -  every notebook CELL 3 parameter, with the notebook value as default
# =============================================================================
def default_dataset_root() -> Path:
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "overney" / "dataset"
    return Path(__file__).resolve().parents[2] / ".." / "02_data" / "dataset"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="text_only_with_tiling - SAM3 text-only prompted Rumex detection, "
                    "with tiling (inference + offline evaluation). Cluster port "
                    "of text_only_with_tiling.ipynb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------- paths -------------------------------------
    g = p.add_argument_group("paths")
    g.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                   help="Folder holding the archive folders (AGS_Multi_Rumex, AgsSpringRumex). "
                        "PHASE 2 uses it as well, for the qualitative figures (CELL 23).")
    g.add_argument("--output-dir", type=Path, required=True,
                   help="RESULTS_ROOT: raw_detections/, metrics/, confusion_matrices/, plots/.")
    g.add_argument("--archives", nargs="*", default=list(ARCHIVES.keys()),
                   help="Subset of archives to run on. Default: both.")

    # ------------------------ experiment identity ---------------------------
    g = p.add_argument_group("experiment identity (CELL 3)")
    g.add_argument("--experiment-name", default="text_only_with_tiling",
                   help="EXPERIMENT_NAME. Written into every CSV row and every NPZ.")
    g.add_argument("--text-prompt", default="Rumex obtusifolius",
                   help="TEXT_PROMPT. Short phrase (SAM3 reads at most 32 tokens).")
    g.add_argument("--n-exemplars", type=int, default=0,
                   help="N_EXEMPLARS. Must be 0 for this text-only experiment.")

    # --------------------------- tiling -------------------------------------
    g = p.add_argument_group("tiling (CELL 3 / CELL 9)")
    g.add_argument("--tile-size", type=int, default=1000,
                   help="TILE_SIZE: tile width/height in pixels.")
    g.add_argument("--overlap", type=int, default=150,
                   help="OVERLAP: overlap between neighbouring tiles.")
    g.add_argument("--batch-size", type=int, default=4,
                   help="BATCH_SIZE: tiles per forward pass. Lower it if you hit CUDA OOM.")
    g.add_argument("--cache-tiles-in-memory", type=str2bool, default=True,
                   help="CACHE_TILES_IN_MEMORY: crop all tiles of an image once and keep them in RAM.")

    # --------------------------- SAM3 inference -----------------------------
    g = p.add_argument_group("sam3 inference (CELL 3 / CELL 5 / CELL 10)")
    g.add_argument("--model-id", default="facebook/sam3",
                   help="HF repo id OR a local snapshot directory.")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="CONFIDENCE_THRESHOLD. SAM3 is executed EXACTLY ONCE per "
                        "tile at this score. In this experiment it is ALSO the "
                        "evaluation operating point (see --operating-confidence).")
    g.add_argument("--mask-threshold", type=float, default=0.40,
                   help="MASK_THRESHOLD. SAM3 mask binarisation. No plausibility filter is "
                        "used here, so masks influence no box and no metric; they are "
                        "discarded right after post-processing (KEEP_MASKS = False).")
    g.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16",
                   help="Model dtype. The notebook loaded with device_map='auto' and no dtype "
                        "(float32 on a T4); bfloat16 is the cluster default (GH200/H100) with "
                        "a negligible accuracy change.")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 15 / CELL 18)")
    g.add_argument("--nms-iou-threshold", type=float, default=0.40,
                   help="NMS_IOU_THRESHOLD, applied OFFLINE in PHASE 2 (fixed, no sweep).")
    g.add_argument("--operating-confidence", type=float, default=0.30,
                   help="The operating point of precision / recall / F1 / IoU1 / IoU2. The "
                        "notebook keeps it EQUAL to --threshold (0.30), so the inference "
                        "threshold and the operating point are the same single value and no "
                        "confidence x NMS sweep is performed.")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot (CELL 3 / CELL 23)")
    g.add_argument("--plot-evaluation-mode", default="all_gt", choices=EVALUATION_MODES,
                   help="PLOT_EVALUATION_MODE: the image-level AP50 of this mode selects "
                        "the image that gets plotted.")
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
                   help="Skip CELL 23 entirely (it is the only PHASE 2 step that reopens the "
                        "original images).")

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

    if args.n_exemplars != 0:
        p.error("--n-exemplars must be 0 for the text-only pipeline")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.tile_size < 64:
        p.error("--tile-size must be >= 64")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        p.error("--overlap must be >= 0 and < --tile-size")

    # PROMPT_TYPE (CELL 3)
    args.prompt_type = "text"
    # USE_TILING (CELL 3) - constant here, kept so it reaches experiment_summary.csv
    args.use_tiling = True

    return args

# =============================================================================
#  CELL 4 - OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections (NPZ, one file per image)
#                          + one runs_manifest_shard<i>.csv per shard
#     metrics/             image / experiment / dataset level CSVs
#     confusion_matrices/  CSV + PNG for all_gt
#     plots/               qualitative GT-vs-prediction figures (CELL 23)
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
#  CELL 6 - DATASET AND YOLO ANNOTATION HELPERS
# =============================================================================
@dataclass(frozen=True)
class ImageRecord:
    """One image plus everything needed to evaluate it."""
    archive: str        # AGS_Multi_Rumex | AgsSpringRumex
    flight: str         # e.g. 20220518_Eschikon ("" if images/ has no sub-folder)
    image_id: str       # "<archive>/<flight>/<image name>"  - unique across both archives
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
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                if int(parts[0]) != class_id:
                    continue
                xc, yc, bw, bh = map(float, parts[1:5])
            except (ValueError, IndexError):
                continue
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
    """
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

# =============================================================================
#  CELL 8 - CORRECT ONE-TO-ONE MATCHING
# =============================================================================
def match_one_to_one(pred_boxes, pred_scores, gt_boxes, iou_threshold: float) -> dict:
    """
    Input : pred_boxes  (P, 4), pred_scores (P,), gt_boxes (G, 4), iou_threshold
    Output: dict with matching arrays and IoUs.
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
        if not gt_free.any():
            break
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
#  CELL 9 - TILES: GENERATED ONCE PER IMAGE
# =============================================================================
def tile_bboxes(img_w: int, img_h: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
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
    return list(dict.fromkeys(tiles))  # remove duplicates, keep order

def build_tile_cache(image: Image.Image, tile_size: int, overlap: int,
                     cache_in_memory: bool) -> List[dict]:
    """
    Build the tile list for ONE already-open image.
    """
    coords = tile_bboxes(image.width, image.height, tile_size, overlap)
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
#  CELL 10 - SAM3 TEXT-ONLY BATCH INFERENCE  (FP16/BF16, no masks kept)
# =============================================================================
def _to_numpy(x) -> np.ndarray:
    """Handles tensors (incl. bf16/fp16), lists of tensors, or plain arrays."""
    import torch
    if torch.is_tensor(x):
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and torch.is_tensor(x[0]):
            x = [t.float() if t.dtype in (torch.bfloat16, torch.float16) else t for t in x]
            return torch.stack([t.detach().cpu() for t in x]).numpy()
        return np.array(x)
    return np.array(x)

class Sam3Runner:
    """Owns the model + processor and performs batched tile forward passes."""
    def __init__(self, model_id: str, device: Optional[str], dtype: str, mask_threshold: float):
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
            self.model_dtype = torch.float32

        self.mask_threshold = mask_threshold

        print(f"Loading SAM3 from '{model_id}' onto {self.device} ({dtype}) ...")
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=self.model_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        print("SAM3 loaded.")
        print("  model device:", next(self.model.parameters()).device)
        print("  model dtype :", next(self.model.parameters()).dtype)

    def run_text_batch(self, tile_images: List[Image.Image], text_prompt: str, threshold: float):
        """
        Run SAM3 on a batch of raw tiles with a text prompt only.
        Returns: list of (boxes, scores) in TILE coordinates.
        """
        torch = self.torch
        inputs = self.processor(
            images=tile_images,
            text=[text_prompt] * len(tile_images),
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

            for res in results:
                boxes = _to_numpy(res["boxes"]).reshape(-1, 4)
                scores = _to_numpy(res["scores"]).reshape(-1)
                if "masks" in res:
                    res["masks"] = None  # masks are never kept
                per_image.append((boxes, scores))

        del inputs, outputs, results
        return per_image

def run_text_prompt_over_tiles(runner: Sam3Runner, tile_cache: List[dict],
                               full_image: Image.Image, text_prompt: str,
                               batch_size: int, threshold: float) -> dict:
    """
    Run SAM3 with ONE text prompt over ALL cached tiles of ONE image.
    Output: dict of numpy arrays, all in ORIGINAL-IMAGE coordinates.
    """
    all_boxes, all_scores, all_tids, all_tboxes = [], [], [], []
    per_tile_counts = {}

    for start in range(0, len(tile_cache), batch_size):
        batch_tiles = tile_cache[start:start + batch_size]
        tile_images = [get_tile_image(tile, full_image) for tile in batch_tiles]

        batch_results = runner.run_text_batch(tile_images, text_prompt, threshold)

        for tile, (boxes, scores) in zip(batch_tiles, batch_results):
            per_tile_counts[tile["tile_id"]] = int(len(scores))
            # tile coords -> original image coords
            for b, s in zip(boxes, scores):
                all_boxes.append([b[0] + tile["x1"], b[1] + tile["y1"],
                                  b[2] + tile["x1"], b[3] + tile["y1"]])
                all_scores.append(float(s))
                all_tids.append(int(tile["tile_id"]))
                all_tboxes.append([tile["x1"], tile["y1"], tile["x2"], tile["y2"]])

        del tile_images, batch_results
        if runner.device.startswith("cuda"):
            runner.torch.cuda.empty_cache()

    return {
        "boxes": np.array(all_boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.array(all_scores, dtype=np.float32).reshape(-1),
        "tile_id": np.array(all_tids, dtype=np.int32).reshape(-1),
        "tile_boxes": np.array(all_tboxes, dtype=np.int32).reshape(-1, 4),
        "per_tile_counts": per_tile_counts,
    }

# =============================================================================
#  CELL 11 - PRE-NMS DETECTION STORAGE
# =============================================================================
def run_npz_path(raw_detections_dir: Path, image_id: str) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one image."""
    return raw_detections_dir / f"{safe_filename(image_id)}.npz"

def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        text_prompt: str, detections: dict, gt_boxes: np.ndarray,
                        image_size: Tuple[int, int],
                        archive: str, flight: str, class_id: int) -> Path:
    """
    Write one run's pre-NMS detections to NPZ. The file is self-contained: it also
    stores the GT boxes, so the whole offline evaluation can run without re-opening
    images or label files.
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
        boxes=detections["boxes"].astype(np.float32),        # x1,y1,x2,y2 (original img)
        scores=detections["scores"].astype(np.float32),      # confidence > 0.30
        tile_id=detections["tile_id"].astype(np.int32),
        tile_boxes=detections["tile_boxes"].astype(np.int32),
    )
    return path

def load_run_detections(path: Path) -> dict:
    """Read one run NPZ back into a plain python dict."""
    with np.load(path, allow_pickle=False) as z:
        run = {
            "image_ID": str(z["image_ID"]),
            "text_prompt": str(z["text_prompt"]),
            "image_width": int(z["image_width"]),
            "image_height": int(z["image_height"]),
            "gt_boxes": z["gt_boxes"].reshape(-1, 4),
            "boxes": z["boxes"].reshape(-1, 4),
            "scores": z["scores"].reshape(-1),
            "tile_id": z["tile_id"].reshape(-1) if "tile_id" in z else np.array([], dtype=np.int32),
            "tile_boxes": z["tile_boxes"].reshape(-1, 4) if "tile_boxes" in z else np.array([], dtype=np.int32).reshape(-1, 4),
        }
        # read while the NPZ is still open (it is closed when the "with" block ends)
        run["archive"] = str(z["archive"]) if "archive" in z else ""
        run["flight"] = str(z["flight"]) if "flight" in z else ""
    return run

# =============================================================================
#  RESUME SUPPORT  (CELL 12, adapted to several shard manifests)
# =============================================================================
def load_done_images(paths: Paths, experiment_name: str) -> set:
    """
    Read every shard manifest and return {image_ID} of the images that are already
    finished.
    """
    done: set = set()
    for csv_path in sorted(paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    try:
                        done.add(row["image_ID"])
                    except (KeyError, ValueError, TypeError):
                        continue
        except OSError:
            continue
    return done

# =============================================================================
#  CELL 12 - MAIN GPU INFERENCE LOOP  (PHASE 1)
# =============================================================================
def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    import torch
    exp = args.experiment_name

    # ---- resume support ------------------------------------------------------
    done_images: set = set()
    if not args.no_resume:
        done_images = load_done_images(paths, exp)
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

    runner = Sam3Runner(args.model_id, args.device, args.dtype, args.mask_threshold)
    device_is_cuda = runner.device.startswith("cuda")

    start_time = time.time()
    n_new_runs = 0
    image_times: List[float] = []
    n_total_images = len(records)

    try:
        for img_idx, rec in enumerate(records, start=1):
            image_t0 = time.time()
            image_id = rec.image_id

            if image_id in done_images:
                continue

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
            tile_cache = build_tile_cache(image, args.tile_size, args.overlap, args.cache_tiles_in_memory)

            run_t0 = time.time()
            detections = run_text_prompt_over_tiles(
                runner, tile_cache, image, args.text_prompt,
                batch_size=args.batch_size, threshold=args.threshold
            )
            run_seconds = time.time() - run_t0

            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, args.text_prompt,
                detections, gt_boxes, (img_w, img_h),
                rec.archive, rec.flight, rec.class_id
            )

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
                "n_tiles": len(tile_cache),
                "n_detections_pre_nms": int(len(detections["scores"])),
                "image_width": img_w,
                "image_height": img_h,
                "npz_file": npz_path.name,
                "inference_seconds": round(run_seconds, 2),
            })
            manifest_file.flush()
            n_new_runs += 1

            n_dets = int(len(detections["scores"]))
            n_tiles_with_dets = sum(1 for v in detections["per_tile_counts"].values() if v > 0)

            # ---------------- release the image and its tiles ----------------------
            n_tiles = len(tile_cache)
            for t in tile_cache:
                if t["image"] is not None:
                    t["image"].close()
            del tile_cache, detections
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
                  f"{n_gt} GT box(es) ({n_gt_large} > {args.overlap}px) | "
                  f"pre-NMS detections={n_dets} from {n_tiles_with_dets}/{n_tiles} tiles | "
                  f"{run_seconds:.1f}s | "
                  f"avg/image={avg_per_image:.1f}s | ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")

    finally:
        manifest_file.close()

    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new images.")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Pre-NMS detections in: {paths.raw_detections}")

# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================
def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading SAM3 ---")
    sample = my_records[:min(len(my_records), 200)]
    total_images, per_archive = 0, {}
    total_tiles = 0
    
    # Estimate tiles for a typical 8192x5460 image
    step = max(1, args.tile_size - args.overlap)
    tiles_per_img = ((8192 + step - 1) // step) * ((5460 + step - 1) // step)

    for rec in sample:
        total_images += 1
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + 1
        
    total_tiles = total_images * tiles_per_img

    print(f"  sampled {len(sample)} image(s) of this shard -> {total_images} image runs "
          f"({per_archive})")
    print(f"  forward passes per image: ~{tiles_per_img} "
          f"(tiling: {args.tile_size}x{args.tile_size} tiles, {args.overlap}px overlap on an 8192x5460 image)")
    print(f"  => ~{total_tiles} SAM3 forward passes for those {len(sample)} images")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{total_images} "
          f"(one per image)")

# =============================================================================
#  CELL 13 - LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
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
        run["archive"] = run.get("archive") or str(row.get("archive", ""))
        run["flight"] = run.get("flight") or str(row.get("flight", ""))
        run["n_gt_larger_than_overlap"] = int(row.get("n_gt_larger_than_overlap", 0))
        runs.append(run)

    if n_missing:
        print(f"  WARNING: {n_missing} manifest row(s) point at a missing NPZ (skipped).")

    print(f"Loaded {len(runs)} runs (= images) for {experiment_name}, prompt '{runs[0]['text_prompt'] if runs else ''}'.")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes over all runs:", int(sum(len(r['gt_boxes']) for r in runs)))
    return runs, manifest

# =============================================================================
#  CELL 14 - OFFLINE NMS (with tile provenance)
# =============================================================================
def nms_with_provenance(boxes, scores, iou_threshold: float):
    """
    Input : boxes (N,4), scores (N,), iou_threshold
    Output: keep (list of kept indices, highest score first)
            suppressed (list of (suppressed_index, suppressor_index))
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
    Output: dict with surviving detections SORTED BY SCORE, plus duplicate-source counters.
    """
    keep, suppressed = nms_with_provenance(run["boxes"], run["scores"], iou_threshold)
    keep = np.array(keep, dtype=int)

    cross_tile_suppressed, same_tile_suppressed = 0, 0
    tile_ids = run.get("tile_id")
    if tile_ids is not None and len(tile_ids) > 0:
        for s, k in suppressed:
            if tile_ids[s] != tile_ids[k]:
                cross_tile_suppressed += 1
            else:
                same_tile_suppressed += 1

    return {
        "boxes": run["boxes"][keep].reshape(-1, 4),
        "scores": run["scores"][keep].reshape(-1),
        "tile_id": run["tile_id"][keep].reshape(-1) if tile_ids is not None else None,
        "tile_boxes": run["tile_boxes"][keep].reshape(-1, 4) if "tile_boxes" in run else None,
        "n_pre_nms": int(len(run["scores"])),
        "n_suppressed_cross_tile": int(cross_tile_suppressed),
        "n_suppressed_same_tile": int(same_tile_suppressed),
    }

# =============================================================================
#  CELL 15 - EVALUATION CORE (all_gt)
# =============================================================================
def evaluate_predictions(pred_boxes, pred_scores, gt_boxes, eval_iou: float) -> dict:
    """
    Evaluate ONE prediction set against ALL GT boxes of the image.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)

    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    status = np.full(n_pred, STATUS_FP, dtype=np.int8)

    match = match_one_to_one(pred_boxes, pred_scores, gt_boxes, eval_iou)
    status[match["pred_match_gt"] >= 0] = STATUS_TP

    tp = int((status == STATUS_TP).sum())
    fp = int((status == STATUS_FP).sum())
    fn = int(n_gt - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = safe_f1(precision, recall)
    matched_ious = match["matched_ious"]
    iou1 = float(np.mean(matched_ious)) if matched_ious else 0.0
    iou2 = float(np.sum(matched_ious) / n_gt) if n_gt > 0 else 0.0

    return {
        "status": status, "pred_match_gt": match["pred_match_gt"],
        "TP": tp, "FP": fp, "FN": fn, "n_gt": n_gt, "n_pred": n_pred,
        "precision": float(precision), "recall": float(recall), "F1": float(f1),
        "IoU1": iou1, "IoU2": iou2,
    }

# =============================================================================
#  CELL 16 - AP50 AND AP50:95  (supervision.metrics.MeanAveragePrecision)
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
    Input : two equally long lists of supervision Detections.
    Output: (AP50, AP50_95). NaN when there is no GT at all to evaluate.
    """
    from supervision.metrics import MeanAveragePrecision
    if len(gt_list) == 0 or sum(len(g) for g in gt_list) == 0:
        return float("nan"), float("nan")
    try:
        result = MeanAveragePrecision().update(pred_list, gt_list).compute()
        ap50, ap5095 = float(result.map50), float(result.map50_95)
        return (ap50 if ap50 >= 0 else float("nan"),
                ap5095 if ap5095 >= 0 else float("nan"))
    except Exception as e:
        print("   (AP computation failed:", e, ")")
        return float("nan"), float("nan")

# =============================================================================
#  CELL 17 - OPERATING-POINT EVALUATION
# =============================================================================
def evaluate_at_operating_point(nms_run, gt_boxes,
                                confidence_threshold: float, eval_iou: float) -> dict:
    """
    Input : nms_run  - output of apply_nms_to_run
            gt_boxes - all GT boxes of the image
    """
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                gt_boxes, eval_iou)

# =============================================================================
#  CELL 23 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right)
# =============================================================================
def _draw_boxes(ax, boxes, color, linewidth=1.5, linestyle="-", labels=None, fontsize=6):
    """Draw [x1,y1,x2,y2] boxes on a matplotlib axis."""
    import matplotlib.patches as patches
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=color, linewidth=linewidth,
                                       linestyle=linestyle))
        if labels is not None:
            ax.text(x1, y1 - 2, labels[k], color="white", fontsize=fontsize,
                    va="bottom", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.8, alpha=0.85))

def select_image_for_plot(image_level_df, mode: str, min_gt: int,
                          archive: Optional[str] = None):
    """
    Output: (image_level_row, selection_rule_text) or (None, reason)
    """
    img_df = image_level_df[image_level_df["evaluation_mode"] == mode].copy()
    if archive is not None:
        img_df = img_df[img_df["archive"] == archive]
    if img_df.empty:
        return None, f"no image for archive={archive}"

    img_df = img_df[img_df["AP50"].notna() & img_df["n_gt"].notna()]
    if img_df.empty:
        return None, "no image with a valid AP50"

    img_df["n_gt"] = img_df["n_gt"].astype(int)
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

    ranked = candidates.sort_values(["AP50", "F1", "n_gt"],
                                    ascending=False, na_position="last")
    print("Top candidates:")
    print(ranked[["image_ID", "n_gt", "n_predictions", "AP50", "F1",
                  "precision", "recall"]].head(5).to_string(index=False))
    return ranked.iloc[0], rule

def plot_gt_vs_predictions(args, paths: Paths, runs, image_level_df,
                           image_paths: dict, plt, scope_label: str,
                           archive: Optional[str] = None) -> None:
    """One qualitative figure for one scope."""
    from matplotlib.lines import Line2D
    mode = args.plot_evaluation_mode
    exp = args.experiment_name

    img_row, rule = select_image_for_plot(image_level_df, mode,
                                          args.plot_min_gt_boxes, archive)
    if img_row is None:
        print(f"Nothing to plot for {scope_label}:", rule)
        return

    plot_image_id = img_row["image_ID"]
    run = next((r for r in runs if r["image_ID"] == plot_image_id), None)
    if run is None:
        print(f"Nothing to plot for {scope_label}: no cached run for {plot_image_id}")
        return

    nms_run = apply_nms_to_run(run, args.nms_iou_threshold)
    keep = nms_run["scores"] >= args.operating_confidence
    pred_boxes, pred_scores = nms_run["boxes"][keep], nms_run["scores"][keep]
    gt_boxes_plot = run["gt_boxes"]

    if plot_image_id not in image_paths:
        print("Image file not found for", plot_image_id)
        return

    with Image.open(image_paths[plot_image_id]) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(1.0, args.plot_max_display_dim / max(w, h))
        disp_w, disp_h = max(1, int(w * s)), max(1, int(h * s))
        display = np.asarray(im.resize((disp_w, disp_h), Image.BILINEAR))
    to_disp = np.array([disp_w / w, disp_h / h, disp_w / w, disp_h / h], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax in axes:
        ax.imshow(display)
        ax.axis("off")

    _draw_boxes(axes[0], gt_boxes_plot * to_disp, color="yellow", linewidth=1.5)
    axes[0].set_title(f"Ground truth: {len(gt_boxes_plot)} Rumex boxes", fontsize=12)

    score_labels = [f"{v:.2f}" for v in pred_scores] if args.plot_show_scores else None
    _draw_boxes(axes[1], pred_boxes * to_disp, color="red", linewidth=1.5,
                labels=score_labels)
    axes[1].set_title(
        f"Predictions: {len(pred_boxes)} boxes | text prompt = '{run['text_prompt']}'\n"
        f"AP50={img_row['AP50']:.3f}  P={img_row['precision']:.3f}  R={img_row['recall']:.3f}  "
        f"F1={img_row['F1']:.3f}  TP={int(img_row['TP'])} FP={int(img_row['FP'])} FN={int(img_row['FN'])}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color="yellow", lw=2, label="ground truth"),
        Line2D([0], [0], color="red", lw=2, label="prediction"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=11,
               frameon=False)
    fig.suptitle(
        f"{exp} | {plot_image_id} | mode={mode} | scope={scope_label}\n"
        f"tiling: tile={args.tile_size}px, overlap={args.overlap}px | "
        f"conf={args.operating_confidence:.2f}, NMS IoU={args.nms_iou_threshold:.2f}, "
        f"mask thr={args.mask_threshold:.2f}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    png_path = paths.plots / (f"best_image_{scope_label}_{safe_filename(plot_image_id)}"
                              f"_{mode}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[{scope_label}] Selected image : {plot_image_id} "
          f"({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Figure saved   : {png_path}")

def make_qualitative_plots(args, paths: Paths, runs, image_level_df, plt) -> None:
    """CELL 23 for every scope: one figure per archive plus one global figure."""
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
        plot_gt_vs_predictions(args, paths, runs, image_level_df,
                               image_paths, plt, scope_label=archive, archive=archive)
    plot_gt_vs_predictions(args, paths, runs, image_level_df,
                           image_paths, plt, scope_label="ALL", archive=None)

# =============================================================================
#  PHASE 2 - THE WHOLE OFFLINE EVALUATION (notebook CELL 13 ... CELL 23)
# =============================================================================
def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2: notebook CELL 13 ... CELL 23, in one process, no GPU."""
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use("Agg")
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

    print("=" * 92)
    print(f" PHASE 2 - OFFLINE EVALUATION | experiment={exp}")
    print("=" * 92)
    print("Operating configuration is fixed (no sweep performed):")
    print(f"  Confidence threshold = {conf:.2f}  (== the SAM3 inference threshold)")
    print(f"  NMS IoU threshold    = {nms_iou:.2f}")
    print(f"  Evaluation IoU       = {eval_iou:.2f}")

    runs, manifest = load_runs(paths, exp)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  CELL 18 - PER-IMAGE METRICS
    # =========================================================================
    print("\n--- CELL 18: image-level metrics ---")
    image_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)

        p_det = make_detections(nms_run["boxes"], nms_run["scores"])
        g_det = make_detections(run["gt_boxes"])
        ap50, ap5095 = compute_ap([p_det], [g_det])

        ev = evaluate_at_operating_point(nms_run, run["gt_boxes"], conf, eval_iou)

        image_rows.append({
            "experiment_name": exp,
            "image_ID": run["image_ID"],
            "archive": run.get("archive", ""),
            "flight": run.get("flight", ""),
            "Prompt_Type": "text",
            "text_prompt": run.get("text_prompt", args.text_prompt),
            "evaluation_mode": "all_gt",
            "confidence_threshold": conf,
            "nms_iou_threshold": nms_iou,
            "n_gt": ev["n_gt"],
            "n_gt_larger_than_overlap": run.get("n_gt_larger_than_overlap", 0),
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
    print("\n--- CELL 19: experiment-level summary ---")
    summary_row = {
        "experiment_name": exp,
        "evaluation_mode": "all_gt",
        "prompt_type": args.prompt_type,
        "text_prompt": args.text_prompt,
        "n_exemplars": args.n_exemplars,
        "use_tiling": args.use_tiling,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "confidence_threshold": conf,
        "nms_iou_threshold": nms_iou,
        "mask_threshold": args.mask_threshold,
        "eval_iou_threshold": eval_iou,
        "n_images": int(image_level_df["image_ID"].nunique()),
        "n_images_with_predictions": int((image_level_df["n_predictions"] > 0).sum()),
    }
    for col in METRIC_COLUMNS:
        summary_row[f"{col}_mean"] = image_level_df[col].mean()
        summary_row[f"{col}_std"] = image_level_df[col].std()

    experiment_summary_df = pd.DataFrame([summary_row])
    experiment_summary_csv = paths.metrics / "experiment_summary.csv"
    experiment_summary_df.to_csv(experiment_summary_csv, index=False)
    print(f"Experiment summary -> {experiment_summary_csv}\n")
    print(experiment_summary_df.T.to_string(header=False))

    # ---- per-archive version of the same table ---------------------------
    per_archive_rows = []
    for (archive, mode), sub in image_level_df.groupby(["archive", "evaluation_mode"]):
        row = {
            "experiment_name": exp,
            "archive": archive,
            "evaluation_mode": mode,
            "n_images": int(sub["image_ID"].nunique()),
            "n_images_with_predictions": int((sub["n_predictions"] > 0).sum()),
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
    #  CELL 20 - POOLED DATASET AP50 / AP50:95
    # =========================================================================
    print("\n--- CELL 20: pooled dataset AP ---")
    pred_list, gt_list = [], []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)
        pred_list.append(make_detections(nms_run["boxes"], nms_run["scores"]))
        gt_list.append(make_detections(run["gt_boxes"]))
    ap50, ap5095 = compute_ap(pred_list, gt_list)
    del pred_list, gt_list
    gc.collect()

    dataset_ap_df = pd.DataFrame([{
        "experiment_name": exp,
        "evaluation_mode": "all_gt",
        "text_prompt": args.text_prompt,
        "n_images": len(runs),
        "confidence_used_for_AP": args.threshold,
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
    print("\n--- CELL 21: confusion matrices ---")
    def plot_confusion_matrix(tp, fp, fn, title, png_path):
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
        plt.close(fig)

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
    cm_df.to_csv(paths.confusion_matrices / "confusion_matrix_all_gt.csv")

    if _HAS_MPL:
        plot_confusion_matrix(
            tp, fp, fn,
            f"{exp} - all_gt\n"
            f"prompt='{args.text_prompt}' | "
            f"conf={conf:.2f}, NMS IoU={nms_iou:.2f}, eval IoU={eval_iou:.2f}",
            paths.confusion_matrices / "confusion_matrix_all_gt.png")

    confusion_summary_df = pd.DataFrame([{
        "experiment_name": exp, "evaluation_mode": "all_gt",
        "TP": tp, "FP": fp, "FN": fn,
        "precision_micro": precision, "recall_micro": recall, "F1_micro": f1,
        "confidence_threshold": conf, "nms_iou_threshold": nms_iou,
        "eval_iou_threshold": eval_iou,
    }])
    confusion_summary_df.to_csv(
        paths.confusion_matrices / "confusion_matrix_summary.csv", index=False)

    print(f"all_gt: TP={tp}  FP={fp}  FN={fn}  "
          f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")
    print("\nConfusion matrices saved to:", paths.confusion_matrices)

    # =========================================================================
    #  CELL 23 - QUALITATIVE FIGURES
    # =========================================================================
    if args.no_plots:
        print("\n--- CELL 23: qualitative figures skipped (--no-plots) ---")
    elif not _HAS_MPL:
        print("\n--- CELL 23: qualitative figures skipped (matplotlib unavailable) ---")
    else:
        print("\n--- CELL 23: qualitative GT-vs-prediction figures ---")
        make_qualitative_plots(args, paths, runs, image_level_df, plt)

    # =========================================================================
    #  CELL 22 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY")
    print("=" * 78)
    print(f"Prompt                   : text only, '{args.text_prompt}'")
    print(f"Tiling                   : {args.use_tiling}  (TILE_SIZE={args.tile_size}, OVERLAP={args.overlap})")
    print(f"Confidence threshold     : {conf:.2f} (fixed; SAM3 run once per image x tile)")
    print(f"Mask threshold           : {args.mask_threshold:.2f}")
    print(f"NMS IoU threshold        : {nms_iou:.2f} (fixed)")
    print(f"Evaluation               : all_gt, IoU >= {eval_iou:.2f}")
    print(f"Images                   : {len(runs)}  "
          f"({summary_row['n_images_with_predictions']} with at least one prediction)")
    print("-" * 78)
    print("EXPERIMENT-LEVEL RESULTS (mean over images, std between images)")
    show = ["AP50_mean", "AP50_std", "AP50_95_mean", "precision_mean",
            "recall_mean", "F1_mean", "F1_std", "IoU1_mean", "IoU2_mean"]
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
        import supervision
        from supervision.metrics import MeanAveragePrecision
        return True
    except Exception:
        return False

def _check_pandas() -> bool:
    try:
        import pandas
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
                  "'Problem B' in text_only_with_tiling_HOW_TO_RUN.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" TEXT_ONLY_WITH_TILING SAM3 PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" text_prompt    : '{args.text_prompt}'")
    print(f" n_exemplars    : {args.n_exemplars}  (text-only, no visual prompts)")
    print(f" tiling         : {args.use_tiling}  (TILE_SIZE={args.tile_size}, OVERLAP={args.overlap})")
    print(f" batch size     : {args.batch_size}  (tiles per forward pass)")
    print(f" cache tiles    : {args.cache_tiles_in_memory}")
    print(f" sam3 threshold : {args.threshold}  (single inference pass per tile)")
    print(f" mask threshold : {args.mask_threshold}")
    print(f" dtype          : {args.dtype}")
    print(f" operating pt   : confidence={args.operating_confidence}, "
          f"NMS IoU={args.nms_iou_threshold} (fixed, no sweep)")
    print(f" eval IoU       : {args.eval_iou_threshold}")
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