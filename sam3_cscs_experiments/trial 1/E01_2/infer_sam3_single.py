#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TILED SINGLE-BOX SAM3 EXEMPLAR 
 
E01_2 is the one-exemplar ablation: tiling ON, but the prompt is a SINGLE crop (the anchor
box alone) instead of three. Everything else is the E02_2 pipeline, so the two experiments
answer exactly one question: how much does the model lose when it sees one example of the
plant instead of three?
 
 
DATASET:
-The dataset root contains four archives; we deliberately use only two of them: AGS_Multi_Rumex && AgsSpringRumex
-the class id that means "rumex" (0 in AGS_Multi_Rumex, 2 in AgsSpringRumex);
 
 
THE PIPELINE, STEP BY STEP:
For every image, and then for every ground-truth box in that image ("anchor"):
 
  1. EXEMPLAR SELECTION       (select_exemplar_indices)
     The anchor box is always exemplar #1. The remaining N_EXEMPLARS-1 slots are
     filled by sampling *other* GT boxes from the same image, without replacement,
     using a seed derived deterministically from (image_ID, anchor_idx) -> the exact
     same exemplar set is reproduced on every rerun, on every machine, in any order.
     At the E01_2 default of 1 exemplar this branch never fires: the set is [anchor].
     (The notebook seeded with `hash((image_id, anchor_idx))`, which Python randomises
     per process unless PYTHONHASHSEED is fixed. Moot at 1 exemplar, wrong above it.)
 
  2. TILING                   (tile_bboxes)
     A 8192x5460 image cannot be fed to SAM3 in one piece without destroying small
     plants (the processor internally resizes to ~1008 px). So the image is cut into
     overlapping TILE_SIZE x TILE_SIZE windows with OVERLAP px of overlap, so that a
     plant lying on a tile border is still fully visible in at least one window.
 
  3. STRIP COMPOSITION        (compose_tile_with_exemplars)
     SAM3 wants its box prompts to live in the SAME image as the search region. So for
     each tile we build a composite: a thin horizontal "strip" on top holding the
     exemplar crops side by side, with the tile pasted underneath.
       - the strip background is real texture sampled from the tile itself, not white,
         so the vision transformer sees no artificial white/photo edge;
       - each exemplar crop is alpha-feathered at its border so it blends in.
 
  4. SAM3 FORWARD PASS        (Sam3Runner.run_on_tile)
     The exemplar boxes (in composite coordinates) are passed as positive boxes
     (label = 1). Outputs are post-processed to instance masks + boxes + scores.
 
  5. STRIP REMOVAL            (keep_only_target_region_detections)
     Any detection that lives in the strip (i.e. detections *of the prompts themselves*)
     is dropped, and the surviving boxes/masks are shifted back into tile coordinates.
 
  6. PLAUSIBILITY FILTER      (filter_implausible_boxes)
     Drops boxes that are degenerate (a few px), absurdly large (> 60% of the tile),
     or whose predicted mask fills < 15% of the proposed box (usually background blobs).
 
  7. TILE -> IMAGE            (run_sam3_pipeline)
     Surviving boxes are translated from tile coordinates into full-image coordinates.
 
  8. NMS MERGE                (nms_merge)
     Because tiles overlap, the same plant is detected several times. A single global
     greedy NMS over all detections of the image removes the duplicates.
 
  9. METRICS                  (compute_detection_metrics)
     mAP50, precision, recall, IoU1 (mean IoU over matched pairs only) and
     IoU2 (sum of matched IoUs / number of GT, so misses count as 0).
     mAP50 is computed with `supervision`
 
 
OUTPUTS:
    <output-dir>/
    |-- run_config_<EXP>.json                       full config snapshot
    |-- runs/
    |   |-- results_<EXP>_shard<k>.csv              raw rows, append-safe, THE resume source
    |   |-- detections_<EXP>_shard<k>.jsonl         boxes + scores per run
    |   `-- masks_<EXP>_shard<k>.jsonl              only if --save-masks
    |-- results_<EXP>_all.csv                       every row, deduplicated
    |-- results_<EXP>_AGS_Multi_Rumex.csv
    |-- results_<EXP>_AgsSpringRumex.csv
    |-- raw_summary_<EXP>_{all,<archive>}.csv       mean/std over every run
    |-- image_level_<EXP>_{all,<archive>}.csv       one row per image
    `-- summary_<EXP>_{all,<archive>}.csv           mean/std over the image-level means
 
The three summary files reproduce the notebook's last three cells: raw_summary_* is the
"every row is an independent observation" table, image_level_* collapses the prompt runs
of one image to one value, and summary_* averages those image-level means so that an
image with 40 plants does not outweigh an image with 2.
 
 
PARALLELISM / RESUME
--num-shards N --shard-index k splits the image list round-robin so N processes (one
per GPU) can run the same experiment concurrently, each writing its own CSV/JSONL. On
startup every shard reads all shard CSVs already present and skips any (image_ID,
anchor_idx) already recorded, so re-submitting an interrupted job simply continues.
"""
 
from __future__ import annotations
 
import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
 
import numpy as np
from PIL import Image
 
 
 
#static configuration
 
ARCHIVES: dict[str, int] = {
    "AGS_Multi_Rumex": 0,
    "AgsSpringRumex": 2,
}
 
IGNORED_ARCHIVES = ("AGS_Multiple_Fields", "AGS_Multiple_Fields_Embeddings")
 
VALID_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
 
NON_LABEL_FILES = {"darknet.labels", "classes.txt", "obj.names"}
 
CSV_COLUMNS = [
    "experiment_name",
    "image_ID",
    "anchor_idx",
    "Prompt_ID",
    "Prompt_Type",
    "mAP50",
    "precision",
    "recall",
    "IoU1",
    "IoU2",
    "archive",
    "flight",
    "source_class_id",
    "n_gt",
    "n_pred",
    "image_width",
    "image_height",
    "n_tiles",          # tiles this image was cut into (E01_2 extra column)
    "runtime_s",
]
 
 
# CLI
 
def default_dataset_root() -> Path:
    scratch = os.getenv("SCRATCH")
    if scratch:
        return Path(scratch) / "overney" / "dataset"
    return Path(__file__).resolve().parents[2] / ".." / "02_data" / "dataset"
 
 
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAM3 tiled, single-exemplar Rumex detection benchmark "
                    "(inference + evaluation). Port of the E01_2 Colab notebook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    #paths
    g = p.add_argument_group("paths")
    g.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                   help="Folder containing the archive folders (AGS_Multi_Rumex, AgsSpringRumex, ...).")
    g.add_argument("--output-dir", type=Path, required=True,
                   help="Where CSVs, detections and summaries are written.")
    g.add_argument("--archives", nargs="*", default=list(ARCHIVES.keys()),
                   help="Subset of archives to run on. Default: both.")
 
    #experiment identity
    g = p.add_argument_group("experiment")
    g.add_argument("--experiment-name", default="E01_2",
                   help="Free-form tag written into every CSV row and into the file names.")
    g.add_argument("--preset", choices=["single", "multi"], default=None,
                   help="Convenience: single = 1 exemplar crop, the anchor alone (E01_2, the "
                        "default), multi = 3 exemplar crops (reproduces E02_2's prompt with "
                        "this script). --n-exemplars always wins over the preset.")
    g.add_argument("--n-exemplars", type=int, default=None,
                   help="Number of exemplar crops per prompt (anchor + n-1 sampled others). "
                        "Default 1.")
 
    # SAM3 / tiling parameters
    g = p.add_argument_group("sam3 / tiling")
    g.add_argument("--model-id", default="facebook/sam3",
                   help="HF repo id OR a local snapshot directory (use a local path when the "
                        "compute node has no internet).")
    g.add_argument("--tile-size", type=int, default=1000)
    g.add_argument("--overlap", type=int, default=150)
    g.add_argument("--threshold", type=float, default=0.3,
                   help="SAM3 confidence threshold.")
    g.add_argument("--mask-threshold", type=float, default=0.4,
                   help="Binarisation threshold for the predicted masks.")
    g.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for TP matching (the '50' in mAP50).")
    g.add_argument("--nms-iou", type=float, default=0.5,
                   help="IoU threshold used when merging detections from overlapping tiles. "
                        "0.5.")
    g.add_argument("--margin", type=int, default=6, help="Padding around exemplar crops in the strip.")
    g.add_argument("--feather-width", type=int, default=8, help="Alpha-blend border of exemplar crops.")
    g.add_argument("--min-fill-ratio", type=float, default=0.15)
    g.add_argument("--max-area-fraction", type=float, default=0.6)
    g.add_argument("--edge-margin", type=int, default=5)
 
    # runtime
    g = p.add_argument_group("runtime")
    g.add_argument("--device", default=None,
                   help="'cuda', 'cuda:0', 'cpu'. Default: cuda if available.")
    g.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32",
                   help="Model dtype. float32 reproduces the notebook exactly; bfloat16 is "
                        "substantially faster on H100/GH200 with a negligible accuracy change.")
    g.add_argument("--num-shards", type=int, default=1,
                   help="Split the image list across this many concurrent processes.")
    g.add_argument("--shard-index", type=int, default=0, help="Which shard this process handles (0-based).")
    g.add_argument("--limit-images", type=int, default=0,
                   help="Debug: process at most this many images (0 = no limit).")
    g.add_argument("--max-anchors-per-image", type=int, default=0,
                   help="Debug / cost control: use at most this many GT boxes as anchors per image "
                        "(0 = every GT box becomes an anchor once, as in the notebook).")
    cache = g.add_mutually_exclusive_group()
    cache.add_argument("--cache-tiles", dest="cache_tiles", action="store_true", default=True,
                       help="Crop the tiles once per image and reuse them for every anchor of "
                            "that image (the notebook's optimisation; ~70 crops instead of "
                            "70 x n_anchors). Costs ~210 MB of RAM at tile-size 1000.")
    cache.add_argument("--no-cache-tiles", dest="cache_tiles", action="store_false",
                       help="Re-crop the tiles for every anchor. Slower, lower memory.")
 
    #  persistence
    g = p.add_argument_group("persistence")
    g.add_argument("--save-detections", dest="save_detections", action="store_true", default=True,
                   help="Append per-run boxes + scores to a JSONL file (default: on).")
    g.add_argument("--no-save-detections", dest="save_detections", action="store_false")
    g.add_argument("--save-masks", action="store_true", default=False,
                   help="Also store every predicted mask losslessly as COCO-style RLE. OFF by "
                        "default: at full resolution this produces very large files.")
    g.add_argument("--no-resume", action="store_true",
                   help="Ignore existing CSV rows and recompute everything.")
 
    #  modes
    g = p.add_argument_group("modes")
    g.add_argument("--dry-run", action="store_true",
                   help="Discover the dataset, print the run plan and exit. No model, no GPU.")
    g.add_argument("--aggregate-only", action="store_true",
                   help="Skip inference; just merge existing shard CSVs and rebuild the summaries.")
    g.add_argument("--no-aggregate", action="store_true",
                   help="Do not run the aggregation step after inference finishes.")
 
    args = p.parse_args(argv)
 
    #  resolve the preset -> concrete values
    preset_table = {                       # n_exemplars
        "single": 1,
        "multi": 3,
    }
    preset_n = preset_table.get(args.preset, 1)
    if args.n_exemplars is None:
        args.n_exemplars = preset_n
 
  
    args.use_tiling = True
 
    if args.n_exemplars < 1:
        p.error("--n-exemplars must be >= 1")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.overlap >= args.tile_size:
        p.error("--overlap must be smaller than --tile-size")
 
    return args
 
 
 
# dataset discovery
@dataclass(frozen=True)
class ImageRecord:
    #One image plus everything needed to evaluate it
    archive: str
    flight: str
    image_id: str
    image_path: Path
    label_path: Path
    class_id: int
 
 
def _index_flat_labels(annotations_root: Path) -> dict[str, Path]:
    #Scan all annotation files once, create {image_name -> label_file} mapping, and make label lookup faster and safer.
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
    # scans the chosen archives, finds valid images with matching annotations,
    # builds the final list of images that will be processed by your SAM3 experiment,
    # and sorts that list in a stable order.
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
        print(f"  WARNING: {len(missing)} image(s) have no matching label file and were skipped. "
              f"First few: {missing[:5]}")
 
    # Deterministic global order -> shard assignment is stable across processes and reruns.
    records.sort(key=lambda r: r.image_id)
    return records
 
 
def load_yolo_boxes(label_path: Path, img_width: int, img_height: int, class_id: int) -> np.ndarray:
        #file line:  class id x y w h            all normalised to [0, 1]
        #returned :  [x1, y1, x2, y2]            absolute pixels, clipped to the image
    boxes: list[list[float]] = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue                      # skip blank lines
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue                      # skip malformed / header lines
            if cls != class_id:
                continue                      # keep only the archive's rumex class
            if len(parts) < 5:
                continue
            xc, yc, bw, bh = map(float, parts[1:5])
            # normalised -> pixels
            xc, yc = xc * img_width, yc * img_height
            bw, bh = bw * img_width, bh * img_height
            # centre form -> corner form, then clip into the image
            x1 = max(0.0, xc - bw / 2.0)
            y1 = max(0.0, yc - bh / 2.0)
            x2 = min(float(img_width), xc + bw / 2.0)
            y2 = min(float(img_height), yc + bh / 2.0)
            if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
                continue
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
 
 
 
#exemplar selection
def stable_seed(*parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "big") % (2 ** 32)
 
 
def select_exemplar_indices(n_gt: int, anchor_idx: int, n_exemplars: int, image_id: str) -> List[int]:
    # Build the exemplar index set for one run:
    #   * the anchor is always first (so the Prompt_ID string always starts with the anchor);
    #   * the remaining n_exemplars-1 slots are filled from the other GT boxes of the SAME
    #     image, sampled without replacement;
    #   * if the image has fewer boxes than requested, we simply use what is available --
    #     no duplication, no crash. (An image with a single plant therefore always runs as
    #     a 1-exemplar prompt, whatever --n-exemplars says.)
    rng = np.random.default_rng(stable_seed(image_id, anchor_idx))
    others = [i for i in range(n_gt) if i != anchor_idx]
    n_needed = min(n_exemplars - 1, len(others))
    chosen = list(rng.choice(others, size=n_needed, replace=False)) if n_needed > 0 else []
    return [anchor_idx] + [int(i) for i in chosen]
 
 
def format_prompt_id(exemplar_indices: Sequence[int]) -> str:
    #  "5"  (single)  |  "5+12+3"  (multiple -- anchor first, order preserved).
    return "+".join(str(int(i)) for i in exemplar_indices)
 
 
 
#image composition
def get_local_background_patch(tile_img: Image.Image, patch_w: int, patch_h: int) -> Image.Image:
 
    # Background for the exemplar strip: REAL texture taken from the tile itself instead of
    # a flat white block. Reasons:
    #   * brightness / white balance / grain match this tile's own lighting, which varies a
    #     lot across a large drone photo;
    #   * there is no artificial white-vs-photo edge for the ViT's self-attention to latch
    #     onto as a spurious feature;
    #   * after the encoder mixes in neighbouring context, the exemplar's pooled feature ends
    #     up looking like "a plant in grass" rather than "a plant in a void"
 
    # We take the tile's top-left corner: the strip sits ABOVE the tile in the composite, so
    # this region is duplicated in appearance but never overlaps the search area.
 
    sample = tile_img.crop((0, 0, min(patch_w, tile_img.width), min(patch_h, tile_img.height)))
    if sample.size != (patch_w, patch_h):
        sample = sample.resize((patch_w, patch_h))
    return sample
 
 
def make_feather_mask(size: Tuple[int, int], feather_width: int) -> Image.Image:
 
    # Soft alpha mask so a pasted exemplar crop fades into the strip background instead of
    # ending in a hard rectangle
 
    w, h = size
    mask = np.full((h, w), 255.0, dtype=np.float32)
    effective = min(feather_width, h // 2, w // 2)
    if effective >= 1:
        for i in range(effective):
            alpha = 255.0 * (i + 1) / effective
            mask[i, :] = np.minimum(mask[i, :], alpha)              # top edge
            mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)  # bottom edge
            mask[:, i] = np.minimum(mask[:, i], alpha)              # left edge
            mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)  # right edge
    return Image.fromarray(mask.astype(np.uint8), mode="L")
 
 
def compose_tile_with_exemplars(tile_img: Image.Image, crop_images: Sequence[Image.Image],
                                margin: int = 6, feather_width: int = 8,
                                warn_wide: bool = True):
    n = len(crop_images)
    strip_h = max(c.height for c in crop_images) + 2 * margin
    strip_content_w = sum(c.width for c in crop_images) + margin * (n + 1)
    canvas_w = max(tile_img.width, strip_content_w)
    canvas_h = strip_h + tile_img.height
 
    if warn_wide and strip_content_w > tile_img.width:
        print(f"      NOTE: exemplar strip ({strip_content_w}px) is wider than the tile "
              f"({tile_img.width}px); the tile will be downscaled more than usual.")
 
    strip_bg = get_local_background_patch(tile_img, canvas_w, strip_h)
 
    composed = Image.new("RGB", (canvas_w, canvas_h))
    composed.paste(strip_bg, (0, 0))        # strip at the very top
    offset = (0, strip_h)                   # the tile now starts at y = strip_h
    composed.paste(tile_img, offset)
 
    crop_boxes: list[list[int]] = []
    cursor_x = margin
    for crop in crop_images:
        feather = make_feather_mask(crop.size, feather_width=feather_width)
        paste_xy = (cursor_x, margin)
        composed.paste(crop, paste_xy, feather)
        crop_boxes.append([paste_xy[0], paste_xy[1],
                           paste_xy[0] + crop.width, paste_xy[1] + crop.height])
        cursor_x += crop.width + margin
    return composed, crop_boxes, offset
 
 
def tile_bboxes(img_w: int, img_h: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    # Overlapping windows covering the whole image.
    step = tile_size - overlap
    tiles: list[Tuple[int, int, int, int]] = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
    return list(dict.fromkeys(tiles))
 
 
 
 #post processing
def keep_only_target_region_detections(boxes, scores, masks, offset, y_tolerance: int = 5):
 
    # Drop detections that live in the exemplar strip (SAM3 happily "detects" the prompts
    # themselves) and remap the survivors from composed coordinates into tile coordinates.
 
 
    import torch
    dx, dy = offset
    kept_boxes, kept_scores, kept_masks = [], [], []
    for box, score, mask in zip(boxes, scores, masks):
        x1, y1, x2, y2 = box.tolist() if torch.is_tensor(box) else list(box)
        if y1 >= dy - y_tolerance:                      # starts inside the real tile
            kept_boxes.append([x1 - dx, max(y1 - dy, 0.0), x2 - dx, y2 - dy])
            kept_scores.append(float(score.item()) if torch.is_tensor(score) else float(score))
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
            kept_masks.append(mask_np[dy:, dx:] if (dx or dy) else mask_np)
    return kept_boxes, kept_scores, kept_masks
 
 
def filter_implausible_boxes(boxes, scores, masks, tile_w: int, tile_h: int,
                             min_fill_ratio: float = 0.15,
                             max_area_fraction: float = 0.6,
                             edge_margin: int = 5):
 
    import torch
 
    kept_boxes, kept_scores, kept_masks = [], [], []
    tile_area = float(tile_w * tile_h)
 
    for box, score, mask in zip(boxes, scores, masks):
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= edge_margin or box_h <= edge_margin:
            continue
        if (box_w * box_h) / tile_area > max_area_fraction:
            continue
 
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
        # clip the box into the mask array before slicing
        x1c, y1c = int(max(0, x1)), int(max(0, y1))
        x2c, y2c = int(min(mask_np.shape[1], x2)), int(min(mask_np.shape[0], y2))
        if x2c <= x1c or y2c <= y1c:
            continue
        region = mask_np[y1c:y2c, x1c:x2c]
        fill_ratio = float((region > 0.5).mean()) if region.size > 0 else 0.0
        if fill_ratio < min_fill_ratio:
            continue
 
        kept_boxes.append(box)
        kept_scores.append(score)
        kept_masks.append(mask)
    return kept_boxes, kept_scores, kept_masks
 
 
def nms_merge(boxes, scores, masks, iou_thresh: float = 0.5):
    import torch
 
    if not boxes:
        return [], [], []
 
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor([float(s.item()) if torch.is_tensor(s) else float(s) for s in scores])
    order = scores_t.argsort(descending=True)       # highest confidence first
    keep: list[int] = []
 
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        # pairwise intersection between the winner and everything still in the running
        xx1 = torch.maximum(boxes_t[i, 0], boxes_t[rest, 0])
        yy1 = torch.maximum(boxes_t[i, 1], boxes_t[rest, 1])
        xx2 = torch.minimum(boxes_t[i, 2], boxes_t[rest, 2])
        yy2 = torch.minimum(boxes_t[i, 3], boxes_t[rest, 3])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        area_i = (boxes_t[i, 2] - boxes_t[i, 0]) * (boxes_t[i, 3] - boxes_t[i, 1])
        area_r = (boxes_t[rest, 2] - boxes_t[rest, 0]) * (boxes_t[rest, 3] - boxes_t[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thresh]
 
    return ([boxes[i] for i in keep],
            [scores[i] for i in keep],
            [masks[i] for i in keep])
 
 
 
#SAM3 wrapper
class Sam3Runner:
    # Holds the model + processor and performs one forward pass per composite tile.
    def __init__(self, model_id: str, device: Optional[str] = None, dtype: str = "float32"):
        import torch
        from transformers import Sam3Model, Sam3Processor
 
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = {"float32": torch.float32,
                       "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[dtype]
 
        print(f"Loading SAM3 from '{model_id}' onto {self.device} ({dtype}) ...")
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        print(f"SAM3 ready. Parameters live on: {next(self.model.parameters()).device}")
 
    def run_on_tile(self, composed_tile: Image.Image, crop_boxes: Sequence[Sequence[float]],
                    offset: Tuple[int, int], threshold: float, mask_threshold: float,
                    text_prompt: Optional[str] = None):
        torch = self.torch
        kwargs = dict(
            images=composed_tile,
            input_boxes=[[[float(c) for c in box] for box in crop_boxes]],
            input_boxes_labels=[[1] * len(crop_boxes)],
            return_tensors="pt",
        )
        if text_prompt:
            kwargs["text"] = text_prompt
 
        inputs = self.processor(**kwargs).to(self.device)
        with torch.no_grad():                       # inference only
            outputs = self.model(**inputs)
 
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
 
        kept = keep_only_target_region_detections(
            results["boxes"], results["scores"], results["masks"], offset
        )
        # free the big intermediates before the next tile
        del inputs, outputs, results
        return kept
 
 
def build_tiles(image: Image.Image, tile_size: int, overlap: int, cache: bool):
 
    # Returns a list of (x1, y1, x2, y2, tile_or_None).
    #
    # With cache=True the PIL crops are materialised now and reused by every anchor of this
    # image -- the notebook's optimisation. With cache=False only the coordinates are kept
    # and each anchor re-crops, which is slower but holds one tile in memory at a time.
 
    boxes = tile_bboxes(image.width, image.height, tile_size, overlap)
    if cache:
        return [(x1, y1, x2, y2, image.crop((x1, y1, x2, y2))) for (x1, y1, x2, y2) in boxes]
    return [(x1, y1, x2, y2, None) for (x1, y1, x2, y2) in boxes]
 
 
def run_sam3_pipeline(runner: Sam3Runner, image: Image.Image, exemplar_crops: Sequence[Image.Image],
                      args, keep_masks: bool, tiles):
 
    # Full detection pipeline for ONE image and ONE exemplar set.
    # `tiles` comes from build_tiles(): (x1, y1, x2, y2, tile_or_None).
    # Returns (boxes, scores, masks) in FULL-IMAGE coordinates, after NMS.
    #
    # No batching: one composed tile per forward pass, as in infer_sam3.py.
 
    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_masks: list = []
 
    for (x1, y1, x2, y2, cached_tile) in tiles:
        tile = cached_tile if cached_tile is not None else image.crop((x1, y1, x2, y2))
        composed, crop_boxes, offset = compose_tile_with_exemplars(
            tile, exemplar_crops, margin=args.margin, feather_width=args.feather_width,
            warn_wide=False,
        )
        boxes, scores, masks = runner.run_on_tile(
            composed, crop_boxes, offset,
            threshold=args.threshold, mask_threshold=args.mask_threshold,
        )
        boxes, scores, masks = filter_implausible_boxes(
            boxes, scores, masks, x2 - x1, y2 - y1,
            min_fill_ratio=args.min_fill_ratio,
            max_area_fraction=args.max_area_fraction,
            edge_margin=args.edge_margin,
        )
        # tile coordinates -> full image coordinates
        for b, s, m in zip(boxes, scores, masks):
            all_boxes.append([b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1])
            all_scores.append(float(s))
            all_masks.append((np.asarray(m), x1, y1) if keep_masks else None)
        del boxes, scores, masks, composed
        if cached_tile is None:
            del tile
 
    # One global NMS at the very end
    return nms_merge(all_boxes, all_scores, all_masks, iou_thresh=args.nms_iou)
 
 
 
#metrics
def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
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
 
try:
    import supervision as _sv
    from supervision.metrics import MeanAveragePrecision as _SvMAP
    _HAS_SUPERVISION = True
except Exception:
    _HAS_SUPERVISION = False
 
SUPERVISION_HINT = (
    "`supervision` is required for inference (it provides mAP50) but is not importable.\n"
    "  On a login node:   pip install --target $SCRATCH/pyextra --no-deps supervision\n"
    "  then point the container's PYTHONPATH at $SCRATCH/pyextra (see HOW_TO_RUN_single.md).\n"
    "  --dry-run and --aggregate-only work without it."
)
 
 
def compute_detection_metrics(final_boxes, final_scores, gt_boxes: np.ndarray,
                              iou_threshold: float = 0.5) -> dict:
    pred_boxes_np = (np.asarray(final_boxes, dtype=np.float32)
                     if len(final_boxes) else np.zeros((0, 4), dtype=np.float32))
    pred_scores_np = (np.asarray(final_scores, dtype=np.float32)
                      if len(final_scores) else np.zeros((0,), dtype=np.float32))
 
    try:
        pred_det = _sv.Detections(
            xyxy=pred_boxes_np,
            confidence=pred_scores_np,
            class_id=np.zeros(len(pred_scores_np), dtype=int),
        )
        gt_det = _sv.Detections(
            xyxy=gt_boxes,
            class_id=np.zeros(len(gt_boxes), dtype=int),
        )
        map50 = float(_SvMAP().update([pred_det], [gt_det]).compute().map50)
    except Exception:
        map50 = float("nan")
    if not np.isfinite(map50):
        map50 = 0.0 if len(gt_boxes) > 0 else float("nan")
 
    num_preds, num_gt = len(pred_boxes_np), len(gt_boxes)
    iou_matrix = compute_iou_matrix(pred_boxes_np, gt_boxes)
 
    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    true_positives = 0
    pred_order = np.argsort(-pred_scores_np) if num_preds > 0 else np.array([], dtype=int)
 
    for pred_idx in pred_order:
        if num_gt == 0:
            break
        best_gt_idx = int(np.argmax(iou_matrix[pred_idx]))
        best_iou = float(iou_matrix[pred_idx, best_gt_idx])
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            true_positives += 1
            matched_ious.append(best_iou)
 
    precision = true_positives / num_preds if num_preds > 0 else 0.0
    recall = true_positives / num_gt if num_gt > 0 else 0.0
    iou1 = float(np.mean(matched_ious)) if matched_ious else 0.0
    iou2 = float(np.sum(matched_ious) / num_gt) if num_gt > 0 else 0.0
 
    return {"map50": map50, "precision": precision, "recall": recall,
            "iou_matched": iou1, "iou_all_gt": iou2}
 
 
#persistence
def encode_rle(mask: np.ndarray) -> dict:
    binary = np.asarray(mask) > 0.5
    flat = binary.flatten(order="F")
    if flat.size == 0:
        return {"size": list(binary.shape), "counts": []}
    # positions where the value changes
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": [int(c) for c in counts]}
 
 
class JsonlWriter:
    def __init__(self, path: Path, enabled: bool = True):
        self.enabled = enabled
        self.path = path
        self._fh = open(path, "a", buffering=1) if enabled else None
 
    def write(self, record: dict) -> None:
        if self._fh is not None:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
 
    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
 
 
 
#resume support
def load_done_keys(runs_dir: Path, experiment_name: str) -> set[Tuple[str, int]]:
    done: set[Tuple[str, int]] = set()
    if not runs_dir.is_dir():
        return done
    for csv_path in sorted(runs_dir.glob(f"results_{experiment_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    try:
                        done.add((row["image_ID"], int(row["anchor_idx"])))
                    except (KeyError, ValueError, TypeError):
                        continue        # ignore a half-written trailing row
        except OSError:
            continue
    return done
 
 
#aggregation
def _summary_row(df, name: str, count_col_name: str, count_value: int) -> "pd.DataFrame":
    import pandas as pd
    row = {"experiment_name": name, count_col_name: count_value}
    for metric, col in (("mAP50", "mAP50"), ("precision", "precision"), ("recall", "recall"),
                        ("IoU1", "IoU1"), ("IoU2", "IoU2")):
        row[f"{metric}_mean"] = df[col].mean()
        row[f"{metric}_std"] = df[col].std()
    return pd.DataFrame([row])
 
 
def aggregate_results(output_dir: Path, experiment_name: str) -> None:
    import pandas as pd
 
    runs_dir = output_dir / "runs"
    shard_files = sorted(runs_dir.glob(f"results_{experiment_name}_shard*.csv"))
    if not shard_files:
        print(f"Nothing to aggregate: no shard CSVs in {runs_dir}")
        return
 
    frames = []
    for path in shard_files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"  WARNING: could not read {path.name}: {exc}")
    if not frames:
        print("Nothing to aggregate: all shard CSVs unreadable.")
        return
 
    df = pd.concat(frames, ignore_index=True)
    df = df[df["experiment_name"] == experiment_name]
    df = df.drop_duplicates(subset=["image_ID", "anchor_idx"], keep="last")
    df = df.sort_values(["archive", "image_ID", "anchor_idx"], kind="stable")
 
    scopes: list[Tuple[str, "pd.DataFrame"]] = [("all", df)]
    for archive in sorted(df["archive"].dropna().unique()):
        scopes.append((str(archive), df[df["archive"] == archive]))
 
    for scope, sub in scopes:
        if sub.empty:
            continue
        tag = f"{experiment_name}_{scope}"
 
        sub.to_csv(output_dir / f"results_{tag}.csv", index=False)
 
        # (1) raw per-run summary => every row is one independent observation
        _summary_row(sub, experiment_name, "n_runs", len(sub)).to_csv(
            output_dir / f"raw_summary_{tag}.csv", index=False)
 
        # (2) image-level table => collapse the prompt runs of each image to one value
        image_level = (
            sub.groupby("image_ID")
               .agg(archive=("archive", "first"),
                    flight=("flight", "first"),
                    n_prompts=("mAP50", "count"),
                    n_gt=("n_gt", "first"),
                    mAP50_image_mean=("mAP50", "mean"),
                    precision_image_mean=("precision", "mean"),
                    recall_image_mean=("recall", "mean"),
                    IoU1_image_mean=("IoU1", "mean"),
                    IoU2_image_mean=("IoU2", "mean"))
               .reset_index()
        )
        image_level.to_csv(output_dir / f"image_level_{tag}.csv", index=False)
 
        # (3) summary over the image-level means
        renamed = image_level.rename(columns={
            "mAP50_image_mean": "mAP50", "precision_image_mean": "precision",
            "recall_image_mean": "recall", "IoU1_image_mean": "IoU1", "IoU2_image_mean": "IoU2"})
        _summary_row(renamed, experiment_name, "n_images", image_level["image_ID"].nunique()).to_csv(
            output_dir / f"summary_{tag}.csv", index=False)
 
        print(f"  [{scope:<22}] runs={len(sub):<7} images={image_level['image_ID'].nunique():<6} "
              f"mAP50(img)={renamed['mAP50'].mean():.4f}  "
              f"P={renamed['precision'].mean():.4f}  R={renamed['recall'].mean():.4f}")
 
    print(f"Aggregation written to {output_dir}")
 
 
 
#main
def main() -> None:
    args = parse_args()
 
    output_dir = args.output_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
 
    exp = args.experiment_name
    prompt_type = "multiple" if args.n_exemplars > 1 else "single"
 
    if args.aggregate_only:
        print(f"=== AGGREGATE ONLY : {exp} ===")
        aggregate_results(output_dir, exp)
        return
 
    print("=" * 92)
    print(f" SAM3 TILED SINGLE-BOX BENCHMARK | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" output_dir     : {output_dir}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" n_exemplars    : {args.n_exemplars}  ({prompt_type} prompt)")
    print(f" prompting      : exemplar crop(s) in a strip above each tile")
    print(f" tiling         : ON  (tile={args.tile_size}, overlap={args.overlap}, "
          f"cache_tiles={args.cache_tiles})")
    print(f" batching       : OFF (one tile per forward pass)")
    print(f" thresholds     : conf={args.threshold}  mask={args.mask_threshold}  "
          f"match_iou={args.iou_threshold}  nms_iou={args.nms_iou}")
    print(f" save_detections: {args.save_detections}   save_masks: {args.save_masks}")
    print(f" supervision    : {'available' if _HAS_SUPERVISION else 'MISSING'}")
    print("=" * 92)
 
    if not args.dry_run and not _HAS_SUPERVISION:
        print("\nERROR: " + SUPERVISION_HINT)
        sys.exit(2)
 
    # dataset
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
 
    #  config snapshot
    if args.shard_index == 0:
        config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        config["archives_class_ids"] = {a: ARCHIVES[a] for a in args.archives}
        config["n_images_total"] = len(records)
        config["supervision_available"] = _HAS_SUPERVISION
        config["variant"] = "tiled_exemplar_strip"
        config["batching"] = False
        (output_dir / f"run_config_{exp}.json").write_text(json.dumps(config, indent=2))
 
    # dry run
    if args.dry_run:
        print("\n--- DRY RUN: counting the work without loading SAM3 ---")
        total_anchors, per_archive = 0, {}
        for rec in my_records[:  # a full count would open every label file; cap the sample
                              min(len(my_records), 200)]:
            with Image.open(rec.image_path) as im:
                w, h = im.size
            n_gt = len(load_yolo_boxes(rec.label_path, w, h, rec.class_id))
            if args.max_anchors_per_image > 0:
                n_gt = min(n_gt, args.max_anchors_per_image)
            total_anchors += n_gt
            per_archive[rec.archive] = per_archive.get(rec.archive, 0) + n_gt
        sampled = min(len(my_records), 200)
        tiles = len(tile_bboxes(8192, 5460, args.tile_size, args.overlap))
        print(f"  sampled {sampled} image(s) of this shard -> {total_anchors} anchor runs "
              f"({per_archive})")
        print(f"  tiles per anchor run at 8192x5460: {tiles}")
        print(f"  => ~{total_anchors * tiles} SAM3 forward passes for those {sampled} images")
        print("  (scale by len(shard)/sampled for the full estimate)")
        return
 
    #  resume
    done_keys: set[Tuple[str, int]] = set()
    if not args.no_resume:
        done_keys = load_done_keys(runs_dir, exp)
        print(f"Resume: {len(done_keys)} run(s) already recorded for {exp}; they will be skipped.")
 
    #  output files for this shard
    csv_path = runs_dir / f"results_{exp}_shard{args.shard_index}.csv"
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if not csv_exists:
        csv_writer.writeheader()
        csv_file.flush()
 
    det_writer = JsonlWriter(runs_dir / f"detections_{exp}_shard{args.shard_index}.jsonl",
                             enabled=args.save_detections)
    mask_writer = JsonlWriter(runs_dir / f"masks_{exp}_shard{args.shard_index}.jsonl",
                              enabled=args.save_masks)
 
    #  model
    runner = Sam3Runner(args.model_id, device=args.device, dtype=args.dtype)
    import torch
    cuda_available = torch.cuda.is_available()
 
    #  main loop
    start_time = time.time()
    n_runs = 0
    image_times: list[float] = []
    n_total = len(my_records)
 
    try:
        for img_idx, rec in enumerate(my_records, start=1):
            image_t0 = time.time()
 
            with Image.open(rec.image_path) as im:
                image = im.convert("RGB")
            img_w, img_h = image.size
            gt_boxes = load_yolo_boxes(rec.label_path, img_w, img_h, rec.class_id)
            n_gt = len(gt_boxes)
 
            if n_gt == 0:
                print(f"[{exp}] ({img_idx}/{n_total}) {rec.image_id}: 0 GT boxes, skipped.")
                image.close()
                continue
 
            # every GT box becomes the anchor once
            anchors = range(n_gt if args.max_anchors_per_image <= 0
                            else min(n_gt, args.max_anchors_per_image))
            image_map50s: list[float] = []
 
            # Cut the tiles ONCE for this image; every anchor below reuses them.
            # Skipped entirely if all of this image's anchors are already recorded.
            pending = [a for a in anchors if (rec.image_id, a) not in done_keys]
            if not pending:
                tiles = build_tiles(image, args.tile_size, args.overlap, cache=False)
            else:
                tiles = build_tiles(image, args.tile_size, args.overlap, cache=args.cache_tiles)
            n_tiles = len(tiles)
 
            for anchor_idx in anchors:
                if (rec.image_id, anchor_idx) in done_keys:
                    continue
 
                run_t0 = time.time()
                exemplar_indices = select_exemplar_indices(n_gt, anchor_idx, args.n_exemplars, rec.image_id)
                prompt_id = format_prompt_id(exemplar_indices)
                exemplar_crops = [image.crop([int(round(v)) for v in gt_boxes[i]])
                                  for i in exemplar_indices]
 
                boxes, scores, masks = run_sam3_pipeline(
                    runner, image, exemplar_crops, args, keep_masks=args.save_masks, tiles=tiles
                )
                metrics = compute_detection_metrics(boxes, scores, gt_boxes,
                                                    iou_threshold=args.iou_threshold)
                image_map50s.append(metrics["map50"])
                run_elapsed = time.time() - run_t0
 
 
                if args.save_detections:
                    det_writer.write({
                        "experiment_name": exp,
                        "archive": rec.archive,
                        "flight": rec.flight,
                        "image_ID": rec.image_id,
                        "anchor_idx": int(anchor_idx),
                        "prompt_id": prompt_id,
                        "image_width": img_w,
                        "image_height": img_h,
                        "n_tiles": n_tiles,
                        "n_pred": len(boxes),
                        "boxes": [[round(float(v), 2) for v in b] for b in boxes],
                        "scores": [round(float(s), 5) for s in scores],
                    })
                if args.save_masks:
                    mask_writer.write({
                        "experiment_name": exp,
                        "image_ID": rec.image_id,
                        "anchor_idx": int(anchor_idx),
                        # tile-local RLE + the tile origin needed to place it in the full image
                        "masks": [{"rle": encode_rle(m), "tile_origin": [int(ox), int(oy)]}
                                  for (m, ox, oy) in masks if m is not None],
                    })
 
                csv_writer.writerow({
                    "experiment_name": exp,
                    "image_ID": rec.image_id,
                    "anchor_idx": int(anchor_idx),
                    "Prompt_ID": prompt_id,
                    "Prompt_Type": prompt_type,
                    "mAP50": metrics["map50"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "IoU1": metrics["iou_matched"],
                    "IoU2": metrics["iou_all_gt"],
                    "archive": rec.archive,
                    "flight": rec.flight,
                    "source_class_id": rec.class_id,
                    "n_gt": n_gt,
                    "n_pred": len(boxes),
                    "image_width": img_w,
                    "image_height": img_h,
                    "n_tiles": n_tiles,
                    "runtime_s": round(run_elapsed, 2),
                })
                csv_file.flush()
                n_runs += 1
 
                print(f"    [{exp}] run #{n_runs} | {rec.image_id} | anchor={anchor_idx} "
                      f"({anchor_idx + 1}/{n_gt}) | prompt={prompt_id} | tiles={n_tiles} | "
                      f"preds={len(boxes)} | mAP50={metrics['map50']:.3f} | {run_elapsed:.1f}s")
 
                del boxes, scores, masks, exemplar_crops
                gc.collect()
                if cuda_available:
                    torch.cuda.empty_cache()
 
            # free this image's cached tiles before moving on
            del tiles
            gc.collect()
            image.close()
 
            #  per-image summary + ETA
            image_elapsed = time.time() - image_t0
            image_times.append(image_elapsed)
            avg_per_image = float(np.mean(image_times))
            eta = (n_total - img_idx) * avg_per_image
            map50_str = f"{np.nanmean(image_map50s):.3f}" if image_map50s else "N/A (already done)"
            print(f"[{exp}] ({img_idx}/{n_total}) {rec.image_id} done | archive={rec.archive} | "
                  f"{n_gt} GT | image_mAP50_mean={map50_str} | {image_elapsed:.1f}s | "
                  f"avg/image={avg_per_image:.1f}s | ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")
 
    finally:
        csv_file.close()
        det_writer.close()
        mask_writer.close()
 
    total = time.time() - start_time
    print(f"\nFinished {exp} shard {args.shard_index}: {n_runs} new row(s) -> {csv_path}")
    print(f"Total time: {total / 60:.1f} min ({total / 3600:.2f} h)")
 
    if not args.no_aggregate and args.shard_index == 0 and args.num_shards == 1:
        print("\n=== AGGREGATION ===")
        aggregate_results(output_dir, exp)
 
 
if __name__ == "__main__":
    main()
 

