#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHOLE-IMAGE MULTI-BOX SAM3 EXEMPLAR (NO TILING)

E02_1_b is the whole-image baseline: NO tiling, NO exemplar strip, NO NMS. The full drone
photo is resized down to MAX_DIM on its long side, the THREE exemplar boxes (the anchor
box plus two other ground-truth boxes from the same image) are scaled into that resized
space and handed to SAM3 as positive box prompts, and ONE forward pass produces every
detection for that run. Predicted boxes are scaled straight back up to full-image pixels.

DATASET:
-The dataset root contains four archives; we deliberately use only two of them: AGS_Multi_Rumex && AgsSpringRumex
-the class id that means "rumex" (0 in AGS_Multi_Rumex, 2 in AgsSpringRumex);

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
from dataclasses import dataclass
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
    "sam_input_width",    # what SAM3 actually received      (E02_1_b extra)
    "sam_input_height",   #                                   (E02_1_b extra)
    "sam_scale",          # full-res -> SAM3 input factor     (E02_1_b extra)
    "max_dim",
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
        description="SAM3 whole-image (no tiling), three-exemplar Rumex detection benchmark "
                    "(inference + evaluation). Port of the E02_1 Colab notebook.",
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
    g.add_argument("--experiment-name", default="E02_1_b",
                   help="Free-form tag written into every CSV row and into the file names.")
    g.add_argument("--preset", choices=["single", "multi"], default=None,
                   help="Convenience: multi = 3 exemplar boxes (E02_1_b, the default), "
                        "single = 1 exemplar box, the anchor alone. --n-exemplars always "
                        "wins over the preset.")
    g.add_argument("--n-exemplars", type=int, default=None,
                   help="Number of exemplar boxes per prompt (anchor + n-1 sampled others). "
                        "Default 3.")

    # SAM3 parameters
    g = p.add_argument_group("sam3")
    g.add_argument("--model-id", default="facebook/sam3",
                   help="HF repo id OR a local snapshot directory (use a local path when the "
                        "compute node has no internet).")
    g.add_argument("--max-dim", type=int, default=1024,
                   help="Long side the whole image is resized to before SAM3 sees it "
                        "(the notebook's MAX_DIM). SAM3's own processor resizes to ~1008 px "
                        "anyway, so values above ~1024 change little.")
    g.add_argument("--threshold", type=float, default=0.4,
                   help="SAM3 confidence threshold. 0.4 here, NOT the 0.3 the tiled "
                        "experiments use -- see the module docstring before comparing.")
    g.add_argument("--mask-threshold", type=float, default=0.5,
                   help="Binarisation threshold for the predicted masks. Affects the stored "
                        "masks only; no filter in this pipeline reads them.")
    g.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for TP matching (the '50' in mAP50).")
    g.add_argument("--nms-iou", type=float, default=0.0,
                   help="OFF by default (0 = disabled), matching the notebook: a single "
                        "whole-image pass has no tile-boundary duplicates to collapse. Set "
                        "e.g. 0.5 only as a deliberate ablation, under a new experiment name.")

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
    g.add_argument("--cache-resized", dest="cache_resized", action="store_true", default=True,
                   help="Resize the image ONCE per image and reuse it for every anchor of that "
                        "image (this pipeline's equivalent of the tiled ports' tile caching). "
                        "Costs a few MB; the resized copy is tiny.")
    g.add_argument("--no-cache-resized", dest="cache_resized", action="store_false",
                   help="Re-resize for every anchor. Slower, no real memory saving.")

    #  persistence
    g = p.add_argument_group("persistence")
    g.add_argument("--save-detections", dest="save_detections", action="store_true", default=True,
                   help="Append per-run boxes + scores to a JSONL file (default: on).")
    g.add_argument("--no-save-detections", dest="save_detections", action="store_false")
    g.add_argument("--save-masks", action="store_true", default=False,
                   help="Also store every predicted mask as COCO-style RLE, at the SAM3 input "
                        "resolution, with the scale factor needed to map it back to full-image "
                        "pixels. OFF by default.")
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
    preset_n = preset_table.get(args.preset, 3)
    if args.n_exemplars is None:
        args.n_exemplars = preset_n

    args.use_tiling = False                # this script is the no-tiling variant, always

    if args.n_exemplars < 1:
        p.error("--n-exemplars must be >= 1")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.max_dim < 64:
        p.error("--max-dim must be >= 64")
    if not (0.0 <= args.nms_iou <= 1.0):
        p.error("--nms-iou must be between 0 and 1 (0 disables NMS)")

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
    #     a 1-exemplar prompt, whatever --n-exemplars says. Its Prompt_ID is then a bare
    #     number, which is how you spot those rows in the CSV.)
    #
    # Seeded from (image_ID, anchor_idx) ONLY -- not from iteration order, not from the
    # number of runs already done. That is what makes this experiment pair row-by-row with
    # the tiled ones, and what makes resume safe.
    rng = np.random.default_rng(stable_seed(image_id, anchor_idx))
    others = [i for i in range(n_gt) if i != anchor_idx]
    n_needed = min(n_exemplars - 1, len(others))
    chosen = list(rng.choice(others, size=n_needed, replace=False)) if n_needed > 0 else []
    return [anchor_idx] + [int(i) for i in chosen]


def format_prompt_id(exemplar_indices: Sequence[int]) -> str:
    #  "5"  (single)  |  "5+12+3"  (multiple -- anchor first, order preserved).
    return "+".join(str(int(i)) for i in exemplar_indices)



#resize helper
def resize_for_sam3(img: Image.Image, max_dim: int) -> Tuple[Image.Image, float]:
    # Scale the whole image so its LONG side is max_dim. Images already smaller are left
    # untouched and get scale 1.0 (the notebook's behaviour).
    w, h = img.size
    scale = max_dim / float(max(w, h))
    if scale >= 1.0:
        return img, 1.0
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.BILINEAR), scale



#post processing
def to_numpy(x):
    # Handles tensors (incl. bf16/fp16), lists of tensors, or plain arrays.
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


def nms_merge(boxes: np.ndarray, scores: np.ndarray, masks, iou_thresh: float):
    # Only used when --nms-iou > 0. A single whole-image pass has no tile-boundary
    # duplicates, so this is off by default; it exists purely as an ablation knob.
    import torch
    if len(boxes) == 0 or iou_thresh <= 0:
        return boxes, scores, masks

    boxes_t = torch.tensor(np.asarray(boxes), dtype=torch.float32)
    scores_t = torch.tensor(np.asarray(scores), dtype=torch.float32)
    order = scores_t.argsort(descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(boxes_t[i, 0], boxes_t[rest, 0])
        yy1 = torch.maximum(boxes_t[i, 1], boxes_t[rest, 1])
        xx2 = torch.minimum(boxes_t[i, 2], boxes_t[rest, 2])
        yy2 = torch.minimum(boxes_t[i, 3], boxes_t[rest, 3])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        area_i = (boxes_t[i, 2] - boxes_t[i, 0]) * (boxes_t[i, 3] - boxes_t[i, 1])
        area_r = (boxes_t[rest, 2] - boxes_t[rest, 0]) * (boxes_t[rest, 3] - boxes_t[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thresh]

    idx = np.asarray(keep, dtype=int)
    kept_masks = None if masks is None else [masks[i] for i in idx]
    return np.asarray(boxes)[idx], np.asarray(scores)[idx], kept_masks



#SAM3 wrapper
class Sam3Runner:
    # Holds the model + processor and performs ONE forward pass per whole image.
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

    def run_whole_image(self, image_sam: Image.Image, exemplar_boxes_scaled: Sequence[Sequence[float]],
                        threshold: float, mask_threshold: float, want_masks: bool = False):
        # image_sam            : the ALREADY-resized image
        # exemplar_boxes_scaled: prompt boxes ALREADY in image_sam's coordinate space
        # Returns boxes/scores in image_sam coordinates, and masks at image_sam resolution
        # (or None). The caller rescales.
        torch = self.torch
        inputs = self.processor(
            images=image_sam,
            input_boxes=[[[float(c) for c in box] for box in exemplar_boxes_scaled]],
            input_boxes_labels=[[1] * len(exemplar_boxes_scaled)],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():                       # inference only
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        boxes = to_numpy(results["boxes"])
        scores = to_numpy(results["scores"])
        masks = to_numpy(results["masks"]) if want_masks else None

        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        del inputs, outputs, results
        return boxes, scores, masks


def run_sam3_pipeline(runner: Sam3Runner, image: Image.Image, gt_boxes: np.ndarray,
                      exemplar_indices: Sequence[int], args, keep_masks: bool,
                      cached_resized: Optional[Tuple[Image.Image, float]] = None):

    # Full detection pipeline for ONE image and ONE exemplar set, whole-image style.
    # Returns (boxes, scores, masks, info) with boxes in FULL-IMAGE coordinates.
    # `info` carries what SAM3 actually saw, so the CSV can record it.

    if cached_resized is not None:
        image_sam, sam_scale = cached_resized
    else:
        image_sam, sam_scale = resize_for_sam3(image, args.max_dim)

    # full-resolution prompt boxes -> resized-image coordinates
    exemplar_boxes_scaled = [[float(c) * sam_scale for c in gt_boxes[i].tolist()]
                             for i in exemplar_indices]

    boxes, scores, masks = runner.run_whole_image(
        image_sam, exemplar_boxes_scaled,
        threshold=args.threshold, mask_threshold=args.mask_threshold,
        want_masks=keep_masks,
    )

    if args.nms_iou > 0:
        boxes, scores, masks = nms_merge(boxes, scores, masks, iou_thresh=args.nms_iou)

    # resized-image coordinates -> full-image coordinates
    if len(boxes) > 0 and sam_scale != 1.0:
        boxes = boxes / sam_scale

    info = {
        "sam_input_width": image_sam.width,
        "sam_input_height": image_sam.height,
        "sam_scale": round(float(sam_scale), 6),
    }
    return boxes, scores, masks, info



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
    "  then point the container's PYTHONPATH at $SCRATCH/pyextra (see HOW_TO_RUN_whole_2.md).\n"
    "  --dry-run and --aggregate-only work without it."
)


def compute_detection_metrics(final_boxes, final_scores, gt_boxes: np.ndarray,
                              iou_threshold: float = 0.5) -> dict:
    pred_boxes_np = (np.asarray(final_boxes, dtype=np.float32).reshape(-1, 4)
                     if len(final_boxes) else np.zeros((0, 4), dtype=np.float32))
    pred_scores_np = (np.asarray(final_scores, dtype=np.float32).reshape(-1)
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
        agg_spec = dict(
            archive=("archive", "first"),
            flight=("flight", "first"),
            n_prompts=("mAP50", "count"),
            n_gt=("n_gt", "first"),
            mAP50_image_mean=("mAP50", "mean"),
            precision_image_mean=("precision", "mean"),
            recall_image_mean=("recall", "mean"),
            IoU1_image_mean=("IoU1", "mean"),
            IoU2_image_mean=("IoU2", "mean"),
        )
        if "n_pred" in sub.columns:
            agg_spec["n_pred_mean"] = ("n_pred", "mean")

        image_level = sub.groupby("image_ID").agg(**agg_spec).reset_index()
        image_level.to_csv(output_dir / f"image_level_{tag}.csv", index=False)

        # (3) summary over the image-level means
        renamed = image_level.rename(columns={
            "mAP50_image_mean": "mAP50", "precision_image_mean": "precision",
            "recall_image_mean": "recall", "IoU1_image_mean": "IoU1", "IoU2_image_mean": "IoU2"})
        _summary_row(renamed, experiment_name, "n_images", image_level["image_ID"].nunique()).to_csv(
            output_dir / f"summary_{tag}.csv", index=False)

        extra = f"  preds/run={sub['n_pred'].mean():.1f}" if "n_pred" in sub.columns else ""
        print(f"  [{scope:<22}] runs={len(sub):<7} images={image_level['image_ID'].nunique():<6} "
              f"mAP50(img)={renamed['mAP50'].mean():.4f}  "
              f"P={renamed['precision'].mean():.4f}  R={renamed['recall'].mean():.4f}{extra}")

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
    print(f" SAM3 WHOLE-IMAGE (NO TILING) BENCHMARK | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" output_dir     : {output_dir}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" n_exemplars    : {args.n_exemplars}  ({prompt_type} prompt)")
    print(f" prompting      : exemplar BOXES scaled into the resized image (no strip)")
    print(f" tiling         : OFF (whole image resized to max_dim={args.max_dim}, "
          f"cache_resized={args.cache_resized})")
    print(f" global pass    : n/a (this IS a whole-image pass)")
    print(f" NMS            : {'OFF (single pass, no tile duplicates)' if args.nms_iou <= 0 else f'ON (iou={args.nms_iou}) -- ABLATION, not the default'}")
    print(f" thresholds     : conf={args.threshold}  mask={args.mask_threshold}  "
          f"match_iou={args.iou_threshold}")
    print(f" save_detections: {args.save_detections}   save_masks: {args.save_masks}")
    print(f" supervision    : {'available' if _HAS_SUPERVISION else 'MISSING'}")
    print("=" * 92)
    if abs(args.threshold - 0.3) > 1e-9:
        print(f" NOTE: conf={args.threshold}. The tiled experiments (E01_2*/E02_2*) use 0.3, so a")
        print(f"       direct comparison against them measures the threshold too. Rerun one of")
        print(f"       them at the other's threshold, under its own EXPERIMENT_NAME, to isolate")
        print(f"       tiling. See HOW_TO_RUN_whole_2.md.")
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
        config["variant"] = "whole_image_no_tiling"
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
        print(f"  sampled {sampled} image(s) of this shard -> {total_anchors} anchor runs "
              f"({per_archive})")
        print(f"  forward passes per anchor run: 1 (no tiling)")
        print(f"  => ~{total_anchors} SAM3 forward passes for those {sampled} images")
        print(f"  (for scale: the tiled experiments need 29-71 passes per anchor run)")
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

            # Resize ONCE for this image; every anchor below reuses it.
            cached_resized = None
            pending = [a for a in anchors if (rec.image_id, a) not in done_keys]
            if pending and args.cache_resized:
                cached_resized = resize_for_sam3(image, args.max_dim)

            for anchor_idx in anchors:
                if (rec.image_id, anchor_idx) in done_keys:
                    continue

                run_t0 = time.time()
                exemplar_indices = select_exemplar_indices(n_gt, anchor_idx, args.n_exemplars, rec.image_id)
                prompt_id = format_prompt_id(exemplar_indices)

                boxes, scores, masks, info = run_sam3_pipeline(
                    runner, image, gt_boxes, exemplar_indices, args,
                    keep_masks=args.save_masks, cached_resized=cached_resized,
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
                        "max_dim": args.max_dim,
                        "sam_input_width": info["sam_input_width"],
                        "sam_input_height": info["sam_input_height"],
                        "sam_scale": info["sam_scale"],
                        "n_pred": int(len(boxes)),
                        "boxes": [[round(float(v), 2) for v in b] for b in boxes],
                        "scores": [round(float(s), 5) for s in scores],
                    })
                if args.save_masks and masks is not None:
                    mask_writer.write({
                        "experiment_name": exp,
                        "image_ID": rec.image_id,
                        "anchor_idx": int(anchor_idx),
                        # RLE at the SAM3 INPUT resolution. Multiply mask coordinates by
                        # 1/sam_scale to reach full-image pixels. (Boxes in the CSV and in
                        # detections_*.jsonl are already full-res -- masks are not, on
                        # purpose: upsampling them to 8192x5460 costs ~45 MB each.)
                        "sam_scale": info["sam_scale"],
                        "masks": [{"rle": encode_rle(m)} for m in masks],
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
                    "n_pred": int(len(boxes)),
                    "image_width": img_w,
                    "image_height": img_h,
                    "sam_input_width": info["sam_input_width"],
                    "sam_input_height": info["sam_input_height"],
                    "sam_scale": info["sam_scale"],
                    "max_dim": args.max_dim,
                    "runtime_s": round(run_elapsed, 2),
                })
                csv_file.flush()
                n_runs += 1

                print(f"    [{exp}] run #{n_runs} | {rec.image_id} | anchor={anchor_idx} "
                      f"({anchor_idx + 1}/{n_gt}) | prompt={prompt_id} | "
                      f"sam_input={info['sam_input_width']}x{info['sam_input_height']} | "
                      f"preds={len(boxes)} | mAP50={metrics['map50']:.3f} | {run_elapsed:.1f}s")

                del boxes, scores, masks
                gc.collect()
                if cuda_available:
                    torch.cuda.empty_cache()

            # free this image's resized copy before moving on
            if cached_resized is not None:
                if cached_resized[0] is not image:
                    cached_resized[0].close()
                del cached_resized
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