#!/usr/bin/env python3
# =============================================================================
#  text_only_no_tiling_infer_yoloe.py
# =============================================================================
#  YOLOE version of the SAM3 cluster script text_only_no_tiling_infer_sam3.py
#  (Colab notebook text_only_no_tiling.ipynb).
#
#  This experiment is the WHOLE-IMAGE, TEXT-ONLY variant: N_EXEMPLARS = 0
#  and USE_TILING = False. There are no tiles, no visual prompts, no exemplar
#  and no plausibility filter. The whole UAV image is downscaled so that its
#  longest side is MAX_DIM = 1024, and YOLOE is prompted with ONLY a text phrase
#  ("Rumex obtusifolius"). The predicted boxes are scaled back to the original
#  resolution immediately after inference. The GT boxes are used ONLY for the
#  evaluation, never as model input -> one run per image, all_gt only.
#
#  HOW YOLOE IS TEXT-PROMPTED
#    YOLOE turns the phrase into ONE class embedding with its text encoder
#    (MobileCLIP + RepRTA, model.get_text_pe) and installs it with
#    set_classes([phrase], embedding). After that every image is a plain
#    model.predict() - exactly how Ultralytics documents YOLOE text prompting.
#    The embedding depends only on the phrase and the checkpoint, never on the
#    image, so:
#      * it is computed ONCE (--prepare-text-pe, run by the dispatcher before the
#        shards start) and cached as a small .pt file (--text-pe-cache) - the SAME
#        file the 1_exemplar_text_*_yoloe experiments use;
#      * every shard LOADS it and installs it ONCE, before its first image.
#    SAM3 instead re-reads the text in every forward pass; the result is the
#    same kind of prompt (text only), just encoded once.
#
#  The notebook is a TWO-PHASE pipeline and this file keeps that separation:
#
#    PHASE 1 - INFERENCE   (notebook CELL 12, GPU)
#        for every image:
#            resize the whole image to MAX_DIM          (CELL 9)
#            run YOLOE once (text prompt installed once per shard)
#            at CONFIDENCE_THRESHOLD=0.30               (CELL 10)
#            rescale the boxes to full resolution       (CELL 10)
#            save the PRE-NMS detections to NPZ         (CELL 11)
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 13 ... CELL 23, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40,
#        evaluate in all_gt mode at the operating point CONFIDENCE_THRESHOLD = 0.30,
#        and write image-level / experiment-level / pooled-AP CSVs, the confusion
#        matrices (CSV + PNG) and the qualitative GT-vs-prediction figures.
#        Runs in ONE process, after every shard has finished. It never touches
#        YOLOE, so it can be repeated as often as you like from the cached NPZs.
#
#  MODES
#    (default)          sharded inference, then the evaluation
#    --prepare-text-pe  compute + cache the text embedding ONCE, then exit
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
    "n_gt", "gt_median_side_px_at_model_input", "n_detections_pre_nms",
    "image_width", "image_height", "model_input_width", "model_input_height",
    "resize_scale", "npz_file", "inference_seconds",
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

# ---- how the prompt is encoded -------------------------------------------------
# ONE text prompt embedding (MobileCLIP + RepRTA), computed once, installed once.
PROMPT_MODE = "text_pe"

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './text_only_no_tiling_run_yoloe.sh download' on a "
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

def default_weights() -> str:
    """
    YOLOE_WEIGHTS. On the cluster the checkpoint is pre-fetched by the download
    mode into $SCRATCH/yoloe_weights (compute nodes are offline). Elsewhere the
    bare name lets Ultralytics download it on first use, as in the notebook.
    """
    scratch = os.getenv("SCRATCH")
    if scratch:
        return str(Path(scratch) / "yoloe_weights" / "yoloe-11l-seg.pt")
    return "yoloe-11l-seg.pt"

def slugify(text: str) -> str:
    """'Rumex obtusifolius' -> 'rumex_obtusifolius' (safe inside a file name)."""
    out = "".join(c.lower() if c.isalnum() else "_" for c in str(text).strip())
    return "_".join(p for p in out.split("_") if p) or "text"


def default_text_pe_cache(weights: str, text: str) -> str:
    """
    TEXT_PE_CACHE: <weights dir>/text_pe/<weights stem>__<text slug>.pt. The key
    contains the checkpoint AND the text, because e_text depends on both (the
    RepRTA text head is part of the YOLOE checkpoint).
    """
    w = Path(str(weights))
    folder = w.parent if str(w.parent) not in ("", ".") else Path(".")
    return str(folder / "text_pe" / f"{w.stem}__{slugify(text)}.pt")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="text_only_no_tiling_yoloe - YOLOE text-only prompted Rumex detection, "
                    "whole image, NO tiling (inference + offline evaluation). YOLOE version "
                    "of the SAM3 notebook text_only_no_tiling.ipynb.",
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
    g.add_argument("--experiment-name", default="text_only_no_tiling_yoloe",
                   help="EXPERIMENT_NAME. Written into every CSV row and every NPZ.")
    g.add_argument("--text-prompt", default="Rumex obtusifolius",
                   help="TEXT_PROMPT (same phrase as the SAM3 notebook). YOLOE's text encoder "
                        "is CLIP-based: keep it a short noun phrase.")
    g.add_argument("--text-pe-cache", type=Path, default=None,
                   help="TEXT_PE_CACHE: the cached text embedding (.pt). Default: "
                        "<weights dir>/text_pe/<weights stem>__<text slug>.pt (shared with the "
                        "1_exemplar_text_*_yoloe experiments).")
    g.add_argument("--n-exemplars", type=int, default=0,
                   help="N_EXEMPLARS. Must be 0 for this text-only experiment.")

    # --------------------------- whole-image input --------------------------
    g = p.add_argument_group("whole-image input (CELL 3 / CELL 9)")
    g.add_argument("--max-dim", type=int, default=1024,
                   help="MAX_DIM: longest side of the downscaled copy fed to YOLOE. The whole "
                        "image is sent in ONE pass; there is no tiling in this experiment.")

    # --------------------------- YOLOE inference ----------------------------
    g = p.add_argument_group("yoloe inference (CELL 3 / CELL 5 / CELL 10)")
    g.add_argument("--weights", default=default_weights(),
                   help="YOLOE_WEIGHTS: path to yoloe-11l-seg.pt (the text-prompt capable "
                        "checkpoint; the *-seg-pf.pt variants have no text prompt).")
    g.add_argument("--imgsz", type=int, default=None,
                   help="IMGSZ, multiple of 32. Default: equal to --max-dim, so the resized "
                        "image is only padded (letterbox), not shrunk again.")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="CONFIDENCE_THRESHOLD. YOLOE is executed EXACTLY ONCE per "
                        "image at this score. In this experiment it is ALSO the "
                        "evaluation operating point (see --operating-confidence).")
    g.add_argument("--predict-nms-iou", type=float, default=0.90,
                   help="YOLOE_PREDICT_NMS_IOU: the NMS inside the Ultralytics predictor, "
                        "permissive on purpose (the offline NMS decides).")
    g.add_argument("--max-det", type=int, default=300, help="MAX_DET per image.")
    g.add_argument("--no-retina-masks", dest="retina_masks", action="store_false",
                   default=True,
                   help="RETINA_MASKS = False. Masks play no role here (no plausibility "
                        "filter); this only changes the predictor's mask resolution.")
    fp = g.add_mutually_exclusive_group()
    fp.add_argument("--fp16", dest="use_fp16", action="store_true", default=True,
                    help="USE_FP16 = True (default): half precision YOLOE inference on the GPU.")
    fp.add_argument("--no-fp16", dest="use_fp16", action="store_false",
                    help="USE_FP16 = False: full precision (use it if a dtype error appears "
                         "while the text embedding is installed).")
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
    g.add_argument("--prepare-text-pe", action="store_true",
                   help="Compute the text embedding of --text-prompt ONCE, cache it in "
                        "--text-pe-cache and exit (the dispatcher runs this before the "
                        "shards start).")

    args = p.parse_args(argv)

    if args.n_exemplars != 0:
        p.error("--n-exemplars must be 0 for the text-only pipeline")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.max_dim < 64:
        p.error("--max-dim must be >= 64")
    if not str(args.text_prompt).strip():
        p.error("--text-prompt must not be empty")

    # IMGSZ = MAX_DIM (as in the other YOLOE no-tiling experiments)
    if args.imgsz is None:
        args.imgsz = args.max_dim
    if args.imgsz % 32 != 0:
        p.error("--imgsz must be a multiple of 32")
    if args.text_pe_cache is None:
        args.text_pe_cache = Path(default_text_pe_cache(args.weights, args.text_prompt))
    args.model_name = Path(str(args.weights)).name

    # PROMPT_TYPE (CELL 3)
    args.prompt_type = "text"
    # USE_TILING (CELL 3) - constant here, kept so it reaches experiment_summary.csv
    args.use_tiling = False

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
# THE RULE
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
#  CELL 9 - RESIZE HELPER (whole-image, no tiling)
# =============================================================================
# The WHOLE image is sent to YOLOE in one pass (no tiles). The large UAV image is
# downscaled so that its longest side is MAX_DIM (same whole-image rule as the
# SAM3 notebook and every other no-tiling experiment).
# Predictions are rescaled back to full resolution right after inference
# (see CELL 10).
# =============================================================================
def resize_for_model(img: Image.Image, max_dim: int) -> Tuple[Image.Image, float]:
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale >= 1:
        return img, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.BILINEAR), scale

# =============================================================================
#  CELL 10 - YOLOE TEXT-ONLY INFERENCE (no tiling, text prompt only)
# =============================================================================
# Pipeline of this experiment:
#   - the text embedding of TEXT_PROMPT is installed ONCE per shard (set_classes)
#   - resize the whole image to MAX_DIM
#   - single YOLOE forward pass (plain predict), no tiling, no visual prompts
#   - scale the predicted boxes back up to full resolution
#
# YOLOE is called ONCE per image at CONFIDENCE_THRESHOLD; NMS is applied offline
# afterwards. imgsz = MAX_DIM, so the resized image is only padded. YOLOE's own
# NMS (PREDICT_NMS_IOU = 0.90) is permissive on purpose so the OFFLINE NMS decides.
# Masks are never read (KEEP_MASKS = False).
#
# Cluster change: each shard process sees exactly ONE GPU via
# CUDA_VISIBLE_DEVICES. CUDA is initialised BEFORE Ultralytics' select_device()
# runs, so that it cannot re-point the process at another physical GPU.
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

class YoloeRunner:
    """Owns the YOLOE model, installs the text prompt once and runs the inference."""

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
              "classes (replaced by the text prompt)")

    # ---------------------------- CELL 13 (text) ----------------------------
    def compute_text_pe(self, text: str):
        """
        Run ONLY the text-prompt encoder on one phrase (MobileCLIP + RepRTA, the
        normal YOLOE text-prompt path).

        Output: torch.Tensor (1, 1, D) on CPU, float32 - same shape as a VPE.

        Ultralytics looks for the text-encoder file in the current directory and
        downloads it there if it is missing. The call therefore runs INSIDE the
        weights folder, where the download mode of the dispatcher puts that file,
        so a compute node without internet still finds it.
        """
        folder = Path(str(self.args.weights)).parent
        old_cwd = os.getcwd()
        try:
            if folder.is_dir():
                os.chdir(folder)
            tpe = self.model.get_text_pe([text])
        finally:
            os.chdir(old_cwd)
        return tpe.detach().float().cpu()

    def install_text_prompt(self, text: str, text_pe) -> None:
        """
        Install the (cached) text embedding as the ONE class of the model. This is
        the standard YOLOE text prompt: set_classes([phrase], text embedding). It
        happens ONCE per shard - every image afterwards is a plain predict().
        """
        tpe = self.torch.nn.functional.normalize(text_pe.detach().float().cpu(), dim=-1, p=2)
        self.model.set_classes([text], tpe)
        self.model.predictor = None        # fresh plain predictor with the new classes
        self.text_pe = tpe

    # ---------------------------- CELL 14 -----------------------------------
    def predict_image(self, image: Image.Image, threshold: float):
        """
        Run the prompted YOLOE on ONE image (here: the resized whole image).

        Output: (boxes, scores) numpy, boxes (N,4) in that image's coordinates,
                every detection with score >= threshold. Masks are not returned.
        """
        a = self.args
        res = self.model.predict(
            image,
            imgsz=a.imgsz,
            conf=threshold,
            iou=a.predict_nms_iou,
            max_det=a.max_det,
            retina_masks=a.retina_masks,
            device=self.device,
            verbose=False,
            **self.precision_kwargs,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            boxes, scores = np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        else:
            boxes = res.boxes.xyxy.detach().float().cpu().numpy().reshape(-1, 4)
            scores = res.boxes.conf.detach().float().cpu().numpy().reshape(-1)
        del res
        return boxes, scores

    # ------------------- CELL 10: the whole-image pass -----------------------
    def run_whole_image_text(self, image: Image.Image, threshold: float):
        """
        resize the whole image to MAX_DIM -> ONE predict() with the installed text
        prompt -> boxes back to full resolution.

        Returns: pred_boxes_fullres (np.ndarray Nx4), pred_scores (np.ndarray N),
                 resize_scale (float), all boxes/scores with score >= threshold.
        """
        resized, resize_scale = resize_for_model(image, self.args.max_dim)
        boxes, scores = self.predict_image(resized, threshold)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) > 0:
            boxes = boxes / resize_scale
        if resized is not image:
            resized.close()
        return boxes, np.asarray(scores, dtype=np.float32).reshape(-1), float(resize_scale)


# =============================================================================
#  CELL 10 (text) - THE TEXT EMBEDDING: COMPUTED ONCE, CACHED ON DISK
# =============================================================================
# e_text depends only on (TEXT_PROMPT, checkpoint). It is computed once by
# --prepare-text-pe (the dispatcher runs that before the shards) and stored as a
# small .pt file together with the text and the checkpoint name it belongs to.
# Every shard loads that file and refuses a file made for another text or
# checkpoint, so a stale cache can never be fused silently.
#
# If the text encoder cannot be obtained on the cluster at all, the same file can
# be produced in Colab (see HOW_TO_RUN, "Problem C") and copied to --text-pe-cache.
# =============================================================================
TEXT_PE_HINT = (
    "Run './text_only_no_tiling_run_yoloe.sh download' on a LOGIN node (it "
    "pre-fetches the YOLOE text encoder) and let the 'run' mode prepare the text "
    "embedding, or create the cache file in Colab - see "
    "text_only_no_tiling_HOW_TO_RUN_yoloe.md, 'Problem C'."
)


def save_text_pe(path: Path, text_pe, text: str, weights_name: str) -> None:
    """Atomic write (tmp file + rename): concurrent readers never see half a file."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save({"text_prompt": text, "weights": weights_name,
                "text_pe": text_pe.detach().float().cpu()}, tmp)
    os.replace(tmp, path)


def load_text_pe(path: Path, text: str, weights_name: str):
    """Load the cached e_text and check it belongs to THIS text and checkpoint."""
    import torch
    blob = torch.load(path, map_location="cpu")
    if str(blob.get("text_prompt")) != text or str(blob.get("weights")) != weights_name:
        raise RuntimeError(
            f"{path} holds the text embedding of '{blob.get('text_prompt')}' / "
            f"{blob.get('weights')}, not of '{text}' / {weights_name}. Delete it or "
            f"point --text-pe-cache elsewhere.")
    return blob["text_pe"].float()


def prepare_text_pe(args) -> None:
    """--prepare-text-pe: compute e_text ONCE (if not cached yet) and cache it."""
    path = Path(args.text_pe_cache)
    if path.is_file():
        tpe = load_text_pe(path, args.text_prompt, args.model_name)
        print(f"Text embedding already cached: {path} {tuple(tpe.shape)} "
              f"('{args.text_prompt}', {args.model_name})")
        return
    print(f"Computing the text embedding of '{args.text_prompt}' with {args.model_name} ...")
    runner = YoloeRunner(args)
    try:
        tpe = runner.compute_text_pe(args.text_prompt)
    except Exception as exc:
        print(f"\nERROR: the text embedding could not be computed: {exc}")
        print("  " + TEXT_PE_HINT)
        sys.exit(3)
    save_text_pe(path, tpe, args.text_prompt, args.model_name)
    print(f"Text embedding cached: {path} {tuple(tpe.shape)}")


# =============================================================================
#  CELL 11 - PRE-NMS DETECTION STORAGE
# =============================================================================
# For every run (= one image) we store the detections AFTER
#   YOLOE inference at CONFIDENCE_THRESHOLD -> conversion to original-image
#   coordinates
# but BEFORE NMS.
# The offline evaluation (PHASE 2) therefore never needs YOLOE again.
# Masks are never stored. NPZ is compact and fast; the metric tables are CSV.
# =============================================================================
def run_npz_path(raw_detections_dir: Path, image_id: str) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one image."""
    return raw_detections_dir / f"{safe_filename(image_id)}.npz"

def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        text_prompt: str, boxes, scores, gt_boxes: np.ndarray,
                        image_size: Tuple[int, int], model_input_size: Tuple[int, int],
                        archive: str, flight: str, class_id: int, resize_scale: float,
                        gt_median_side_model: float) -> Path:
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
        model_input_width=np.array(int(model_input_size[0])),
        model_input_height=np.array(int(model_input_size[1])),
        archive=np.array(archive),
        flight=np.array(flight),
        source_class_id=np.array(int(class_id)),
        resize_scale=np.array(float(resize_scale)),
        gt_median_side_px_at_model_input=np.array(float(gt_median_side_model)),
        gt_boxes=gt_boxes.astype(np.float32),
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),   # x1,y1,x2,y2 (original img)
        scores=np.asarray(scores, dtype=np.float32).reshape(-1),    # confidence > 0.30
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
        }
        # cluster additions; tolerate older NPZs.
        run["archive"] = str(z["archive"]) if "archive" in z else ""
        run["flight"] = str(z["flight"]) if "flight" in z else ""
        run["resize_scale"] = float(z["resize_scale"]) if "resize_scale" in z else float("nan")
        run["gt_median_side_px_at_model_input"] = float(z["gt_median_side_px_at_model_input"]) if "gt_median_side_px_at_model_input" in z else 0.0
    return run

# =============================================================================
#  RESUME SUPPORT  (CELL 12, adapted to several shard manifests)
# =============================================================================
def load_done_images(paths: Paths, experiment_name: str, text_prompt: str) -> set:
    """
    Read every shard manifest and return {image_ID} of the images that are already
    finished. Every shard reads ALL manifests, so a resubmission after the walltime
    never repeats work, even if the shard assignment changed because --num-gpus was
    different.

    Guard (as in the SAM3 text notebooks): if a manifest holds runs made with
    ANOTHER text prompt, the job stops instead of silently mixing the two.
    """
    done: set = set()
    other_prompts: set = set()
    for csv_path in sorted(paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    if str(row.get("text_prompt", "")) != text_prompt:
                        other_prompts.add(str(row.get("text_prompt", "")))
                        continue
                    try:
                        done.add(row["image_ID"])
                    except (KeyError, ValueError, TypeError):
                        continue            # ignore a half-written trailing row
        except OSError:
            continue
    if other_prompts:
        raise RuntimeError(
            f"{paths.raw_detections} already holds runs made with another text prompt "
            f"{sorted(other_prompts)}. Use a new EXPERIMENT_NAME (and OUTPUT_DIR) for "
            f"'{text_prompt}'.")
    return done

# =============================================================================
#  CELL 12 - MAIN GPU INFERENCE LOOP  (PHASE 1)
# =============================================================================
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE
#     read its GT boxes ONCE
#     run YOLOE ONCE over the whole (resized) image (text prompt already installed)
#     save the PRE-NMS detections (NPZ)
#     release the image
#
# NO NMS and NO metric computation happens here - that is all done offline in
# PHASE 2. The loop is resumable: finished images are listed in the shard manifests.
# =============================================================================
def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    import torch

    exp = args.experiment_name

    # ---- resume support ------------------------------------------------------
    done_images: set = set()
    if not args.no_resume:
        done_images = load_done_images(paths, exp, args.text_prompt)
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

    runner = YoloeRunner(args)
    device_is_cuda = runner.device.startswith("cuda")

    # ---- the text prompt: computed ONCE (cached), installed ONCE here ----------
    tpe_path = Path(args.text_pe_cache)
    if tpe_path.is_file():
        text_pe = load_text_pe(tpe_path, args.text_prompt, args.model_name)
        print(f"Text embedding loaded from cache: {tpe_path}")
    else:
        # normally prepared by the dispatcher; computed here only as a fallback
        print(f"Text embedding cache {tpe_path} not found -> computing it now ...")
        try:
            text_pe = runner.compute_text_pe(args.text_prompt)
        except Exception as exc:
            manifest_file.close()
            raise RuntimeError(f"text embedding unavailable ({exc}). {TEXT_PE_HINT}") from exc
        save_text_pe(tpe_path, text_pe, args.text_prompt, args.model_name)
    runner.install_text_prompt(args.text_prompt, text_pe)
    print(f"Text prompt '{args.text_prompt}' installed -> e_text {tuple(runner.text_pe.shape)}; "
          f"classes = {runner.model.names}")

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

            run_t0 = time.time()

            # ---------------- run YOLOE (text prompt already installed) -----------
            pred_boxes_np, pred_scores_np, resize_scale = runner.run_whole_image_text(
                image, threshold=args.threshold)

            # compute median GT side at the model input resolution for the manifest
            gt_side = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1])
            gt_median_side_model = float(np.median(gt_side) * resize_scale) if len(gt_side) > 0 else 0.0

            # get the actual model input size
            image_resized_for_size, _ = resize_for_model(image, args.max_dim)
            model_input_size = image_resized_for_size.size
            if image_resized_for_size is not image:
                image_resized_for_size.close()

            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, args.text_prompt,
                pred_boxes_np, pred_scores_np, gt_boxes,
                (img_w, img_h), model_input_size,
                rec.archive, rec.flight, rec.class_id, resize_scale, gt_median_side_model)

            run_seconds = time.time() - run_t0

            manifest_writer.writerow({
                "experiment_name": exp,
                "image_ID": image_id,
                "Prompt_Type": args.prompt_type,
                "text_prompt": args.text_prompt,
                "archive": rec.archive,
                "flight": rec.flight,
                "source_class_id": rec.class_id,
                "n_gt": n_gt,
                "gt_median_side_px_at_model_input": round(gt_median_side_model, 1),
                "n_detections_pre_nms": int(len(pred_scores_np)),
                "image_width": img_w,
                "image_height": img_h,
                "model_input_width": model_input_size[0],
                "model_input_height": model_input_size[1],
                "resize_scale": round(resize_scale, 6),
                "npz_file": npz_path.name,
                "inference_seconds": round(run_seconds, 2),
            })
            manifest_file.flush()                  # this image is on disk -> resumable

            n_new_runs += 1
            print(f"  [{exp}] shard{args.shard_index} run #{n_new_runs} | {image_id} | "
                  f"{n_gt} GT box(es) | pre-NMS detections={len(pred_scores_np)} | {run_seconds:.1f}s")

            del pred_boxes_np, pred_scores_np
            gc.collect()
            if device_is_cuda:
                torch.cuda.empty_cache()

            # ---------------- release the image ------------------------------------
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
                  f"{image_elapsed:.1f}s | avg/image={avg_per_image:.1f}s | "
                  f"ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")

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
    print("\n--- DRY RUN: counting the work without loading YOLOE ---")
    sample = my_records[:min(len(my_records), 200)]
    total_images, per_archive = 0, {}
    for rec in sample:
        total_images += 1
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + 1

    print(f"  sampled {len(sample)} image(s) of this shard -> {total_images} image runs "
          f"({per_archive})")
    print(f"  forward passes per image run: 1 (whole image, no tiling, "
          f"longest side -> {args.max_dim} px)")
    print(f"  => ~{total_images} YOLOE forward passes for those {len(sample)} images "
          f"(+ the text embedding ONCE per job)")
    weights = Path(str(args.weights))
    if weights.is_absolute() or weights.parent != Path("."):
        print(f"  weights file: {weights} -> {'FOUND' if weights.is_file() else 'MISSING'}")
    tpe = Path(args.text_pe_cache)
    print(f"  text prompt : '{args.text_prompt}' (text only)")
    print(f"  text embedding cache: {tpe} -> "
          f"{'FOUND' if tpe.is_file() else 'not yet (prepared by the run mode)'}")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{total_images} "
          f"(one per image)")

# =============================================================================
#  CELL 13 - LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
# =============================================================================
# From here on YOLOE is never touched again. Everything below works on the NPZ
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
        run["gt_median_side_px_at_model_input"] = float(row.get("gt_median_side_px_at_model_input", 0.0))
        runs.append(run)

    if n_missing:
        print(f"  WARNING: {n_missing} manifest row(s) point at a missing NPZ (skipped).")

    print(f"Loaded {len(runs)} runs (= images) for {experiment_name}.")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes over all runs:", int(sum(len(r['gt_boxes']) for r in runs)))
    return runs, manifest

# =============================================================================
#  CELL 14 - OFFLINE NMS
# =============================================================================
# YOLOE can return several overlapping boxes for the same plant. NMS keeps the
# highest-scoring box of each overlapping group. The NMS IoU threshold is fixed
# (NMS_IOU_THRESHOLD = 0.40) and applied offline, so the raw pre-NMS detections
# stay untouched on disk.
# =============================================================================
def nms(boxes, scores, iou_threshold: float):
    """
    Input : boxes (N,4), scores (N,), iou_threshold
    Output: keep (list of kept indices, highest score first),
            n_suppressed (int)
    A detection is suppressed when its IoU with an already kept, higher-scoring
    detection is GREATER than the threshold.
    """
    n = len(boxes)
    if n == 0:
        return [], 0
    order = list(np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable"))
    keep, n_suppressed = [], 0
    while order:
        i = int(order[0])
        keep.append(i)
        rest = np.array(order[1:], dtype=int)
        if rest.size == 0:
            break
        ious = compute_iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        n_suppressed += int((ious > iou_threshold).sum())
        order = list(rest[ious <= iou_threshold])
    return keep, n_suppressed

def apply_nms_to_run(run: dict, iou_threshold: float) -> dict:
    """
    Apply NMS to one run's pre-NMS detections.
    Output: dict with 'boxes' and 'scores' of the surviving detections SORTED BY
            SCORE (high -> low), plus 'n_pre_nms' and 'n_suppressed'.
    """
    keep, n_suppressed = nms(run["boxes"], run["scores"], iou_threshold)
    keep = np.array(keep, dtype=int)
    return {
        "boxes": run["boxes"][keep].reshape(-1, 4),
        "scores": run["scores"][keep].reshape(-1),
        "n_pre_nms": int(len(run["scores"])),
        "n_suppressed": int(n_suppressed),
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
# AP is an area under the precision-recall curve, built by walking through ALL
# detections ordered by confidence. It therefore uses every post-NMS prediction
# that YOLOE returned (score >= CONFIDENCE_THRESHOLD).
# Precision / recall / F1 / IoU1 / IoU2 describe ONE operating point.
# Here the operating point IS the inference threshold (0.30), so both use exactly
# the same prediction set - they still answer different questions (ranking
# quality vs. deployed behaviour).
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
    the detections of all images are ranked together in one PR curve.
    """
    from supervision.metrics import MeanAveragePrecision
    if len(gt_list) == 0 or sum(len(g) for g in gt_list) == 0:
        return float("nan"), float("nan")
    try:
        result = MeanAveragePrecision().update(pred_list, gt_list).compute()
        ap50, ap5095 = float(result.map50), float(result.map50_95)
        # supervision returns -1.0 when a metric is undefined -> report NaN instead
        return (ap50 if ap50 >= 0 else float("nan"),
                ap5095 if ap5095 >= 0 else float("nan"))
    except Exception as e:
        print("   (AP computation failed:", e, ")")
        return float("nan"), float("nan")

# =============================================================================
#  CELL 17 - OPERATING-POINT EVALUATION
# =============================================================================
# Keeps the predictions with score >= CONFIDENCE_THRESHOLD and evaluates them.
# The operating point is fixed from the start (no sweep, no "best" selection):
#     confidence threshold = CONFIDENCE_THRESHOLD (0.30)
#     NMS IoU threshold    = NMS_IOU_THRESHOLD    (0.40)
# =============================================================================
def evaluate_at_operating_point(nms_run, gt_boxes,
                                confidence_threshold: float, eval_iou: float) -> dict:
    """
    Input : nms_run  - output of apply_nms_to_run (sorted by score, high -> low)
            gt_boxes - all GT boxes of the image
    Output: dict of evaluate_predictions for the thresholded predictions.
    """
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                gt_boxes, eval_iou)

# =============================================================================
#  CELL 23 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right)
# =============================================================================
# IMAGE SELECTION (per-image AP50):
#   1. keep only images that have at least PLOT_MIN_GT_BOXES (7) GT boxes
#      and take the one with the highest AP50
#   2. if NO image has 7 GT boxes: keep the images with the HIGHEST number of GT
#      boxes and take the one among them with the highest AP50
#   ties are broken by F1, then by the number of GT boxes.
#
# Right panel = post-NMS predictions of that image with
# score >= CONFIDENCE_THRESHOLD, i.e. exactly the set evaluated in CELL 18.
# The image is downscaled for DISPLAY only (PLOT_MAX_DISPLAY_DIM).
# =============================================================================
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

def select_image_for_plot(image_level_df, mode: str, min_gt: int,
                          archive: Optional[str] = None):
    """
    Output: (image_level_row, selection_rule_text) or (None, reason)
    'archive' restricts the candidates to one archive; None = the whole dataset.
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
    """One qualitative figure for one scope (a single archive, or the whole dataset)."""
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

    # ---- open the image, downscale for display only ---------------------------
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

    # ---- figure: GT (left) | predictions (right) -------------------------------
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
        f"no tiling (MAX_DIM={args.max_dim}px) | conf={args.operating_confidence:.2f}, "
        f"NMS IoU={args.nms_iou_threshold:.2f} | YOLOE {args.model_name} @ imgsz={args.imgsz}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    png_path = paths.plots / (f"best_image_{scope_label}_{safe_filename(plot_image_id)}"
                              f"_{mode}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

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
#  The operating point is frozen and identical to the YOLOE inference threshold:
#  confidence 0.30, NMS IoU 0.40. No sweep is performed.
# =============================================================================
def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2: notebook CELL 13 ... CELL 23, in one process, no GPU."""
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

    print("=" * 92)
    print(f" PHASE 2 - OFFLINE EVALUATION | experiment={exp}")
    print("=" * 92)
    print("Operating configuration is fixed (no sweep performed):")
    print(f"  Confidence threshold = {conf:.2f}  (== the YOLOE inference threshold)")
    print(f"  NMS IoU threshold    = {nms_iou:.2f}")
    print(f"  Evaluation IoU       = {eval_iou:.2f}")

    runs, manifest = load_runs(paths, exp)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  CELL 18 - PER-IMAGE METRICS  (one run = one image)
    # =========================================================================
    # Text-only prompting has no anchors, so every image is evaluated exactly once.
    # Everything is evaluated at the fixed configuration.
    #   AP50 / AP50_95 : all post-NMS predictions, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : predictions >= CONFIDENCE_THRESHOLD
    # =========================================================================
    print("\n--- CELL 18: image-level metrics ---")
    image_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)

        # ---- AP: every post-NMS prediction ---------------------------------------
        p_det = make_detections(nms_run["boxes"], nms_run["scores"])
        g_det = make_detections(run["gt_boxes"])
        ap50, ap5095 = compute_ap([p_det], [g_det])

        # ---- operating point ------------------------------------------------------
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
            "gt_median_side_px_at_model_input": run.get("gt_median_side_px_at_model_input", 0.0),
            "n_predictions_pre_nms": nms_run["n_pre_nms"],
            "n_suppressed_by_nms": nms_run["n_suppressed"],
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
    # Mean over images and std BETWEEN images of every per-image metric, so every
    # UAV image contributes exactly the same weight regardless of how many GT boxes
    # it contains.
    # =========================================================================
    print("\n--- CELL 19: experiment-level summary ---")
    summary_row = {
        "experiment_name": exp,
        "evaluation_mode": "all_gt",
        "prompt_type": args.prompt_type,
        "text_prompt": args.text_prompt,
        "n_exemplars": args.n_exemplars,
        "use_tiling": args.use_tiling,
        "max_dim": args.max_dim,
        "confidence_threshold": conf,
        "nms_iou_threshold": nms_iou,
        "model": args.model_name,
        "imgsz": args.imgsz,
        "prompt_mode": PROMPT_MODE,
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

    # ---- per-archive version of the same table (cluster addition) -----------
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
    # This is NOT the mean of the per-image AP values. All images are handed to
    # supervision at once, so every detection of the whole dataset is ranked in ONE
    # precision-recall curve.
    # =========================================================================
    print("\n--- CELL 20: pooled dataset AP ---")
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
    # One class (Rumex) plus a background row/column:
    #     Actual Rumex      -> Predicted Rumex      = TP
    #     Actual Rumex      -> Predicted Background = FN  (missed plants)
    #     Actual Background -> Predicted Rumex      = FP  (spurious detections)
    #     Actual Background -> Predicted Background = not defined for detection
    #                                                 (there are no true negatives)
    # Counts are pooled over every image at the fixed configuration.
    # =========================================================================
    print("\n--- CELL 21: confusion matrices ---")
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
            f"{exp} - all_gt\nprompt='{args.text_prompt}' | "
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
    #  CELL 23 - QUALITATIVE FIGURES (one per archive + one global)
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
    print(f"Tiling                   : {args.use_tiling}  (whole image, longest side -> {args.max_dim}px)")
    print(f"Detector                 : YOLOE {args.model_name} @ imgsz={args.imgsz}, text "
          f"prompt embedding installed once, in-predictor NMS {args.predict_nms_iou} (permissive)")
    print(f"Confidence threshold     : {conf:.2f} (fixed; YOLOE run once per image)")
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

    # ---------------- prepare the text embedding only -------------------------
    if args.prepare_text_pe:
        prepare_text_pe(args)
        return

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
                  "'Problem B' in text_only_no_tiling_HOW_TO_RUN_yoloe.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" TEXT_ONLY_NO_TILING YOLOE PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" text_prompt    : '{args.text_prompt}'")
    print(f" n_exemplars    : {args.n_exemplars}  (text-only, no visual prompts)")
    print(f" tiling         : {args.use_tiling}  (whole image, longest side -> {args.max_dim} px)")
    print(f" model          : {args.weights} @ imgsz={args.imgsz}  ({PROMPT_MODE})")
    print(f" text emb. cache: {args.text_pe_cache}")
    print(f" yoloe threshold: {args.threshold}  (single inference pass per image)")
    print(f" in-pred. NMS   : {args.predict_nms_iou}  (permissive on purpose)")
    print(f" fp16           : {args.use_fp16}")
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