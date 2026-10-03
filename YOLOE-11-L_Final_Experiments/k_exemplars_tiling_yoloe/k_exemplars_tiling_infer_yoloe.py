#!/usr/bin/env python3
# =============================================================================
#  k_exemplars_tiling_infer_yoloe.py
# =============================================================================
#
#  This is the K-EXEMPLAR TILING experiment with YOLOE instead of SAM3:
#  N_EXEMPLARS = 3 and USE_TILING = True. Each run shows YOLOE exactly K=3
#  example plants (the anchor itself + 2 randomly sampled other GT boxes from the
#  same image). It is the cluster port of
#  E01_single_image_several_exemplar_YOLOE_11.ipynb, extended from ONE image with
#  hand-picked exemplars to the WHOLE dataset with every GT box used as the anchor
#  once (the exemplar selection logic is the one of k_exemplars_tiling_infer_sam3.py,
#  seeded with THIS experiment's own EXPERIMENT_NAME, so the experiment is
#  self-contained).
#
#  WHAT IS DIFFERENT FROM THE SAM3 VERSION (from the notebook, CELL 3)
#    * THERE IS NO EXEMPLAR STRIP. YOLOE takes its exemplars from a SEPARATE
#      reference image. One TILE_SIZE x TILE_SIZE reference window is cut around
#      each exemplar (exemplars inside the same window share it), each window is
#      encoded into a visual prompt embedding (VPE), the VPEs are averaged and
#      L2-normalised and installed on the model with set_classes(). Every tile is
#      then a plain model.predict(tile) call.
#    * No strip -> no strip/tile region rule. Only the plausibility filter stays.
#    * YOLOE runs its own NMS inside the predictor. It is set very permissive
#      (PREDICT_NMS_IOU = 0.90) so that the OFFLINE NMS of PHASE 2 decides.
#
#  The notebook is a TWO-PHASE pipeline and this file keeps that separation:
#
#    PHASE 1 - INFERENCE   (notebook CELL 12 ... CELL 17, GPU)
#        for every image x every anchor:
#            select the exemplars deterministically       (SAM3 CELL 6)
#            build the reference windows                  (CELL 12)
#            encode + install the visual prompt (VPE)     (CELL 13)
#            run YOLOE over all tiles at YOLOE_INFERENCE_THRESHOLD = 0.30
#                                                         (CELL 15 + 17)
#            plausibility filter                          (CELL 14)
#            save the PRE-NMS detections to NPZ
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 18 ... CELL 24, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40,
#        evaluate in both modes (all_gt / held_out) at CONFIDENCE_THRESHOLD =
#        0.30, and write run-level / image-level / experiment-level / pooled-AP
#        CSVs, the confusion matrices (CSV + PNG) and the qualitative
#        GT-vs-prediction figures.
#        Runs in ONE process, after every shard has finished. It never touches
#        YOLOE, so it can be repeated as often as you like from the cached NPZs.
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
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# =============================================================================
#  STATIC CONFIGURATION  (dataset layout - from the cluster scripts)
# =============================================================================
# The notebook pointed at ONE image with RUMEX_CLASS_ID = 0. On the cluster the
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
    "experiment_name", "image_ID", "anchor_idx", "Prompt_ID", "Prompt_Type",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt", "n_reference_windows", "n_detections_pre_nms", "n_tiles",
    "image_width", "image_height", "npz_file", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
# held_out : the GT instances used as visual prompts are IGNORED, and so are the
#            predictions that fall on them.

# ---- image/experiment-level metric columns ---------------------------------
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2"]

# ---- notebook CELL 19 ------------------------------------------------------
STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2

# ---- notebook CELL 3: the only prompt encoding that is active --------------
PROMPT_MODE = "per_exemplar_vpe"

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './k_exemplars_tiling_run_yoloe.sh download' on a "
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

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="k_exemplars_tiling_yoloe - YOLOE k-exemplar visual-prompted Rumex "
                    "detection, tiling ON (inference + offline evaluation), cluster port "
                    "of E01_single_image_several_exemplar_YOLOE_11.ipynb.",
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
    g.add_argument("--experiment-name", default="k_exemplar_tiling_yoloe",
                   help="EXPERIMENT_NAME. Written into every CSV row, every NPZ and "
                        "into the deterministic exemplar seed.")
    g.add_argument("--n-exemplars", type=int, default=3,
                   help="N_EXEMPLARS. This experiment uses 3 (multiple visual prompts: "
                        "the anchor + 2 random GT boxes). Set 1 for a single-prompt variant.")

    tiling = g.add_mutually_exclusive_group()
    tiling.add_argument("--tiling", dest="use_tiling", action="store_true", default=True,
                        help="USE_TILING = True (default): overlapping tiles.")
    tiling.add_argument("--no-tiling", dest="use_tiling", action="store_false",
                        help="USE_TILING = False: the whole image is one single tile.")

    # ------------------------------ tiling ----------------------------------
    g = p.add_argument_group("tiling (CELL 3 / CELL 9)")
    g.add_argument("--tile-size", type=int, default=1000, help="TILE_SIZE")
    g.add_argument("--overlap", type=int, default=150, help="OVERLAP")
    g.add_argument("--no-cache-tiles", dest="cache_tiles_in_memory", action="store_false",
                   default=True,
                   help="CACHE_TILES_IN_MEMORY = False: crop tiles on demand (less RAM).")

    # --------------------------- YOLOE inference ----------------------------
    g = p.add_argument_group("yoloe inference (CELL 3 / CELL 4 / CELL 15)")
    g.add_argument("--weights", default=default_weights(),
                   help="YOLOE_WEIGHTS: path to yoloe-11l-seg.pt (a visual-prompt capable "
                        "checkpoint; the *-seg-pf.pt variants CANNOT take visual prompts).")
    g.add_argument("--prompt-class-name", default="rumex",
                   help="PROMPT_CLASS_NAME (cosmetic).")
    g.add_argument("--imgsz", type=int, default=1024,
                   help="IMGSZ, multiple of 32. 1024 >= TILE_SIZE keeps a tile at "
                        "essentially native resolution. Do NOT drop to 640.")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="YOLOE_INFERENCE_THRESHOLD. YOLOE is executed EXACTLY ONCE per "
                        "(image, anchor) at this score; higher operating thresholds are "
                        "applied offline afterwards.")
    g.add_argument("--predict-nms-iou", type=float, default=0.90,
                   help="YOLOE_PREDICT_NMS_IOU: the per-tile NMS inside the Ultralytics "
                        "predictor, permissive on purpose (the offline NMS decides).")
    g.add_argument("--max-det", type=int, default=300, help="MAX_DET per tile.")
    g.add_argument("--no-retina-masks", dest="retina_masks", action="store_false",
                   default=True,
                   help="RETINA_MASKS = False (masks at network resolution; they are "
                        "resized to the tile before the fill ratio is computed).")
    g.add_argument("--mask-binarise", type=float, default=0.50,
                   help="MASK_BINARISE (masks are only used transiently for the fill ratio).")
    g.add_argument("--batch-size", type=int, default=4,
                   help="BATCH_SIZE: tiles per predict() call.")
    fp = g.add_mutually_exclusive_group()
    fp.add_argument("--fp16", dest="use_fp16", action="store_true", default=True,
                    help="USE_FP16 = True (default): half precision inference on the GPU.")
    fp.add_argument("--no-fp16", dest="use_fp16", action="store_false",
                    help="USE_FP16 = False: use this if you hit a dtype error while the "
                         "visual prompt embedding is applied.")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")

    # ------------------------ reference windows -----------------------------
    g = p.add_argument_group("visual prompt (CELL 3 / CELL 12 / CELL 13)")
    g.add_argument("--reference-window-size", type=int, default=None,
                   help="REFERENCE_WINDOW_SIZE. Default: equal to --tile-size, so the "
                        "exemplar is encoded at the same pixel scale as the tiles.")

    # --------------------------- filters ------------------------------------
    g = p.add_argument_group("filters (CELL 3 / CELL 14)")
    g.add_argument("--min-fill-ratio", type=float, default=0.15, help="MIN_FILL_RATIO")
    g.add_argument("--max-area-fraction", type=float, default=0.80, help="MAX_AREA_FRACTION")
    g.add_argument("--edge-margin", type=int, default=5, help="EDGE_MARGIN")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 19 / CELL 20)")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")
    g.add_argument("--prompt-ignore-iou", type=float, default=0.50,
                   help="PROMPT_IGNORE_IOU: held_out mode ignore rule.")
    g.add_argument("--nms-iou-threshold", type=float, default=0.40,
                   help="BEST_NMS_IOU, applied OFFLINE in PHASE 2 (fixed, no sweep).")
    g.add_argument("--operating-confidence", type=float, default=0.30,
                   help="BEST_CONFIDENCE: the operating point of precision / recall / F1 / "
                        "IoU1 / IoU2. The notebook keeps it EQUAL to --threshold (0.30), so "
                        "no confidence x NMS sweep is performed.")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot")
    g.add_argument("--plot-evaluation-mode", default="all_gt", choices=EVALUATION_MODES,
                   help="PLOT_EVALUATION_MODE: the image-level AP50 of this mode selects "
                        "the image that gets plotted.")
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
    g.add_argument("--max-anchors-per-image", type=int, default=0,
                   help="Debug / cost control: use at most this many GT boxes as anchors per "
                        "image (0 = every GT box becomes an anchor once).")
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
    if args.imgsz % 32 != 0:
        p.error("--imgsz must be a multiple of 32")

    # REFERENCE_WINDOW_SIZE = TILE_SIZE (CELL 3)
    if args.reference_window_size is None:
        args.reference_window_size = args.tile_size
    # PROMPT_TYPE (CELL 3)
    args.prompt_type = "multiple" if args.n_exemplars > 1 else "single"
    args.model_name = Path(str(args.weights)).name
    return args

# =============================================================================
#  OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections (NPZ, one file per image x anchor)
#                          + one runs_manifest_shard<i>.csv per shard
#     metrics/             run / image / experiment / dataset level CSVs
#     confusion_matrices/  CSV + PNG for all_gt and held_out
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
    One manifest PER SHARD, so four processes never append to the same file; the
    resume step simply reads all of them back (see load_done_runs).
    """
    return paths.raw_detections / f"runs_manifest_{experiment_name}_shard{shard_index}.csv"

# =============================================================================
#  CELL 5 - STABLE REPRODUCIBILITY HELPERS
# =============================================================================

def stable_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)

def select_exemplar_indices(n_gt: int, anchor_idx: int, n_exemplars: int,
                            image_id: str, experiment_name: str) -> List[int]:
    """
    Choose which GT instances of ONE image are used as visual prompts.
    (Replaces the MANUAL ANCHOR_INDEX / EXTRA_EXEMPLAR_INDICES of the notebook;
    same logic as k_exemplars_tiling_infer_sam3.py, seeded with this experiment's
    own name.)

    Input : n_gt        - number of GT boxes in the image
            anchor_idx  - index of the GT box this run is "about" (always a prompt)
            n_exemplars - how many prompts in total (1 or k)
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
    Human-readable id of a prompt set. The ANCHOR is always the first number.
      single   -> "9"
      multiple -> "9+3+14"
    """
    return "+".join(str(int(i)) for i in exemplar_indices)

def seed_run(run_seed: int) -> None:
    """Notebook CELL 6: torch.manual_seed + np.random.seed for this run."""
    import torch
    torch.manual_seed(run_seed)
    np.random.seed(run_seed % (2 ** 32))

# =============================================================================
#  CELL 6 - DATASET AND YOLO ANNOTATION HELPERS
# =============================================================================
#  The notebook opened ONE image with a single RUMEX_CLASS_ID. On the cluster
#  both archives are pooled into one dataset, each with its own class id and a
#  FLAT annotations_yolo folder, so discover_images() is the archive-aware
#  version from the cluster scripts. load_yolo_boxes / safe_filename are
#  unchanged notebook code.
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
#  CELL 7 - IoU HELPERS
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

def safe_f1(precision: float, recall: float) -> float:
    """F1 = 2PR/(P+R) with a safe zero denominator (returns 0.0)."""
    denom = precision + recall
    return float(2.0 * precision * recall / denom) if denom > 0 else 0.0

# =============================================================================
#  CELL 8 -  ONE-TO-ONE MATCHING
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

# =============================================================================
#  CELL 9 - TILES: GENERATED ONCE PER IMAGE, REUSED BY EVERY ANCHOR
# =============================================================================
# Open the image once -> build the tile list once -> reuse it for every anchor.
# Tiles are kept as CPU/PIL images (never on the GPU) and are handed to YOLOE
# EXACTLY as they are - no strip is pasted on top of them.
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
#  CELL 12 - REFERENCE WINDOWS FOR THE VISUAL PROMPT
# =============================================================================
# THIS REPLACES THE WHOLE EXEMPLAR-STRIP MACHINERY OF THE SAM3 VERSION (local
# background sampling, feathering, canvas composition, strip/tile region rule).
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
#  CELL 14 - PLAUSIBILITY FILTER
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
#  CELL 4 + CELL 13 + CELL 15 - YOLOE MODEL, VISUAL PROMPT AND BATCHED INFERENCE
# =============================================================================
#  * the exemplars are encoded ONCE PER RUN (image x anchor) into one visual
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
#  CELL 17 - ONE PROMPT SET OVER ALL TILES OF ONE IMAGE
# =============================================================================
# For every tile:
#     YOLOE   (prompted model, plain predict, threshold 0.30)
#     plausibility filter
#     tile coords -> ORIGINAL image coords
# There is no "compose" step and no strip/tile region remapping.
# =============================================================================
def run_anchor_over_tiles(runner: YoloeRunner, tile_cache: List[dict],
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
#  PRE-NMS DETECTION STORAGE
# =============================================================================
# For every run (= one image x one anchor) we store the detections AFTER
#   YOLOE inference at 0.30 -> plausibility filtering
#   -> conversion to original-image coordinates
# but BEFORE
#   any operating confidence threshold and BEFORE the offline NMS.
#
# That is exactly the state needed to replay any (confidence, NMS IoU) pair
# offline without ever running YOLOE again. Masks are never stored.
# =============================================================================
def run_npz_path(raw_detections_dir: Path, image_id: str, anchor_idx: int) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one run."""
    return raw_detections_dir / f"{safe_filename(image_id)}__anchor{int(anchor_idx):03d}.npz"

def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        anchor_idx: int, detections: dict, gt_boxes: np.ndarray,
                        prompt_indices: Sequence[int], image_size: Tuple[int, int],
                        archive: str, flight: str, class_id: int,
                        reference_windows: Sequence[Tuple[int, int, int, int]]) -> Path:
    """
    Write one run's pre-NMS detections to NPZ. The file is self-contained: it also
    stores the GT boxes, the prompt indices and the reference windows, so the whole
    offline evaluation can run without re-opening images or label files.
    """
    path = run_npz_path(raw_detections_dir, image_id, anchor_idx)
    np.savez_compressed(
        path,
        experiment_name=np.array(experiment_name),
        image_ID=np.array(image_id),
        anchor_idx=np.array(int(anchor_idx)),
        prompt_indices=np.array(prompt_indices, dtype=np.int32),
        reference_windows=np.array(reference_windows, dtype=np.int32).reshape(-1, 4),
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
        run["archive"] = str(z["archive"]) if "archive" in z else ""
        run["flight"] = str(z["flight"]) if "flight" in z else ""
    return run

# =============================================================================
#  RESUME SUPPORT  (adapted to several shard manifests)
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
#  MAIN GPU INFERENCE LOOP  (PHASE 1)
# =============================================================================
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE
#     read its GT boxes ONCE
#     build the overlapping tiles ONCE (cached on CPU)
#     FOR EACH anchor:
#         select the exemplars deterministically
#         build the reference windows around them (CELL 12)
#         encode + install the visual prompt (CELL 13)
#         run YOLOE over the cached tiles in batches (threshold 0.30)
#         save the PRE-NMS detections (NPZ)
#     release the image and the tile cache
#
# NO offline NMS and NO metric computation happens here - that is all done in
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

    runner = YoloeRunner(args)
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

            # deterministic prompt selection (SHA-256 based, see CELL 5)
            exemplar_indices = select_exemplar_indices(n_gt, anchor_idx, args.n_exemplars,
                                                       image_id, exp)
            prompt_id = format_prompt_id(exemplar_indices)
            seed_run(stable_seed(exp, image_id, prompt_id))

            # CELL 12 + CELL 13: reference windows -> one VPE -> installed on the model
            reference_windows = build_reference_windows(image, exemplar_indices, gt_boxes,
                                                        args.reference_window_size)
            runner.install_prompt(reference_windows)

            detections = run_anchor_over_tiles(runner, tile_cache, image, args)

            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, anchor_idx, detections, gt_boxes,
                exemplar_indices, (img_w, img_h), rec.archive, rec.flight, rec.class_id,
                [w["window"] for w in reference_windows])

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
                "n_reference_windows": len(reference_windows),
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
                  f"ref windows={len(reference_windows)} | tiles={n_tiles} | "
                  f"pre-NMS detections={len(detections['scores'])} | {run_seconds:.1f}s")

            for w in reference_windows:
                w["image"] = None
            del detections, reference_windows
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
    print("\n--- DRY RUN: counting the work without loading YOLOE ---")
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
    print(f"  => ~{total_anchors * tiles} YOLOE tile forward passes for those {len(sample)} images")
    print(f"     (+ 1..{args.n_exemplars} reference-window VPE encodings per anchor run)")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{total_anchors} "
          f"(one per image x anchor)")
    weights = Path(str(args.weights))
    if weights.is_absolute() or weights.parent != Path("."):
        print(f"  weights file: {weights} -> {'FOUND' if weights.is_file() else 'MISSING'}")

# =============================================================================
#  LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
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
# group. YOLOE's own per-tile NMS was left permissive (0.90) precisely so that
# this stage is what actually decides.
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
# held_out is still the right protocol even though the exemplars come from a
# separate reference window: the prompt plants are real GT instances of THIS
# image, so finding them again is not evidence of generalisation.
#
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
#   saved predictions with score >= 0.30 (the YOLOE inference threshold), after
#   the selected NMS - the same 0.30 floor as the SAM3 version.
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
    All post-NMS predictions with score >= YOLOE_INFERENCE_THRESHOLD are used;
    in held_out mode the predictions that were IGNORED (they belong to prompt
    plants) are removed first, exactly like in the operating-point evaluation.
    """
    ev = evaluate_run_predictions(nms_run["boxes"], nms_run["scores"],
                                  eval_gt, prompt_gt, eval_iou, ignore_iou)
    keep = ev["status"] != STATUS_IGNORED
    return (make_detections(nms_run["boxes"][keep], nms_run["scores"][keep]),
            make_detections(eval_gt))

def evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, confidence_threshold: float,
                                eval_iou: float, ignore_iou: float) -> dict:
    """
    Drop every post-NMS prediction below the operating confidence threshold, then
    evaluate. This is what produces Precision, Recall, F1, IoU1 and IoU2.
    """
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_run_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                    eval_gt, prompt_gt, eval_iou, ignore_iou)

# =============================================================================
#  QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right)
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
#   is displayed. Its predictions are the post-NMS boxes (merged over all tiles)
#   with score >= CONFIDENCE_THRESHOLD; the exemplar (prompt) boxes are drawn dashed.
#
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

def select_image_for_plot(run_level_df, image_level_df, mode: str, min_gt: int,
                          archive: Optional[str] = None):
    """
    Output: (image_level_row, selection_rule_text) or (None, reason)
    'archive' restricts the candidates to one archive; None = the whole dataset.
    """
    img_df = image_level_df[image_level_df["evaluation_mode"] == mode].copy()
    runs_df = run_level_df[run_level_df["evaluation_mode"] == mode]
    if archive is not None:
        img_df = img_df[img_df["archive"] == archive]
        runs_df = runs_df[runs_df["archive"] == archive]
    if img_df.empty:
        return None, f"no image for archive={archive}"

    n_gt_per_image = runs_df.groupby("image_ID")["n_gt_total"].first()
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
                           image_paths: dict, plt, scope_label: str,
                           archive: Optional[str] = None) -> None:
    """One qualitative figure for one scope (a single archive, or the whole dataset)."""
    from matplotlib.lines import Line2D

    mode = args.plot_evaluation_mode
    exp = args.experiment_name

    img_row, rule = select_image_for_plot(run_level_df, image_level_df, mode,
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
    run = next((r for r in runs
                if r["image_ID"] == plot_image_id and r["anchor_idx"] == anchor), None)
    if run is None:
        print(f"Nothing to plot for {scope_label}: no cached run for "
              f"{plot_image_id} anchor {anchor}")
        return

    nms_run = apply_nms_to_run(run, args.nms_iou_threshold)
    keep = nms_run["scores"] >= args.operating_confidence
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
        f"{exp} | {plot_image_id} | mode={mode} | scope={scope_label} | "
        f"image-level AP50_mean={img_row['AP50_mean']:.3f} "
        f"(over {int(img_row['n_runs_valid_for_macro'])} anchors)\n"
        f"YOLOE {args.model_name} @ imgsz={args.imgsz} | {args.n_exemplars} exemplars "
        f"({PROMPT_MODE}) | tile={args.tile_size}px, overlap={args.overlap}px | "
        f"conf={args.operating_confidence:.2f}, NMS IoU={args.nms_iou_threshold:.2f}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    png_path = paths.plots / (f"best_image_{scope_label}_{safe_filename(plot_image_id)}"
                              f"_anchor{anchor:03d}_{mode}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

    print(f"\n[{scope_label}] Selected image : {plot_image_id} "
          f"({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Shown anchor   : {anchor} (best run-level AP50 of this image)")
    print(f"[{scope_label}] Figure saved   : {png_path}")

def make_qualitative_plots(args, paths: Paths, runs, run_level_df, image_level_df, plt) -> None:
    """The qualitative plot for every scope: one figure per archive plus one global figure."""
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

    archives_present = sorted(str(a) for a in run_level_df["archive"].dropna().unique() if str(a))
    for archive in archives_present:
        plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df,
                               image_paths, plt, scope_label=archive, archive=archive)
    plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df,
                           image_paths, plt, scope_label="ALL", archive=None)

def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2: notebook CELL 18 ... CELL 24 over the whole dataset, one process, no GPU."""
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
        print("         Confusion-matrix CSVs will still be written; the PNGs will not.")

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

    runs, manifest = load_runs(paths, exp)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  RUN-LEVEL METRICS  (one run = one image x one anchor/prompt set)
    # =========================================================================
    #   AP50 / AP50_95 : all post-NMS predictions >= 0.30, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : only predictions >= CONFIDENCE_THRESHOLD
    #
    # Special case (held_out with no evaluable GT, i.e. every plant of the image
    # was used as a prompt): the metrics are written as NaN and valid_for_macro is
    # False so they are excluded from every mean/std, but TP/FN = 0 and the real FP
    # count are kept, because such a run can still produce false positives that
    # must show up in the pooled counts and in the confusion matrix.
    # =========================================================================
    print("\n--- run-level metrics ---")
    run_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)
        for mode in EVALUATION_MODES:
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)

            # ---- AP: every prediction >= 0.30 after NMS (ignored ones removed) ----
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            ap50, ap5095 = compute_ap([p_det], [g_det])

            # ---- operating point -------------------------------------------------
            ev = evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, conf,
                                             eval_iou, ignore_iou)
            valid = ev["valid_for_macro"]
            nan = float("nan")

            run_rows.append({
                "experiment_name": exp,
                "model": args.model_name,
                "image_ID": run["image_ID"],
                "archive": run.get("archive", ""),
                "flight": run.get("flight", ""),
                "anchor_idx": run["anchor_idx"],
                "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"],
                "evaluation_mode": mode,
                "confidence_threshold": conf,
                "nms_iou_threshold": nms_iou,
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
    #  IMAGE-LEVEL METRICS
    # =========================================================================
    # All anchor runs of the same image are averaged into ONE value per image and
    # per evaluation mode. The std here is the spread BETWEEN the different
    # anchor/prompt selections of the SAME image, i.e. "how sensitive is the
    # result to which plant was used as the visual prompt?".
    # NaN rows (held_out runs with no evaluable GT) are ignored by pandas mean/std.
    # std is NaN when an image has only one valid run - that is expected.
    # =========================================================================
    print("\n--- image-level metrics ---")
    image_rows = []
    for (image_id, mode), grp in run_level_df.groupby(["image_ID", "evaluation_mode"]):
        row = {
            "experiment_name": exp,
            "image_ID": image_id,
            "archive": grp["archive"].iloc[0],
            "flight": grp["flight"].iloc[0],
            "evaluation_mode": mode,
            "confidence_threshold": conf,
            "nms_iou_threshold": nms_iou,
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
    #  EXPERIMENT-LEVEL SUMMARY
    # =========================================================================
    # Computed from the IMAGE-LEVEL values, not from the raw run rows, so that
    # every UAV image contributes exactly the same weight regardless of how many
    # GT boxes (and therefore how many anchor runs) it contains.
    # The std here is the variation BETWEEN UAV images.
    # =========================================================================
    print("\n--- experiment-level summary ---")
    summary_rows = []
    for mode in EVALUATION_MODES:
        sub = image_level_df[image_level_df["evaluation_mode"] == mode]
        row = {
            "experiment_name": exp,
            "model": args.model_name,
            "evaluation_mode": mode,
            "prompt_type": args.prompt_type,
            "n_exemplars": args.n_exemplars,
            "prompt_mode": PROMPT_MODE,
            "use_tiling": args.use_tiling,
            "confidence_threshold": conf,
            "nms_iou_threshold": nms_iou,
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

    # ---- per-archive version of the same table ------------------------------
    # Both archives are pooled here, so the same numbers are also written per
    # archive - the pooled rows above stay exactly as defined.
    per_archive_rows = []
    for (archive, mode), sub_df in image_level_df.groupby(["archive", "evaluation_mode"]):
        row = {
            "experiment_name": exp,
            "archive": archive,
            "evaluation_mode": mode,
            "n_images": int(sub_df["image_ID"].nunique()),
            "n_runs": int(sub_df["n_runs_total"].sum()),
            "n_runs_valid_for_macro": int(sub_df["n_runs_valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub_df[f"{col}_mean"].mean()
            row[f"{col}_std"] = sub_df[f"{col}_mean"].std()
        per_archive_rows.append(row)

    per_archive_df = pd.DataFrame(per_archive_rows)
    per_archive_csv = paths.metrics / "experiment_summary_per_archive.csv"
    per_archive_df.to_csv(per_archive_csv, index=False)
    print(f"\nPer-archive summary -> {per_archive_csv}\n")
    if not per_archive_df.empty:
        print(per_archive_df[["archive", "evaluation_mode", "n_images", "n_runs",
                              "AP50_mean", "precision_mean", "recall_mean",
                              "F1_mean"]].to_string(index=False))

    # =========================================================================
    #  POOLED DATASET AP50 / AP50:95
    # =========================================================================
    # This is NOT the mean of the image-level AP values. All runs are handed to
    # supervision as evaluation EPISODES at once, so every detection of the whole
    # dataset is ranked in ONE precision-recall curve.
    # =========================================================================
    print("\n--- pooled dataset AP ---")
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

        ap50, ap5095 = compute_ap(pred_list, gt_list)
        dataset_rows.append({
            "experiment_name": exp,
            "evaluation_mode": mode,
            "n_images": len(images_used),
            "n_runs": len(runs),
            "confidence_used_for_AP": args.threshold,     # AP always uses >= 0.30
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
    print("\nFor comparison, the MEAN of the image-level AP50 values (a different quantity):")
    print(image_level_df.groupby("evaluation_mode")["AP50_mean"].mean().to_string())

    # =========================================================================
    #  CELL 23 - DATASET-LEVEL CONFUSION MATRICES
    # =========================================================================
    #     Actual Rumex      -> Predicted Rumex      = TP
    #     Actual Rumex      -> Predicted Background = FN  (missed plants)
    #     Actual Background -> Predicted Rumex      = FP  (spurious detections)
    #     Actual Background -> Predicted Background = not defined for detection
    #                                                 (there are no true negatives)
    # Counts are pooled over every run at the frozen configuration. In held_out
    # mode the prompt plants and the ignored detections do not appear anywhere.
    # =========================================================================
    print("\n--- confusion matrices ---")

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
                f"{exp} - {mode}\nconf={conf:.2f}, "
                f"NMS IoU={nms_iou:.2f}, eval IoU={eval_iou:.2f}",
                paths.confusion_matrices / f"confusion_matrix_{mode}.png")

        confusion_summary.append({
            "experiment_name": exp, "model": args.model_name, "evaluation_mode": mode,
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
    #  QUALITATIVE FIGURES (one per archive + one global)
    # =========================================================================
    if args.no_plots:
        print("\n--- qualitative figures skipped (--no-plots) ---")
    elif not _HAS_MPL:
        print("\n--- qualitative figures skipped (matplotlib unavailable) ---")
    else:
        print("\n--- qualitative GT-vs-prediction figures ---")
        make_qualitative_plots(args, paths, runs, run_level_df, image_level_df, plt)

    # =========================================================================
    #  CELL 24 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY")
    print("=" * 78)
    print(f"Model                    : {args.model_name} @ imgsz={args.imgsz}")
    print(f"Prompts per run          : {args.n_exemplars} ({args.prompt_type}), "
          f"encoding={PROMPT_MODE} (NO exemplar strip)")
    print(f"Tiling                   : {args.use_tiling}  (tile={args.tile_size}px, "
          f"overlap={args.overlap}px)")
    print(f"YOLOE inference threshold: {args.threshold} (executed once per image x anchor)")
    print(f"In-predictor NMS IoU     : {args.predict_nms_iou} (permissive; offline NMS decides)")
    print(f"Operating point          : confidence={conf:.2f} (== the inference "
          f"threshold), NMS IoU={nms_iou:.2f} (both fixed)")
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
                  "'Problem B' in k_exemplars_tiling_HOW_TO_RUN_yoloe.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" K_EXEMPLARS_WITH_TILING YOLOE PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" model          : {args.weights} @ imgsz={args.imgsz}")
    print(f" n_exemplars    : {args.n_exemplars}  ({args.prompt_type} prompt, {PROMPT_MODE}, "
          f"reference window={args.reference_window_size}px)")
    print(f" tiling         : {args.use_tiling}  (tile={args.tile_size}, overlap={args.overlap})")
    print(f" yoloe threshold: {args.threshold}  (single inference pass per image x anchor)")
    print(f" in-pred. NMS   : {args.predict_nms_iou}  (permissive on purpose)")
    print(f" batch / fp16   : {args.batch_size} / {args.use_fp16}")
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
