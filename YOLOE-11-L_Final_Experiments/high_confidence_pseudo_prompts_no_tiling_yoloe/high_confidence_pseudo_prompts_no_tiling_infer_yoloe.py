#!/usr/bin/env python3
# =============================================================================
#  high_confidence_pseudo_prompts_no_tiling_infer_yoloe.py
# =============================================================================
#
#  This is the HIGH-CONFIDENCE PSEUDO-PROMPT NO-TILING experiment with YOLOE
#  instead of SAM3: TWO YOLOE ROUNDS PER IMAGE, NO tiling - the WHOLE UAV image
#  is resized to MAX_DIM = 1024 on its longest side.
#
#    ROUND 1: the resized image is prompted with the K_EXEMPLARS = 3 size-based
#             GT boxes (S, M, L) -> one YOLOE pass.
#    SELF-PROMPTS: the round-1 detections are NMS-merged, the ones with score >=
#             0.30 that fall on a prompt plant are dropped (IoU >=
#             SELF_PROMPT_EXCLUDE_IOU with any S/M/L box), and the N_SELF_PROMPTS
#             = 2 highest-confidence remaining boxes are kept - these are YOLOE's
#             OWN predictions, NOT ground truth.
#    ROUND 2: the same resized image is prompted again with 3 + 2 = 5 boxes (S, M,
#             L + the 2 self-prompts) -> one YOLOE pass.
#    The FINAL predictions are the round-2 output (round 1 is NOT merged in);
#    they are evaluated against the original human GT, and round 1 is evaluated
#    too, for comparison. If round 1 leaves no eligible box, round 2 would repeat
#    round 1 exactly, so it is skipped and the round-1 output is final
#    (second_run = False).
#
#  The S/M/L choice is deterministic, so there is exactly ONE run per image (no
#  anchors, no random sampling, no seeds). It is the cluster port of
#  E01_single_image_pseudo_prompts_YOLOE_11_no_tiling.ipynb (the YOLOE version of
#  the SAM3 notebook high_confidence_pseudo_prompts_no_tiling), extended from ONE
#  image to the WHOLE dataset.
#
#  WHAT IS IDENTICAL TO THE SAM3 NOTEBOOK
#    * the S / M / L rule, the self-prompt rule, the whole image at MAX_DIM with
#      the prompt boxes on the SAME image, NO plausibility filter, plain offline
#      NMS (no tile provenance), round-1 metrics for comparison
#
#  WHAT IS DIFFERENT, AND WHY
#    * YOLOE first encodes the prompt boxes (3 in round 1, 5 in round 2), scaled
#      into the resized image, into ONE visual prompt embedding (VPE) from that
#      same image ("same_image_vpe"), installs it with set_classes(), and then
#      runs a plain model.predict() on the image - the YOLOE equivalent of SAM3's
#      positive box prompts in one forward pass.
#    * YOLOE runs its own NMS inside the predictor. It is set very permissive
#      (PREDICT_NMS_IOU = 0.90) so that the OFFLINE NMS decides.
#
#  The notebook is a TWO-PHASE pipeline and this file keeps that separation:
#
#    PHASE 1 - INFERENCE   (notebook CELL 5 ... CELL 19, GPU)
#        for every image:
#            select the S / M / L exemplars by area       (CELL 5)
#            resize the whole image to MAX_DIM ONCE       (CELL 9)
#            ROUND 1: VPE from S/M/L -> YOLOE on the resized image (CELL 12-16)
#            NMS on round 1 -> pick the self-prompts      (CELL 17 + 18)
#            ROUND 2: VPE from S/M/L + self-prompts -> YOLOE again  (CELL 19)
#            boxes back to full resolution
#            save round 1, the self-prompts and the final detections to NPZ
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 20 ... CELL 26, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40,
#        evaluate the FINAL and the ROUND-1 detections in both modes (all_gt /
#        held_out) at CONFIDENCE_THRESHOLD = 0.30, and write per-image /
#        experiment-level / pooled-AP CSVs (with *_round1 columns and deltas), the
#        confusion matrices (CSV + PNG), the recall per size group, the prompt
#        re-detection table and the qualitative GT-vs-prediction figures.
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
    "experiment_name", "image_ID", "Prompt_ID", "Prompt_Type",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt", "area_S_px2", "area_M_px2", "area_L_px2", "median_gt_area_px2",
    "n_detections_r1_pre_nms", "n_detections_r1_post_nms",
    "n_self_prompts", "self_prompt_scores", "second_run",
    "n_detections_pre_nms",
    "image_width", "image_height", "resized_width", "resized_height", "resize_scale",
    "npz_file", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
# held_out : the GT instances used as visual prompts are IGNORED, and so are the
#            predictions that fall on them.

# ---- image/experiment-level metric columns ---------------------------------
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2"]

# ---- figure colours of the three prompt roles (notebook CELL 5) --------------
# (the self-prompts are drawn dashed WHITE)
ROLE_COLOURS = {"S": "orange", "M": "deepskyblue", "L": "lime"}
ROLE_NAMES = {"S": "smallest", "M": "medium", "L": "largest"}

# ---- notebook CELL 19 ------------------------------------------------------
STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2

# ---- notebook CELL 3: the size-based prompt ---------------------------------
# The full prompt type is f"size_SML+{N_SELF_PROMPTS}self" (set in parse_args,
# because it depends on --n-self-prompts).
PROMPT_TYPE_BASE = "size_SML"
SIZE_MEASURE = "area"          # box size = width x height (full-resolution pixels)

# ---- notebook CELL 3: the prompt encoding ----------------------------------
# The exemplar boxes are scaled into the resized image, and YOLOE encodes that
# SAME image + boxes into ONE visual prompt embedding (one prompt group, class 0).
PROMPT_MODE = "same_image_vpe"

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh download' on a "
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
        description="high_confidence_pseudo_prompts_no_tiling_yoloe - YOLOE visual-prompted "
                    "Rumex detection, TWO rounds per image: S/M/L GT exemplars, then S/M/L "
                    "+ the highest-confidence round-1 predictions as self-prompts; NO "
                    "tiling (whole image resized to MAX_DIM) (inference + offline "
                    "evaluation), cluster port of "
                    "E01_single_image_pseudo_prompts_YOLOE_11_no_tiling.ipynb.",
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
    g.add_argument("--experiment-name", default="high_confidence_pseudo_prompts_no_tiling_yoloe",
                   help="EXPERIMENT_NAME. Written into every CSV row and every NPZ.")

    # --------------------------- whole-image resize -------------------------
    g = p.add_argument_group("whole-image resize (CELL 3 / CELL 9)")
    g.add_argument("--max-dim", type=int, default=1024,
                   help="MAX_DIM: the whole image is resized to this size on its longest "
                        "side (8192 x 5460 -> 1024 x 682). No tiling.")

    # --------------------------- YOLOE inference ----------------------------
    g = p.add_argument_group("yoloe inference (CELL 3 / CELL 4 / CELL 14)")
    g.add_argument("--weights", default=default_weights(),
                   help="YOLOE_WEIGHTS: path to yoloe-11l-seg.pt (a visual-prompt capable "
                        "checkpoint; the *-seg-pf.pt variants CANNOT take visual prompts).")
    g.add_argument("--prompt-class-name", default="rumex",
                   help="PROMPT_CLASS_NAME (cosmetic).")
    g.add_argument("--imgsz", type=int, default=None,
                   help="IMGSZ, multiple of 32. Default: equal to --max-dim, so the "
                        "resized image is only padded (letterbox), not shrunk again.")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="YOLOE_INFERENCE_THRESHOLD. Used in BOTH rounds (one whole-image "
                        "pass each); higher operating thresholds are applied offline "
                        "afterwards.")
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
                    help="USE_FP16 = True (default): half precision inference on the GPU.")
    fp.add_argument("--no-fp16", dest="use_fp16", action="store_false",
                    help="USE_FP16 = False: use this if you hit a dtype error while the "
                         "visual prompt embedding is applied.")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")

    # ------------------- self-prompts (round 1 -> round 2) ------------------
    g = p.add_argument_group("self-prompts (CELL 3 / CELL 18)")
    g.add_argument("--n-self-prompts", type=int, default=2,
                   help="N_SELF_PROMPTS: highest-confidence round-1 predictions added to "
                        "the round-2 prompt.")
    g.add_argument("--self-prompt-exclude-iou", type=float, default=0.50,
                   help="SELF_PROMPT_EXCLUDE_IOU: a round-1 prediction whose IoU with ANY "
                        "S/M/L prompt box is >= this value is NOT eligible as a self-prompt.")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 18 / CELL 19)")
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
                   help="PLOT_EVALUATION_MODE: the per-image AP50 of this mode selects "
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
    if args.max_dim < 32:
        p.error("--max-dim must be >= 32")

    # IMGSZ = MAX_DIM (CELL 3)
    if args.imgsz is None:
        args.imgsz = args.max_dim
    if args.imgsz % 32 != 0:
        p.error("--imgsz must be a multiple of 32")
    # USE_TILING = False (CELL 3) - always whole-image in this experiment
    args.use_tiling = False
    # K_EXEMPLARS / PROMPT_TYPE (CELL 3) - fixed by the S/M/L rule
    args.k_exemplars = 3
    if args.n_self_prompts < 0:
        p.error("--n-self-prompts must be >= 0")
    args.prompt_type = f"{PROMPT_TYPE_BASE}+{args.n_self_prompts}self"
    args.model_name = Path(str(args.weights)).name
    return args

# =============================================================================
#  OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections (NPZ, one file per image)
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
#  CELL 5 - SIZE-BASED EXEMPLAR SELECTION (S / M / L)
# =============================================================================
# Same rule as the SAM3 notebook k_size_exemplars_no_tiling.
# Size = box AREA (width x height) in full-resolution pixels. ALL GT boxes of the
# image are candidates - boxes cut by the image border are treated like any other
# GT box.
#   S = smallest area
#   L = largest area (another box than S)
#   M = among the remaining boxes, the one whose area is closest to the MEDIAN
#       area of ALL GT boxes of the image
# Ties are broken by the lower GT index -> fully deterministic, so there are no
# anchors and no seeds: exactly ONE run per image.
# Fewer than 3 GT boxes: 1 box -> [S], 2 boxes -> [S, L].
# The prompt order is always S, M, L.
# =============================================================================
def box_areas(boxes) -> np.ndarray:
    """(N,4) [x1,y1,x2,y2] -> (N,) areas in px^2."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

def select_size_exemplars(gt_boxes) -> dict:
    """
    Input : gt_boxes (G,4) full-resolution GT boxes of ONE image
    Output: dict with
              'indices' - GT indices in prompt order (S, M, L)
              'roles'   - ["S", "M", "L"] (shorter when G < 3)
              'areas'   - area of each selected box
              'median_area', 'all_areas'
    """
    areas = box_areas(gt_boxes)
    n = len(areas)
    if n == 0:
        raise ValueError("The image has no GT boxes -> no size-based exemplars.")

    small = int(np.argmin(areas))                          # first index on ties
    if n == 1:
        indices, roles = [small], ["S"]
    else:
        masked = areas.astype(np.float64).copy()
        masked[small] = -np.inf
        large = int(np.argmax(masked))                     # first index on ties, != S
        if n == 2:
            indices, roles = [small, large], ["S", "L"]
        else:
            dist = np.abs(areas.astype(np.float64) - float(np.median(areas)))
            dist[[small, large]] = np.inf
            medium = int(np.argmin(dist))                  # first index on ties
            indices, roles = [small, medium, large], ["S", "M", "L"]

    return {
        "indices": indices,
        "roles": roles,
        "areas": areas[indices],
        "median_area": float(np.median(areas)),
        "all_areas": areas,
    }

def format_size_prompt_id(selection: dict) -> str:
    """e.g. {'indices': [12, 5, 40], 'roles': ['S','M','L']} -> 'S12+M5+L40'."""
    return "+".join(f"{r}{i}" for r, i in zip(selection["roles"], selection["indices"]))

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
#  CELL 9 - RESIZE HELPER (whole-image, no tiling)
# =============================================================================
# Same helper as the SAM3 no-tiling notebook: the WHOLE image is downscaled to
# MAX_DIM on its longest side (8192 x 5460 -> 1024 x 682, scale = 0.125). It is
# done ONCE per image. The prompt is encoded from, and
# the prediction is run on, THIS resized image; predicted boxes are rescaled back
# to full resolution right after inference.
# =============================================================================
def resize_for_model(img: Image.Image, max_dim: int) -> Tuple[Image.Image, float]:
    """
    Output: (resized PIL image, scale) with scale = max_dim / longest side.
    Images already smaller than max_dim are returned unchanged (scale = 1.0).
    """
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale >= 1:
        return img, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.BILINEAR), scale

# =============================================================================
#  CELL 12 - THE VISUAL PROMPT: SAME RESIZED IMAGE + SCALED EXEMPLAR BOXES
# =============================================================================
# SAM3 no-tiling passes the exemplar GT boxes, scaled into the resized image, as
# box prompts on that SAME image. The YOLOE equivalent: the resized image itself
# is the reference image, and the exemplar boxes are scaled by the same factor.
# No strip and no reference windows are needed - there are no tiles, so the
# exemplar plants are always visible in the image that is searched.
# =============================================================================
def scale_exemplar_boxes(gt_boxes: np.ndarray, exemplar_indices: Sequence[int],
                         resize_scale: float) -> np.ndarray:
    """Exemplar GT boxes (full resolution) -> boxes in the resized image, (N, 4)."""
    return np.asarray([[c * resize_scale for c in gt_boxes[i]] for i in exemplar_indices],
                      dtype=np.float32).reshape(-1, 4)

def build_visual_prompts(boxes_in_ref) -> dict:
    """
    The dict Ultralytics expects. ALL exemplars carry class id 0 because they are
    all examples of the SAME concept (Rumex): they form one visual-prompt group
    and YOLOE aggregates them into one visual prompt embedding.
    """
    boxes = np.asarray(boxes_in_ref, dtype=np.float32).reshape(-1, 4)
    return {"bboxes": boxes, "cls": np.zeros(len(boxes), dtype=np.int64)}

# =============================================================================
#  CELL 4 + CELL 13 + CELL 14 - YOLOE MODEL, VISUAL PROMPT AND WHOLE-IMAGE INFERENCE
# =============================================================================
#  * the exemplars are encoded ONCE PER RUN (= one image) into one visual
#    prompt embedding (VPE) from the resized image, and installed with
#    set_classes(); the prediction is then a plain model.predict(resized_image)
#  * imgsz=IMGSZ (= MAX_DIM): the resized image is only padded, not shrunk again
#  * conf=YOLOE_INFERENCE_THRESHOLD (0.30): one pass, every higher operating
#    threshold is replayed offline later
#  * iou=YOLOE_PREDICT_NMS_IOU (0.90) is deliberately permissive so the OFFLINE
#    NMS is the one that decides
#  * NO plausibility filter and therefore no fill ratio: every detection >= 0.30
#    is kept, and masks are never read (KEEP_MASKS = False on the cluster).
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

class YoloeRunner:
    """Owns the YOLOE model, installs the visual prompt and runs the prediction."""

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

        Input : reference_image - PIL image (here: the resized whole image)
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
        # all exemplar boxes of this image form ONE visual-prompt group
        vp.set_prompts(build_visual_prompts(boxes_in_ref))
        vpe = vp.get_vpe(reference_image)
        del vp
        return vpe.detach().float().cpu()

    def install_prompt(self, reference_image: Image.Image, boxes_in_ref):
        """
        PROMPT_MODE = "same_image_vpe": ONE VPE from the resized image + the scaled
        exemplar boxes, L2-normalised, installed on the model with set_classes().
        """
        torch = self.torch
        vpe_final = self.extract_vpe(reference_image, boxes_in_ref)
        vpe_final = torch.nn.functional.normalize(vpe_final, dim=-1, p=2)
        self.model.set_classes([self.args.prompt_class_name], vpe_final)
        self.model.predictor = None        # drop the VP predictor; use the plain one
        return vpe_final

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

# =============================================================================
#  CELL 16 - ONE PROMPT SET OVER THE WHOLE (RESIZED) IMAGE (used by both rounds)
# =============================================================================
# ONE plain predict() on the resized image, then the boxes are scaled back to
# full resolution (same as the SAM3 no-tiling notebook). No tiles, no filter.
# =============================================================================
def run_image_whole(runner: YoloeRunner, resized_image: Image.Image,
                           resize_scale: float, args) -> dict:
    """
    Run the (already prompted) YOLOE once on the resized whole image.

    Output: dict of numpy arrays in ORIGINAL-IMAGE coordinates:
            boxes (N,4), scores (N,)
            These are the PRE-NMS detections (no confidence filtering beyond 0.30,
            no offline NMS) that get written to disk for the offline evaluation.
    """
    boxes, scores = runner.predict_image(resized_image, args.threshold)

    # boxes came back in the resized image's coordinates -> rescale to full-res
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) > 0:
        boxes = boxes / resize_scale

    return {
        "boxes": boxes.reshape(-1, 4),
        "scores": np.asarray(scores, dtype=np.float32).reshape(-1),
    }

# =============================================================================
#  CELL 17 - OFFLINE NMS
# =============================================================================
# Even without tiling, the model can propose more than one overlapping box for
# the same plant in a single forward pass. NMS keeps the highest-scoring box of
# each overlapping group. YOLOE's own NMS was left permissive (0.90) precisely
# so that this stage is what actually decides. Same code as the SAM3 no-tiling
# notebook: no tile provenance, since there are no tiles.
# Used twice: on the round-1 detections when picking the self-prompts (PHASE 1)
# and on the final + round-1 detections during the evaluation (PHASE 2).
# =============================================================================
def nms(boxes, scores, iou_threshold: float):
    """
    Input : boxes (N,4), scores (N,), iou_threshold
    Output: keep - list of kept indices, highest score first.
    A detection is suppressed when its IoU with an already kept, higher-scoring
    detection is GREATER than the threshold.
    """
    n = len(boxes)
    if n == 0:
        return []
    order = list(np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable"))
    keep = []
    while order:
        i = int(order[0])
        keep.append(i)
        rest = np.array(order[1:], dtype=int)
        if rest.size == 0:
            break
        ious = compute_iou_matrix(boxes[i:i + 1], boxes[rest])[0]
        order = list(rest[ious <= iou_threshold])
    return keep

def apply_nms_to_run(run: dict, iou_threshold: float) -> dict:
    """
    Apply NMS to one run's pre-NMS detections.

    Output: dict with 'boxes', 'scores' of the surviving detections SORTED BY
            SCORE (high -> low), plus 'n_pre_nms'. Sorting by score means that
            applying an operating confidence threshold later is just a prefix
            selection.
    """
    keep = np.array(nms(run["boxes"], run["scores"], iou_threshold), dtype=int)
    return {
        "boxes": run["boxes"][keep].reshape(-1, 4),
        "scores": run["scores"][keep].reshape(-1),
        "n_pre_nms": int(len(run["scores"])),
    }

# =============================================================================
#  CELL 18 - SELF-PROMPT SELECTION (round 1 -> round 2)
# =============================================================================
#   round-1 detections
#     -> NMS (NMS_IOU_THRESHOLD) so near-duplicates cannot both be picked
#     -> keep score >= CONFIDENCE_THRESHOLD
#     -> drop every box that is just a prompt plant again: IoU >=
#        SELF_PROMPT_EXCLUDE_IOU (0.50) with ANY of the S/M/L prompt boxes
#     -> keep the N_SELF_PROMPTS (2) highest-confidence remaining boxes
# These boxes are YOLOE's own predictions, NOT ground truth: a false positive
# here is fed back as a positive example, which is why PHASE 2 also reports the
# round-1 metrics and how many self-prompts actually sit on a GT plant.
# =============================================================================
def select_self_prompts(nms_run: dict, prompt_boxes, n_self: int, exclude_iou: float,
                        confidence_threshold: float) -> dict:
    """
    Input : nms_run      - output of apply_nms_to_run on the ROUND-1 detections
                           (boxes/scores sorted by score, high -> low)
            prompt_boxes - the S/M/L boxes used in round 1 (full-res coords)
    Output: dict with
              'boxes'  (n,4) the selected self-prompt boxes (full-res coords)
              'scores' (n,)  their confidences
              'n_eligible'   how many post-NMS boxes were eligible at all
              'n_on_prompt'  how many were dropped for sitting on a prompt plant
    """
    boxes = np.asarray(nms_run["boxes"], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(nms_run["scores"], dtype=np.float32).reshape(-1)

    keep = scores >= confidence_threshold
    boxes, scores = boxes[keep], scores[keep]

    prompt_boxes = np.asarray(prompt_boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) and len(prompt_boxes):
        best_prompt_iou = compute_iou_matrix(boxes, prompt_boxes).max(axis=1)
        on_prompt = best_prompt_iou >= exclude_iou
    else:
        on_prompt = np.zeros(len(boxes), dtype=bool)

    eligible_boxes, eligible_scores = boxes[~on_prompt], scores[~on_prompt]
    order = np.argsort(-eligible_scores, kind="stable")[:n_self]   # highest confidence first

    return {
        "boxes": eligible_boxes[order].reshape(-1, 4),
        "scores": eligible_scores[order].reshape(-1),
        "n_eligible": int(len(eligible_scores)),
        "n_on_prompt": int(on_prompt.sum()),
    }

# =============================================================================
#  PRE-NMS DETECTION STORAGE (both rounds)
# =============================================================================
# For every image we store
#   - the ROUND-1 detections (before NMS),
#   - the self-prompt boxes and their confidences,
#   - the FINAL (round-2) detections; when round 2 was skipped because no eligible
#     self-prompt existed, the final arrays are the round-1 ones and second_run
#     is False,
# all in original-image coordinates, at CONFIDENCE_THRESHOLD, before NMS. PHASE 2
# therefore never needs YOLOE again. Masks are never stored.
# =============================================================================
def run_npz_path(raw_detections_dir: Path, image_id: str) -> Path:
    """Path of the NPZ holding the detections of one image."""
    return raw_detections_dir / f"{safe_filename(image_id)}.npz"

def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        r1: dict, self_prompts: dict, final: dict, second_run: bool,
                        gt_boxes: np.ndarray, selection: dict,
                        image_size: Tuple[int, int], archive: str, flight: str,
                        class_id: int, resize_scale: float) -> Path:
    """
    Write one image's two-round result to NPZ. The file is self-contained: it also
    stores the GT boxes and the S/M/L prompt indices, so the whole offline
    evaluation can run without re-opening images or label files.
    """
    path = run_npz_path(raw_detections_dir, image_id)
    np.savez_compressed(
        path,
        experiment_name=np.array(experiment_name),
        image_ID=np.array(image_id),
        prompt_indices=np.array(selection["indices"], dtype=np.int32),   # order: S, M, L
        prompt_roles=np.array(selection["roles"]),                       # ["S","M","L"]
        prompt_areas=np.asarray(selection["areas"], dtype=np.float32),
        median_gt_area=np.array(float(selection["median_area"]), dtype=np.float32),
        image_width=np.array(int(image_size[0])),
        image_height=np.array(int(image_size[1])),
        resize_scale=np.array(float(resize_scale)),
        archive=np.array(archive),
        flight=np.array(flight),
        source_class_id=np.array(int(class_id)),
        gt_boxes=gt_boxes.astype(np.float32),
        # ---- round 1 -----------------------------------------------------------
        boxes_round1=r1["boxes"].astype(np.float32),
        scores_round1=r1["scores"].astype(np.float32),
        # ---- self-prompts taken from round 1 ------------------------------------
        self_prompt_boxes=np.asarray(self_prompts["boxes"], dtype=np.float32).reshape(-1, 4),
        self_prompt_scores=np.asarray(self_prompts["scores"], dtype=np.float32).reshape(-1),
        second_run=np.array(bool(second_run)),
        # ---- final (round 2, or round 1 when round 2 was skipped) ---------------
        boxes=final["boxes"].astype(np.float32),              # x1,y1,x2,y2 (original img)
        scores=final["scores"].astype(np.float32),            # confidence >= 0.30
    )
    return path

def load_run_detections(path: Path) -> dict:
    """Read one image NPZ back into a plain python dict (final + 'round1')."""
    with np.load(path, allow_pickle=False) as z:
        run = {
            "image_ID": str(z["image_ID"]),
            "prompt_indices": z["prompt_indices"].astype(int),
            "prompt_roles": [str(r) for r in z["prompt_roles"]],
            "prompt_areas": z["prompt_areas"].reshape(-1),
            "median_gt_area": float(z["median_gt_area"]),
            "image_width": int(z["image_width"]),
            "image_height": int(z["image_height"]),
            "gt_boxes": z["gt_boxes"].reshape(-1, 4),
            "self_prompt_boxes": z["self_prompt_boxes"].reshape(-1, 4),
            "self_prompt_scores": z["self_prompt_scores"].reshape(-1),
            "second_run": bool(z["second_run"]),
            "boxes": z["boxes"].reshape(-1, 4),
            "scores": z["scores"].reshape(-1),
            "round1": {
                "boxes": z["boxes_round1"].reshape(-1, 4),
                "scores": z["scores_round1"].reshape(-1),
            },
        }
        run["archive"] = str(z["archive"]) if "archive" in z else ""
        run["flight"] = str(z["flight"]) if "flight" in z else ""
    return run

def round1_as_run(run: dict) -> dict:
    """The stored ROUND-1 detections in the same shape as a final run (for NMS / eval)."""
    return {**run, **run["round1"]}

# =============================================================================
#  RESUME SUPPORT  (adapted to several shard manifests)
# =============================================================================
def load_done_images(paths: Paths, experiment_name: str, prompt_type: str) -> set:
    """
    Read every shard manifest and return the image_IDs that are already finished
    (one run per image). Every shard reads ALL manifests, so a resubmission after
    the walltime never repeats work, even if the shard assignment changed because
    --num-gpus was different. Manifests that hold another prompt setting stop the
    run, so results are never mixed (same guard as the SAM3 notebook).
    """
    done: set = set()
    for csv_path in sorted(paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    if row.get("Prompt_Type") and row["Prompt_Type"] != prompt_type:
                        raise RuntimeError(
                            f"{csv_path} holds results for another prompt setting "
                            f"({row['Prompt_Type']}). Use a new EXPERIMENT_NAME / OUTPUT_DIR.")
                    if row.get("image_ID"):
                        done.add(row["image_ID"])
        except OSError:
            continue
    return done

# =============================================================================
#  MAIN GPU INFERENCE LOOP  (PHASE 1, two YOLOE rounds per image)
# =============================================================================
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE, read its GT boxes ONCE
#     pick the S / M / L exemplars by area (CELL 5) - deterministic, ONE run
#     resize the whole image to MAX_DIM ONCE (CELL 9)
#     ROUND 1 : VPE from the S/M/L boxes on the resized image -> YOLOE on it
#     NMS on the round-1 detections -> pick the self-prompts (CELL 18)
#     ROUND 2 : VPE from S/M/L + the self-prompts on the same image -> YOLOE again
#               (skipped when no eligible box exists -> the round-1 output is final)
#     save round 1, the self-prompts and the final detections (NPZ)
#     release the image
#
# Images without any Rumex GT box are skipped. NO metric computation happens
# here; the loop is resumable through the shard manifests.
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

    runner = YoloeRunner(args)
    device_is_cuda = runner.device.startswith("cuda")

    start_time = time.time()
    n_new_runs = 0
    n_without_second_run = 0
    image_times: List[float] = []
    n_total_images = len(records)

    try:
        for img_idx, rec in enumerate(records, start=1):
            image_id = rec.image_id
            if image_id in done_images:
                print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id}: already done, skipped.")
                continue
            image_t0 = time.time()

            # ---------------- open the original image exactly once ------------
            image = Image.open(rec.image_path).convert("RGB")
            img_w, img_h = image.size
            gt_boxes = load_yolo_boxes(rec.label_path, img_w, img_h, rec.class_id)
            n_gt = len(gt_boxes)

            if n_gt == 0:
                print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id}: 0 GT boxes, skipped.")
                image.close(); del image; gc.collect()
                continue

            # ---------------- size-based prompt selection (CELL 5) ------------
            selection = select_size_exemplars(gt_boxes)
            exemplar_indices = selection["indices"]
            prompt_boxes_sml = np.asarray([gt_boxes[i] for i in exemplar_indices], dtype=np.float32)
            prompt_id = format_size_prompt_id(selection)

            # ---------------- resize the whole image exactly once -------------
            resized_image, resize_scale = resize_for_model(image, args.max_dim)
            res_w, res_h = resized_image.size

            # ---------------- ROUND 1: S / M / L ------------------------------
            runner.install_prompt(resized_image, prompt_boxes_sml * resize_scale)
            r1 = run_image_whole(runner, resized_image, resize_scale, args)

            # ---------------- self-prompts from round 1 (CELL 18) -------------
            r1_nms = apply_nms_to_run(r1, args.nms_iou_threshold)
            self_prompts = select_self_prompts(r1_nms, prompt_boxes_sml, args.n_self_prompts,
                                               args.self_prompt_exclude_iou,
                                               args.operating_confidence)

            # ---------------- ROUND 2: S / M / L + self-prompts ---------------
            if len(self_prompts["boxes"]):
                round2_boxes = np.vstack([prompt_boxes_sml, self_prompts["boxes"]]).astype(np.float32)
                runner.install_prompt(resized_image, round2_boxes * resize_scale)
                final = run_image_whole(runner, resized_image, resize_scale, args)
                second_run = True
            else:
                # no eligible box -> round 2 would repeat round 1 exactly
                final, second_run = r1, False
                n_without_second_run += 1

            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, r1, self_prompts, final, second_run,
                gt_boxes, selection, (img_w, img_h), rec.archive, rec.flight, rec.class_id,
                resize_scale)

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
                "area_S_px2": round(float(areas_by_role["S"])),
                "area_M_px2": round(float(areas_by_role["M"])) if "M" in areas_by_role else "",
                "area_L_px2": round(float(areas_by_role["L"])) if "L" in areas_by_role else "",
                "median_gt_area_px2": round(float(selection["median_area"])),
                "n_detections_r1_pre_nms": int(len(r1["scores"])),
                "n_detections_r1_post_nms": int(len(r1_nms["scores"])),
                "n_self_prompts": int(len(self_prompts["boxes"])),
                "self_prompt_scores": "+".join(f"{s:.3f}" for s in self_prompts["scores"]),
                "second_run": second_run,
                "n_detections_pre_nms": int(len(final["scores"])),
                "image_width": img_w,
                "image_height": img_h,
                "resized_width": res_w,
                "resized_height": res_h,
                "resize_scale": round(float(resize_scale), 6),
                "npz_file": npz_path.name,
                "inference_seconds": round(run_seconds, 2),
            })
            manifest_file.flush()                  # this image is on disk -> resumable
            n_new_runs += 1

            if len(exemplar_indices) < args.k_exemplars:
                print(f"    NOTE: only {n_gt} GT box(es) -> {len(exemplar_indices)} GT prompt(s) "
                      f"({'+'.join(selection['roles'])}).")
            if not second_run:
                print(f"    NOTE: no eligible round-1 box ({self_prompts['n_eligible']} eligible, "
                      f"{self_prompts['n_on_prompt']} on a prompt plant) -> round 2 skipped, "
                      f"round-1 output kept as final.")
            n_r1, n_final = int(len(r1["scores"])), int(len(final["scores"]))
            sp_txt = ", ".join(f"{s:.2f}" for s in self_prompts["scores"]) or "-"

            # ---------------- release the image -------------------------------
            if resized_image is not image:
                resized_image.close()
            image.close()
            del image, resized_image, r1, r1_nms, final
            gc.collect()
            if device_is_cuda:
                torch.cuda.empty_cache()

            image_times.append(time.time() - image_t0)
            avg_per_image = float(np.mean(image_times))
            eta = (n_total_images - img_idx) * avg_per_image
            print(f"  [{exp}] shard{args.shard_index} ({img_idx}/{n_total_images}) {image_id} | "
                  f"{n_gt} GT box(es) | prompts={prompt_id} | input={res_w}x{res_h} | "
                  f"round1={n_r1} det -> self-prompts={int(len(self_prompts['boxes']))} ({sp_txt}) "
                  f"-> final={n_final} det | {run_seconds:.1f}s | avg/image={avg_per_image:.1f}s | "
                  f"ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")
    finally:
        manifest_file.close()                      # also closed if the loop crashes

    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new images "
          f"({n_without_second_run} without a second round).")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Detections in: {paths.raw_detections}")

# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================
def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading YOLOE ---")
    sample = my_records[:min(len(my_records), 200)]
    n_with_gt, n_short, per_archive = 0, 0, {}
    for rec in sample:
        with Image.open(rec.image_path) as im:
            w, h = im.size
        n_gt = len(load_yolo_boxes(rec.label_path, w, h, rec.class_id))
        if n_gt == 0:
            continue
        n_with_gt += 1
        n_short += int(n_gt < args.k_exemplars)
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + 1

    scale = min(1.0, args.max_dim / 8192)
    print(f"  sampled {len(sample)} image(s) of this shard -> {n_with_gt} run(s) "
          f"(one per image with GT; {per_archive})")
    print(f"  images with fewer than {args.k_exemplars} GT boxes (fewer prompts): {n_short}")
    print(f"  no tiling: an 8192x5460 image is resized to "
          f"{int(8192 * scale)}x{int(5460 * scale)} (scale {scale:.3f})")
    print(f"  => up to ~{2 * n_with_gt} YOLOE forward passes (+ up to {2 * n_with_gt} VPE "
          f"encodings) for those {len(sample)} images - two rounds per image; round 2 is "
          f"skipped when no self-prompt is eligible")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{n_with_gt} (one per image)")
    weights = Path(str(args.weights))
    if weights.is_absolute() or weights.parent != Path("."):
        print(f"  weights file: {weights} -> {'FOUND' if weights.is_file() else 'MISSING'}")

# =============================================================================
#  CELL 20 (start) - LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2, both rounds)
# =============================================================================
# From here on YOLOE is never touched again. Everything below works on the NPZ
# files written in PHASE 1, so the complete evaluation can be redone in minutes
# on a login node or in a small CPU allocation.
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
    print("Images with fewer than 3 GT prompts:",
          int(sum(1 for r in runs if len(r["prompt_indices"]) < 3)))
    print("Images with a second round:",
          int(sum(1 for r in runs if r["second_run"])), "/", len(runs))
    n_sp = pd.Series([len(r["self_prompt_boxes"]) for r in runs]).value_counts().sort_index()
    print("Self-prompts per image:", {int(k): int(v) for k, v in n_sp.items()})
    return runs, manifest

# =============================================================================
#  CELL 18 - EVALUATION CORE: all_gt AND held_out
# =============================================================================
# held_out matters even more here: the exemplars are encoded from the SAME image
# that is searched. It is still the right protocol even though the exemplars come from a
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
#  CELL 19 - AP50 AND AP50:95  (supervision.metrics.MeanAveragePrecision)
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
#  CELL 23 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right)
# =============================================================================
# IMAGE SELECTION (per-image AP50, mode = PLOT_EVALUATION_MODE):
#   1. keep only images that have at least PLOT_MIN_GT_BOXES (7) GT boxes
#      and take the one with the highest AP50
#   2. if NO image has 7 GT boxes: keep the images with the HIGHEST number of GT
#      boxes and take the one among them with the highest AP50
#   ties are broken by F1, then by the number of GT boxes.
#
# Right panel = the FINAL (round-2) post-NMS predictions of that image with
# score >= CONFIDENCE_THRESHOLD, plus its S / M / L prompts (dashed, role colour)
# and the self-prompts fed back into round 2 (dashed white).
# One run per image, so there is no "best anchor" to choose.
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
    Output: (per-image row, selection_rule_text) or (None, reason)
    'archive' restricts the candidates to one archive; None = the whole dataset.
    """
    img_df = image_level_df[(image_level_df["evaluation_mode"] == mode) &
                            image_level_df["AP50"].notna()].copy()
    if archive is not None:
        img_df = img_df[img_df["archive"] == archive]
    if img_df.empty:
        return None, f"no image with a valid AP50 (archive={archive})"

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

    ranked = candidates.sort_values(["AP50", "F1", "n_gt"], ascending=False, na_position="last")
    print("Top candidates:")
    print(ranked[["image_ID", "n_gt", "Prompt_ID", "n_predictions", "AP50", "F1",
                  "precision", "recall"]].head(5).to_string(index=False))
    return ranked.iloc[0], rule

def plot_gt_vs_predictions(args, paths: Paths, runs_by_image: dict, image_level_df,
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

    gt_d = gt_boxes_plot * to_disp
    _draw_boxes(axes[0], gt_d, color="yellow", linewidth=1.5)
    axes[0].set_title(f"Ground truth: {len(gt_boxes_plot)} Rumex boxes", fontsize=12)

    score_labels = [f"{v:.2f}" for v in pred_scores] if args.plot_show_scores else None
    _draw_boxes(axes[1], pred_boxes * to_disp, color="red", linewidth=1.5,
                labels=score_labels)
    for role, idx in zip(run["prompt_roles"], run["prompt_indices"]):
        _draw_boxes(axes[1], gt_d[[int(idx)]], color=ROLE_COLOURS[role], linewidth=2.5,
                    linestyle="--")
    if len(run["self_prompt_boxes"]):
        _draw_boxes(axes[1], run["self_prompt_boxes"] * to_disp, color="white", linewidth=2.5,
                    linestyle="--")
    axes[1].set_title(
        f"Final (round {2 if run['second_run'] else 1}): {len(pred_boxes)} boxes | "
        f"prompts {run['Prompt_ID']} + {len(run['self_prompt_boxes'])} self\n"
        f"AP50={row['AP50']:.3f}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
        f"F1={row['F1']:.3f}  TP={int(row['TP'])} FP={int(row['FP'])} FN={int(row['FN'])}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color="yellow", lw=2, label="ground truth"),
        Line2D([0], [0], color="red", lw=2, label="prediction"),
    ] + [Line2D([0], [0], color=ROLE_COLOURS[r], lw=2.5, linestyle="--",
                label=f"{r} prompt ({ROLE_NAMES[r]})") for r in run["prompt_roles"]] + (
        [Line2D([0], [0], color="white", lw=2.5, linestyle="--",
                label=f"self-prompts ({len(run['self_prompt_boxes'])})")]
        if len(run["self_prompt_boxes"]) else [])
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=11, frameon=False)
    fig.suptitle(
        f"{exp} | {plot_image_id} | mode={mode} | scope={scope_label}\n"
        f"YOLOE {args.model_name} @ imgsz={args.imgsz} | round 1: size-based prompts "
        f"{run['Prompt_ID']}, round 2: + {len(run['self_prompt_boxes'])} self-prompts "
        f"({', '.join(f'{s:.2f}' for s in run['self_prompt_scores']) or '-'}) | "
        f"{PROMPT_MODE} | no tiling, MAX_DIM={args.max_dim}px | "
        f"conf={args.operating_confidence:.2f}, NMS IoU={args.nms_iou_threshold:.2f}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])

    png_path = paths.plots / f"best_image_{scope_label}_{safe_filename(plot_image_id)}_{mode}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

    print(f"\n[{scope_label}] Selected image : {plot_image_id} ({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Prompts        : {run['Prompt_ID']} + "
          f"{len(run['self_prompt_boxes'])} self-prompt(s), round 2 run: {run['second_run']}")
    print(f"[{scope_label}] Figure saved   : {png_path}")

def make_qualitative_plots(args, paths: Paths, runs, image_level_df, plt) -> None:
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

    runs_by_image = {r["image_ID"]: r for r in runs}
    archives_present = sorted(str(a) for a in image_level_df["archive"].dropna().unique() if str(a))
    for archive in archives_present:
        plot_gt_vs_predictions(args, paths, runs_by_image, image_level_df, image_paths, plt,
                               scope_label=archive, archive=archive)
    plot_gt_vs_predictions(args, paths, runs_by_image, image_level_df, image_paths, plt,
                           scope_label="ALL", archive=None)

# =============================================================================
#  PHASE 2 - OFFLINE EVALUATION  (notebook CELL 20 ... CELL 26)
# =============================================================================
def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2 over the whole dataset, one process, no GPU."""
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

    runs, manifest = load_runs(paths, exp, args.prompt_type)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  CELL 22 - PER-IMAGE METRICS  (final = round 2, with round 1 for comparison)
    # =========================================================================
    # The evaluated predictions are the FINAL ones (round 2, or round 1 when
    # round 2 was skipped). The same metrics are also computed for the ROUND-1
    # output and written into *_round1 columns (delta_AP50 = AP50 - AP50_round1).
    # n_self_prompts_on_gt counts the self-prompts that sit on a real plant
    # (IoU >= EVAL_IOU_THRESHOLD with any GT box) - information only.
    # The S/M/L choice is deterministic, so every image is evaluated exactly once
    # and this single table replaces the run-level + image-level tables of the
    # anchor experiments.
    #   AP50 / AP50_95 : all post-NMS predictions >= 0.30, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : predictions >= CONFIDENCE_THRESHOLD
    # held_out with no evaluable GT (every plant was a prompt): the metrics are
    # NaN and valid_for_macro is False, but the real FP count is kept.
    # =========================================================================
    print("\n--- per-image metrics ---")
    image_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)
        nms_r1 = apply_nms_to_run(round1_as_run(run), nms_iou)
        gt = run["gt_boxes"]
        sp = run["self_prompt_boxes"]
        n_sp_on_gt = (int((compute_iou_matrix(sp, gt).max(axis=1) >= eval_iou).sum())
                      if len(sp) and len(gt) else 0)
        for mode in EVALUATION_MODES:
            eval_gt, prompt_gt = split_gt_for_mode(gt, run["prompt_indices"], mode)

            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            ap50, ap5095 = compute_ap([p_det], [g_det])

            ev = evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, conf,
                                             eval_iou, ignore_iou)
            valid = ev["valid_for_macro"]
            nan = float("nan")

            # ---- round 1, same metrics, for comparison -----------------------
            p1_det, g1_det = ap_inputs_for_run(nms_r1, eval_gt, prompt_gt, eval_iou, ignore_iou)
            ap50_r1, ap5095_r1 = compute_ap([p1_det], [g1_det])
            ev1 = evaluate_at_operating_point(nms_r1, eval_gt, prompt_gt, conf,
                                              eval_iou, ignore_iou)

            image_rows.append({
                "experiment_name": exp,
                "model": args.model_name,
                "image_ID": run["image_ID"],
                "archive": run.get("archive", ""),
                "flight": run.get("flight", ""),
                "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"],
                "evaluation_mode": mode,
                "confidence_threshold": conf,
                "nms_iou_threshold": nms_iou,
                "n_gt": int(len(gt)),
                "n_prompt_gt": int(len(run["prompt_indices"])),
                "median_gt_area_px2": round(run["median_gt_area"]),
                "n_self_prompts": int(len(sp)),
                "self_prompt_scores": "+".join(f"{s:.3f}" for s in run["self_prompt_scores"]),
                "n_self_prompts_on_gt": n_sp_on_gt,
                "second_run": run["second_run"],
                "n_eval_gt": ev["n_eval_gt"],
                "n_predictions_pre_nms": nms_run["n_pre_nms"],
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
                # ---- round 1, same metrics, for comparison ---------------------
                "AP50_round1": ap50_r1 if valid else nan,
                "AP50_95_round1": ap5095_r1 if valid else nan,
                "precision_round1": ev1["precision"] if valid else nan,
                "recall_round1": ev1["recall"] if valid else nan,
                "F1_round1": ev1["F1"] if valid else nan,
                "IoU1_round1": ev1["IoU1"] if valid else nan,
                "IoU2_round1": ev1["IoU2"] if valid else nan,
                "TP_round1": ev1["TP"], "FP_round1": ev1["FP"], "FN_round1": ev1["FN"],
                "n_predictions_round1": ev1["n_pred"],
                "delta_AP50": (ap50 - ap50_r1) if valid else nan,
                "delta_F1": (ev["F1"] - ev1["F1"]) if valid else nan,
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
              f"averaging, AP50_mean={sub['AP50'].mean():.4f}, F1_mean={sub['F1'].mean():.4f}")
        better = int((sub["delta_AP50"] > 0).sum())
        worse = int((sub["delta_AP50"] < 0).sum())
        same = int((sub["delta_AP50"] == 0).sum())
        print(f"             AP50 round1={sub['AP50_round1'].mean():.4f} -> final="
              f"{sub['AP50'].mean():.4f} (mean delta {sub['delta_AP50'].mean():+.4f}) | "
              f"images better/worse/unchanged: {better}/{worse}/{same}")

    # =========================================================================
    #  CELL 22 - EXPERIMENT-LEVEL SUMMARY
    # =========================================================================
    # Mean over images and std BETWEEN images of every per-image metric, so every
    # UAV image contributes exactly the same weight regardless of how many GT
    # boxes it contains.
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
            "n_exemplars": args.k_exemplars,
            "n_self_prompts": args.n_self_prompts,
            "self_prompt_exclude_iou": args.self_prompt_exclude_iou,
            "n_images_with_second_round": int(sub["second_run"].sum()),
            "self_prompts_on_gt_share": (sub["n_self_prompts_on_gt"].sum() /
                                         max(1, sub["n_self_prompts"].sum())),
            "size_measure": SIZE_MEASURE,
            "prompt_mode": PROMPT_MODE,
            "use_tiling": args.use_tiling,
            "max_dim": args.max_dim,
            "imgsz": args.imgsz,
            "confidence_threshold": conf,
            "nms_iou_threshold": nms_iou,
            "eval_iou_threshold": eval_iou,
            "n_images": int(sub["image_ID"].nunique()),
            "n_images_valid_for_macro": int(sub["valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std()      # spread between images
            row[f"{col}_round1_mean"] = sub[f"{col}_round1"].mean()
        row["delta_AP50_mean"] = sub["delta_AP50"].mean()
        row["delta_F1_mean"] = sub["delta_F1"].mean()
        summary_rows.append(row)

    experiment_summary_df = pd.DataFrame(summary_rows)
    experiment_summary_csv = paths.metrics / "experiment_summary.csv"
    experiment_summary_df.to_csv(experiment_summary_csv, index=False)
    print(f"Experiment summary -> {experiment_summary_csv}\n")
    print(experiment_summary_df.to_string(index=False))

    # ---- per-archive version of the same table (cluster addition) -----------
    per_archive_rows = []
    for (archive, mode), sub_df in image_level_df.groupby(["archive", "evaluation_mode"]):
        row = {
            "experiment_name": exp,
            "archive": archive,
            "evaluation_mode": mode,
            "n_images": int(sub_df["image_ID"].nunique()),
            "n_images_valid_for_macro": int(sub_df["valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub_df[col].mean()
            row[f"{col}_std"] = sub_df[col].std()
            row[f"{col}_round1_mean"] = sub_df[f"{col}_round1"].mean()
        row["delta_AP50_mean"] = sub_df["delta_AP50"].mean()
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
    #  CELL 22 - POOLED DATASET AP50 / AP50:95
    # =========================================================================
    # This is NOT the mean of the per-image AP values. All images are handed to
    # supervision at once, so every detection of the whole dataset is ranked in
    # ONE precision-recall curve. Here one episode = one image.
    # =========================================================================
    print("\n--- pooled dataset AP ---")
    dataset_rows = []
    for mode in EVALUATION_MODES:
        pred_list, gt_list, images_used = [], [], set()
        pred_list_r1, gt_list_r1 = [], []
        for run in runs:
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)
            for det, pl, gl in [(run, pred_list, gt_list),
                                (round1_as_run(run), pred_list_r1, gt_list_r1)]:
                nms_run = apply_nms_to_run(det, nms_iou)
                p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
                pl.append(p_det)
                gl.append(g_det)
            images_used.add(run["image_ID"])

        ap50, ap5095 = compute_ap(pred_list, gt_list)          # one PR curve over ALL images
        ap50_r1, ap5095_r1 = compute_ap(pred_list_r1, gt_list_r1)
        dataset_rows.append({
            "experiment_name": exp,
            "evaluation_mode": mode,
            "prompt_type": args.prompt_type,
            "n_images": len(images_used),
            "confidence_used_for_AP": args.threshold,     # AP always uses >= 0.30
            "nms_iou_threshold": nms_iou,
            "dataset_AP50": ap50,
            "dataset_AP50_95": ap5095,
            "dataset_AP50_round1": ap50_r1,
            "dataset_AP50_95_round1": ap5095_r1,
        })
        del pred_list, gt_list, pred_list_r1, gt_list_r1
        gc.collect()

    dataset_ap_df = pd.DataFrame(dataset_rows)
    dataset_ap_csv = paths.metrics / "dataset_ap_metrics.csv"
    dataset_ap_df.to_csv(dataset_ap_csv, index=False)
    print(f"Dataset pooled AP -> {dataset_ap_csv}\n")
    print(dataset_ap_df.to_string(index=False))
    print("\nFor comparison, the MEAN of the per-image AP50 values (a different quantity):")
    print(image_level_df.groupby("evaluation_mode")["AP50"].mean().to_string())

    # =========================================================================
    #  CELL 24 - DATASET-LEVEL CONFUSION MATRICES
    # =========================================================================
    #     Actual Rumex      -> Predicted Rumex      = TP
    #     Actual Rumex      -> Predicted Background = FN  (missed plants)
    #     Actual Background -> Predicted Rumex      = FP  (spurious detections)
    #     Actual Background -> Predicted Background = not defined for detection
    # Counts are pooled over every image at the fixed configuration. In held_out
    # mode the S/M/L prompt plants and the ignored detections do not appear.
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
                f"{exp} - {mode}\nS/M/L + self-prompts (final) | conf={conf:.2f}, "
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
    #  CELL 25 - RECALL PER SIZE GROUP (does the S/M/L mix help across plant sizes?)
    # =========================================================================
    # For every image the GT boxes are split at that image's MEDIAN area:
    #   "area <= median" (the smaller half) and "area > median" (the larger half).
    # Matching uses all_gt at the fixed operating point (same matcher as
    # everywhere). The prompt plants themselves are reported separately, because
    # the model usually re-detects them and they would flatter the recall of
    # their size group.
    # =========================================================================
    print("\n--- recall per size group ---")
    size_rows, prompt_rows = [], []
    for run in runs:
        gt = run["gt_boxes"]
        nms_run = apply_nms_to_run(run, nms_iou)
        keep = nms_run["scores"] >= conf
        match = match_one_to_one(nms_run["boxes"][keep], nms_run["scores"][keep], gt, eval_iou)
        found = match["gt_match_pred"] >= 0

        areas = box_areas(gt)
        is_prompt = np.isin(np.arange(len(gt)), run["prompt_indices"])
        small_half = areas <= run["median_gt_area"]
        for name, grp in [("area <= median", small_half), ("area >  median", ~small_half)]:
            mask = grp & ~is_prompt                      # non-prompt GT boxes only
            size_rows.append({
                "image_ID": run["image_ID"], "archive": run.get("archive", ""),
                "size_group": name,
                "n_gt_non_prompt": int(mask.sum()), "found": int((found & mask).sum()),
                "recall": float((found & mask).sum() / mask.sum()) if mask.sum() else float("nan"),
            })
        for role, idx in zip(run["prompt_roles"], run["prompt_indices"]):
            prompt_rows.append({"image_ID": run["image_ID"], "archive": run.get("archive", ""),
                                "role": role, "gt_index": int(idx),
                                "area_px2": float(areas[idx]),
                                "re_detected": bool(found[idx])})

    size_group_df = pd.DataFrame(size_rows)
    prompt_hits_df = pd.DataFrame(prompt_rows)
    size_group_csv = paths.metrics / "size_group_recall.csv"
    prompt_hits_csv = paths.metrics / "prompt_plants_redetected.csv"
    size_group_df.to_csv(size_group_csv, index=False)
    prompt_hits_df.to_csv(prompt_hits_csv, index=False)

    pooled = (size_group_df.groupby("size_group")[["n_gt_non_prompt", "found"]].sum()
              .assign(recall_pooled=lambda d: d["found"] / d["n_gt_non_prompt"]))
    per_image_mean = (size_group_df.groupby("size_group")["recall"].mean()
                      .rename("recall_mean_over_images"))
    size_group_summary = pooled.join(per_image_mean)
    print("Recall of NON-PROMPT GT boxes per size group (all_gt matching):")
    print(size_group_summary.to_string())
    print("\nPrompt plants re-detected (share per role):")
    print(prompt_hits_df.groupby("role")["re_detected"].agg(["count", "sum", "mean"]).to_string())
    print("\nSaved:", size_group_csv, "and", prompt_hits_csv)

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
    #  CELL 26 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY")
    print("=" * 78)
    print(f"Model                    : {args.model_name} @ imgsz={args.imgsz}")
    print(f"Round 1 prompts          : {args.k_exemplars} GT boxes by {SIZE_MEASURE} "
          f"(smallest / medium / largest), encoding={PROMPT_MODE}")
    print(f"Tiling                   : {args.use_tiling}  (whole image resized to "
          f"MAX_DIM={args.max_dim}px, imgsz={args.imgsz})")
    print(f"Plausibility filter      : none (as in the SAM3 no-tiling notebook)")
    print(f"Round 2 prompts          : S/M/L + up to {args.n_self_prompts} self-prompts "
          f"(IoU < {args.self_prompt_exclude_iou} to S/M/L), same resized image")
    print(f"Images with 2 rounds     : {int(sum(1 for r in runs if r['second_run']))} / {len(runs)}")
    n_sp_all = int(image_level_df[image_level_df['evaluation_mode'] == 'all_gt']['n_self_prompts'].sum())
    n_sp_gt = int(image_level_df[image_level_df['evaluation_mode'] == 'all_gt']['n_self_prompts_on_gt'].sum())
    print(f"Self-prompts on a GT plant: {n_sp_gt} / {n_sp_all} (the rest are false positives fed back)")
    print(f"YOLOE inference threshold: {args.threshold} (both rounds, one whole-image pass each)")
    print(f"In-predictor NMS IoU     : {args.predict_nms_iou} (permissive; offline NMS decides)")
    print(f"Operating point          : confidence={conf:.2f} (== the inference "
          f"threshold), NMS IoU={nms_iou:.2f} (both fixed)")
    print(f"Evaluation IoU           : {eval_iou:.2f}")
    print(f"Images                   : {len(runs)}")
    print(f"Archives                 : "
          f"{', '.join(sorted(str(a) for a in image_level_df['archive'].dropna().unique()))}")
    print("-" * 78)
    print("EXPERIMENT-LEVEL RESULTS (mean over images, std between images)")
    show = ["evaluation_mode", "AP50_mean", "AP50_std", "AP50_95_mean", "precision_mean",
            "recall_mean", "F1_mean", "F1_std", "IoU1_mean", "IoU2_mean"]
    print(experiment_summary_df[show].to_string(index=False))
    print("-" * 78)
    print("ROUND 1 vs FINAL (mean over images)")
    print(experiment_summary_df[["evaluation_mode", "AP50_round1_mean", "AP50_mean",
                                 "delta_AP50_mean", "F1_round1_mean", "F1_mean",
                                 "delta_F1_mean"]].to_string(index=False))
    print("-" * 78)
    print("POOLED DATASET AP")
    print(dataset_ap_df[["evaluation_mode", "dataset_AP50_round1", "dataset_AP50",
                         "dataset_AP50_95"]].to_string(index=False))
    print("-" * 78)
    print("POOLED CONFUSION COUNTS")
    print(confusion_summary_df[["evaluation_mode", "TP", "FP", "FN",
                                "precision_micro", "recall_micro",
                                "F1_micro"]].to_string(index=False))
    print("-" * 78)
    print("RECALL PER SIZE GROUP (non-prompt GT, all_gt)")
    print(size_group_summary.to_string())
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

    # ---------------- PHASE 2 only -------------------------------------------
    if args.evaluate_only:
        if not _check_supervision():
            print("\nERROR: " + SUPERVISION_HINT)
            sys.exit(2)
        if not _check_pandas():
            print("\nERROR: pandas is required for the evaluation. See "
                  "'Problem B' in high_confidence_pseudo_prompts_no_tiling_HOW_TO_RUN_yoloe.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" HIGH_CONFIDENCE_PSEUDO_PROMPTS_NO_TILING YOLOE PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" model          : {args.weights} @ imgsz={args.imgsz}")
    print(f" prompts        : {args.k_exemplars} GT boxes by {SIZE_MEASURE} (S / M / L), "
          f"{PROMPT_MODE} -> ONE run per image")
    print(f" round 2        : S/M/L + up to {args.n_self_prompts} self-prompts "
          f"(IoU < {args.self_prompt_exclude_iou} to S/M/L) -> prompt type {args.prompt_type}")
    print(f" tiling         : {args.use_tiling}  (whole image resized to MAX_DIM={args.max_dim})")
    print(f" filter         : none (as in the SAM3 no-tiling notebook)")
    print(f" yoloe threshold: {args.threshold}  (single inference pass per image)")
    print(f" in-pred. NMS   : {args.predict_nms_iou}  (permissive on purpose)")
    print(f" fp16           : {args.use_fp16}")
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
        config["size_measure"] = SIZE_MEASURE
        config["prompt_type"] = args.prompt_type
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
