#!/usr/bin/env python3
# =============================================================================
#  high_confidence_pseudo_prompts_no_tiling_infer_countGDpp.py
# =============================================================================
#
#  This is the HIGH-CONFIDENCE PSEUDO-PROMPTS NO-TILING experiment with CountGD++
#  (CVPR 2026), SAM3 notebook high_confidence_pseudo_prompts_no_tiling: TWO
#  CountGD++ rounds per image, USE_TILING = False.
#
#    ROUND 1       the 3 SIZE-BASED GT exemplars of the image, chosen by box AREA
#                  (width x height, full-resolution pixels) among ALL its GT boxes:
#                    S = the SMALLEST box
#                    M = the remaining box whose area is closest to the median
#                        area of all GT boxes of that image
#                    L = the LARGEST box
#                  (ties -> lower GT index; 1 GT box -> [S], 2 GT boxes -> [S, L]).
#                  Round 1 is exactly k_size_exemplars_no_tiling_countGDpp.
#    SELF-PROMPTS  the round-1 detections go through NMS (0.40), the ones that sit
#                  on a prompt plant (IoU >= SELF_PROMPT_EXCLUDE_IOU = 0.50 with any
#                  S/M/L box) are dropped, and the N_SELF_PROMPTS = 2 highest-
#                  confidence remaining boxes are kept. They are CountGD++'s OWN
#                  predictions, not ground truth.
#    ROUND 2       CountGD++ again on the same image with 3 + 2 = 5 exemplar boxes
#                  (S, M, L + the 2 self-prompts) in ONE forward pass.
#
#  The FINAL predictions are the round-2 output (round 1 is NOT merged in); they
#  are evaluated against the original GT. If round 1 leaves no eligible box, round 2
#  would repeat round 1 exactly, so it is skipped and the round-1 output is final
#  (second_run = False). Everything is deterministic: ONE run per image.
#
#  The WHOLE UAV image is resized ONCE with CountGD++'s own resize (short side
#  800, long side <= 1333: 8192 x 5460 -> 1200 x 800) and used by both rounds. The
#  exemplar boxes lie at their real locations INSIDE that same image: CountGD++'s
#  native way of prompting with several exemplars (exemplar image = the input
#  image, k boxes drawn on it -> k exemplar tokens, one prompt). No tiles, no crop,
#  no mosaic, no plausibility filter (as in the notebook). The predicted boxes are
#  scaled back to the original resolution and clipped to the image.
#
#  The pipeline is TWO-PHASE:
#
#    PHASE 1 - INFERENCE   (notebook CELL 10 ... CELL 15, GPU)
#        for every image (ONE run per image):
#            pick the S / M / L exemplars by area            (CELL 10)
#            resize the whole image ONCE (1200 x 800)
#            ROUND 1: CountGD++ with the 3 GT boxes, 0.30     (CELL 11)
#            NMS on the round-1 detections -> the 2 self-
#              prompts                                       (CELL 12 + 13)
#            ROUND 2: CountGD++ with the 3 GT boxes + the
#              2 self-prompts, 0.30 (skipped if none)        (CELL 11)
#            boxes back to original pixels + clip to the image
#            save round 1, the self-prompts and the final
#              PRE-NMS detections to NPZ                      (CELL 14)
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 16 ... CELL 27, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40,
#        evaluate the FINAL detections in both modes (all_gt / held_out) at
#        CONFIDENCE_THRESHOLD = 0.30 - and the round-1 detections the same way, for
#        comparison (*_round1 columns, delta_AP50 / delta_F1) - and write the
#        per-image / experiment-level / pooled-AP CSVs, the size-group recall
#        (CELL 25), the confusion matrices (CSV + PNG), the qualitative
#        GT-vs-prediction figures (CELL 27) and the exemplar preview.
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
    "experiment_name", "image_ID", "Prompt_ID", "Prompt_Type",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt", "area_S_px2", "area_M_px2", "area_L_px2", "median_gt_area_px2",
    "n_detections_r1_pre_nms", "n_detections_r1_post_nms",
    "n_self_prompts", "self_prompt_scores", "second_run",
    "n_detections_pre_nms", "image_width", "image_height", "input_width", "input_height",
    "npz_file", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
# PROMPT_TYPE = f"size_SML+{N_SELF_PROMPTS}self" is built in parse_args.
SIZE_MEASURE = "area"         # box size = width x height (full-resolution pixels)
K_EXEMPLARS = 3               # S + M + L (GT boxes, round 1 and round 2)
N_SELF_PROMPTS = 2            # highest-confidence round-1 predictions added in round 2
SELF_PROMPT_EXCLUDE_IOU = 0.50  # a round-1 prediction with IoU >= this with ANY S/M/L
                                # prompt box is NOT eligible (it is the prompt plant again)

EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated - this is the main number
#            ("final predictions vs ALL original human GT annotations").
# held_out : the 3 GT instances used as S/M/L prompts are IGNORED, and so are the
#            predictions that fall on them. The self-prompts are NOT GT boxes, so
#            they never change the GT set.

# ---- notebook CELL 21 ------------------------------------------------------
# count_abs_error = |predicted count - GT count| at the operating point (the
# CountGD++ notebook adds it because CountGD++ is a counting model).
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2",
                  "count_abs_error"]

# ---- notebook CELL 17 ------------------------------------------------------
STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2

# ---- notebook CELL 12 ------------------------------------------------------
DOT_TOKEN_ID = 1012   # BERT id of "." - separates the positive prompt from negatives

# ---- notebook CELL 4 -------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # the repo's own normalisation
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './high_confidence_pseudo_prompts_no_tiling_run_countGDpp.sh download' on a "
    "LOGIN node, then './high_confidence_pseudo_prompts_no_tiling_run_countGDpp.sh build' inside the "
    "container, which installs it into $COUNTGD_PYEXTRA."
)

COUNTGD_HINT = (
    "Run './high_confidence_pseudo_prompts_no_tiling_run_countGDpp.sh download' on a LOGIN node (repo + "
    "checkpoint + BERT), then './high_confidence_pseudo_prompts_no_tiling_run_countGDpp.sh build' inside the "
    "container (packages + BERT folder + CUDA op). See high_confidence_pseudo_prompts_no_tiling_HOW_TO_RUN_countGDpp.md."
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
        description="high_confidence_pseudo_prompts_no_tiling_countGDpp - CountGD++ Rumex detection "
                    "in TWO rounds, NO tiling: round 1 prompted with the smallest / medium / "
                    "largest GT box of every image, round 2 with those 3 boxes + the 2 "
                    "highest-confidence round-1 predictions (self-prompts), all inside the whole "
                    "image (CountGD++ resize) (inference + offline evaluation).",
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
    g.add_argument("--experiment-name", default="high_confidence_pseudo_prompts_no_tiling_countGDpp",
                   help="EXPERIMENT_NAME. Written into every CSV row and every NPZ.")
    g.add_argument("--n-exemplars", type=int, default=K_EXEMPLARS,
                   help="K_EXEMPLARS = 3 by definition (S + M + L; fewer only when the image "
                        "has fewer GT boxes). The only accepted value is 3.")

    # ------------------- self-prompt selection (round 1 -> 2) ---------------
    g = p.add_argument_group("self-prompt selection, round 1 -> round 2 (CELL 3 / CELL 13)")
    g.add_argument("--n-self-prompts", type=int, default=N_SELF_PROMPTS,
                   help="N_SELF_PROMPTS: the highest-confidence round-1 predictions (after NMS, "
                        "not on a prompt plant) added to the S/M/L boxes in round 2. Part of "
                        "PROMPT_TYPE (size_SML+<n>self), so PHASE 2 must get the same value.")
    g.add_argument("--self-prompt-exclude-iou", type=float, default=SELF_PROMPT_EXCLUDE_IOU,
                   help="SELF_PROMPT_EXCLUDE_IOU: a round-1 prediction whose IoU with ANY S/M/L "
                        "prompt box is >= this value is not eligible as a self-prompt.")

    # ------------------------- CountGD++ inference --------------------------
    g = p.add_argument_group("countgd++ inference (CELL 3 / CELL 10 / CELL 11)")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="CONFIDENCE_THRESHOLD used INSIDE the CountGD++ post-processing "
                        "(replaces the repo default 0.23), in BOTH rounds. Round 1 -> round 2 "
                        "only uses predictions above it.")
    g.add_argument("--batch-size", type=int, default=1,
                   help="BATCH_SIZE. FIXED at 1: CountGD++ inserts the exemplar tokens for "
                        "sample 0 only (add_exemplar_tokens is called with labels=[0]), so its "
                        "code cannot run several images in one forward pass.")
    g.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="float32 = USE_FP16 False (the notebook; CountGD++ is released and "
                        "evaluated in fp32). float16 = USE_FP16 True (autocast, not validated).")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")
    g.add_argument("--text-prompt", default="",
                   help="TEXT_PROMPT. '' = exemplar only, which matches the SAM3 / YOLOE "
                        "exemplar experiments.")
    g.add_argument("--model-short-side", type=int, default=800,
                   help="MODEL_SHORT_SIDE: official resize of the WHOLE image, short side -> "
                        "800 px (8192 x 5460 -> 1200 x 800). Replaces the notebook's MAX_DIM.")
    g.add_argument("--model-max-size", type=int, default=1333,
                   help="MODEL_MAX_SIZE: official resize, long side capped at 1333 px.")
    g.add_argument("--seed", type=int, default=42,
                   help="SEED: same seed as the official inference scripts.")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 12 / CELL 17)")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")
    g.add_argument("--prompt-ignore-iou", type=float, default=0.50,
                   help="PROMPT_IGNORE_IOU: held_out mode ignore rule.")
    g.add_argument("--nms-iou-threshold", type=float, default=0.40,
                   help="NMS_IOU_THRESHOLD (fixed, no sweep). Used TWICE: in PHASE 1 on the "
                        "round-1 detections to pick the self-prompts, and offline in PHASE 2 on "
                        "the final (and round-1) detections.")
    g.add_argument("--operating-confidence", type=float, default=0.30,
                   help="The operating point of precision / recall / F1 / IoU1 / IoU2 / "
                        "count_abs_error. The notebook keeps it EQUAL to --threshold (0.30), so "
                        "the inference threshold and the operating point are the same single "
                        "value and no confidence x NMS sweep is performed.")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot (CELL 27)")
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

    if args.n_exemplars != K_EXEMPLARS:
        p.error("--n-exemplars must be 3: the size-based prompts are S + M + L by definition.")
    if args.n_self_prompts < 1:
        p.error("--n-self-prompts must be >= 1 (with 0, round 2 would repeat round 1).")
    if not (0.0 < args.self_prompt_exclude_iou <= 1.0):
        p.error("--self-prompt-exclude-iou must be in (0, 1].")
    if args.batch_size != 1:
        p.error("--batch-size must be 1: CountGD++ inserts the exemplar tokens for sample 0 "
                "only, so its code does not support several images per forward pass.")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")

    if args.checkpoint is None:
        args.checkpoint = args.countgd_repo / "checkpoints" / "countgd_plusplus.pth"

    # PROMPT_TYPE (CELL 3): S/M/L GT boxes + the round-1 self-prompts
    args.prompt_type = f"size_SML+{args.n_self_prompts}self"
    args.use_tiling = False                 # whole image, no tiles
    return args


# =============================================================================
#  OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections of both rounds (NPZ, one file per image)
#                          + one runs_manifest_<exp>_shard<i>.csv per shard
#     metrics/             image / experiment / dataset level CSVs + size groups
#     confusion_matrices/  CSV + PNG for all_gt and held_out
#     plots/               qualitative GT-vs-prediction figures + exemplar previews
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
    (see load_done_runs).
    """
    return paths.raw_detections / f"runs_manifest_{experiment_name}_shard{shard_index}.csv"


# =============================================================================
#  CELL 10 - SIZE-BASED EXEMPLAR SELECTION (S / M / L)
# =============================================================================
# Size = box AREA (width x height) in full-resolution pixels. ALL GT boxes of the
# image are candidates - boxes cut by the image border are treated like any other
# GT box.
#   S = smallest area
#   L = largest area (another box than S)
#   M = among the remaining boxes, the one whose area is closest to the MEDIAN
#       area of ALL GT boxes of the image
# Ties are broken by the lower GT index -> fully deterministic.
# Fewer than 3 GT boxes: 1 box -> [S], 2 boxes -> [S, L].
# The prompt order (= the order of the exemplar boxes given to CountGD++) is always S, M, L.
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


# --- quick self-test on a toy image (notebook CELL 10): fails loudly if broken ---
_toy = np.array([[0, 0, 10, 10], [0, 0, 50, 50], [0, 0, 30, 30],
                 [0, 0, 20, 20], [0, 0, 40, 40]], dtype=np.float32)   # areas 100..2500
assert select_size_exemplars(_toy)["indices"] == [0, 2, 1]   # S = 10x10, M = 30x30, L = 50x50
del _toy


# =============================================================================
#  CELL 6 - DATASET AND YOLO ANNOTATION HELPERS
# =============================================================================
#  The notebook opened ONE image of ONE folder with a single RUMEX_CLASS_ID. On
#  the cluster both archives are pooled into one dataset, each with its own class
#  id and a FLAT annotations_yolo folder, so discover_images() is the
#  archive-aware version from the cluster scripts. load_yolo_boxes / safe_crop
#  are unchanged notebook code.
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


def safe_crop(image: Image.Image, box, min_size: int = 2) -> Image.Image:
    """
    Crop an exemplar from the full image, clamped to the image borders and to a
    minimum size. The crop is EXACTLY the GT box: no padding / context is added.
    """
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(x1, image.width - min_size))
    y1 = max(0, min(y1, image.height - min_size))
    x2 = min(image.width, max(x2, x1 + min_size))
    y2 = min(image.height, max(y2, y1 + min_size))
    return image.crop((x1, y1, x2, y2))


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
#  CELL 9 - RESIZE HELPER (whole image, no tiling)
# =============================================================================
# The WHOLE image is sent to CountGD++ in one pass (no tiles). It is resized ONCE
# per image with CountGD++'s own resize (get_size_with_aspect_ratio in
# datasets/transforms_app.py): short side -> 800, long side <= 1333, i.e.
# 8192 x 5460 -> 1200 x 800 (factor ~0.146, the same scale as the global pass of
# the extra-path experiments). The SAM3 notebook used MAX_DIM = 1024 instead; the
# CountGD++ standard input size is used here. The exemplar boxes (S / M / L, and
# the self-prompts in round 2) are scaled into that resized image, and the
# predictions are scaled back to full resolution.
# These helpers are pure geometry (no torch): PHASE 2 re-uses them for the
# exemplar preview.
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


def image_scale_factor(img_w: int, img_h: int, short_side: int,
                       max_size: Optional[int]) -> Tuple[float, float, int, int]:
    """
    (sx, sy, new_w, new_h) of the official resize applied to the WHOLE image
    (8192 x 5460 -> 1200 x 800: sx = sy ~ 0.146).
    """
    new_h, new_w = countgd_resize_hw(img_w, img_h, short_side, max_size)
    return new_w / img_w, new_h / img_h, new_w, new_h


def clip_boxes_to_image(boxes, img_w: int, img_h: int) -> np.ndarray:
    """Clip predicted boxes (original-image coordinates) to the image borders."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)
    return boxes


def cxcywh_norm_to_xyxy(boxes, w: float, h: float) -> np.ndarray:
    """Normalised (cx, cy, bw, bh) -> pixel [x1, y1, x2, y2] in a w x h image."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


# =============================================================================
#  CELL 5 + CELL 11 - COUNTGD++ MODEL AND WHOLE-IMAGE INFERENCE
# =============================================================================
#  The model is built exactly like build_model_and_transforms() in the official
#  test_dataset.py: cfg_app.py (Swin-B backbone, 900 queries, BERT text encoder)
#  + countgd_plusplus.pth loaded with strict=False (as in the official code).
#
#  One forward pass per round (BATCH_SIZE = 1). Post-processing = the official
#  get_boxes_from_prediction():
#    stage 1: keep queries whose best POSITIVE-token probability > threshold
#             (threshold = CONFIDENCE_THRESHOLD = 0.30 instead of the default 0.23)
#    stage 2: keep queries more similar to the positive than to any negative
#             prompt (there is no negative prompt here, so stage 2 keeps all)
#    score  : the highest token probability of the kept query
#
#  Exemplars: CountGD++'s native prompting - the exemplar image IS the input
#  image and the exemplar boxes are drawn on it (in the pixels of the resized
#  image), exactly like boxes drawn on the image in the official app: round 1 the
#  S / M / L GT boxes, round 2 the same 3 boxes + the self-prompts. The image tensor
#  is built once per image and used by both rounds.
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
    """Owns CountGD++ and its input normalisation; performs one forward pass per round."""

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
        # (the official short side 800 / long side 1333 rule), once per image.
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

    # ---------------------------- CELL 9 -------------------------------------
    def prepare_image_tensor(self, image: Image.Image):
        """
        Resize (official rule) + normalise the WHOLE image, once per image.
        Output: tensor (3, H', W') and (sx, sy) = (W'/W, H'/H).
        """
        a = self.args
        sx, sy, new_w, new_h = image_scale_factor(image.width, image.height,
                                                  a.model_short_side, a.model_max_size)
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        tensor, _ = self.normalize(resized, None)
        resized.close()
        return tensor, (sx, sy)

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

        # the first "." closes the positive prompt (text + exemplar tokens)
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

    def infer_image(self, image_t, exemplar_box_t, img_w: int, img_h: int):
        """
        Run CountGD++ ONCE on the whole (resized) image with the k exemplar boxes in it.
        Input : image_t        - (3,H',W') tensor of the resized image (prepare_image_tensor)
                exemplar_box_t - (k,4) tensor: the k exemplar GT boxes in the RESIZED
                                 image's pixels -> k exemplar tokens of one prompt
                img_w, img_h   - ORIGINAL image size
        Output: boxes (K,4) in ORIGINAL pixel coordinates, scores (K,)
        """
        torch = self.torch
        caption = self.args.text_prompt + " . "   # official format: "<positive text> . "

        autocast = (torch.autocast("cuda", dtype=torch.float16) if self.use_fp16
                    else contextlib.nullcontext())
        with torch.inference_mode():
            with autocast:
                out = self.model(
                    self.nested_tensor_from_tensor_list([image_t.to(self.device)]),  # image to count in
                    self.nested_tensor_from_tensor_list([image_t.to(self.device)]),  # exemplar image = the SAME image
                    [exemplar_box_t.to(self.device)],                                # the k exemplar boxes drawn on it
                    [],                                                              # no negative exemplar images
                    [],                                                              # no negative exemplar boxes
                    captions=[caption],
                )
            boxes_n, scores = self.postprocess(out, self.args.threshold)
        del out
        # pred_boxes are normalised to the (unpadded) resized image; the resize keeps
        # the aspect ratio, so multiplying by the ORIGINAL size maps them back.
        return cxcywh_norm_to_xyxy(boxes_n, img_w, img_h), scores


def run_exemplars_whole_image(runner: CountGDRunner, image_t, scale: Tuple[float, float],
                           exemplar_boxes, img_w: int, img_h: int) -> dict:
    """
    Run CountGD++ for ONE exemplar set on the whole image (notebook CELL 11,
    run_sam3_whole_image): round 1 = the S / M / L boxes, round 2 = those + the
    self-prompts.

    Input : image_t        - tensor of the resized image (built once per image)
            scale          - (sx, sy) of that resize
            exemplar_boxes - (k, 4) [x1, y1, x2, y2] of the exemplars in ORIGINAL
                             pixels, in prompt order (S, M, L[, self-prompts])
    Output: dict of numpy arrays in ORIGINAL-IMAGE coordinates: boxes (N,4),
            scores (N,). These are the PRE-NMS detections (score >= 0.30, no NMS).
    """
    torch = runner.torch
    sx, sy = scale
    boxes_in = np.asarray(exemplar_boxes, dtype=np.float32).reshape(-1, 4) * np.array(
        [sx, sy, sx, sy], dtype=np.float32)
    box_t = torch.tensor(boxes_in.tolist(), dtype=torch.float32)

    boxes, scores = runner.infer_image(image_t, box_t, img_w, img_h)
    boxes = clip_boxes_to_image(boxes, img_w, img_h)       # no plausibility filter
    return {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.asarray(scores, dtype=np.float32).reshape(-1),
    }


# =============================================================================
#  CELL 12 - NMS
# =============================================================================
# CountGD++ can return several overlapping boxes for the same plant. NMS keeps the
# highest-scoring box of each overlapping group. The NMS IoU threshold is fixed
# (NMS_IOU_THRESHOLD = 0.40). It is used TWICE: in PHASE 1 on the round-1
# detections when picking the self-prompts (CELL 13), and offline in PHASE 2 on the
# final (and round-1) detections. The raw pre-NMS detections of both rounds stay
# untouched on disk. (No tile provenance is tracked here: without tiling every
# detection comes from the same single input image.)
# =============================================================================

def nms(boxes, scores, iou_threshold: float):
    """
    Input : boxes (N,4), scores (N,), iou_threshold
    Output: keep (list of kept indices, highest score first), n_suppressed (int)
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
#  CELL 13 - SELF-PROMPT SELECTION (round 1 -> round 2)
# =============================================================================
#   round-1 detections
#     -> NMS (NMS_IOU_THRESHOLD) so near-duplicates cannot both be picked
#     -> drop every box that is just a prompt plant again: IoU >=
#        SELF_PROMPT_EXCLUDE_IOU (0.50) with ANY of the S/M/L prompt boxes
#     -> keep the N_SELF_PROMPTS (2) highest-confidence remaining boxes
#     -> they are added to the S/M/L boxes as exemplar boxes in round 2
# These boxes are CountGD++'s own predictions, NOT ground truth: a false positive
# here is fed back as a positive example, so the per-image table also reports the
# round-1 metrics to make that visible. No plausibility filter (as in the notebook).
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
#  CELL 14 - PRE-NMS DETECTION STORAGE (both rounds)
# =============================================================================
# For every image we store
#   - the ROUND-1 pre-NMS detections,
#   - the self-prompt boxes and their confidences,
#   - the FINAL (round-2) pre-NMS detections; when round 2 was skipped because no
#     eligible self-prompt existed, the final arrays are the round-1 ones and
#     second_run is False,
# all AFTER CountGD++ inference at 0.30 -> conversion to original-image
# coordinates -> clipping to the image, but BEFORE any operating confidence
# threshold and BEFORE NMS. The offline evaluation therefore never needs
# CountGD++ again.
# =============================================================================

def run_npz_path(raw_detections_dir: Path, image_id: str) -> Path:
    """Path of the NPZ holding the two-round detections of one image (= one run)."""
    return raw_detections_dir / f"{safe_filename(image_id)}.npz"


def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        r1: dict, self_prompts: dict, final: dict, second_run: bool,
                        gt_boxes: np.ndarray, selection: dict,
                        image_size: Tuple[int, int], input_size: Tuple[int, int],
                        archive: str, flight: str, class_id: int) -> Path:
    """
    Write one image's two-round result to NPZ. The file is self-contained: it also
    stores the GT boxes and the S/M/L prompt indices (in prompt order), their roles
    and areas, so the whole offline evaluation and the final plots can run without
    re-opening label files or re-selecting the exemplars.
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
        input_width=np.array(int(input_size[0])),             # what CountGD++ saw
        input_height=np.array(int(input_size[1])),
        archive=np.array(archive),
        flight=np.array(flight),
        source_class_id=np.array(int(class_id)),
        gt_boxes=gt_boxes.astype(np.float32),
        # ---- round 1 -----------------------------------------------------------
        boxes_round1=np.asarray(r1["boxes"], dtype=np.float32).reshape(-1, 4),
        scores_round1=np.asarray(r1["scores"], dtype=np.float32).reshape(-1),
        # ---- self-prompts taken from round 1 ------------------------------------
        self_prompt_boxes=np.asarray(self_prompts["boxes"], dtype=np.float32).reshape(-1, 4),
        self_prompt_scores=np.asarray(self_prompts["scores"], dtype=np.float32).reshape(-1),
        second_run=np.array(bool(second_run)),
        # ---- final (round 2, or round 1 when round 2 was skipped) ---------------
        boxes=np.asarray(final["boxes"], dtype=np.float32).reshape(-1, 4),    # x1,y1,x2,y2 (original img)
        scores=np.asarray(final["scores"], dtype=np.float32).reshape(-1),     # confidence >= 0.30
    )
    return path


def load_run_detections(path: Path) -> dict:
    """Read one image NPZ back into a plain python dict."""
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
            "boxes_round1": z["boxes_round1"].reshape(-1, 4),
            "scores_round1": z["scores_round1"].reshape(-1),
            "self_prompt_boxes": z["self_prompt_boxes"].reshape(-1, 4),
            "self_prompt_scores": z["self_prompt_scores"].reshape(-1),
            "second_run": bool(z["second_run"]),
            "boxes": z["boxes"].reshape(-1, 4),               # FINAL detections
            "scores": z["scores"].reshape(-1),
            "input_width": int(z["input_width"]),
            "input_height": int(z["input_height"]),
            "archive": str(z["archive"]),
            "flight": str(z["flight"]),
        }
    return run


# =============================================================================
#  RESUME SUPPORT  (several shard manifests)
# =============================================================================

def load_done_images(paths: Paths, experiment_name: str, prompt_type: str) -> set:
    """
    Read every shard manifest and return {image_ID} of the images that are already
    finished (one run per image). Every shard reads ALL manifests, so a
    resubmission after the walltime never repeats work, even if the shard
    assignment changed because NUM_GPUS was different.

    Refuses to continue when the manifests already hold runs of ANOTHER prompt
    setting (notebook CELL 15): two prompt settings must never be mixed in one
    results folder.
    """
    done: set = set()
    other_types: set = set()
    for csv_path in sorted(paths.raw_detections.glob(f"runs_manifest_{experiment_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != experiment_name:
                        continue
                    # only complete rows (last column present) are checked, so a row
                    # cut short by the walltime cannot trigger a false alarm
                    if row.get("inference_seconds") and row.get("Prompt_Type") != prompt_type:
                        other_types.add(row.get("Prompt_Type"))
                    if row.get("image_ID"):
                        done.add(row["image_ID"])
        except OSError:
            continue
    if other_types:
        raise RuntimeError(
            f"{paths.raw_detections} already holds results for another prompt setting "
            f"{sorted(other_types)}. Use a new EXPERIMENT_NAME and OUTPUT_DIR for "
            f"{prompt_type!r} (or delete the old results folder) so the runs are not mixed.")
    return done


# =============================================================================
#  CELL 15 - MAIN GPU INFERENCE LOOP  (PHASE 1, two CountGD++ rounds per image)
# =============================================================================
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE
#     read its GT boxes ONCE
#     pick the S / M / L exemplars by area (CELL 10) - deterministic, so there is
#         exactly ONE run per image
#     resize the whole image ONCE (CountGD++ resize, 1200 x 800)
#     ROUND 1 : CountGD++ with the 3 GT boxes (threshold 0.30)
#     NMS on the round-1 detections -> pick the 2 highest-confidence boxes that do
#         not sit on a prompt plant (CELL 13)
#     ROUND 2 : CountGD++ with the 3 GT boxes + those 2 self-prompts (5 exemplar
#         boxes, threshold 0.30); skipped when no eligible box exists -> the
#         round-1 output is final
#     save round 1, the self-prompts and the final detections (NPZ)
#     release the image
#
# Images without any Rumex GT box are skipped, exactly like in the other
# experiments. The only NMS here is the one that picks the self-prompts; NO
# metric computation happens here - that is all done offline in PHASE 2. The
# loop is resumable: finished images are listed in the shard manifests.
# =============================================================================

def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    exp = args.experiment_name

    # ---- resume support (+ the prompt-setting guard, checked even with --no-resume) ----
    done_images: set = load_done_images(paths, exp, args.prompt_type)
    if args.no_resume:
        done_images = set()
    else:
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

    try:
        runner = CountGDRunner(args)
        torch = runner.torch
        device_is_cuda = runner.device.startswith("cuda")

        start_time = time.time()
        n_new_runs = 0
        n_without_second_run = 0
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

            # ---------------- size-based prompt selection (CELL 10) ---------------
            selection = select_size_exemplars(gt_boxes)
            exemplar_indices = selection["indices"]
            exemplar_boxes = gt_boxes[exemplar_indices]
            prompt_id = format_size_prompt_id(selection)

            # ---------------- whole image, resized ONCE for both rounds ------------
            image_t, scale = runner.prepare_image_tensor(image)
            in_w, in_h = int(image_t.shape[-1]), int(image_t.shape[-2])

            # ---------------- ROUND 1: the S / M / L GT boxes ----------------------
            r1 = run_exemplars_whole_image(runner, image_t, scale, exemplar_boxes, img_w, img_h)

            # ---------------- self-prompts from round 1 (CELL 13) ------------------
            r1_nms = apply_nms_to_run(r1, args.nms_iou_threshold)
            self_prompts = select_self_prompts(r1_nms, exemplar_boxes, args.n_self_prompts,
                                               args.self_prompt_exclude_iou, args.threshold)

            # ---------------- ROUND 2: S/M/L + the self-prompts --------------------
            if len(self_prompts["boxes"]):
                round2_boxes = np.concatenate([exemplar_boxes, self_prompts["boxes"]], axis=0)
                final = run_exemplars_whole_image(runner, image_t, scale, round2_boxes,
                                                  img_w, img_h)
                second_run = True
            else:
                # no eligible box -> round 2 would repeat round 1 exactly
                final, second_run = r1, False
                n_without_second_run += 1

            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, r1, self_prompts, final, second_run,
                gt_boxes, selection, (img_w, img_h), (in_w, in_h),
                rec.archive, rec.flight, rec.class_id)

            areas_by_role = dict(zip(selection["roles"], selection["areas"]))
            n_r1 = int(len(r1["scores"]))
            n_self = int(len(self_prompts["boxes"]))
            n_dets = int(len(final["scores"]))
            self_scores_txt = "+".join(f"{v:.3f}" for v in self_prompts["scores"])
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
                "n_detections_r1_pre_nms": n_r1,
                "n_detections_r1_post_nms": int(len(r1_nms["scores"])),
                "n_self_prompts": n_self,
                "self_prompt_scores": self_scores_txt,
                "second_run": second_run,
                "n_detections_pre_nms": n_dets,
                "image_width": img_w,
                "image_height": img_h,
                "input_width": in_w,
                "input_height": in_h,
                "npz_file": npz_path.name,
                "inference_seconds": round(run_seconds, 2),
            })
            manifest_file.flush()                  # this image is on disk -> resumable
            n_new_runs += 1

            # ---------------- release the image ---------------------------------------
            del r1, r1_nms, final, image_t
            image.close()
            del image
            gc.collect()
            if device_is_cuda:
                torch.cuda.empty_cache()

            image_times.append(time.time() - image_t0)
            avg_per_image = float(np.mean(image_times))
            eta = (n_total_images - img_idx) * avg_per_image
            print(f"[{exp}] shard{args.shard_index} ({img_idx}/{n_total_images}) {image_id} | "
                  f"{n_gt} GT box(es) | prompts={prompt_id} | input={in_w}x{in_h} | "
                  f"round1={n_r1} det -> self-prompts={n_self} "
                  f"({', '.join(f'{v:.2f}' for v in self_prompts['scores']) or '-'}) "
                  f"-> final={n_dets} det | {run_seconds:.1f}s | "
                  f"avg/image={avg_per_image:.1f}s | ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")
            if len(exemplar_indices) < K_EXEMPLARS:
                print(f"    NOTE: only {n_gt} GT box(es) -> {len(exemplar_indices)} GT prompt(s) "
                      f"({'+'.join(selection['roles'])}).")
            if not second_run:
                print(f"    NOTE: no eligible round-1 box ({self_prompts['n_eligible']} eligible, "
                      f"{self_prompts['n_on_prompt']} on a prompt plant) -> round 2 skipped, "
                      f"round-1 output kept as final.")
    finally:
        manifest_file.close()                      # also closed if the loop crashes

    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new images "
          f"({n_without_second_run} without a second round).")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Pre-NMS detections in: {paths.raw_detections}")


# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================

def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading CountGD++ ---")
    sample = my_records[:min(len(my_records), 200)]
    n_runs, n_few, per_archive = 0, 0, {}
    for rec in sample:
        with Image.open(rec.image_path) as im:
            w, h = im.size
        n_gt = len(load_yolo_boxes(rec.label_path, w, h, rec.class_id))
        if n_gt == 0:
            continue
        n_runs += 1                                     # ONE run per image
        n_few += int(n_gt < K_EXEMPLARS)
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + 1
    sx, _, in_w, in_h = image_scale_factor(8192, 5460, args.model_short_side, args.model_max_size)
    print(f"  sampled {len(sample)} image(s) of this shard -> {n_runs} runs, ONE per image "
          f"({per_archive})")
    print(f"  images with fewer than {K_EXEMPLARS} GT boxes (fewer prompts): {n_few}")
    print(f"  CountGD++ input for an 8192x5460 image: {in_w}x{in_h} (scale {sx:.3f}), no tiles")
    print(f"  => ~{2 * n_runs} CountGD++ forward passes (TWO per image: round 1 + round 2; "
          f"round 2 is skipped when round 1 leaves no eligible box) for those "
          f"{len(sample)} images")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{n_runs} (one per image)")


# =============================================================================
#  CELL 16 - LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
# =============================================================================
# From here on CountGD++ is never touched again. Everything below works on the
# NPZ files written in PHASE 1 (both rounds), so the complete evaluation can be redone in
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
    # only the runs of THIS experiment and THIS prompt setting (notebook CELL 16)
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
        runs.append(run)

    if n_missing:
        print(f"  WARNING: {n_missing} manifest row(s) point at a missing NPZ (skipped).")

    print(f"Loaded {len(runs)} runs (= images) for {experiment_name} ({prompt_type}).")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes:", int(sum(len(r['gt_boxes']) for r in runs)))
    print(f"Images with fewer than {K_EXEMPLARS} GT prompts:",
          int(sum(1 for r in runs if len(r["prompt_indices"]) < K_EXEMPLARS)))
    print("Images with a second round:",
          int(sum(1 for r in runs if r["second_run"])), "/", len(runs))
    n_self = pd.Series([len(r["self_prompt_boxes"]) for r in runs], dtype=int)
    print("Self-prompts per image:",
          {int(k): int(v) for k, v in n_self.value_counts().sort_index().items()})
    return runs, manifest


# =============================================================================
#  CELL 17 - EVALUATION CORE: all_gt AND held_out
# =============================================================================
#   1. one-to-one matching against the evaluated (non-prompt) GT at IoU 0.50
#   2. still-unmatched predictions with IoU >= 0.50 to a PROMPT GT box -> IGNORED
#   3. the rest are false positives. Prompt GT boxes are never false negatives.
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
      IoU2 - sum of matched IoUs divided by the number of evaluated GT boxes
      valid_for_macro - False when there is no GT left to evaluate (held_out runs
             in which every plant of the image was used as a prompt)
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    eval_gt_boxes = np.asarray(eval_gt_boxes, dtype=np.float32).reshape(-1, 4)
    prompt_gt_boxes = np.asarray(prompt_gt_boxes, dtype=np.float32).reshape(-1, 4)

    n_pred, n_eval_gt = len(pred_boxes), len(eval_gt_boxes)
    status = np.full(n_pred, STATUS_FP, dtype=np.int8)

    # step 1 - one-to-one matching against the evaluated GT
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
    matched_ious = match["matched_ious"]
    return {
        "status": status, "pred_match_gt": match["pred_match_gt"],
        "TP": tp, "FP": fp, "FN": fn, "n_ignored": n_ignored,
        "n_eval_gt": n_eval_gt, "n_pred": n_pred,
        "precision": float(precision), "recall": float(recall),
        "F1": safe_f1(precision, recall),
        "IoU1": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "IoU2": float(np.sum(matched_ious) / n_eval_gt) if n_eval_gt > 0 else 0.0,
        "valid_for_macro": bool(n_eval_gt > 0),
    }


# =============================================================================
#  CELL 18 + CELL 19 - AP50 AND AP50:95 + OPERATING POINT
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


def ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou: float, ignore_iou: float):
    """AP episode of ONE run; in held_out the IGNORED predictions are removed first."""
    ev = evaluate_run_predictions(nms_run["boxes"], nms_run["scores"],
                                  eval_gt, prompt_gt, eval_iou, ignore_iou)
    keep = ev["status"] != STATUS_IGNORED
    return (make_detections(nms_run["boxes"][keep], nms_run["scores"][keep]),
            make_detections(eval_gt))


def evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, confidence_threshold: float,
                                eval_iou: float, ignore_iou: float) -> dict:
    """Precision / recall / F1 / IoU1 / IoU2 for predictions >= confidence_threshold."""
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_run_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                    eval_gt, prompt_gt, eval_iou, ignore_iou)


def evaluate_detection_set(det: dict, gt, prompt_indices, mode: str, args):
    """
    AP + operating-point metrics of ONE pre-NMS detection set (the final or the
    round-1 detections of an image) in one mode (notebook CELL 21).
    Output: nms_run, ev (evaluate_run_predictions dict), ap50, ap5095
    """
    nms_run = apply_nms_to_run(det, args.nms_iou_threshold)
    eval_gt, prompt_gt = split_gt_for_mode(gt, prompt_indices, mode)
    p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt,
                                     args.eval_iou_threshold, args.prompt_ignore_iou)
    ap50, ap5095 = compute_ap([p_det], [g_det])
    ev = evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, args.operating_confidence,
                                     args.eval_iou_threshold, args.prompt_ignore_iou)
    return nms_run, ev, ap50, ap5095


# =============================================================================
#  CELL 20 + CELL 27 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs
#  PREDICTIONS (right)
# =============================================================================
# Colours used everywhere:
#   yellow = GT Rumex boxes
#   cyan   = S (smallest) exemplar prompt
#   lime   = M (medium) exemplar prompt
#   orange = L (largest) exemplar prompt
#   magenta= self-prompts (round-1 predictions fed back in round 2)
#   red    = predictions
#
# IMAGE SELECTION (per-image AP50, mode = PLOT_EVALUATION_MODE):
#   1. keep only images that have at least PLOT_MIN_GT_BOXES (7) GT boxes
#      and take the one with the highest AP50
#   2. if NO image has 7 GT boxes: keep the images with the HIGHEST number of GT
#      boxes and take the one among them with the highest AP50
#   ties are broken by F1, then by the number of GT boxes.
#
# Right panel = the FINAL (round-2) post-NMS predictions of that image with
# score >= CONFIDENCE_THRESHOLD, plus its S / M / L prompts (dashed) and the
# self-prompts that were fed back into round 2 (magenta dashed).
#
# For that same image, the exemplar preview is saved as well:
#   left   : all GT boxes with their index, the S / M / L prompts and the
#            self-prompts highlighted
#   right  : the round-2 exemplar regions (S, M, L, self1, self2) at original
#            resolution and as CountGD++ sees them inside the 1200 x 800 input
#            (side by side, display only), each coloured by its role
#
# The image is downscaled for DISPLAY only (PLOT_MAX_DISPLAY_DIM).
# =============================================================================

COLOR_GT, COLOR_PRED, COLOR_SELF = "yellow", "red", "magenta"
ROLE_COLORS = {"S": "cyan", "M": "lime", "L": "orange"}
ROLE_NAMES = {"S": "smallest", "M": "medium", "L": "largest"}


def role_color(role: str) -> str:
    """S / M / L colours; every self-prompt ('self1', 'self2', ...) is magenta."""
    return ROLE_COLORS.get(role, COLOR_SELF)


def _draw_boxes(ax, boxes, color, linewidth=1.5, linestyle="-", labels=None, fontsize=6):
    """Draw [x1,y1,x2,y2] boxes (display coordinates) on a matplotlib axis."""
    import matplotlib.patches as patches
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes).reshape(-1, 4)):
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=color, linewidth=linewidth,
                                       linestyle=linestyle))
        if labels is not None:
            ax.text(x1, y1 - 2, labels[k], color="white" if color == COLOR_PRED else "black",
                    fontsize=fontsize, va="bottom", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none", pad=0.8, alpha=0.85))


def _draw_role_boxes(ax, boxes, roles, linewidth=2.5, linestyle="--", labels=None, fontsize=9):
    """Draw the prompt boxes (S / M / L, self-prompts), each in the colour of its role."""
    boxes = np.asarray(boxes).reshape(-1, 4)
    for k, role in enumerate(roles):
        _draw_boxes(ax, boxes[k:k + 1], role_color(role), linewidth=linewidth,
                    linestyle=linestyle,
                    labels=None if labels is None else [labels[k]], fontsize=fontsize)


def _display_copy(image: Image.Image, max_display_dim: int):
    """Downscaled numpy copy of the image + factor mapping full-res boxes onto it."""
    w, h = image.size
    s = min(1.0, max_display_dim / max(w, h))
    disp_w, disp_h = max(1, int(w * s)), max(1, int(h * s))
    display = np.asarray(image.resize((disp_w, disp_h), Image.BILINEAR))
    return display, np.array([disp_w / w, disp_h / h, disp_w / w, disp_h / h], dtype=np.float32)


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
    print(ranked[["image_ID", "n_gt", "Prompt_ID", "n_predictions", "AP50", "F1",
                  "precision", "recall"]].head(5).to_string(index=False))
    return ranked.iloc[0], rule


def _crops_side_by_side(crops: Sequence[Image.Image], gap: int = 6):
    """Display-only strip of the exemplar regions (never fed to the model).
    Output: strip image, (k, 4) boxes of every region in it."""
    w = sum(c.width for c in crops) + gap * (len(crops) + 1)
    h = max(c.height for c in crops) + 2 * gap
    strip = Image.new("RGB", (w, h), (255, 255, 255))
    boxes, x = [], gap
    for c in crops:
        strip.paste(c, (x, gap))
        boxes.append([x, gap, x + c.width, gap + c.height])
        x += c.width + gap
    return strip, np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def plot_exemplar_preview(display, to_disp, gt_boxes, prompt_indices, prompt_roles,
                          self_boxes, strip: Image.Image, strip_boxes, strip_fed: Image.Image,
                          fed_boxes, strip_roles, title: str, png_path: Path, plt) -> None:
    """
    Check of the prompt (saved, never shown):
      left   : all GT boxes with their index, the S / M / L prompts and the
               self-prompts highlighted
      right  : the round-2 exemplar regions at original resolution and as CountGD++
               sees them inside the resized whole image (display-only strips, in
               prompt order: S, M, L, self1, self2)
    """
    fig = plt.figure(figsize=(22, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(display)
    ax.axis("off")
    gt_d = np.asarray(gt_boxes).reshape(-1, 4) * to_disp
    _draw_boxes(ax, gt_d, COLOR_GT, linewidth=1.0,
                labels=[str(i) for i in range(len(gt_d))], fontsize=5)
    idx = [int(i) for i in prompt_indices]
    _draw_role_boxes(ax, gt_d[idx], prompt_roles, linewidth=3, linestyle="-",
                     labels=[f"{r} ({i})" for r, i in zip(prompt_roles, idx)])
    self_d = np.asarray(self_boxes).reshape(-1, 4) * to_disp
    _draw_role_boxes(ax, self_d, [f"self{k + 1}" for k in range(len(self_d))],
                     linewidth=3, linestyle="--",
                     labels=[f"self{k + 1}" for k in range(len(self_d))])
    ax.set_title(f"GT boxes (yellow, indexed), the {len(idx)} size prompts "
                 f"(S cyan, M lime, L orange) and the {len(self_d)} self-prompts (magenta)",
                 fontsize=12)

    for row, (img, bxs, name) in enumerate([(strip, strip_boxes, "round-2 exemplar boxes (original pixels)"),
                                            (strip_fed, fed_boxes, "as CountGD++ sees them")]):
        cax = fig.add_subplot(gs[row, 1])
        cax.imshow(np.asarray(img))
        _draw_role_boxes(cax, bxs, strip_roles, linewidth=1.5, linestyle="--",
                         labels=list(strip_roles), fontsize=7)
        cax.set_title(f"{name}: {img.width} x {img.height} px, {len(bxs)} boxes", fontsize=10)
        cax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown


def plot_gt_vs_predictions(args, paths: Paths, runs_by_image: dict, image_level_df,
                           image_paths: dict, plt, scope_label: str,
                           archive: Optional[str] = None) -> None:
    """One qualitative figure (+ its exemplar preview) for one scope."""
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

    nms_run = apply_nms_to_run(run, args.nms_iou_threshold)          # FINAL detections
    keep = nms_run["scores"] >= args.operating_confidence
    pred_boxes, pred_scores = nms_run["boxes"][keep], nms_run["scores"][keep]
    gt_boxes_plot = run["gt_boxes"]
    roles = run["prompt_roles"]
    prompt_boxes = gt_boxes_plot[np.asarray(run["prompt_indices"], dtype=int)]
    self_boxes = run["self_prompt_boxes"]
    n_self = len(self_boxes)
    # the round-2 exemplar set, in prompt order: S, M, L, self1, self2
    strip_roles = list(roles) + [f"self{k + 1}" for k in range(n_self)]
    strip_src = np.concatenate([prompt_boxes, self_boxes], axis=0)

    # ---- open the image: display copy + the exemplar regions as CountGD++ saw them --
    if plot_image_id not in image_paths:
        print("Image file not found for", plot_image_id)
        return
    with Image.open(image_paths[plot_image_id]) as im:
        im = im.convert("RGB")
        w, h = im.size
        display, to_disp = _display_copy(im, args.plot_max_display_dim)
        crops = [safe_crop(im, b) for b in strip_src]          # S, M, L, self-prompt regions
    sx, sy, in_w, in_h = image_scale_factor(w, h, args.model_short_side, args.model_max_size)
    fed = [c.resize((max(1, int(round(c.width * sx))), max(1, int(round(c.height * sy)))),
                    Image.BILINEAR) for c in crops]
    strip, strip_boxes = _crops_side_by_side(crops)          # display only
    strip_fed, fed_boxes = _crops_side_by_side(fed, gap=2)

    # ---- figure: GT (left) | predictions (right) -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax in axes:
        ax.imshow(display)
        ax.axis("off")

    _draw_boxes(axes[0], gt_boxes_plot * to_disp, COLOR_GT, linewidth=1.5)
    axes[0].set_title(f"Ground truth: {len(gt_boxes_plot)} Rumex boxes", fontsize=12)

    score_labels = [f"{v:.2f}" for v in pred_scores] if args.plot_show_scores else None
    _draw_boxes(axes[1], pred_boxes * to_disp, COLOR_PRED, linewidth=1.5, labels=score_labels)
    _draw_role_boxes(axes[1], prompt_boxes * to_disp, roles, linewidth=2.5, linestyle="--")
    if n_self:
        _draw_boxes(axes[1], self_boxes * to_disp, COLOR_SELF, linewidth=2.5, linestyle="--")
    axes[1].set_title(
        f"Final (round {2 if run['second_run'] else 1}): {len(pred_boxes)} boxes | "
        f"prompts {run['Prompt_ID']} + {n_self} self\n"
        f"AP50={row['AP50']:.3f}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
        f"F1={row['F1']:.3f}  TP={int(row['TP'])} FP={int(row['FP'])} FN={int(row['FN'])}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color=COLOR_GT, lw=2, label="ground truth"),
        Line2D([0], [0], color=COLOR_PRED, lw=2, label="prediction"),
    ] + [Line2D([0], [0], color=ROLE_COLORS[r], lw=2.5, linestyle="--",
                label=f"{r} prompt ({ROLE_NAMES[r]})") for r in roles] \
        + ([Line2D([0], [0], color=COLOR_SELF, lw=2.5, linestyle="--",
                   label=f"self-prompts ({n_self})")] if n_self else [])
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=11, frameon=False)
    fig.suptitle(
        f"{exp} | {plot_image_id} | CountGD++ | mode={mode} | scope={scope_label}\n"
        f"size-based prompts {run['Prompt_ID']} + {n_self} self-prompts, all inside the "
        f"image | NO tiling: whole image -> {in_w}x{in_h} (CountGD++ resize) | "
        f"conf={args.operating_confidence:.2f}, NMS IoU={args.nms_iou_threshold:.2f}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])

    stem = f"{scope_label}_{safe_filename(plot_image_id)}"
    png_path = paths.plots / f"best_image_{stem}_{mode}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

    # ---- exemplar preview for the same image ----------------------------------
    preview_path = paths.plots / f"exemplar_preview_{stem}.png"
    plot_exemplar_preview(
        display, to_disp, gt_boxes_plot, run["prompt_indices"], roles, self_boxes,
        strip, strip_boxes, strip_fed, fed_boxes, strip_roles,
        title=(f"{exp} | {plot_image_id}\n"
               f"round {2 if run['second_run'] else 1}: size-based prompts {run['Prompt_ID']} "
               f"+ {n_self} self-prompts, their boxes INSIDE the image (exemplar image = the "
               f"input image) | whole image -> {in_w}x{in_h}"),
        png_path=preview_path, plt=plt)

    print(f"\n[{scope_label}] Selected image : {plot_image_id} "
          f"({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Prompts        : {run['Prompt_ID']} + {n_self} self-prompt(s) "
          f"({', '.join(f'{v:.2f}' for v in run['self_prompt_scores']) or '-'})")
    print(f"[{scope_label}] Exemplars fed  : "
          + ", ".join(f"{r} {c.width}x{c.height}->{f.width}x{f.height}"
                      for r, c, f in zip(strip_roles, crops, fed))
          + f" px inside the {in_w} x {in_h} input (scale {sx:.3f})")
    strip.close()
    strip_fed.close()
    print(f"[{scope_label}] Figure saved   : {png_path}")
    print(f"[{scope_label}] Preview saved  : {preview_path}")


def make_qualitative_plots(args, paths: Paths, runs, image_level_df, plt) -> None:
    """One figure per archive plus one global figure."""
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
    print(f"  Confidence threshold = {conf:.2f}  (== the CountGD++ inference threshold)")
    print(f"  NMS IoU threshold    = {nms_iou:.2f}")
    print(f"  Evaluation IoU       = {eval_iou:.2f}")

    runs, manifest = load_runs(paths, exp, args.prompt_type)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  CELL 21 - PER-IMAGE METRICS  (final = round 2, with round 1 for comparison)
    # =========================================================================
    # The evaluated predictions are the FINAL ones (round 2, or round 1 when
    # round 2 was skipped), compared against the original human GT annotations.
    # The same metrics are also computed for the ROUND-1 output and written into
    # *_round1 columns, so the effect of the self-prompts is visible per image
    # (delta_AP50 = AP50 - AP50_round1, delta_F1 = F1 - F1_round1).
    # Everything is deterministic, so every image is evaluated exactly once and
    # this single table replaces the run-level + image-level tables of the
    # anchor-based experiments.
    #   AP50 / AP50_95 : all post-NMS predictions >= 0.30, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : only predictions >= CONFIDENCE_THRESHOLD
    #   count_abs_error : |evaluated predictions - evaluated GT| at the operating
    #                     point (ignored predictions are not counted)
    #
    # Special case (held_out with no evaluable GT, i.e. the image has at most 3
    # plants and all of them are prompts): the metrics are written as NaN and
    # valid_for_macro is False so they are excluded from every mean/std, but
    # TP/FN = 0 and the real FP count are kept, because such an image can still
    # produce false positives that must show up in the pooled counts and in the
    # confusion matrix.
    # =========================================================================
    print("\n--- per-image metrics ---")
    image_rows = []
    for run in runs:
        gt = run["gt_boxes"]
        final_det = {"boxes": run["boxes"], "scores": run["scores"]}
        round1_det = {"boxes": run["boxes_round1"], "scores": run["scores_round1"]}
        for mode in EVALUATION_MODES:
            # AP: every post-NMS prediction >= 0.30 (ignored ones removed);
            # operating point: predictions >= CONFIDENCE_THRESHOLD
            nms_run, ev, ap50, ap5095 = evaluate_detection_set(
                final_det, gt, run["prompt_indices"], mode, args)
            _, ev1, ap50_r1, ap5095_r1 = evaluate_detection_set(
                round1_det, gt, run["prompt_indices"], mode, args)
            valid = ev["valid_for_macro"]
            nan = float("nan")
            n_counted = ev["n_pred"] - ev["n_ignored"]       # predictions that are evaluated
            n_counted_r1 = ev1["n_pred"] - ev1["n_ignored"]

            row = {
                "experiment_name": exp,
                "image_ID": run["image_ID"],
                "archive": run["archive"],
                "flight": run["flight"],
                "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"],
                "evaluation_mode": mode,
                "confidence_threshold": conf,
                "nms_iou_threshold": nms_iou,
                "n_gt": int(len(gt)),
                "n_prompt_gt": int(len(run["prompt_indices"])),
                "n_self_prompts": int(len(run["self_prompt_boxes"])),
                "self_prompt_scores": "+".join(f"{v:.3f}" for v in run["self_prompt_scores"]),
                "second_run": run["second_run"],
                "median_gt_area_px2": round(run["median_gt_area"]),
                "n_eval_gt": ev["n_eval_gt"],
                "n_predictions_pre_nms": nms_run["n_pre_nms"],
                "n_suppressed": nms_run["n_suppressed"],
                "n_predictions": ev["n_pred"],
                "n_ignored_predictions": ev["n_ignored"],
                "AP50": ap50 if valid else nan,
                "AP50_95": ap5095 if valid else nan,
                "precision": ev["precision"] if valid else nan,
                "recall": ev["recall"] if valid else nan,
                "F1": ev["F1"] if valid else nan,
                "IoU1": ev["IoU1"] if valid else nan,
                "IoU2": ev["IoU2"] if valid else nan,
                "count_abs_error": abs(n_counted - ev["n_eval_gt"]) if valid else nan,
                "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"],
                "valid_for_macro": valid,
                # ---- round 1, same metrics, for comparison --------------------------
                "AP50_round1": ap50_r1 if valid else nan,
                "AP50_95_round1": ap5095_r1 if valid else nan,
                "precision_round1": ev1["precision"] if valid else nan,
                "recall_round1": ev1["recall"] if valid else nan,
                "F1_round1": ev1["F1"] if valid else nan,
                "IoU1_round1": ev1["IoU1"] if valid else nan,
                "IoU2_round1": ev1["IoU2"] if valid else nan,
                "count_abs_error_round1": abs(n_counted_r1 - ev1["n_eval_gt"]) if valid else nan,
                "TP_round1": ev1["TP"], "FP_round1": ev1["FP"], "FN_round1": ev1["FN"],
            }
            row["delta_AP50"] = row["AP50"] - row["AP50_round1"] if valid else nan
            row["delta_F1"] = row["F1"] - row["F1_round1"] if valid else nan
            image_rows.append(row)

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
              f"averaging, AP50_mean={sub['AP50'].mean():.4f}, F1_mean={sub['F1'].mean():.4f}, "
              f"count_abs_error_mean={sub['count_abs_error'].mean():.2f}")
        better = int((sub["delta_AP50"] > 0).sum())
        worse = int((sub["delta_AP50"] < 0).sum())
        same = int((sub["delta_AP50"] == 0).sum())
        print(f"  {mode:9s}: AP50 round1={sub['AP50_round1'].mean():.4f} -> "
              f"final={sub['AP50'].mean():.4f} (mean delta {sub['delta_AP50'].mean():+.4f}) | "
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
            "evaluation_mode": mode,
            "prompt_type": args.prompt_type,
            "n_exemplars": args.n_exemplars,
            "n_self_prompts": args.n_self_prompts,
            "self_prompt_exclude_iou": args.self_prompt_exclude_iou,
            "n_images_with_second_round": int(sub.drop_duplicates("image_ID")["second_run"].sum()),
            "size_measure": SIZE_MEASURE,
            "use_tiling": args.use_tiling,
            "model_short_side": args.model_short_side,
            "model_max_size": args.model_max_size,
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
            "n_images_valid_for_macro": int(sub_df["valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub_df[col].mean()
            row[f"{col}_std"] = sub_df[col].std()
            row[f"{col}_round1_mean"] = sub_df[f"{col}_round1"].mean()
        row["delta_AP50_mean"] = sub_df["delta_AP50"].mean()
        row["delta_F1_mean"] = sub_df["delta_F1"].mean()
        per_archive_rows.append(row)
    per_archive_df = pd.DataFrame(per_archive_rows)
    per_archive_csv = paths.metrics / "experiment_summary_per_archive.csv"
    per_archive_df.to_csv(per_archive_csv, index=False)
    print(f"\nPer-archive summary -> {per_archive_csv}\n")
    if not per_archive_df.empty:
        print(per_archive_df[["archive", "evaluation_mode", "n_images",
                              "AP50_round1_mean", "AP50_mean", "precision_mean", "recall_mean",
                              "F1_mean", "count_abs_error_mean"]].to_string(index=False))

    # =========================================================================
    #  CELL 23 - POOLED DATASET AP50 / AP50:95
    # =========================================================================
    # This is NOT the mean of the per-image AP values. All images are handed to
    # supervision at once, so every detection of the whole dataset is ranked in
    # ONE precision-recall curve. Here one episode = one image. Computed for the
    # final detections and, for comparison, for the round-1 detections.
    # =========================================================================
    print("\n--- pooled dataset AP ---")
    dataset_rows = []
    for mode in EVALUATION_MODES:
        pred_list, gt_list, images_used = [], [], set()
        pred_list_r1, gt_list_r1 = [], []
        for run in runs:
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)
            for det, pl, gl in [({"boxes": run["boxes"], "scores": run["scores"]},
                                 pred_list, gt_list),
                                ({"boxes": run["boxes_round1"], "scores": run["scores_round1"]},
                                 pred_list_r1, gt_list_r1)]:
                nms_run = apply_nms_to_run(det, nms_iou)
                p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
                pl.append(p_det)
                gl.append(g_det)
            images_used.add(run["image_ID"])
        ap50, ap5095 = compute_ap(pred_list, gt_list)   # one PR curve over ALL images
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
    #                                                 (there are no true negatives)
    # Counts are pooled over every image at the fixed configuration (FINAL
    # detections). In held_out mode the S/M/L prompt plants and the ignored
    # detections do not appear.
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
                f"{exp} - {mode}\nS/M/L + {args.n_self_prompts} self-prompts | conf={conf:.2f}, "
                f"NMS IoU={nms_iou:.2f}, eval IoU={eval_iou:.2f}",
                paths.confusion_matrices / f"confusion_matrix_{mode}.png")

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
    #  CELL 25 - RECALL PER SIZE GROUP (does the S/M/L mix help across plant sizes?)
    # =========================================================================
    # For every image the GT boxes are split at that image's MEDIAN area:
    #   "area <= median" (the smaller half) and "area >  median" (the larger half).
    # Matching uses the FINAL detections, all_gt, at the fixed operating point (same
    # matcher as everywhere).
    # The prompt plants themselves are reported separately, because CountGD++
    # usually re-detects them and they would flatter the recall of their size group.
    # =========================================================================
    print("\n--- recall per size group ---")
    size_rows, prompt_rows = [], []
    for run in runs:
        gt = run["gt_boxes"]
        nms_run = apply_nms_to_run({"boxes": run["boxes"], "scores": run["scores"]}, nms_iou)
        keep = nms_run["scores"] >= conf
        match = match_one_to_one(nms_run["boxes"][keep], nms_run["scores"][keep], gt, eval_iou)
        found = match["gt_match_pred"] >= 0

        areas = box_areas(gt)
        is_prompt = np.isin(np.arange(len(gt)), run["prompt_indices"])
        small_half = areas <= run["median_gt_area"]
        for name, grp in [("area <= median", small_half), ("area >  median", ~small_half)]:
            mask = grp & ~is_prompt                      # non-prompt GT boxes only
            size_rows.append({
                "image_ID": run["image_ID"], "archive": run["archive"], "size_group": name,
                "n_gt_non_prompt": int(mask.sum()), "found": int((found & mask).sum()),
                "recall": float((found & mask).sum() / mask.sum()) if mask.sum() else float("nan"),
            })
        for role, idx in zip(run["prompt_roles"], run["prompt_indices"]):
            prompt_rows.append({"image_ID": run["image_ID"], "archive": run["archive"],
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

    print(f"Recall of NON-PROMPT GT boxes per size group (all_gt matching, conf={conf:.2f}):")
    print(pooled.join(per_image_mean).to_string())
    print("\nPrompt plants re-detected (share per role):")
    print(prompt_hits_df.groupby("role")["re_detected"].agg(["count", "sum", "mean"]).to_string())
    print("\nSaved:", size_group_csv, "and", prompt_hits_csv)

    # =========================================================================
    #  CELL 27 - QUALITATIVE FIGURES (one per archive + one global)
    # =========================================================================
    if args.no_plots:
        print("\n--- qualitative figures skipped (--no-plots) ---")
    elif not _HAS_MPL:
        print("\n--- qualitative figures skipped (matplotlib unavailable) ---")
    else:
        print("\n--- qualitative GT-vs-prediction figures + exemplar previews ---")
        make_qualitative_plots(args, paths, runs, image_level_df, plt)

    # =========================================================================
    #  CELL 26 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY (CountGD++)")
    print("=" * 78)
    print(f"Round 1 prompts          : {args.n_exemplars} GT boxes by {SIZE_MEASURE} "
          f"(smallest / medium / largest), no text, no negatives")
    print(f"Round 2 prompts          : those {args.n_exemplars} + up to {args.n_self_prompts} "
          f"self-prompts (highest-confidence round-1 predictions after NMS, IoU < "
          f"{args.self_prompt_exclude_iou} to the prompts)")
    print(f"Images with 2 rounds     : {int(sum(1 for r in runs if r['second_run']))} / {len(runs)}")
    print(f"Exemplars                : the boxes INSIDE the input image (CountGD++ native "
          f"exemplars), S, M, L[, self-prompts] order")
    print(f"Tiling                   : {args.use_tiling}  (whole image, CountGD++ resize: short "
          f"side {args.model_short_side}px, max {args.model_max_size}px -> 1200 x 800 for "
          f"8192 x 5460)")
    print(f"Plausibility filter      : none (boxes only clipped to the image), as in the notebook")
    print(f"CountGD++ threshold      : {args.threshold} (both rounds; "
          f"CountGD++ default is 0.23)")
    print(f"Operating point          : confidence={conf:.2f} (== the inference "
          f"threshold), NMS IoU={nms_iou:.2f} (both fixed)")
    print(f"Evaluation IoU           : {eval_iou:.2f}")
    print(f"Images                   : {len(runs)}")
    print(f"Archives                 : "
          f"{', '.join(sorted(str(a) for a in image_level_df['archive'].dropna().unique()))}")
    print("-" * 78)
    print("EXPERIMENT-LEVEL RESULTS (mean over images, std between images)")
    show = ["evaluation_mode", "AP50_mean", "AP50_std", "AP50_95_mean", "precision_mean",
            "recall_mean", "F1_mean", "F1_std", "IoU1_mean", "IoU2_mean",
            "count_abs_error_mean"]
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
                  "'Problem B' in high_confidence_pseudo_prompts_no_tiling_HOW_TO_RUN_countGDpp.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" HIGH_CONFIDENCE_PSEUDO_PROMPTS_NO_TILING COUNTGD++ PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {args.dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" countgd_repo   : {args.countgd_repo}")
    print(f" checkpoint     : {args.checkpoint}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" round 1        : {args.n_exemplars} GT boxes, smallest / medium / largest by "
          f"{SIZE_MEASURE}, inside the image")
    print(f" round 2        : the same + {args.n_self_prompts} self-prompts (highest-confidence "
          f"round-1 predictions after NMS, IoU < {args.self_prompt_exclude_iou} to the prompts)"
          f"  [{args.prompt_type}]")
    print(f" text prompt    : {args.text_prompt!r}")
    print(f" tiling         : {args.use_tiling}  (whole image, TWO forward passes per image)")
    print(f" model input    : short side {args.model_short_side}px (max {args.model_max_size}px)")
    print(f" threshold      : {args.threshold}  (both rounds)")
    print(f" batch / dtype  : {args.batch_size} / {args.dtype}")
    print(f" operating pt   : confidence={args.operating_confidence}, "
          f"NMS IoU={args.nms_iou_threshold} (fixed, no sweep)")
    print(f" eval IoU       : {args.eval_iou_threshold}  "
          f"(prompt ignore IoU={args.prompt_ignore_iou})")
    print(f" supervision    : {'available' if has_supervision else 'MISSING'}")
    print("=" * 92)

    # ---------------- prompt-setting guard -----------------------------------
    # Refuse a different prompt setting BEFORE anything is written (config
    # snapshot, manifest, NPZ), so an existing experiment is never touched by a
    # mixed run.
    load_done_images(paths, exp, args.prompt_type)

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
        config["evaluation_modes"] = EVALUATION_MODES
        config["size_measure"] = SIZE_MEASURE
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
