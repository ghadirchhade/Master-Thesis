#!/usr/bin/env python3
# =============================================================================
#  1_exemplars_text_tiling_extra_path_infer_countGDpp.py
# =============================================================================
#
#  This is the SINGLE-EXEMPLAR + TEXT TILING + EXTRA PATH experiment with
#  CountGD++ (CVPR 2026): N_EXEMPLARS = 1, TEXT_PROMPT = " rumex obtusifolius ",
#  USE_TILING = True with TILE_SIZE = 1536 / OVERLAP = 384, PLUS the "extra
#  path": a downsampled whole-image GLOBAL-CONTEXT pass (ADD_GLOBAL_CONTEXT_PASS)
#  whose detections are merged into the tiled detections BEFORE NMS. Each run
#  shows CountGD++ exactly ONE example plant (the anchor itself) AND the text
#  prompt, in the same forward pass, for every tile and for the global pass. The
#  crop is given to CountGD++ as an EXTERNAL exemplar image (no strip, no
#  padding: the exemplar box is the whole crop).
#
#  TEXT + EXEMPLAR: the caption is "<TEXT_PROMPT> . ". CountGD++ inserts the
#  visual exemplar token right after the text tokens, before the ".", so the
#  positive prompt is "rumex obtusifolius <exemplar>" (official CountGD++ way of
#  combining both prompts).
#
#  EXTRA PATH (SAM3 notebook 1_exemplar_text_tiling_extra_path, CELL 16-17): the
#  whole image is downscaled by GLOBAL_DOWNSCALE (2) and sent through the SAME
#  exemplar + text + filter pipeline as one big "tile". Its detections are
#  rescaled to full resolution and appended to the SAME pre-NMS pool, tagged
#  tile_id = -1, so NMS and the whole evaluation need no special-casing. The
#  exemplar crop of the global pass is scaled by the TOTAL factor applied to the
#  global image (downscale x CountGD++ resize), so the plant has the same
#  apparent size as the plants in the global image (same idea as match_tile).
#
#  The pipeline is TWO-PHASE:
#
#    PHASE 1 - INFERENCE   (notebook CELL 11 + CELL 12 + CELL 19, GPU)
#        for every image x every anchor:
#            select the exemplar deterministically           (CELL 5)
#            crop it, exactly the GT box, no padding         (CELL 6)
#            resize it by the SAME factor as the tile        (CELL 11)
#            run CountGD++ on every tile with the exemplar AND
#              the text, ONE tile per forward pass, at
#              CONFIDENCE_THRESHOLD = 0.30                    (CELL 12)
#            clip to the tile + plausibility filter          (CELL 10)
#            + the global-context pass (same steps, whole image / 2)
#            save the PRE-NMS detections to NPZ
#        A RAM guard (MEM_STOP_THRESHOLD_PCT = 70 %) stops a shard cleanly
#        before the node runs out of host memory; resubmitting resumes.
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 13 ... CELL 22, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40,
#        evaluate in both modes (all_gt / held_out) at CONFIDENCE_THRESHOLD =
#        0.30, and write run-level / image-level / experiment-level / pooled-AP
#        CSVs, the confusion matrices (CSV + PNG), the qualitative
#        GT-vs-prediction figures (CELL 22) and the exemplar preview (CELL 18).
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
import hashlib
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
    "experiment_name", "image_ID", "anchor_idx", "Prompt_ID", "Prompt_Type", "text_prompt",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt", "n_detections_pre_nms", "n_tiles", "used_global_pass",
    "image_width", "image_height", "npz_file", "inference_seconds",
]

# ---- notebook CELL 17: tile_id of the detections of the global-context pass ----
GLOBAL_PASS_TILE_ID = -1

# ---- notebook CELL 3 -------------------------------------------------------
EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
# held_out : the GT plant used as the visual prompt is IGNORED, and so are the
#            predictions that fall on it.

# ---- notebook CELL 20 ------------------------------------------------------
# count_abs_error = |predicted count - GT count| at the operating point (the
# CountGD++ notebook adds it because CountGD++ is a counting model).
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2",
                  "count_abs_error"]

# ---- notebook CELL 14 ------------------------------------------------------
STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2

# ---- notebook CELL 12 ------------------------------------------------------
DOT_TOKEN_ID = 1012   # BERT id of "." - separates the positive prompt from negatives

# ---- notebook CELL 4 -------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # the repo's own normalisation
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './1_exemplars_text_tiling_extra_path_run_countGDpp.sh download' on a "
    "LOGIN node, then './1_exemplars_text_tiling_extra_path_run_countGDpp.sh build' inside the "
    "container, which installs it into $COUNTGD_PYEXTRA."
)

COUNTGD_HINT = (
    "Run './1_exemplars_text_tiling_extra_path_run_countGDpp.sh download' on a LOGIN node (repo + "
    "checkpoint + BERT), then './1_exemplars_text_tiling_extra_path_run_countGDpp.sh build' inside the "
    "container (packages + BERT folder + CUDA op). See 1_exemplars_text_tiling_extra_path_HOW_TO_RUN_countGDpp.md."
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
        description="1_exemplars_text_tiling_extra_path_countGDpp - CountGD++ single-exemplar + text "
                    "prompted Rumex detection, tiling ON (1536/384) + downsampled global-context "
                    "pass (inference + offline evaluation).",
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
    g.add_argument("--experiment-name", default="1_exemplars_text_tiling_extra_path_countGDpp",
                   help="EXPERIMENT_NAME. Written into every CSV row, every NPZ and "
                        "into the deterministic exemplar seed.")
    g.add_argument("--n-exemplars", type=int, default=1,
                   help="N_EXEMPLARS. The CountGD++ notebook implements the 1-exemplar "
                        "experiment only (CELL 3 assert), so 1 is the only accepted value.")
    tiling = g.add_mutually_exclusive_group()
    tiling.add_argument("--tiling", dest="use_tiling", action="store_true", default=True,
                        help="USE_TILING = True (default): overlapping tiles.")
    tiling.add_argument("--no-tiling", dest="use_tiling", action="store_false",
                        help="USE_TILING = False: the whole image is one single tile.")

    # ------------------------------ tiling ----------------------------------
    g = p.add_argument_group("tiling (CELL 3 / CELL 9)")
    g.add_argument("--tile-size", type=int, default=1536,
                   help="TILE_SIZE (the extra-path notebook uses 1536 instead of 1000).")
    g.add_argument("--overlap", type=int, default=384,
                   help="OVERLAP: every plant whose width AND height are <= 384 px lies "
                        "completely inside at least one tile; larger plants are seen whole by "
                        "the global-context pass.")
    g.add_argument("--no-cache-tiles", dest="cache_tiles_in_memory", action="store_false",
                   default=True,
                   help="CACHE_TILES_IN_MEMORY = False: crop tiles on demand (less RAM).")

    # ------------------- extra path: global-context pass --------------------
    g = p.add_argument_group("extra path: global-context pass (CELL 3 / CELL 16 / CELL 17)")
    gp = g.add_mutually_exclusive_group()
    gp.add_argument("--global-pass", dest="use_global_pass", action="store_true", default=True,
                    help="ADD_GLOBAL_CONTEXT_PASS = True (default): one extra pass over the "
                         "whole image, downscaled by --global-downscale, merged before NMS.")
    gp.add_argument("--no-global-pass", dest="use_global_pass", action="store_false",
                    help="ADD_GLOBAL_CONTEXT_PASS = False: tiles only.")
    g.add_argument("--global-downscale", type=int, default=2,
                   help="GLOBAL_DOWNSCALE: shrink factor of the global-pass image.")

    # ------------------------------ safety net ------------------------------
    g = p.add_argument_group("safety net (CELL 3 / CELL 19)")
    g.add_argument("--mem-stop-threshold-pct", type=float, default=70.0,
                   help="MEM_STOP_THRESHOLD_PCT: a shard stops CLEANLY (manifest closed, "
                        "completed runs kept) when system RAM exceeds this percentage; "
                        "resubmitting resumes. Needs psutil; disabled with a warning without it.")

    # ------------------------- CountGD++ inference --------------------------
    g = p.add_argument_group("countgd++ inference (CELL 3 / CELL 4 / CELL 11 / CELL 12)")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="CONFIDENCE_THRESHOLD used INSIDE the CountGD++ post-processing "
                        "(replaces the repo default 0.23). CountGD++ is executed EXACTLY ONCE "
                        "per (image, anchor) at this score.")
    g.add_argument("--batch-size", type=int, default=1,
                   help="BATCH_SIZE. FIXED at 1: CountGD++ inserts the exemplar tokens for "
                        "sample 0 only (add_exemplar_tokens is called with labels=[0]), so its "
                        "code cannot run several tiles in one forward pass.")
    g.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="float32 = USE_FP16 False (the notebook; CountGD++ is released and "
                        "evaluated in fp32). float16 = USE_FP16 True (autocast, not validated).")
    g.add_argument("--device", default=None, help="'cuda', 'cuda:0', 'cpu'. Default: auto.")
    g.add_argument("--text-prompt", default=" rumex obtusifolius ",
                   help="TEXT_PROMPT, sent together with the exemplar in the same forward "
                        "pass (caption = TEXT_PROMPT + ' . '). Must not be empty: an empty text "
                        "is the exemplar-only experiment 1_exemplars_tiling_countGDpp.")
    g.add_argument("--exemplar-scale-mode", choices=["match_tile", "model_default"],
                   default="match_tile",
                   help="EXEMPLAR_SCALE_MODE. match_tile: the crop is resized by EXACTLY the "
                        "factor applied to the tile (800/1536 here), so the plant has the same "
                        "apparent size in the exemplar and in the tile. model_default: the crop "
                        "goes through the standard 800 px resize like any image.")
    g.add_argument("--model-short-side", type=int, default=800,
                   help="MODEL_SHORT_SIDE: official resize, short side -> 800 px.")
    g.add_argument("--model-max-size", type=int, default=1333,
                   help="MODEL_MAX_SIZE: official resize, long side capped at 1333 px.")
    g.add_argument("--seed", type=int, default=42,
                   help="SEED: same seed as the official inference scripts.")

    # --------------------------- filters ------------------------------------
    g = p.add_argument_group("plausibility filter (CELL 3 / CELL 10)")
    g.add_argument("--max-area-fraction", type=float, default=0.80, help="MAX_AREA_FRACTION")
    g.add_argument("--edge-margin", type=int, default=5, help="EDGE_MARGIN")

    # -------------------------- evaluation ----------------------------------
    g = p.add_argument_group("evaluation (CELL 3 / CELL 14 / CELL 15)")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")
    g.add_argument("--prompt-ignore-iou", type=float, default=0.50,
                   help="PROMPT_IGNORE_IOU: held_out mode ignore rule.")
    g.add_argument("--nms-iou-threshold", type=float, default=0.40,
                   help="NMS_IOU_THRESHOLD, applied OFFLINE in PHASE 2 (fixed, no sweep).")
    g.add_argument("--operating-confidence", type=float, default=0.30,
                   help="The operating point of precision / recall / F1 / IoU1 / IoU2 / "
                        "count_abs_error. The notebook keeps it EQUAL to --threshold (0.30), so "
                        "the inference threshold and the operating point are the same single "
                        "value and no confidence x NMS sweep is performed.")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot (CELL 16 / CELL 18 / CELL 22)")
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

    if args.n_exemplars != 1:
        p.error("--n-exemplars must be 1: the CountGD++ notebook implements the 1-exemplar "
                "experiment only (one exemplar crop = one exemplar image).")
    if args.batch_size != 1:
        p.error("--batch-size must be 1: CountGD++ inserts the exemplar tokens for sample 0 "
                "only, so its code does not support several tiles per forward pass "
                "(notebook CELL 3).")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.use_tiling and args.overlap >= args.tile_size:
        p.error("--overlap must be smaller than --tile-size")
    if not args.text_prompt.strip():
        p.error("--text-prompt is empty: exemplar-only prompting is the experiment "
                "1_exemplars_tiling_countGDpp, not this one.")

    if args.checkpoint is None:
        args.checkpoint = args.countgd_repo / "checkpoints" / "countgd_plusplus.pth"

    if args.global_downscale < 1:
        p.error("--global-downscale must be >= 1")

    # PROMPT_TYPE (CELL 3): exemplar(s) + text
    args.prompt_type = ("multiple" if args.n_exemplars > 1 else "single") + "+text"
    return args


# =============================================================================
#  OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections (NPZ, one file per image x anchor)
#                          + one runs_manifest_<exp>_shard<i>.csv per shard
#     metrics/             run / image / experiment / dataset level CSVs
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
#  CELL 5 - STABLE REPRODUCIBILITY HELPERS
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
            n_exemplars - how many prompts in total (1 here)
            image_id    - "<archive>/<flight>/<image name>"
    Output: list of GT indices, ANCHOR FIRST, then the randomly sampled others.
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
    return "+".join(str(int(i)) for i in exemplar_indices)


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
#  CELL 9 - TILES: GENERATED ONCE PER IMAGE, REUSED BY EVERY ANCHOR
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
#  CELL 10 - PLAUSIBILITY FILTER
# =============================================================================
# Compared with the SAM3 experiments there is NO exemplar strip (the exemplar is
# given to CountGD++ as its own image), so there is no target-region filter:
# every prediction lies in the tile. There is also NO mask-fill-ratio criterion:
# CountGD++ outputs boxes only.
# KEPT (same thresholds): tiny boxes and boxes covering > 80 % of the tile.
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
#  CELL 11 - COUNTGD++ INPUTS: TILE + EXTERNAL EXEMPLAR CROP
# =============================================================================
# CountGD++ receives TWO images per forward pass:
#   1) the image to count in            -> the TILE
#   2) the image the exemplar is taken from + the exemplar box inside it
# Here image 2 is the exemplar CROP itself (cut exactly at the GT box, no
# padding) and the exemplar box is the WHOLE crop: [0, 0, crop_w, crop_h].
# CountGD++ RoI-aligns that box on the crop's backbone features to obtain the
# visual exemplar token (that is how the official code handles "external
# exemplars" coming from another image).
#
# SCALE: the tile is resized the way the official code does (short side 800,
# long side <= 1333): a 1536 x 1536 tile -> 800 x 800, factor 0.52. With
# EXEMPLAR_SCALE_MODE = "match_tile" the crop is resized by the SAME factor, so
# a plant looks equally large in the exemplar image and in the tile.
#
# The three helpers below are pure geometry (no torch): PHASE 2 re-uses them to
# draw the exemplar exactly as it was fed to the model (CELL 18 preview).
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


def tile_scale_factor(tile_w: int, tile_h: int, short_side: int,
                      max_size: Optional[int]) -> Tuple[float, float]:
    """(sx, sy) applied to a tile by the official resize (1536 x 1536 -> 0.52, 0.52)."""
    new_h, new_w = countgd_resize_hw(tile_w, tile_h, short_side, max_size)
    return new_w / tile_w, new_h / tile_h


def exemplar_fed_size(crop_w: int, crop_h: int, tile_scale: Tuple[float, float],
                      mode: str, short_side: int, max_size: Optional[int]) -> Tuple[int, int]:
    """
    Size (new_w, new_h) of the exemplar crop as it is fed to CountGD++.
      match_tile    -> resized by EXACTLY the factor applied to the tile
      model_default -> the standard 800 px resize, like any image
                       (a 41 x 53 px plant would be upscaled ~15x)
    """
    if mode == "match_tile":
        return (max(1, int(round(crop_w * tile_scale[0]))),
                max(1, int(round(crop_h * tile_scale[1]))))
    if mode == "model_default":
        new_h, new_w = countgd_resize_hw(crop_w, crop_h, short_side, max_size)
        return new_w, new_h
    raise ValueError(f"Unknown EXEMPLAR_SCALE_MODE: {mode}")


def global_pass_geometry(img_w: int, img_h: int, downscale: int, short_side: int,
                         max_size: Optional[int]):
    """
    Geometry of the global-context pass (notebook CELL 16).
    Output: small_w, small_h - size of the downscaled whole image
            total_scale      - (sx, sy) from ORIGINAL pixels to what CountGD++ sees:
                               (1 / downscale) x the official resize of the small
                               image. The exemplar crop is resized by this factor
                               (match_tile), so the exemplar plant has the same
                               apparent size as the plants in the global image.
    """
    small_w, small_h = max(1, img_w // downscale), max(1, img_h // downscale)
    sx, sy = tile_scale_factor(small_w, small_h, short_side, max_size)
    return small_w, small_h, (sx * small_w / img_w, sy * small_h / img_h)


def cxcywh_norm_to_xyxy(boxes, w: float, h: float) -> np.ndarray:
    """Normalised (cx, cy, bw, bh) -> pixel [x1, y1, x2, y2] in a w x h image."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    cx, cy, bw, bh = boxes[:, 0] * w, boxes[:, 1] * h, boxes[:, 2] * w, boxes[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


# =============================================================================
#  CELL 4 + CELL 12 - COUNTGD++ MODEL AND TILE INFERENCE
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
        # so that the tile and the exemplar crop get the SAME scale factor.
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

    # ---------------------------- CELL 11 ------------------------------------
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

    def prepare_exemplar_tensor(self, crop_img: Image.Image, tile_scale):
        """
        Turn the exemplar crop into CountGD++'s exemplar inputs.
        Input : crop_img   - PIL crop of the GT plant (no padding)
                tile_scale - (sx, sy) applied to the tile (from prepare_tile_tensor)
        Output: image tensor (3, h, w) of the (resized) crop
                box tensor (1, 4) = [0, 0, w, h] -> the exemplar covers the whole crop
        """
        torch = self.torch
        a = self.args
        new_w, new_h = exemplar_fed_size(crop_img.width, crop_img.height, tile_scale,
                                         a.exemplar_scale_mode, a.model_short_side,
                                         a.model_max_size)
        resized = crop_img.resize((new_w, new_h), Image.BILINEAR)
        tensor, _ = self.normalize(resized, None)
        box = torch.tensor([[0.0, 0.0, float(new_w), float(new_h)]], dtype=torch.float32)
        return tensor, box

    # ---------------------------- CELL 12 ------------------------------------
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

    def infer_tile(self, tile_img: Image.Image, exemplar_img_t, exemplar_box_t,
                   tile_t=None):
        """
        Run CountGD++ on ONE tile with ONE external exemplar.
        Input : tile_img       - PIL tile (original resolution)
                exemplar_img_t - (3,h,w) tensor of the exemplar crop
                exemplar_box_t - (1,4) tensor [0,0,w,h] (the whole crop)
        Output: boxes (K,4) in TILE pixel coordinates, scores (K,)
        """
        torch = self.torch
        if tile_t is None:
            tile_t, _ = self.prepare_tile_tensor(tile_img)

        # official format: "<positive text> . ". CountGD++ inserts the exemplar token
        # right after the text tokens (before the "."), so the positive prompt is
        # "<TEXT_PROMPT> <exemplar>". BERT is uncased and ignores surrounding spaces.
        caption = self.args.text_prompt + " . "

        autocast = (torch.autocast("cuda", dtype=torch.float16) if self.use_fp16
                    else contextlib.nullcontext())
        with torch.inference_mode():
            with autocast:
                out = self.model(
                    self.nested_tensor_from_tensor_list([tile_t.to(self.device)]),          # image to count in
                    self.nested_tensor_from_tensor_list([exemplar_img_t.to(self.device)]),  # exemplar image (= crop)
                    [exemplar_box_t.to(self.device)],                                       # exemplar box in it
                    [],                                                                     # no negative exemplar images
                    [],                                                                     # no negative exemplar boxes
                    captions=[caption],
                )
            boxes_n, scores = self.postprocess(out, self.args.threshold)
        del out
        # pred_boxes are normalised to the (unpadded) resized tile; the resize keeps
        # the aspect ratio, so multiplying by the ORIGINAL tile size maps them back.
        return cxcywh_norm_to_xyxy(boxes_n, tile_img.width, tile_img.height), scores


# =============================================================================
#  CELL 16 - GLOBAL-CONTEXT PASS (the "extra path", ADD_GLOBAL_CONTEXT_PASS)
# =============================================================================
# An EXTRA single pass over the whole image, downscaled by GLOBAL_DOWNSCALE, run
# through the exact same exemplar + text + CountGD++ + filter pipeline as a tile
# (it IS treated as one big "tile" - no cropping). Its detections are rescaled
# back to full-resolution coordinates so the caller can append them to the tiled
# pre-NMS pool.
#
# CountGD++ resizes every input to short side 800 (long side <= 1333), so an
# 8192 x 5460 image becomes 4096 x 2730 here and then 1200 x 800 inside the model.
# The exemplar crop is resized by that TOTAL factor (global_pass_geometry), so the
# exemplar plant has the same apparent size as the plants in the global image.
# =============================================================================

def run_global_context_pass(runner: CountGDRunner, full_image: Image.Image,
                            exemplar_crop: Image.Image, args):
    """
    Input : full_image    - the open, full-resolution PIL image
            exemplar_crop - PIL crop of the anchor plant (same one used for tiles)
    Output: boxes (N,4) in ORIGINAL-IMAGE coordinates and scores (N,), after the
            clipping + plausibility filter (applied in global-image coordinates,
            exactly like the notebook). Empty arrays if nothing is detected.
    """
    img_w, img_h = full_image.size
    small_w, small_h, total_scale = global_pass_geometry(
        img_w, img_h, args.global_downscale, args.model_short_side, args.model_max_size)
    small_img = full_image.resize((small_w, small_h), Image.BILINEAR)

    small_t, _ = runner.prepare_tile_tensor(small_img)
    ex_img_t, ex_box_t = runner.prepare_exemplar_tensor(exemplar_crop, total_scale)

    boxes, scores = runner.infer_tile(small_img, ex_img_t, ex_box_t, tile_t=small_t)
    boxes = clip_boxes_to_tile(boxes, small_w, small_h)
    boxes, scores = filter_implausible_boxes(boxes, scores, small_w, small_h,
                                             args.max_area_fraction, args.edge_margin)

    # small_img coords -> original image coords (same single factor as the notebook)
    scale = img_w / small_w
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) * scale

    small_img.close()
    del small_t, ex_img_t, ex_box_t
    return boxes, np.asarray(scores, dtype=np.float32).reshape(-1)


# =============================================================================
#  CELL 17 - RUN ONE ANCHOR OVER ALL TILES (+ the global-context pass)
# =============================================================================

def run_anchor_over_tiles(runner: CountGDRunner, tile_cache: List[dict],
                          full_image: Image.Image, exemplar_crop: Image.Image, args):
    """
    Run CountGD++ for ONE exemplar over ALL cached tiles (notebook CELL 12,
    run_exemplar_over_tiles) and - if ADD_GLOBAL_CONTEXT_PASS is on - ALSO run
    the global-context pass and append its detections to the SAME pre-NMS pool,
    tagged tile_id = -1 and tile_boxes = the whole image extent.

    Input : tile_cache    - list of tile dicts (CELL 9), built once per image
            full_image    - the open PIL image (used only if tiles are not cached,
                            and by the global-context pass)
            exemplar_crop - PIL crop of the anchor plant (the visual prompt)
    Output: (dict, used_global_pass)
            dict of numpy arrays, all in ORIGINAL-IMAGE coordinates:
            boxes (N,4), scores (N,), tile_id (N,), tile_boxes (N,4)
            These are the PRE-NMS detections (no confidence filtering beyond 0.30,
            no NMS) that get written to disk for the offline evaluation.
            used_global_pass = True when the global pass contributed at least one
            detection (same definition as the notebook).

    The exemplar tensors do not depend on the tile content, only on the tile size,
    so they are prepared once per tile size and re-used.
    """
    all_boxes, all_scores, all_tids, all_tboxes = [], [], [], []
    exemplar_cache = {}   # (tile_w, tile_h) -> exemplar tensors at that tile's scale

    for tile in tile_cache:
        tile_img = get_tile_image(tile, full_image)
        tw, th = tile_img.size
        tile_t, tile_scale = runner.prepare_tile_tensor(tile_img)
        if (tw, th) not in exemplar_cache:
            exemplar_cache[(tw, th)] = runner.prepare_exemplar_tensor(exemplar_crop, tile_scale)
        ex_img_t, ex_box_t = exemplar_cache[(tw, th)]

        boxes, scores = runner.infer_tile(tile_img, ex_img_t, ex_box_t, tile_t=tile_t)
        # 1) clip to the tile, 2) plausibility filter (same thresholds as SAM3)
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

    # ---- CELL 17: the global-context pass joins the SAME pre-NMS pool -----------
    used_global_pass = False
    if args.use_global_pass:
        g_boxes, g_scores = run_global_context_pass(runner, full_image, exemplar_crop, args)
        used_global_pass = len(g_scores) > 0
        for b, s in zip(g_boxes, g_scores):
            all_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            all_scores.append(float(s))
            all_tids.append(GLOBAL_PASS_TILE_ID)          # sentinel: global-pass detection
            all_tboxes.append([0, 0, full_image.width, full_image.height])

    return {
        "boxes": np.array(all_boxes, dtype=np.float32).reshape(-1, 4),
        "scores": np.array(all_scores, dtype=np.float32).reshape(-1),
        "tile_id": np.array(all_tids, dtype=np.int32).reshape(-1),
        "tile_boxes": np.array(all_tboxes, dtype=np.int32).reshape(-1, 4),
    }, used_global_pass


# =============================================================================
#  PRE-NMS DETECTION STORAGE
# =============================================================================
# For every run (= one image x one anchor) we store the detections AFTER
#   CountGD++ inference at 0.30 -> clipping + plausibility filtering
#   -> conversion to original-image coordinates
# but BEFORE
#   any operating confidence threshold and BEFORE NMS.
#
# That is exactly the state needed to replay any (confidence, NMS IoU) pair
# offline without ever running CountGD++ again.
# =============================================================================

def run_npz_path(raw_detections_dir: Path, image_id: str, anchor_idx: int) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one run."""
    return raw_detections_dir / f"{safe_filename(image_id)}__anchor{int(anchor_idx):03d}.npz"


def save_run_detections(raw_detections_dir: Path, experiment_name: str, image_id: str,
                        anchor_idx: int, detections: dict, gt_boxes: np.ndarray,
                        prompt_indices: Sequence[int], image_size: Tuple[int, int],
                        archive: str, flight: str, class_id: int, text_prompt: str) -> Path:
    """
    Write one run's pre-NMS detections to NPZ. The file is self-contained: it also
    stores the GT boxes, the prompt indices and the text prompt, so the whole
    offline evaluation can run without re-opening images or label files.
    """
    path = run_npz_path(raw_detections_dir, image_id, anchor_idx)
    np.savez_compressed(
        path,
        experiment_name=np.array(experiment_name),
        image_ID=np.array(image_id),
        anchor_idx=np.array(int(anchor_idx)),
        text_prompt=np.array(text_prompt),
        prompt_indices=np.array(prompt_indices, dtype=np.int32),
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
    """Read one run NPZ back into a plain python dict."""
    with np.load(path, allow_pickle=False) as z:
        run = {
            "image_ID": str(z["image_ID"]),
            "anchor_idx": int(z["anchor_idx"]),
            "text_prompt": str(z["text_prompt"]),
            "prompt_indices": z["prompt_indices"].astype(int),
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

def load_done_runs(paths: Paths, experiment_name: str, text_prompt: str) -> set:
    """
    Read every shard manifest and return {(image_ID, anchor_idx)} of the runs that
    are already finished. Every shard reads ALL manifests, so a resubmission after
    the walltime never repeats work, even if the shard assignment changed because
    NUM_GPUS was different.

    Refuses to continue when the manifests already hold runs made with ANOTHER text
    prompt (notebook CELL 16): runs of two different prompts must never be mixed in
    one results folder.
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
                    try:
                        done.add((row["image_ID"], int(row["anchor_idx"])))
                    except (KeyError, ValueError, TypeError):
                        continue            # ignore a half-written trailing row
        except OSError:
            continue
    if other_prompts:
        raise RuntimeError(
            f"{paths.raw_detections} already holds results for another text prompt "
            f"{sorted(other_prompts)}. Use a new EXPERIMENT_NAME and OUTPUT_DIR for "
            f"{text_prompt!r} (or delete the old results folder) so the runs are not mixed.")
    return done


# =============================================================================
#  CELL 17 + CELL 19 - MAIN GPU INFERENCE LOOP  (PHASE 1)
# =============================================================================
# The notebook ran CELL 17 (load the image, its GT, the exemplar, the tiles) and
# CELL 19 (the run) for ONE image and ONE anchor. Here:
#
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE
#     read its GT boxes ONCE
#     build the overlapping tiles ONCE (cached on CPU)
#     FOR EACH anchor (= every GT box, once):
#         select the exemplar deterministically
#         crop it from the already-open image (exactly the GT box)
#         run CountGD++ over the cached tiles, one tile per pass (threshold 0.30)
#         + the global-context pass
#         save the PRE-NMS detections (NPZ)
#     release the image and the tile cache
#
# NO NMS and NO metric computation happens here - that is all done offline in
# PHASE 2. The loop is resumable: finished runs are listed in the shard manifests.
#
# RAM GUARD (MEM_STOP_THRESHOLD_PCT, notebook CELL 19): 1536 px tiles + the global
# pass use more host RAM. Before every image and every anchor the system RAM is
# checked; above the threshold the shard stops CLEANLY (completed runs are saved,
# the manifest is closed) and exits with code 3, so the job is reported as
# incomplete. Resubmitting resumes from the manifests.
# =============================================================================

EXIT_STOPPED_BY_RAM = 3


def run_inference(args, paths: Paths, records: List[ImageRecord]) -> bool:
    """PHASE 1 for this shard. Returns True when the RAM guard stopped it early."""
    exp = args.experiment_name

    # ---- RAM guard -------------------------------------------------------------
    try:
        import psutil
    except Exception as exc:
        psutil = None
        print(f"WARNING: psutil unavailable ({exc}) -> the RAM guard is DISABLED.")
    stopped_by_ram = False

    # ---- resume support (+ the text-prompt guard, checked even with --no-resume) ----
    done_runs: set = load_done_runs(paths, exp, args.text_prompt)
    if args.no_resume:
        done_runs = set()
    else:
        print(f"Resuming: {len(done_runs)} run(s) already finished for {exp} "
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

    runner = CountGDRunner(args)
    torch = runner.torch
    device_is_cuda = runner.device.startswith("cuda")

    start_time = time.time()
    n_new_runs = 0
    image_times: List[float] = []
    n_total_images = len(records)

    for img_idx, rec in enumerate(records, start=1):
        image_t0 = time.time()
        image_id = rec.image_id

        mem_pct = psutil.virtual_memory().percent if psutil is not None else 0.0
        if mem_pct > args.mem_stop_threshold_pct:
            print(f"RAM at {mem_pct:.0f}% (threshold {args.mem_stop_threshold_pct:.0f}%) -- "
                  f"stopping cleanly before {image_id} to avoid a hard crash. "
                  f"Resubmit to resume from where it stopped.")
            stopped_by_ram = True
            break

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

            mem_pct = psutil.virtual_memory().percent if psutil is not None else 0.0
            if mem_pct > args.mem_stop_threshold_pct:
                print(f"RAM at {mem_pct:.0f}% mid-image at {image_id} anchor={anchor_idx} -- "
                      f"stopping cleanly. Completed anchors are already saved; resubmit to resume.")
                stopped_by_ram = True
                break
            run_t0 = time.time()

            # deterministic prompt selection (SHA-256 based, see CELL 5);
            # with N_EXEMPLARS = 1 this is [anchor_idx]
            exemplar_indices = select_exemplar_indices(n_gt, anchor_idx, args.n_exemplars,
                                                       image_id, exp)
            prompt_id = format_prompt_id(exemplar_indices)
            exemplar_crop = safe_crop(image, gt_boxes[anchor_idx])   # exactly the GT box

            # exemplar + text prompt in the same forward pass (args.text_prompt),
            # over every tile AND the global-context pass
            detections, used_global_pass = run_anchor_over_tiles(
                runner, tile_cache, image, exemplar_crop, args)

            npz_path = save_run_detections(
                paths.raw_detections, exp, image_id, anchor_idx, detections, gt_boxes,
                exemplar_indices, (img_w, img_h), rec.archive, rec.flight, rec.class_id,
                args.text_prompt)

            run_seconds = time.time() - run_t0
            manifest_writer.writerow({
                "experiment_name": exp,
                "image_ID": image_id,
                "anchor_idx": anchor_idx,
                "Prompt_ID": prompt_id,
                "Prompt_Type": args.prompt_type,
                "text_prompt": args.text_prompt,
                "archive": rec.archive,
                "flight": rec.flight,
                "source_class_id": rec.class_id,
                "n_gt": n_gt,
                "n_prompt_gt": len(exemplar_indices),
                "n_detections_pre_nms": int(len(detections["scores"])),
                "n_tiles": n_tiles,
                "used_global_pass": used_global_pass,
                "image_width": img_w,
                "image_height": img_h,
                "npz_file": npz_path.name,
                "inference_seconds": round(run_seconds, 2),
            })
            manifest_file.flush()
            n_new_runs += 1

            print(f"  [{exp}] shard{args.shard_index} run #{n_new_runs} | {image_id} | "
                  f"anchor={anchor_idx} ({anchor_idx + 1}/{n_anchors}) | prompt={prompt_id}+text | "
                  f"tiles={n_tiles} | global_pass={used_global_pass} | "
                  f"pre-NMS detections={len(detections['scores'])} | {run_seconds:.1f}s")

            del detections, exemplar_crop
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

        if stopped_by_ram:                 # the RAM guard fired mid-image
            break

        image_elapsed = time.time() - image_t0
        image_times.append(image_elapsed)
        avg_per_image = float(np.mean(image_times))
        eta = (n_total_images - img_idx) * avg_per_image
        rss = (f" | [MEM] RSS={psutil.Process().memory_info().rss / 1e9:.2f} GB"
               if psutil is not None else "")
        print(f"[{exp}] ({img_idx}/{n_total_images}) {image_id} done | "
              f"{n_gt} GT box(es) | {image_elapsed:.1f}s | avg/image={avg_per_image:.1f}s | "
              f"ETA={eta / 60:.1f} min ({eta / 3600:.2f} h){rss}")

    manifest_file.close()
    total_elapsed = time.time() - start_time
    if stopped_by_ram:
        print(f"\nInference STOPPED by the RAM guard for {exp} (shard {args.shard_index}): "
              f"{n_new_runs} new runs saved. Resubmit to resume.")
    else:
        print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new runs.")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Pre-NMS detections in: {paths.raw_detections}")
    return stopped_by_ram


# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================

def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading CountGD++ ---")
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
    passes = tiles + (1 if args.use_global_pass else 0)
    print(f"  sampled {len(sample)} image(s) of this shard -> {total_anchors} anchor runs "
          f"({per_archive})")
    print(f"  tiles per anchor run at 8192x5460: {tiles}"
          f"{' + 1 global-context pass' if args.use_global_pass else ''}")
    print(f"  => ~{total_anchors * passes} CountGD++ forward passes (1 image each) for those "
          f"{len(sample)} images")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{total_anchors} "
          f"(one per image x anchor)")


# =============================================================================
#  LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
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
    # only the runs of THIS experiment and THIS text prompt (notebook CELL 17)
    manifest = manifest[(manifest["experiment_name"] == experiment_name) &
                        (manifest["text_prompt"].astype(str) == text_prompt)].copy()
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
        runs.append(run)

    if n_missing:
        print(f"  WARNING: {n_missing} manifest row(s) point at a missing NPZ (skipped).")

    print(f"Loaded {len(runs)} runs "
          f"({manifest['image_ID'].nunique()} images) for {experiment_name}, "
          f"text prompt {text_prompt!r}.")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes over all runs:", int(sum(len(r['gt_boxes']) for r in runs)))
    return runs, manifest


# =============================================================================
#  CELL 13 - NMS (with tile provenance)
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
#  CELL 14 - EVALUATION CORE: all_gt AND held_out
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
#  CELL 15 - AP50 AND AP50:95 + OPERATING POINT
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


# =============================================================================
#  CELL 16 + CELL 18 + CELL 22 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs
#  PREDICTIONS (right)
# =============================================================================
#   yellow = GT boxes | red = predictions | lime (dashed) = exemplar prompt
#
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
#   with score >= CONFIDENCE_THRESHOLD; the exemplar (prompt) box is drawn dashed.
#
# For that same image and anchor, the CELL 18 exemplar preview is saved as well:
#   left   : all GT boxes with their index, the exemplar highlighted
#   right  : the exemplar crop as cut from the image (no padding) and the
#            exemplar image exactly as fed to CountGD++ (after resizing)
#
# The image is downscaled for DISPLAY only (PLOT_MAX_DISPLAY_DIM).
# =============================================================================

COLOR_GT, COLOR_PRED, COLOR_EXEMPLAR = "yellow", "red", "lime"


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


def _display_copy(image: Image.Image, max_display_dim: int):
    """Downscaled numpy copy of the image + factor mapping full-res boxes onto it."""
    w, h = image.size
    s = min(1.0, max_display_dim / max(w, h))
    disp_w, disp_h = max(1, int(w * s)), max(1, int(h * s))
    display = np.asarray(image.resize((disp_w, disp_h), Image.BILINEAR))
    return display, np.array([disp_w / w, disp_h / h, disp_w / w, disp_h / h], dtype=np.float32)


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


def plot_exemplar_preview(display, to_disp, gt_boxes, anchor_idx: int,
                          exemplar_crop: Image.Image, crop_as_fed: Image.Image,
                          title: str, png_path: Path, plt,
                          crop_as_fed_global: Optional[Image.Image] = None) -> None:
    """
    Notebook CELL 18 - check of the prompt (saved, never shown):
      left   : all GT boxes with their index, the anchor/exemplar highlighted
      right  : the exemplar crop as cut from the image (no padding), the
               exemplar image exactly as fed to CountGD++ for the tiles and, when
               the global-context pass is on, as fed for the global pass
    """
    panels = [(exemplar_crop, "crop (original pixels)"),
              (crop_as_fed, "as fed to CountGD++ (tiles)")]
    if crop_as_fed_global is not None:
        panels.append((crop_as_fed_global, "as fed to CountGD++ (global pass)"))

    fig = plt.figure(figsize=(22, 8.5))
    gs = fig.add_gridspec(len(panels), 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(display)
    ax.axis("off")
    gt_d = np.asarray(gt_boxes).reshape(-1, 4) * to_disp
    _draw_boxes(ax, gt_d, COLOR_GT, linewidth=1.0,
                labels=[str(i) for i in range(len(gt_d))], fontsize=5)
    _draw_boxes(ax, gt_d[[anchor_idx]], COLOR_EXEMPLAR, linewidth=3,
                labels=[f"exemplar ({anchor_idx})"], fontsize=9)
    ax.set_title("GT boxes (yellow, indexed) and the exemplar (lime)", fontsize=12)

    for row, (img, name) in enumerate(panels):
        cax = fig.add_subplot(gs[row, 1])
        cax.imshow(np.asarray(img))
        cax.set_title(f"{name}: {img.width} x {img.height} px", fontsize=10)
        cax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown


def plot_gt_vs_predictions(args, paths: Paths, runs, run_level_df, image_level_df,
                           image_paths: dict, plt, scope_label: str,
                           archive: Optional[str] = None) -> None:
    """One qualitative figure (+ its exemplar preview) for one scope."""
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
    prompt_id = format_prompt_id(run["prompt_indices"])

    # ---- open the image: display copy + the exemplar exactly as fed ----------
    if plot_image_id not in image_paths:
        print("Image file not found for", plot_image_id)
        return
    with Image.open(image_paths[plot_image_id]) as im:
        im = im.convert("RGB")
        w, h = im.size
        display, to_disp = _display_copy(im, args.plot_max_display_dim)
        exemplar_crop = safe_crop(im, gt_boxes_plot[anchor])   # exactly the GT box
    # what CountGD++ saw for the first tile (notebook CELL 17)
    if args.use_tiling:
        tx1, ty1, tx2, ty2 = tile_bboxes(w, h, args.tile_size, args.overlap)[0]
    else:
        tx1, ty1, tx2, ty2 = 0, 0, w, h
    tile_scale = tile_scale_factor(tx2 - tx1, ty2 - ty1,
                                   args.model_short_side, args.model_max_size)
    fed_w, fed_h = exemplar_fed_size(exemplar_crop.width, exemplar_crop.height, tile_scale,
                                     args.exemplar_scale_mode, args.model_short_side,
                                     args.model_max_size)
    crop_as_fed = exemplar_crop.resize((fed_w, fed_h), Image.BILINEAR)
    # ... and for the global-context pass (CELL 16): the TOTAL scale of the global image
    crop_as_fed_global, global_scale = None, None
    if args.use_global_pass:
        _, _, global_scale = global_pass_geometry(w, h, args.global_downscale,
                                                  args.model_short_side, args.model_max_size)
        gfed_w, gfed_h = exemplar_fed_size(exemplar_crop.width, exemplar_crop.height,
                                           global_scale, args.exemplar_scale_mode,
                                           args.model_short_side, args.model_max_size)
        crop_as_fed_global = exemplar_crop.resize((gfed_w, gfed_h), Image.BILINEAR)

    # ---- figure: GT (left) | predictions (right) -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5))
    for ax in axes:
        ax.imshow(display)
        ax.axis("off")

    _draw_boxes(axes[0], gt_boxes_plot * to_disp, color=COLOR_GT, linewidth=1.5)
    axes[0].set_title(f"Ground truth: {len(gt_boxes_plot)} Rumex boxes", fontsize=12)

    score_labels = [f"{v:.2f}" for v in pred_scores] if args.plot_show_scores else None
    _draw_boxes(axes[1], pred_boxes * to_disp, color=COLOR_PRED, linewidth=1.5,
                labels=score_labels)
    _draw_boxes(axes[1], exemplar_boxes * to_disp, color=COLOR_EXEMPLAR, linewidth=2.5,
                linestyle="--")
    axes[1].set_title(
        f"Predictions: {len(pred_boxes)} boxes | anchor (exemplar) = {anchor} "
        f"+ text {run['text_prompt']!r}\n"
        f"run AP50={run_row['AP50']:.3f}  P={run_row['precision']:.3f}  "
        f"R={run_row['recall']:.3f}  F1={run_row['F1']:.3f}  "
        f"TP={int(run_row['TP'])} FP={int(run_row['FP'])} FN={int(run_row['FN'])}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color=COLOR_GT, lw=2, label="ground truth"),
        Line2D([0], [0], color=COLOR_PRED, lw=2, label="prediction"),
        Line2D([0], [0], color=COLOR_EXEMPLAR, lw=2.5, linestyle="--", label="exemplar prompt"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=11,
               frameon=False)
    fig.suptitle(
        f"{exp} | {plot_image_id} | CountGD++ | mode={mode} | scope={scope_label} | "
        f"image-level AP50_mean={img_row['AP50_mean']:.3f} "
        f"(over {int(img_row['n_runs_valid_for_macro'])} anchors)\n"
        f"exemplar GT {prompt_id} (scale {args.exemplar_scale_mode}) + text "
        f"{run['text_prompt']!r} | tile={args.tile_size}px, overlap={args.overlap}px, "
        f"global pass={args.use_global_pass} (x1/{args.global_downscale}) | "
        f"conf={args.operating_confidence:.2f}, NMS IoU={args.nms_iou_threshold:.2f}\n"
        f"selection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    stem = f"{scope_label}_{safe_filename(plot_image_id)}_anchor{anchor:03d}"
    png_path = paths.plots / f"best_image_{stem}_{mode}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

    # ---- CELL 18: exemplar preview for the same image and anchor ---------------
    preview_path = paths.plots / f"exemplar_preview_{stem}.png"
    plot_exemplar_preview(
        display, to_disp, gt_boxes_plot, anchor, exemplar_crop, crop_as_fed,
        title=(f"{exp} | {plot_image_id}\n"
               f"exemplar = GT {prompt_id} (external exemplar image = the crop, "
               f"box = whole crop) | scale mode {args.exemplar_scale_mode} | "
               f"+ text {run['text_prompt']!r}"),
        png_path=preview_path, plt=plt, crop_as_fed_global=crop_as_fed_global)

    print(f"\n[{scope_label}] Selected image : {plot_image_id} "
          f"({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Shown anchor   : {anchor} (best run-level AP50 of this image), "
          f"text {run['text_prompt']!r}")
    print(f"[{scope_label}] Exemplar fed   : crop {exemplar_crop.width} x {exemplar_crop.height} px "
          f"-> {crop_as_fed.width} x {crop_as_fed.height} px "
          f"(tile scale {tile_scale[0]:.3f}, mode {args.exemplar_scale_mode})")
    if crop_as_fed_global is not None:
        print(f"[{scope_label}] Exemplar fed   : global pass -> {crop_as_fed_global.width} x "
              f"{crop_as_fed_global.height} px (total scale {global_scale[0]:.3f})")
    print(f"[{scope_label}] Figure saved   : {png_path}")
    print(f"[{scope_label}] Preview saved  : {preview_path}")


def make_qualitative_plots(args, paths: Paths, runs, run_level_df, image_level_df, plt) -> None:
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

    archives_present = sorted(str(a) for a in run_level_df["archive"].dropna().unique() if str(a))
    for archive in archives_present:
        plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df,
                               image_paths, plt, scope_label=archive, archive=archive)
    plot_gt_vs_predictions(args, paths, runs, run_level_df, image_level_df,
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

    runs, manifest = load_runs(paths, exp, args.text_prompt)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  RUN-LEVEL METRICS  (notebook CELL 20; one run = one image x one anchor)
    # =========================================================================
    #   AP50 / AP50_95 : all post-NMS predictions >= 0.30, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : only predictions >= CONFIDENCE_THRESHOLD
    #   count_abs_error : |evaluated predictions - evaluated GT| at the operating
    #                     point (ignored predictions are not counted)
    #   exemplar_redetected : was the anchor plant itself found? (all_gt matching
    #                     at the operating point, same value in both mode rows)
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

        op_keep = nms_run["scores"] >= conf
        match_all = match_one_to_one(nms_run["boxes"][op_keep], nms_run["scores"][op_keep],
                                     run["gt_boxes"], eval_iou)
        exemplar_redetected = bool(match_all["gt_match_pred"][run["anchor_idx"]] >= 0)

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
            n_counted = ev["n_pred"] - ev["n_ignored"]       # predictions that are evaluated

            run_rows.append({
                "experiment_name": exp,
                "image_ID": run["image_ID"],
                "archive": run["archive"],
                "flight": run["flight"],
                "anchor_idx": run["anchor_idx"],
                "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"],
                "text_prompt": run["text_prompt"],
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
                "count_abs_error": abs(n_counted - ev["n_eval_gt"]) if valid else nan,
                "TP": ev["TP"], "FP": ev["FP"], "FN": ev["FN"],
                "exemplar_redetected": exemplar_redetected,
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
              f"F1_mean={sub['F1'].mean():.4f}, "
              f"count_abs_error_mean={sub['count_abs_error'].mean():.2f}")
    redetected = run_level_df.loc[run_level_df["evaluation_mode"] == "all_gt",
                                  "exemplar_redetected"]
    print(f"  exemplar plant re-detected in {100.0 * redetected.mean():.1f}% of the runs "
          f"(all_gt matching at conf={conf:.2f})")

    # =========================================================================
    #  IMAGE-LEVEL METRICS
    # =========================================================================
    # All anchor runs of the same image are averaged into ONE value per image and
    # per evaluation mode. The std here is the spread BETWEEN the different
    # anchor selections of the SAME image, i.e. "how sensitive is the result to
    # which plant was used as the visual prompt?".
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
        ["AP50_mean", "precision_mean", "recall_mean", "F1_mean", "IoU1_mean", "IoU2_mean",
         "count_abs_error_mean"]
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
            "evaluation_mode": mode,
            "prompt_type": args.prompt_type,
            "text_prompt": args.text_prompt,
            "n_exemplars": args.n_exemplars,
            "use_tiling": args.use_tiling,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "add_global_context_pass": args.use_global_pass,
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
            "text_prompt": args.text_prompt,
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
                              "F1_mean", "count_abs_error_mean"]].to_string(index=False))

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
            "text_prompt": args.text_prompt,
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
    #  CELL 21 - DATASET-LEVEL CONFUSION MATRICES
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
    #  CELL 22 (+ CELL 18) - QUALITATIVE FIGURES (one per archive + one global)
    # =========================================================================
    if args.no_plots:
        print("\n--- qualitative figures skipped (--no-plots) ---")
    elif not _HAS_MPL:
        print("\n--- qualitative figures skipped (matplotlib unavailable) ---")
    else:
        print("\n--- qualitative GT-vs-prediction figures + exemplar previews ---")
        make_qualitative_plots(args, paths, runs, run_level_df, image_level_df, plt)

    # =========================================================================
    #  CELL 23 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY (CountGD++)")
    print("=" * 78)
    print(f"Prompts per run          : {args.n_exemplars} exemplar(s) + text ({args.prompt_type}), "
          f"external crop, no padding, scale {args.exemplar_scale_mode}")
    print(f"Text prompt              : {args.text_prompt!r} (same forward pass as the exemplar)")
    print(f"Tiling                   : {args.use_tiling}  (tile={args.tile_size}px, "
          f"overlap={args.overlap}px)")
    print(f"Global context pass      : {args.use_global_pass} "
          f"(downscale={args.global_downscale}, detections merged before NMS, tile_id=-1)")
    print(f"Model input              : short side {args.model_short_side}px "
          f"(max {args.model_max_size}px), one tile per forward pass")
    print(f"CountGD++ threshold      : {args.threshold} (executed once per image x anchor; "
          f"CountGD++ default is 0.23)")
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
            "recall_mean", "F1_mean", "F1_std", "IoU1_mean", "IoU2_mean",
            "count_abs_error_mean"]
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
                  "'Problem B' in 1_exemplars_text_tiling_extra_path_HOW_TO_RUN_countGDpp.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" 1_EXEMPLARS_TEXT_TILING_EXTRA_PATH COUNTGD++ PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {args.dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" countgd_repo   : {args.countgd_repo}")
    print(f" checkpoint     : {args.checkpoint}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" n_exemplars    : {args.n_exemplars} + text  ({args.prompt_type} prompt, external crop, "
          f"scale {args.exemplar_scale_mode})")
    print(f" text prompt    : {args.text_prompt!r}")
    print(f" tiling         : {args.use_tiling}  (tile={args.tile_size}, overlap={args.overlap})")
    print(f" global pass    : {args.use_global_pass}  (downscale={args.global_downscale})")
    print(f" RAM guard      : stop cleanly above {args.mem_stop_threshold_pct:.0f}% system RAM")
    print(f" model input    : short side {args.model_short_side}px (max {args.model_max_size}px)")
    print(f" threshold      : {args.threshold}  (single inference pass per image x anchor)")
    print(f" batch / dtype  : {args.batch_size} / {args.dtype}")
    print(f" operating pt   : confidence={args.operating_confidence}, "
          f"NMS IoU={args.nms_iou_threshold} (fixed, no sweep)")
    print(f" eval IoU       : {args.eval_iou_threshold}  "
          f"(prompt ignore IoU={args.prompt_ignore_iou})")
    print(f" supervision    : {'available' if has_supervision else 'MISSING'}")
    print("=" * 92)

    # ---------------- text-prompt guard ---------------------------------------
    # Refuse a different text prompt BEFORE anything is written (config snapshot,
    # manifest, NPZ), so an existing experiment is never touched by a mixed run.
    load_done_runs(paths, exp, args.text_prompt)

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
    stopped_by_ram = run_inference(args, paths, my_records)

    # ---------------- PHASE 2 -------------------------------------------------
    if not args.no_evaluate:
        run_evaluation(args, paths)

    # A shard stopped by the RAM guard exits non-zero, so the job is reported as
    # incomplete (FAIL email) and is simply resubmitted.
    if stopped_by_ram:
        sys.exit(EXIT_STOPPED_BY_RAM)


if __name__ == "__main__":
    main()
