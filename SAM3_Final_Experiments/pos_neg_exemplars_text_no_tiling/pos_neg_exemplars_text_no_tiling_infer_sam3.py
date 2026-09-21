#!/usr/bin/env python3
# =============================================================================
#  pos_neg_exemplars_text_no_tiling_infer_sam3.py
# =============================================================================
#  Cluster (CSCS) port of the Colab notebook  pos_neg_exemplars_text_no_tiling.ipynb
#
#  This experiment is the WHOLE-IMAGE, POSITIVE + NEGATIVE + TEXT variant:
#  USE_TILING = False. There are no tiles, no exemplar strip and no plausibility
#  filter. The whole UAV image is downscaled so that its longest side is
#  MAX_DIM = 1024, and ONE SAM3 forward pass per (image, anchor) receives, at the
#  same time and at their REAL LOCATION in the image:
#      - K_POSITIVES = 2 GT Rumex boxes of the SAME image as POSITIVE box prompts
#        (label 1): the anchor + 1 seeded random other GT box
#      - J_NEGATIVES = 3 automatically generated NON-Rumex boxes as NEGATIVE box
#        prompts (label 0), generated anew for every run (seeded)
#      - the text prompt TEXT_PROMPT = "Rumex obtusifolius"
#  The predicted boxes are scaled back to the original resolution immediately.
#
#  NEGATIVE EXEMPLARS (notebook CELL 11), generated automatically for every run:
#      GT Rumex boxes
#        -> candidate regions (sizes drawn from this image's own GT boxes)
#        -> remove candidates overlapping Rumex (0 overlap, gap >= NEG_MIN_GAP_PX)
#        -> prefer candidates close to Rumex (gap <= NEG_MAX_GAP_FACTOR x median
#           plant size), to avoid picking bare soil
#        -> select J_NEGATIVES spatially diverse ones (farthest-point sampling)
#        -> SAM3
#  NOTE: the generator uses ALL GT boxes of the image (that is what guarantees
#  zero overlap), i.e. label information of the evaluated plants.
#
#  The notebook is a TWO-PHASE pipeline and this file keeps that separation:
#
#    PHASE 1 - INFERENCE   (notebook CELL 14, GPU)
#        for every image x every anchor:
#            select the positives deterministically     (CELL 6)
#            generate the negatives deterministically   (CELL 11)
#            resize the whole image to MAX_DIM          (CELL 10)
#            run SAM3 once at CONFIDENCE_THRESHOLD=0.30 with
#                positive boxes + negative boxes + text (CELL 12)
#            rescale the boxes to full resolution       (CELL 12)
#            save the PRE-NMS detections + prompts to NPZ (CELL 13)
#        This phase is sharded: one process per GPU, round-robin over the
#        (deterministically sorted) image list. Each shard writes its own
#        manifest so the phase is crash-safe and resumable.
#
#    PHASE 2 - EVALUATION  (notebook CELL 15 ... CELL 26, no GPU)
#        load the NPZ files, apply offline NMS at NMS_IOU_THRESHOLD = 0.40,
#        evaluate in both modes (all_gt / held_out) at the operating point
#        CONFIDENCE_THRESHOLD = 0.30, and write run-level / image-level /
#        experiment-level / pooled-AP CSVs, the confusion matrices (CSV + PNG)
#        and the qualitative GT-vs-prediction figures.
#        Runs in ONE process, after every shard has finished. It never touches
#        SAM3, so it can be repeated as often as you like from the cached NPZs.
#
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


# UAV images are very large; disable PIL's decompression-bomb guard.
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
# archive / flight / source_class_id / sam_scale are cluster additions.
MANIFEST_COLUMNS = [
    "experiment_name", "image_ID", "anchor_idx", "Prompt_ID", "Prompt_Type", "text_prompt",
    "archive", "flight", "source_class_id",
    "n_gt", "n_prompt_gt", "n_negatives", "neg_max_gap_px", "n_detections_pre_nms",
    "image_width", "image_height", "sam_scale", "npz_file", "inference_seconds",
]

# ---- notebook CELL 3 -------------------------------------------------------
EVALUATION_MODES = ["all_gt", "held_out"]
# all_gt   : every GT box of the image is evaluated (classical evaluation).
# held_out : the K_POSITIVES GT instances used as positive prompts are IGNORED, and
#            so are the predictions that fall on them. Answers "how well does SAM3
#            find the REMAINING Rumex plants after being shown K examples?".
#            (Negatives are not GT boxes, so they do not change the evaluation.)

# ---- notebook CELL 21 ------------------------------------------------------
METRIC_COLUMNS = ["AP50", "AP50_95", "precision", "recall", "F1", "IoU1", "IoU2"]

# ---- notebook CELL 17 ------------------------------------------------------
STATUS_FP, STATUS_TP, STATUS_IGNORED = 0, 1, 2

SUPERVISION_HINT = (
    "the 'supervision' package is required for AP50 / AP50:95. Compute nodes "
    "have no internet: run './pos_neg_exemplars_text_no_tiling_run_sam3.sh download' "
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
        description="pos_neg_exemplars_text_no_tiling - SAM3 prompted with K positive GT "
                    "boxes + J generated negative boxes + a text prompt, whole image, NO "
                    "tiling (inference + offline evaluation). Cluster port of "
                    "pos_neg_exemplars_text_no_tiling.ipynb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------- paths -------------------------------------
    g = p.add_argument_group("paths")
    g.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                   help="Folder holding the archive folders (AGS_Multi_Rumex, AgsSpringRumex). "
                        "PHASE 2 uses it as well, for the qualitative figures (CELL 26).")
    g.add_argument("--output-dir", type=Path, required=True,
                   help="RESULTS_ROOT: raw_detections/, metrics/, confusion_matrices/, plots/.")
    g.add_argument("--archives", nargs="*", default=list(ARCHIVES.keys()),
                   help="Subset of archives to run on. Default: both.")

    # ------------------------ experiment identity ---------------------------
    g = p.add_argument_group("experiment identity (CELL 3)")
    g.add_argument("--experiment-name", default="pos_neg_exemplars_text_no_tiling",
                   help="EXPERIMENT_NAME. Written into every CSV row, every NPZ and into "
                        "the deterministic positive / negative seeds.")
    g.add_argument("--k-positives", type=int, default=2,
                   help="K_POSITIVES = N_EXEMPLARS: positive box prompts per run (the anchor "
                        "+ K-1 seeded random other GT boxes).")
    g.add_argument("--j-negatives", type=int, default=3,
                   help="J_NEGATIVES: negative box prompts per run (generated, non-Rumex, new "
                        "for every run).")
    g.add_argument("--text-prompt", default="Rumex obtusifolius",
                   help="TEXT_PROMPT sent together with the boxes (short phrase, SAM3 reads at "
                        "most 32 tokens). The same text is used for both archives.")

    # --------------------------- whole-image input --------------------------
    g = p.add_argument_group("whole-image input (CELL 3 / CELL 10)")
    g.add_argument("--max-dim", type=int, default=1024,
                   help="MAX_DIM: longest side of the downscaled copy fed to SAM3. The whole "
                        "image is sent in ONE pass; there is no tiling in this experiment.")

    # ----------------------- negative exemplar generator --------------------
    g = p.add_argument_group("negative exemplar generator (CELL 3 / CELL 11)")
    g.add_argument("--neg-num-candidates", type=int, default=2000,
                   help="NEG_NUM_CANDIDATES: random candidate boxes generated over the image.")
    g.add_argument("--neg-min-gap-px", type=float, default=10,
                   help="NEG_MIN_GAP_PX: a negative must stay at least this far (px, full-res) "
                        "from EVERY Rumex GT box -> zero overlap guaranteed.")
    g.add_argument("--neg-max-gap-factor", type=float, default=1.0,
                   help="NEG_MAX_GAP_FACTOR: 'close to Rumex' = gap <= factor x median plant "
                        "size (doubled automatically, up to 3 times, if too few candidates "
                        "qualify).")

    # --------------------------- SAM3 inference -----------------------------
    g = p.add_argument_group("sam3 inference (CELL 3 / CELL 5 / CELL 12)")
    g.add_argument("--model-id", default="facebook/sam3",
                   help="HF repo id OR a local snapshot directory.")
    g.add_argument("--threshold", type=float, default=0.30,
                   help="CONFIDENCE_THRESHOLD. SAM3 is executed EXACTLY ONCE per "
                        "(image, anchor) at this score. In this experiment it is ALSO the "
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
    g = p.add_argument_group("evaluation (CELL 3 / CELL 16 / CELL 19)")
    g.add_argument("--nms-iou-threshold", type=float, default=0.40,
                   help="NMS_IOU_THRESHOLD, applied OFFLINE in PHASE 2 (fixed, no sweep).")
    g.add_argument("--operating-confidence", type=float, default=0.30,
                   help="The operating point of precision / recall / F1 / IoU1 / IoU2. The "
                        "notebook keeps it EQUAL to --threshold (0.30), so the inference "
                        "threshold and the operating point are the same single value and no "
                        "confidence x NMS sweep is performed.")
    g.add_argument("--eval-iou-threshold", type=float, default=0.50,
                   help="EVAL_IOU_THRESHOLD: IoU needed for a prediction to count as TP.")
    g.add_argument("--prompt-ignore-iou", type=float, default=0.50,
                   help="PROMPT_IGNORE_IOU: held_out mode ignore rule. An unmatched "
                        "prediction whose best IoU with a PROMPT GT box is >= this value is "
                        "IGNORED (neither TP nor FP).")

    # ------------------------ qualitative plot ------------------------------
    g = p.add_argument_group("qualitative plot (CELL 3 / CELL 26)")
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
                   help="Skip CELL 26 entirely (it is the only PHASE 2 step that reopens the "
                        "original images).")

    # ---------------------------- runtime -----------------------------------
    g = p.add_argument_group("runtime")
    g.add_argument("--num-shards", type=int, default=1,
                   help="Split the image list across this many concurrent processes.")
    g.add_argument("--shard-index", type=int, default=0, help="0-based shard of this process.")
    g.add_argument("--limit-images", type=int, default=0,
                   help="Debug: process at most this many images (0 = no limit).")
    g.add_argument("--max-anchors-per-image", type=int, default=0,
                   help="Debug / cost control: use at most this many GT boxes as anchors per "
                        "image (0 = every GT box becomes an anchor once, as in the notebook).")
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

    if args.k_positives < 1:
        p.error("--k-positives must be >= 1")
    if args.j_negatives < 0:
        p.error("--j-negatives must be >= 0")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.max_dim < 64:
        p.error("--max-dim must be >= 64")

    # N_EXEMPLARS (CELL 3): positive exemplars = the GT boxes used as prompts
    args.n_exemplars = args.k_positives
    # PROMPT_TYPE (CELL 3)
    args.prompt_type = f"{args.k_positives}pos+{args.j_negatives}neg+text"
    # EXPERIMENT_DIR_NAME (CELL 3): file-system-safe version of the name, used ONLY
    # inside file names (a "/" would be read as a sub-folder)
    args.experiment_dir_name = args.experiment_name.replace("/", "-")
    # USE_TILING (CELL 3) - constant here, kept so it reaches experiment_summary.csv
    args.use_tiling = False
    args.keep_masks = False
    return args


# =============================================================================
#  CELL 4 - OUTPUT FOLDERS
# =============================================================================
#  <output-dir>/
#     raw_detections/      pre-NMS detections (NPZ, one file per image x anchor)
#                          + one runs_manifest_shard<i>.csv per shard
#     metrics/             run / image / experiment / dataset level CSVs
#     confusion_matrices/  CSV + PNG for all_gt and held_out
#     plots/               qualitative GT-vs-prediction figures (CELL 26)
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


def manifest_path(paths: Paths, experiment_dir_name: str, shard_index: int) -> Path:
    """
    One manifest PER SHARD. The notebook had a single runs_manifest.csv, which is
    not safe when four processes append to it at the same time; the resume step
    simply reads all of them back (see load_done_runs).
    """
    return paths.raw_detections / f"runs_manifest_{experiment_dir_name}_shard{shard_index}.csv"


# =============================================================================
#  CELL 6 - STABLE REPRODUCIBILITY HELPERS
# =============================================================================
# The positive-exemplar selection AND the negative generator are seeded with a
# SHA-256 digest of a plain text key (not Python's hash(), which is randomised per
# process), so every run gets the same prompts in every shard, session and
# machine. With N_EXEMPLARS = K_POSITIVES = 2 every run uses the anchor + 1
# random GT box.
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

    Input : n_gt        - number of GT boxes in the image
            anchor_idx  - index of the GT box this run is "about" (always a prompt)
            n_exemplars - how many positive prompts in total (K_POSITIVES here)
            image_id    - "<archive>/<flight>/<image name>"
    Output: list of GT indices, ANCHOR FIRST, then the randomly sampled others.
            With n_exemplars = 1 this is simply [anchor_idx].
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
    Human-readable id of a positive prompt set.
      single   -> "5"
      multiple -> "5+12"  (the ANCHOR is always the first number)
    """
    return "+".join(str(int(i)) for i in exemplar_indices)


# =============================================================================
#  CELL 7 - DATASET AND YOLO ANNOTATION HELPERS
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
#  CELL 8 - IoU
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
#  CELL 9 - CORRECT ONE-TO-ONE MATCHING
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
#  CELL 10 - RESIZE HELPER (whole-image, no tiling)
# =============================================================================
# The WHOLE image is sent to SAM3 in one pass (no tiles). The large UAV image is
# downscaled so that its longest side is MAX_DIM, purely for SAM3 input size /
# speed (it keeps the mask upsampling inside the post-processing cheap).
# Predictions are rescaled back to full resolution right after inference
# (see CELL 12).
# =============================================================================

def resize_for_sam3(img: Image.Image, max_dim: int) -> Tuple[Image.Image, float]:
    w, h = img.size
    scale = max_dim / max(w, h)
    if scale >= 1:
        return img, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.BILINEAR), scale


# =============================================================================
#  CELL 11 - NEGATIVE EXEMPLAR GENERATOR
# =============================================================================
#   GT Rumex boxes
#         |
#   1. generate NEG_NUM_CANDIDATES candidate regions: random positions inside the
#      image, sizes drawn from this image's own GT boxes (plant-sized regions)
#         |
#   2. remove every candidate that overlaps Rumex: zero intersection with ALL GT
#      boxes AND a gap of at least NEG_MIN_GAP_PX (10 px) to the nearest one
#         |
#   3. prefer candidates close to Rumex (vegetated surroundings, not bare soil):
#      keep gap <= NEG_MAX_GAP_FACTOR x median plant size; if fewer than
#      J_NEGATIVES qualify, the limit is doubled (up to 3 times)
#         |
#   4. select J_NEGATIVES spatially diverse ones by farthest-point sampling on
#      the box centres, starting from the candidate closest to Rumex; selected
#      negatives never overlap each other
#         |
#   SAM3 (label 0 box prompts)
# =============================================================================

def box_gap_matrix(boxes_a, boxes_b) -> np.ndarray:
    """
    Euclidean gap between axis-aligned boxes.
    Input : boxes_a (N,4), boxes_b (M,4) as [x1,y1,x2,y2]
    Output: (N,M) distance in px between the box borders; 0 when they touch/overlap.
    """
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


def generate_negative_exemplars(gt_boxes, img_w: int, img_h: int, n_negatives: int,
                                seed: int, n_candidates: int, min_gap: float,
                                max_gap_factor: float) -> dict:
    """
    Input : gt_boxes (G,4) full-res Rumex boxes, image size, number of negatives,
            deterministic seed and generator parameters
    Output: dict with
              'boxes'      (n,4) selected negative boxes, full-res [x1,y1,x2,y2]
              'gaps'       (n,)  gap of each negative to its nearest Rumex box (px)
              'pool_boxes' (P,4) the close, non-overlapping candidate pool
              'stats'      counters for the log
    """
    gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    if len(gt) == 0:
        raise ValueError("The image has no Rumex GT boxes -> no negatives can be placed near Rumex.")
    rng = np.random.default_rng(seed)

    # ---- 1. candidate regions (plant-sized) -----------------------------------
    gt_w = gt[:, 2] - gt[:, 0]
    gt_h = gt[:, 3] - gt[:, 1]
    size_idx = rng.integers(0, len(gt), n_candidates)
    cw = np.minimum(gt_w[size_idx], img_w - 1)
    ch = np.minimum(gt_h[size_idx], img_h - 1)
    cx1 = rng.uniform(0, img_w - cw)
    cy1 = rng.uniform(0, img_h - ch)
    cand = np.stack([cx1, cy1, cx1 + cw, cy1 + ch], axis=1).astype(np.float32)

    # ---- 2. no overlap with Rumex (+ minimum gap) -----------------------------
    gap_to_rumex = box_gap_matrix(cand, gt).min(axis=1)
    overlap_to_rumex = intersection_area_matrix(cand, gt).max(axis=1)
    valid = (overlap_to_rumex == 0) & (gap_to_rumex >= min_gap)

    # ---- 3. close to Rumex -----------------------------------------------------
    median_plant = float(np.median(np.maximum(gt_w, gt_h)))
    factor = max_gap_factor
    for _ in range(4):
        max_gap = factor * median_plant
        close = valid & (gap_to_rumex <= max_gap)
        if close.sum() >= n_negatives:
            break
        factor *= 2.0
    pool = np.where(close)[0]
    if len(pool) < n_negatives:                    # last resort: nearest valid ones
        pool = np.where(valid)[0]
    pool = pool[np.argsort(gap_to_rumex[pool], kind="stable")]   # nearest first

    # ---- 4. spatially diverse selection (farthest-point sampling) --------------
    centres = np.stack([(cand[:, 0] + cand[:, 2]) / 2, (cand[:, 1] + cand[:, 3]) / 2], axis=1)
    selected = []
    if len(pool):
        selected.append(int(pool[0]))              # start with the one closest to Rumex
    while len(selected) < n_negatives and len(selected) < len(pool):
        sel = np.array(selected)
        d = np.linalg.norm(centres[pool][:, None, :] - centres[sel][None, :, :], axis=2).min(axis=1)
        clash = intersection_area_matrix(cand[pool], cand[sel]).max(axis=1) > 0
        d[np.isin(pool, sel) | clash] = -1.0      # never re-pick or overlap a negative
        best = int(np.argmax(d))
        if d[best] < 0:
            break
        selected.append(int(pool[best]))

    boxes = cand[selected].reshape(-1, 4)
    # hard guarantee: zero overlap with every Rumex GT box
    assert (intersection_area_matrix(boxes, gt) == 0).all()

    return {
        "boxes": boxes,
        "gaps": gap_to_rumex[selected],
        "pool_boxes": cand[pool].reshape(-1, 4),
        "stats": {
            "n_candidates": int(n_candidates),
            "n_no_overlap": int(valid.sum()),
            "n_close": int(len(pool)),
            "median_plant_px": median_plant,
            "max_gap_px": float(max_gap),
            "n_selected": int(len(selected)),
        },
    }


# =============================================================================
#  CELL 12 - SAM3 WHOLE-IMAGE INFERENCE (no tiling, positive + negative boxes + text)
# =============================================================================
# Pipeline of this experiment:
#   - resize the whole image to MAX_DIM
#   - scale the positive (GT) and negative (generated) boxes into that space
#   - single SAM3 forward pass, no tiling, no exemplar-strip composition:
#     positive boxes (label 1), negative boxes (label 0) and the text prompt
#     are all passed in the SAME call -> SAM3 conditions on all of them at once
#   - scale the predicted boxes back up to full resolution
#
# SAM3 is called ONCE per (image_ID, anchor_idx) at CONFIDENCE_THRESHOLD;
# NMS is applied offline afterwards.
#
# Masks are requested from SAM3 internally (the post-processing needs them) but
# NEVER stored: KEEP_MASKS = False
#
# Cluster change: the notebook used device_map="auto" (accelerate). Here each
# shard process sees exactly ONE GPU via CUDA_VISIBLE_DEVICES, so the model is
# placed explicitly with .to(device) - simpler and it cannot silently offload.
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
    """Owns the model + processor and performs one whole-image forward pass."""

    def __init__(self, model_id: str, device: Optional[str], dtype: str,
                 mask_threshold: float, max_dim: int):
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
            self.model_dtype = torch.float32     # half precision is pointless on CPU
        self.mask_threshold = mask_threshold
        self.max_dim = max_dim

        print(f"Loading SAM3 from '{model_id}' onto {self.device} ({dtype}) ...")
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=self.model_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

        print("SAM3 loaded.")
        print("  model device:", next(self.model.parameters()).device)
        print("  model dtype :", next(self.model.parameters()).dtype)

    def run_whole_image(self, image: Image.Image,
                        pos_boxes_fullres: Sequence[Sequence[float]],
                        neg_boxes_fullres: Sequence[Sequence[float]],
                        text_prompt: str, threshold: float):
        """
        pos_boxes_fullres: list of POSITIVE [x1,y1,x2,y2] boxes (ORIGINAL image pixel coords)
        neg_boxes_fullres: list of NEGATIVE [x1,y1,x2,y2] boxes (ORIGINAL image pixel coords)
        text_prompt: text phrase sent together with the boxes.

        Returns: pred_boxes_fullres (np.ndarray Nx4), pred_scores (np.ndarray N),
                 sam_scale (float), all boxes/scores with score > threshold.
                 Masks are requested from SAM3 internally (needed by
                 post-processing) but never stored.
        """
        torch = self.torch
        image_sam, sam_scale = resize_for_sam3(image, self.max_dim)

        all_boxes_fullres = list(pos_boxes_fullres) + list(neg_boxes_fullres)
        input_boxes_xyxy = [[float(c) * sam_scale for c in box] for box in all_boxes_fullres]
        input_boxes = [input_boxes_xyxy]                           # [batch=1, num_boxes, 4]
        input_boxes_labels = [[1] * len(pos_boxes_fullres)         # 1 = positive prompt
                              + [0] * len(neg_boxes_fullres)]      # 0 = negative prompt

        inputs = self.processor(
            images=image_sam,
            text=text_prompt,                                    # text prompt ...
            input_boxes=input_boxes,                             # ... + positive/negative boxes
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.device)

        # match the pixel tensor dtype to the (possibly half precision) weights
        if self.model_dtype != torch.float32 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model_dtype)

        with torch.inference_mode():
            if self.model_dtype != torch.float32 and self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=self.model_dtype):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            pred_boxes_np = _to_numpy(results["boxes"]).reshape(-1, 4)
            pred_scores_np = _to_numpy(results["scores"]).reshape(-1)

            # masks are never kept (KEEP_MASKS = False) -> drop them immediately
            if "masks" in results:
                results["masks"] = None

        # Boxes came back in image_sam's (resized) coordinate space -> rescale to full-res
        if len(pred_boxes_np) > 0:
            pred_boxes_np = pred_boxes_np / sam_scale

        del results, outputs, inputs
        if image_sam is not image:
            image_sam.close()

        return pred_boxes_np, pred_scores_np, float(sam_scale)


# =============================================================================
#  CELL 13 - PRE-NMS DETECTION STORAGE
# =============================================================================
# For every run (= one image x one anchor) we store the detections AFTER
#   SAM3 inference at CONFIDENCE_THRESHOLD -> conversion to original-image
#   coordinates
# but BEFORE NMS.
# The offline evaluation (PHASE 2) therefore never needs SAM3 again.
# Masks are never stored. NPZ is compact and fast; the metric tables are CSV.
# =============================================================================

def run_npz_path(raw_detections_dir: Path, image_id: str, anchor_idx: int) -> Path:
    """Path of the NPZ holding the pre-NMS detections of one run."""
    return raw_detections_dir / f"{safe_filename(image_id)}__anchor{int(anchor_idx):03d}.npz"


def save_run_detections(raw_detections_dir: Path, experiment_name: str, text_prompt: str,
                        image_id: str, anchor_idx: int, boxes, scores,
                        gt_boxes: np.ndarray, prompt_indices: Sequence[int], neg_boxes,
                        image_size: Tuple[int, int], archive: str, flight: str,
                        class_id: int, sam_scale: float) -> Path:
    """
    Write one run's pre-NMS detections to NPZ. The file is self-contained: it also
    stores the GT boxes, the positive prompt indices and the negative prompt boxes,
    so the whole offline evaluation (and the final plots) can run without
    re-opening label files or re-generating the negatives.
    """
    path = run_npz_path(raw_detections_dir, image_id, anchor_idx)
    np.savez_compressed(
        path,
        experiment_name=np.array(experiment_name),
        image_ID=np.array(image_id),
        anchor_idx=np.array(int(anchor_idx)),
        text_prompt=np.array(text_prompt),
        prompt_indices=np.array(prompt_indices, dtype=np.int32),
        neg_boxes=np.asarray(neg_boxes, dtype=np.float32).reshape(-1, 4),
        image_width=np.array(int(image_size[0])),
        image_height=np.array(int(image_size[1])),
        archive=np.array(archive),
        flight=np.array(flight),
        source_class_id=np.array(int(class_id)),
        sam_scale=np.array(float(sam_scale)),
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
            "anchor_idx": int(z["anchor_idx"]),
            "text_prompt": str(z["text_prompt"]),
            "prompt_indices": z["prompt_indices"].astype(int),
            "neg_boxes": z["neg_boxes"].reshape(-1, 4),
            "image_width": int(z["image_width"]),
            "image_height": int(z["image_height"]),
            "gt_boxes": z["gt_boxes"].reshape(-1, 4),
            "boxes": z["boxes"].reshape(-1, 4),
            "scores": z["scores"].reshape(-1),
        }
        # archive / flight / sam_scale are cluster additions; tolerate older NPZs.
        run["archive"] = str(z["archive"]) if "archive" in z else ""
        run["flight"] = str(z["flight"]) if "flight" in z else ""
        run["sam_scale"] = float(z["sam_scale"]) if "sam_scale" in z else float("nan")
    return run


# =============================================================================
#  RESUME SUPPORT  (CELL 14, adapted to several shard manifests)
# =============================================================================

def load_done_runs(paths: Paths, args) -> set:
    """
    Read every shard manifest and return {(image_ID, anchor_idx)} of the runs that
    are already finished. Every shard reads ALL manifests, so a resubmission after
    the walltime never repeats work, even if the shard assignment changed because
    NUM_GPUS was different.

    The notebook's guard is kept: if a manifest holds results produced with a
    DIFFERENT text prompt or a DIFFERENT prompt type, the run is aborted instead of
    silently mixing them.
    """
    exp = args.experiment_name
    done: set = set()
    other_prompts: set = set()
    other_types: set = set()
    for csv_path in sorted(paths.raw_detections.glob(
            f"runs_manifest_{args.experiment_dir_name}_shard*.csv")):
        try:
            with open(csv_path, newline="") as fh:
                for row in csv.DictReader(fh):
                    if row.get("experiment_name") != exp:
                        continue
                    row_text = str(row.get("text_prompt", ""))
                    row_type = str(row.get("Prompt_Type", ""))
                    if row_text and row_text != args.text_prompt:
                        other_prompts.add(row_text)
                    if row_type and row_type != args.prompt_type:
                        other_types.add(row_type)
                    try:
                        done.add((row["image_ID"], int(row["anchor_idx"])))
                    except (KeyError, ValueError, TypeError):
                        continue            # ignore a half-written trailing row
        except OSError:
            continue
    if other_prompts or other_types:
        raise RuntimeError(
            f"{paths.raw_detections} already holds results for another prompt setting "
            f"(text {sorted(other_prompts) or [args.text_prompt]}, "
            f"type {sorted(other_types) or [args.prompt_type]}). "
            f"Use a new EXPERIMENT_NAME (and OUTPUT_DIR) for '{args.text_prompt}' / "
            f"{args.prompt_type} so the runs are not mixed.")
    return done


# =============================================================================
#  CELL 14 - MAIN GPU INFERENCE LOOP  (PHASE 1)
# =============================================================================
# FOR EACH IMAGE OF THIS SHARD:
#     open the original image ONCE
#     read its GT boxes ONCE
#     FOR EACH anchor (= every GT box, once):
#         select the K_POSITIVES positives deterministically (stable_seed, CELL 6)
#         generate J_NEGATIVES negatives deterministically (CELL 11)
#         run SAM3 ONCE over the whole (resized) image with
#             positive boxes + negative boxes + text
#         save the PRE-NMS detections + the prompts (NPZ)
#     release the image
#
# NO NMS and NO metric computation happens here - that is all done offline in
# PHASE 2. The loop is resumable: finished runs are listed in the shard manifests.
# =============================================================================

def run_inference(args, paths: Paths, records: List[ImageRecord]) -> None:
    import torch

    exp = args.experiment_name

    # ---- resume support ------------------------------------------------------
    done_runs: set = set()
    if not args.no_resume:
        done_runs = load_done_runs(paths, args)
        print(f"Resuming: {len(done_runs)} run(s) already finished for {exp}; skipped.")

    # ---- this shard's manifest ----------------------------------------------
    manifest_csv = manifest_path(paths, args.experiment_dir_name, args.shard_index)
    manifest_exists = manifest_csv.exists() and manifest_csv.stat().st_size > 0
    manifest_file = open(manifest_csv, "a", newline="")
    manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS,
                                     extrasaction="ignore")
    if not manifest_exists:
        manifest_writer.writeheader()
        manifest_file.flush()

    runner = Sam3Runner(args.model_id, args.device, args.dtype, args.mask_threshold,
                        args.max_dim)
    device_is_cuda = runner.device.startswith("cuda")

    start_time = time.time()
    n_new_runs = 0
    image_times: List[float] = []
    n_total_images = len(records)

    try:
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

            for anchor_idx in range(n_anchors):        # every GT box is an anchor once
                if (image_id, anchor_idx) in done_runs:
                    continue
                run_t0 = time.time()

                # deterministic positive selection (SHA-256 based, see CELL 6)
                exemplar_indices = select_exemplar_indices(n_gt, anchor_idx, args.n_exemplars,
                                                           image_id, exp)
                prompt_id = format_prompt_id(exemplar_indices)
                pos_boxes_fullres = [gt_boxes[i].tolist() for i in exemplar_indices]

                # deterministic negative generation, new for every run (CELL 11)
                neg = generate_negative_exemplars(
                    gt_boxes, img_w, img_h, args.j_negatives,
                    seed=stable_seed(exp, image_id, anchor_idx, "negatives"),
                    n_candidates=args.neg_num_candidates,
                    min_gap=args.neg_min_gap_px,
                    max_gap_factor=args.neg_max_gap_factor)
                neg_boxes_fullres = neg["boxes"].tolist()

                pred_boxes_np, pred_scores_np, sam_scale = runner.run_whole_image(
                    image, pos_boxes_fullres, neg_boxes_fullres,
                    text_prompt=args.text_prompt, threshold=args.threshold)

                npz_path = save_run_detections(
                    paths.raw_detections, exp, args.text_prompt, image_id, anchor_idx,
                    pred_boxes_np, pred_scores_np, gt_boxes, exemplar_indices, neg["boxes"],
                    (img_w, img_h), rec.archive, rec.flight, rec.class_id, sam_scale)

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
                    "n_negatives": int(len(neg["boxes"])),
                    "neg_max_gap_px": round(neg["stats"]["max_gap_px"], 1),
                    "n_detections_pre_nms": int(len(pred_scores_np)),
                    "image_width": img_w,
                    "image_height": img_h,
                    "sam_scale": round(sam_scale, 6),
                    "npz_file": npz_path.name,
                    "inference_seconds": round(run_seconds, 2),
                })
                manifest_file.flush()                  # this run is on disk -> resumable
                n_new_runs += 1

                print(f"  [{exp}] shard{args.shard_index} run #{n_new_runs} | {image_id} | "
                      f"anchor={anchor_idx} ({anchor_idx + 1}/{n_anchors}) | "
                      f"prompt=POS {prompt_id} + {len(neg['boxes'])} NEG + text | "
                      f"pre-NMS detections={len(pred_scores_np)} | {run_seconds:.1f}s")

                if len(neg["boxes"]) < args.j_negatives:
                    print(f"    WARNING: only {len(neg['boxes'])} of {args.j_negatives} "
                          f"negatives could be placed.")

                del pred_boxes_np, pred_scores_np, neg
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
                  f"{n_gt} GT box(es) | {image_elapsed:.1f}s | avg/image={avg_per_image:.1f}s | "
                  f"ETA={eta / 60:.1f} min ({eta / 3600:.2f} h)")
    finally:
        manifest_file.close()                    # also closed if the loop crashes

    total_elapsed = time.time() - start_time
    print(f"\nInference finished for {exp} (shard {args.shard_index}): {n_new_runs} new runs.")
    print(f"Total time: {total_elapsed / 60:.1f} min ({total_elapsed / 3600:.2f} h)")
    print(f"Pre-NMS detections in: {paths.raw_detections}")


# =============================================================================
#  DRY RUN - dataset report + cost estimate (no model, no GPU)
# =============================================================================

def dry_run(args, records: List[ImageRecord], my_records: List[ImageRecord]) -> None:
    print("\n--- DRY RUN: counting the work without loading SAM3 ---")
    sample = my_records[:min(len(my_records), 200)]
    total_anchors, per_archive = 0, {}
    n_short_negatives, n_checked = 0, 0
    for rec in sample:
        with Image.open(rec.image_path) as im:
            w, h = im.size
        gt = load_yolo_boxes(rec.label_path, w, h, rec.class_id)
        n_gt = len(gt)
        n_anchors = n_gt if args.max_anchors_per_image <= 0 else min(n_gt, args.max_anchors_per_image)
        total_anchors += n_anchors
        per_archive[rec.archive] = per_archive.get(rec.archive, 0) + n_anchors
        # probe the negative generator on anchor 0 of every image (CPU, cheap)
        if n_gt > 0:
            neg = generate_negative_exemplars(
                gt, w, h, args.j_negatives,
                seed=stable_seed(args.experiment_name, rec.image_id, 0, "negatives"),
                n_candidates=args.neg_num_candidates, min_gap=args.neg_min_gap_px,
                max_gap_factor=args.neg_max_gap_factor)
            n_checked += 1
            if len(neg["boxes"]) < args.j_negatives:
                n_short_negatives += 1
    print(f"  sampled {len(sample)} image(s) of this shard -> {total_anchors} anchor runs "
          f"({per_archive})")
    print(f"  prompts per run: {args.k_positives} positive + {args.j_negatives} negative + "
          f"text '{args.text_prompt}' ({args.prompt_type})")
    print(f"  negative generator probed on {n_checked} image(s): "
          f"{n_short_negatives} could not place all {args.j_negatives} negatives")
    print(f"  forward passes per anchor run: 1 (whole image, no tiling, "
          f"longest side -> {args.max_dim} px)")
    print(f"  => ~{total_anchors} SAM3 forward passes for those {len(sample)} images")
    print("  (scale by len(shard)/sampled for the full estimate)")
    print(f"  NPZ files that will be written by this shard: ~{total_anchors} "
          f"(one per image x anchor)")


# =============================================================================
#  CELL 15 - LOAD CACHED PRE-NMS DETECTIONS  (start of PHASE 2)
# =============================================================================
# From here on SAM3 is never touched again. Everything below works on the NPZ
# files written in PHASE 1, so the complete evaluation can be redone in minutes
# on a login node or in a small CPU allocation.
# =============================================================================

def load_runs(paths: Paths, args):
    import pandas as pd

    exp = args.experiment_name
    manifest_files = sorted(paths.raw_detections.glob(
        f"runs_manifest_{args.experiment_dir_name}_shard*.csv"))
    if not manifest_files:
        print(f"No manifest found in {paths.raw_detections} for {exp}.")
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
    manifest = manifest[(manifest["experiment_name"] == exp) &
                        (manifest["text_prompt"].astype(str) == args.text_prompt)].copy()
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
          f"({manifest['image_ID'].nunique()} images) for {exp}, "
          f"text prompt '{args.text_prompt}'.")
    print("Total pre-NMS detections:", int(sum(len(r['scores']) for r in runs)))
    print("Total GT boxes over all runs:", int(sum(len(r['gt_boxes']) for r in runs)))
    return runs, manifest


# =============================================================================
#  CELL 16 - OFFLINE NMS
# =============================================================================
# SAM3 can return several overlapping boxes for the same plant. NMS keeps the
# highest-scoring box of each overlapping group. The NMS IoU threshold is fixed
# (NMS_IOU_THRESHOLD = 0.40) and applied offline, so the raw pre-NMS detections
# stay untouched on disk.
# (No tile provenance is tracked here: without tiling every detection comes
#  from the same single input image.)
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
#  CELL 17 - EVALUATION CORE: all_gt AND held_out
# =============================================================================
# all_gt   : classical evaluation, every GT box of the image counts.
#
# held_out : the GT instances shown to SAM3 as POSITIVE prompts are REMOVED
#            from the GT set, and predictions that fall on those plants are
#            IGNORED (neither TP nor FP). The positives lie INSIDE the evaluated
#            image and SAM3 usually re-detects them. It answers: "after being
#            shown K examples, how well does SAM3 find the REMAINING Rumex plants?"
#            Negative prompts are not GT boxes and play no role here.
#
# ORDER OF OPERATIONS (the agreed protocol):
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
             on an image whose every plant was a prompt)
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    eval_gt_boxes = np.asarray(eval_gt_boxes, dtype=np.float32).reshape(-1, 4)
    prompt_gt_boxes = np.asarray(prompt_gt_boxes, dtype=np.float32).reshape(-1, 4)

    n_pred, n_eval_gt = len(pred_boxes), len(eval_gt_boxes)
    # start with everything as FP -> valid matches become TP -> leftovers on the
    # prompt plant become IGNORED -> the rest stays FP
    status = np.full(n_pred, STATUS_FP, dtype=np.int8)

    # step 1 - corrected one-to-one matching against the evaluated GT
    match = match_one_to_one(pred_boxes, pred_scores, eval_gt_boxes, eval_iou)
    status[match["pred_match_gt"] >= 0] = STATUS_TP

    # step 2 - ignore the unmatched leftovers that sit on a PROMPT plant
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
#  CELL 18 - AP50 AND AP50:95  (supervision.metrics.MeanAveragePrecision)
# =============================================================================
# AP is an area under the precision-recall curve, built by walking through ALL
# detections ordered by confidence. It therefore always uses every post-NMS
# prediction that SAM3 returned (score > CONFIDENCE_THRESHOLD).
# Precision / recall / F1 / IoU1 / IoU2 describe ONE operating point.
# In this experiment the operating point IS the inference threshold (0.30), so
# both use exactly the same prediction set - they still answer different
# questions (ranking quality vs. deployed behaviour).
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
    All post-NMS predictions are used; in held_out mode the predictions that were
    IGNORED (they belong to a prompt plant) are removed first, exactly like in the
    operating-point evaluation.
    """
    ev = evaluate_run_predictions(nms_run["boxes"], nms_run["scores"], eval_gt, prompt_gt,
                                  eval_iou, ignore_iou)
    keep = ev["status"] != STATUS_IGNORED
    return (make_detections(nms_run["boxes"][keep], nms_run["scores"][keep]),
            make_detections(eval_gt))


# =============================================================================
#  CELL 19 - OPERATING-POINT EVALUATION
# =============================================================================
# Keeps the predictions with score >= CONFIDENCE_THRESHOLD and evaluates them
# (precision, recall, F1, IoU1, IoU2, TP, FP, FN).
# The operating point is fixed from the start - no confidence x NMS sweep and no
# "best configuration" selection is performed in this experiment:
#     confidence threshold = CONFIDENCE_THRESHOLD (0.30)
#     NMS IoU threshold    = NMS_IOU_THRESHOLD    (0.40)
# =============================================================================

def evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, confidence_threshold: float,
                                eval_iou: float, ignore_iou: float) -> dict:
    """
    Input : nms_run  - output of apply_nms_to_run (sorted by score, high -> low)
            eval_gt / prompt_gt - from split_gt_for_mode
            confidence_threshold - the operating point
    Output: the dict of evaluate_run_predictions for the thresholded predictions.
    """
    keep = nms_run["scores"] >= confidence_threshold
    return evaluate_run_predictions(nms_run["boxes"][keep], nms_run["scores"][keep],
                                    eval_gt, prompt_gt, eval_iou, ignore_iou)


# =============================================================================
#  CELL 24 (helper) - CONFUSION-MATRIX FIGURE
# =============================================================================

def plot_confusion_matrix(tp, fp, fn, title, png_path, plt) -> None:
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


# =============================================================================
#  CELL 26 - QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right)
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
#   is displayed. Its predictions are the post-NMS boxes with
#   score >= CONFIDENCE_THRESHOLD; the positive prompts (lime) and the
#   negative prompts (magenta) of that run are drawn dashed.
#
# The image is downscaled for DISPLAY only (PLOT_MAX_DISPLAY_DIM).
#
# The notebook produced ONE figure. Both archives are pooled here, so the same
# selection is run once per archive and once over everything.
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
    neg_boxes_plot = run["neg_boxes"]

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
    _draw_boxes(axes[1], neg_boxes_plot * to_disp, color="magenta", linewidth=2.5,
                linestyle="--")
    axes[1].set_title(
        f"Predictions: {len(pred_boxes)} boxes | anchor = {anchor} | "
        f"POS {run['Prompt_ID']} + {len(neg_boxes_plot)} NEG + text '{run['text_prompt']}'\n"
        f"run AP50={run_row['AP50']:.3f}  P={run_row['precision']:.3f}  "
        f"R={run_row['recall']:.3f}  F1={run_row['F1']:.3f}  "
        f"TP={int(run_row['TP'])} FP={int(run_row['FP'])} FN={int(run_row['FN'])}",
        fontsize=12)

    legend_handles = [
        Line2D([0], [0], color="yellow", lw=2, label="ground truth"),
        Line2D([0], [0], color="red", lw=2, label="prediction"),
        Line2D([0], [0], color="lime", lw=2.5, linestyle="--",
               label=f"positive prompts ({len(exemplar_boxes)})"),
        Line2D([0], [0], color="magenta", lw=2.5, linestyle="--",
               label=f"negative prompts ({len(neg_boxes_plot)})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=11,
               frameon=False)
    fig.suptitle(
        f"{exp} | {plot_image_id} | mode={mode} | scope={scope_label} | "
        f"image-level AP50_mean={img_row['AP50_mean']:.3f} "
        f"(over {int(img_row['n_runs_valid_for_macro'])} anchors)\n"
        f"{args.k_positives} pos + {args.j_negatives} neg + text, no tiling "
        f"(MAX_DIM={args.max_dim}px) | "
        f"conf={args.operating_confidence:.2f}, NMS IoU={args.nms_iou_threshold:.2f}, "
        f"mask thr={args.mask_threshold:.2f}\nselection: {rule}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.91])

    png_path = paths.plots / (f"best_image_{scope_label}_{safe_filename(plot_image_id)}"
                              f"_anchor{anchor:03d}_{mode}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)                     # headless: saved, never shown

    print(f"\n[{scope_label}] Selected image : {plot_image_id} "
          f"({len(gt_boxes_plot)} GT boxes)")
    print(f"[{scope_label}] Selection rule : {rule}")
    print(f"[{scope_label}] Shown anchor   : {anchor} (best run-level AP50 of this image), "
          f"POS {run['Prompt_ID']} + {len(neg_boxes_plot)} NEG + text '{run['text_prompt']}'")
    print(f"[{scope_label}] Figure saved   : {png_path}")


def make_qualitative_plots(args, paths: Paths, runs, run_level_df, image_level_df, plt) -> None:
    """CELL 26 for every scope: one figure per archive plus one global figure."""
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
#  PHASE 2 - THE WHOLE OFFLINE EVALUATION (notebook CELL 15 ... CELL 26)
# =============================================================================
#  The operating point is frozen and identical to the SAM3 inference threshold:
#  confidence 0.30, NMS IoU 0.40. No sweep is performed.
# =============================================================================

def run_evaluation(args, paths: Paths) -> None:
    """PHASE 2: notebook CELL 15 ... CELL 26, in one process, no GPU."""
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
    ignore_iou = args.prompt_ignore_iou

    print("=" * 92)
    print(f" PHASE 2 - OFFLINE EVALUATION | experiment={exp}")
    print("=" * 92)
    print("Operating configuration is fixed (no sweep performed):")
    print(f"  Confidence threshold = {conf:.2f}  (== the SAM3 inference threshold)")
    print(f"  NMS IoU threshold    = {nms_iou:.2f}")
    print(f"  Evaluation IoU       = {eval_iou:.2f}")

    runs, manifest = load_runs(paths, args)
    if not runs:
        print("Nothing to evaluate.")
        return

    # =========================================================================
    #  CELL 20 - RUN-LEVEL METRICS  (one run = one image x one anchor)
    # =========================================================================
    # Everything is evaluated at the fixed configuration from CELL 3.
    #   AP50 / AP50_95 : all post-NMS predictions, confidence-ranked
    #   P / R / F1 / IoU1 / IoU2 / TP / FP / FN : predictions >= CONFIDENCE_THRESHOLD
    #
    # Special case (held_out with no evaluable GT, i.e. every plant of the image
    # was used as a positive prompt): the metrics are written as NaN and
    # valid_for_macro = False so they are excluded from every mean/std, but
    # TP/FN = 0 and the real FP count are kept, because such a run can still
    # produce false positives that must show up in the pooled counts and in the
    # confusion matrix.
    # =========================================================================
    print("\n--- CELL 20: run-level metrics ---")
    run_rows = []
    for run in runs:
        nms_run = apply_nms_to_run(run, nms_iou)
        for mode in EVALUATION_MODES:
            eval_gt, prompt_gt = split_gt_for_mode(run["gt_boxes"], run["prompt_indices"], mode)

            # ---- AP: every post-NMS prediction (ignored ones removed) ------------
            p_det, g_det = ap_inputs_for_run(nms_run, eval_gt, prompt_gt, eval_iou, ignore_iou)
            ap50, ap5095 = compute_ap([p_det], [g_det])

            # ---- operating point -------------------------------------------------
            ev = evaluate_at_operating_point(nms_run, eval_gt, prompt_gt, conf,
                                             eval_iou, ignore_iou)
            valid = ev["valid_for_macro"]
            nan = float("nan")

            run_rows.append({
                "experiment_name": exp,
                "image_ID": run["image_ID"],
                "archive": run.get("archive", ""),
                "flight": run.get("flight", ""),
                "anchor_idx": run["anchor_idx"],
                "Prompt_ID": run["Prompt_ID"],
                "Prompt_Type": run["Prompt_Type"],
                "text_prompt": run["text_prompt"],
                "n_negatives": int(len(run["neg_boxes"])),
                "evaluation_mode": mode,
                "confidence_threshold": conf,
                "nms_iou_threshold": nms_iou,
                "n_gt_total": int(len(run["gt_boxes"])),
                "n_prompt_gt": int(len(run["prompt_indices"])) if mode == "held_out" else 0,
                "n_eval_gt": ev["n_eval_gt"],
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
    #  CELL 21 - IMAGE-LEVEL METRICS
    # =========================================================================
    # All anchor runs of the same image are averaged into ONE value per image and
    # per evaluation mode. The std here is the spread BETWEEN the different anchors
    # of the SAME image, i.e. "how sensitive is the result to which plant was used
    # as the visual prompt?".
    # NaN rows (held_out runs with no evaluable GT) are ignored by pandas mean/std.
    # std is NaN when an image has only one valid run - that is expected.
    # =========================================================================
    print("\n--- CELL 21: image-level metrics ---")
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
    #  CELL 22 - EXPERIMENT-LEVEL SUMMARY
    # =========================================================================
    # Computed from the IMAGE-LEVEL values, not from the raw run rows, so that every
    # UAV image contributes exactly the same weight regardless of how many GT boxes
    # (and therefore how many anchor runs) it contains.
    # The std here is the variation BETWEEN UAV images.
    # =========================================================================
    print("\n--- CELL 22: experiment-level summary ---")
    summary_rows = []
    for mode in EVALUATION_MODES:
        sub = image_level_df[image_level_df["evaluation_mode"] == mode]
        row = {
            "experiment_name": exp,
            "evaluation_mode": mode,
            "prompt_type": args.prompt_type,
            "text_prompt": args.text_prompt,
            "n_exemplars": args.n_exemplars,
            "n_negatives": args.j_negatives,
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

    # ---- per-archive version of the same table (cluster addition) -----------
    # The notebook ran on ONE archive, so a single summary row per mode was the
    # whole story. Both archives are pooled here, so the same numbers are also
    # written per archive - the pooled rows above stay exactly as the notebook
    # defines them.
    per_archive_rows = []
    for (archive, mode), sub in image_level_df.groupby(["archive", "evaluation_mode"]):
        row = {
            "experiment_name": exp,
            "archive": archive,
            "evaluation_mode": mode,
            "text_prompt": args.text_prompt,
            "n_images": int(sub["image_ID"].nunique()),
            "n_runs": int(sub["n_runs_total"].sum()),
            "n_runs_valid_for_macro": int(sub["n_runs_valid_for_macro"].sum()),
        }
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = sub[f"{col}_mean"].mean()
            row[f"{col}_std"] = sub[f"{col}_mean"].std()
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
    #  CELL 23 - POOLED DATASET AP50 / AP50:95
    # =========================================================================
    # This is NOT the mean of the image-level AP values. All runs are handed to
    # supervision as evaluation EPISODES at once, so every detection of the whole
    # dataset is ranked in ONE precision-recall curve.
    #
    # NOTE for the thesis text: because every GT box of an image becomes an anchor
    # once, the same UAV image appears in several episodes (once per anchor).
    # The pooled AP is therefore computed over "pooled evaluation episodes", not
    # over unique images - it measures the ranking quality of the whole experiment.
    # =========================================================================
    print("\n--- CELL 23: pooled dataset AP ---")
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
        ap50, ap5095 = compute_ap(pred_list, gt_list)   # one PR curve over ALL runs
        dataset_rows.append({
            "experiment_name": exp,
            "evaluation_mode": mode,
            "text_prompt": args.text_prompt,
            "n_images": len(images_used),
            "n_runs": len(runs),
            "confidence_used_for_AP": args.threshold,
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
    print("\nFor comparison, the MEAN of the image-level AP50 values "
          "(a different quantity):")
    print(image_level_df.groupby("evaluation_mode")["AP50_mean"].mean().to_string())

    # =========================================================================
    #  CELL 24 - DATASET-LEVEL CONFUSION MATRICES
    # =========================================================================
    # One class (Rumex) plus a background row/column:
    #     Actual Rumex      -> Predicted Rumex      = TP
    #     Actual Rumex      -> Predicted Background = FN  (missed plants)
    #     Actual Background -> Predicted Rumex      = FP  (spurious detections)
    #     Actual Background -> Predicted Background = not defined for detection
    #                                                 (there are no true negatives)
    # Counts are pooled over every run at the fixed configuration. In held_out mode
    # the prompt plants and the detections that were ignored do not appear anywhere.
    # =========================================================================
    print("\n--- CELL 24: confusion matrices ---")
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
                paths.confusion_matrices / f"confusion_matrix_{mode}.png", plt)

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
    #  CELL 26 - QUALITATIVE FIGURES (one per archive + one global)
    # =========================================================================
    if args.no_plots:
        print("\n--- CELL 26: qualitative figures skipped (--no-plots) ---")
    elif not _HAS_MPL:
        print("\n--- CELL 26: qualitative figures skipped (matplotlib unavailable) ---")
    else:
        print("\n--- CELL 26: qualitative GT-vs-prediction figures ---")
        make_qualitative_plots(args, paths, runs, run_level_df, image_level_df, plt)

    # =========================================================================
    #  CELL 25 - FINAL OUTPUT SUMMARY
    # =========================================================================
    print("=" * 78)
    print(f"EXPERIMENT {exp} - FINAL SUMMARY")
    print("=" * 78)
    print(f"Prompts per run          : {args.k_positives} positive + {args.j_negatives} "
          f"negative + text ({args.prompt_type})")
    print(f"Negative generator       : gap >= {args.neg_min_gap_px:g}px, close <= "
          f"{args.neg_max_gap_factor:g} x median plant, spatially diverse, new per run")
    print(f"Text prompt              : '{args.text_prompt}'")
    print(f"Tiling                   : {args.use_tiling}  (whole image, longest side -> "
          f"{args.max_dim}px)")
    print(f"Confidence threshold     : {conf:.2f} (fixed; SAM3 run once per image x anchor)")
    print(f"Mask threshold           : {args.mask_threshold:.2f}")
    print(f"NMS IoU threshold        : {nms_iou:.2f} (fixed)")
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
                  "'Problem B' in pos_neg_exemplars_text_no_tiling_HOW_TO_RUN.md.")
            sys.exit(2)
        run_evaluation(args, paths)
        return

    has_supervision = _check_supervision()

    print("=" * 92)
    print(f" POS_NEG_EXEMPLARS_TEXT_NO_TILING SAM3 PIPELINE | experiment={exp} | "
          f"shard {args.shard_index + 1}/{args.num_shards}")
    print("=" * 92)
    print(f" dataset_root   : {dataset_root}")
    print(f" results_root   : {paths.results_root}")
    print(f" archives       : {', '.join(args.archives)}")
    print(f" prompts        : {args.k_positives} positive + {args.j_negatives} negative + "
          f"text '{args.text_prompt}' ({args.prompt_type})")
    print(f" negatives      : {args.neg_num_candidates} candidates, gap >= "
          f"{args.neg_min_gap_px:g}px, close = gap <= {args.neg_max_gap_factor:g} x median "
          f"plant size, spatially diverse, new per run")
    print(f" tiling         : {args.use_tiling}  (whole image, longest side -> {args.max_dim} px)")
    print(f" sam3 threshold : {args.threshold}  (single inference pass per image x anchor)")
    print(f" mask threshold : {args.mask_threshold}  (masks stored: {args.keep_masks})")
    print(f" dtype          : {args.dtype}")
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
        (paths.results_root / f"run_config_{args.experiment_dir_name}.json").write_text(
            json.dumps(config, indent=2))

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