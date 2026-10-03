HOW TO RUN — Experiment `text_only_no_tiling_yoloe`
=====================================================

What this experiment is: YOLOE (`yoloe-11l-seg.pt`, imgsz 1024) prompted with a TEXT prompt
ONLY — `"Rumex obtusifolius"`, no exemplar, no visual prompt — NO tiling: the whole image is
resized to 1024 px on its longest side and processed in ONE pass per image — run over both
`AGS_Multi_Rumex` and `AgsSpringRumex` pooled as one dataset. The GT boxes are used ONLY for
the evaluation, never as model input, so there are no anchors: ONE run = ONE image, evaluated
against ALL its GT boxes (`all_gt`).
This is the YOLOE version of the SAM3 notebook `text_only_no_tiling` (its SAM3 cluster script
is the base: storage and evaluation are identical; only the detector changes).

HOW YOLOE IS TEXT-PROMPTED. YOLOE turns the phrase into ONE class embedding with its text
encoder (`model.get_text_pe`) and installs it with `set_classes([phrase], embedding)`; after
that every image is a plain `predict()` — the standard Ultralytics YOLOE text prompt. The
embedding depends only on the phrase and the checkpoint, never on the image, so:

- it is computed ONCE (STEP 0, before the shards start) and cached in
  `$SCRATCH/yoloe_weights/text_pe/` — the SAME file the `1_exemplar_text_*_yoloe` experiments
  use, so if one of them already ran, STEP 0 just verifies it;
- every shard loads it and installs it ONCE, before its first image.

   STEP 0   TEXT EMBEDDING (GPU, once)  "Rumex obtusifolius" -> e_text -> cached .pt file
   PHASE 1  INFERENCE   (GPU)      per shard: install e_text once; per image: resize ONCE to
                                   1024 px -> YOLOE ONCE on it at score 0.30 -> boxes back
                                   to full res -> NPZ
   PHASE 2  EVALUATION  (no GPU)   NMS 0.40 -> metrics at confidence 0.30 (all_gt only)
                                   -> image / experiment / pooled-AP CSVs
                                   -> confusion matrix (CSV + PNG)
                                   -> qualitative GT-vs-prediction figures

Phase 2 never touches YOLOE, so once Phase 1 is done you can rebuild every number in minutes.

Same as the SAM3 `text_only_no_tiling` pipeline: whole image at 1024 px, text prompt only, NO
plausibility filter, plain offline NMS, `all_gt` only (there is no prompt plant, so `held_out`
does not exist). Differences:
- SAM3 reads the text in every forward pass; YOLOE encodes it once into a class embedding
- YOLOE's own NMS is set permissive (0.90), so the offline NMS (0.40) decides
- the SAM3 columns `sam_scale`, `sam_input_width/height` and
  `gt_median_side_px_at_sam_input` are called `resize_scale`, `model_input_width/height` and
  `gt_median_side_px_at_model_input` here (same values: the input is the same 1024 px image)

   PHASE A — SETUP            do once, ~20 minutes
   ├─ Step 1   copy the files onto the cluster
   ├─ Step 2   put the dataset on $SCRATCH
   ├─ Step 3   download the weights, the text encoder and the extra packages
   ├─ Step 4   open an interactive session inside the container
   ├─ Step 5   check the container has what it needs
   └─ Step 6   smoke test on 2 images

   PHASE B — RUN              every time
   ├─ Step 7   submit the job
   ├─ Step 8   watch it
   ├─ Step 9   collect the results
   ├─ Step 10  resubmit if it hit the 24 h limit
   └─ Step 11  re-run only the evaluation (optional, cheap)


PHASE A — SETUP (once)
======================

Step 1 — Copy the files onto the cluster
-----------------------------------------
Put the three scripts here:

    $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe/
    ├── text_only_no_tiling_infer_yoloe.py
    ├── text_only_no_tiling_run_yoloe.sh
    └── text_only_no_tiling_submit_yoloe.sh

Then, on a login node:

    mkdir -p $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    chmod +x text_only_no_tiling_run_yoloe.sh text_only_no_tiling_submit_yoloe.sh
    ls -l

You should see: three files, with the two `.sh` marked executable (`-rwxr-xr-x`).

If you edited the `.sh` files on Windows, strip the Windows line endings once:

    sed -i 's/\r$//' text_only_no_tiling_run_yoloe.sh text_only_no_tiling_submit_yoloe.sh


Step 2 — Put the dataset on `$SCRATCH`
---------------------------------------
If the SAM3 experiments or another YOLOE experiment already ran, the dataset is already
there — skip this step.

Otherwise extract the two archives side by side under one folder:

    mkdir -p $SCRATCH/overney/dataset
    cd $SCRATCH/overney/dataset
    tar -xzf /path/to/AGS_Multi_Rumex.tar.gz
    tar -xzf /path/to/AgsSpringRumex.tar.gz

The result must look exactly like this:

    $SCRATCH/overney/dataset/
    ├── AGS_Multi_Rumex/
    │   ├── images/
    │   │   └── 20220518_Eschikon/        (one folder per flight)
    │   │       └── DJI_0001.JPG          (8192 × 5460)
    │   └── annotations_yolo/             (FLAT: DJI_0001.txt, ... + darknet.labels)
    └── AgsSpringRumex/
        ├── images/
        │   └── 20230410_Lindau/
        │       └── DJI_1001.JPG
        └── annotations_yolo/             (FLAT)

Important: the folder names are case-sensitive. `AGS_Multi_Rumex` uses class id 0 and
`AgsSpringRumex` class id 2. The other two archives (`AGS_Multiple_Fields`,
`AGS_Multiple_Fields_Embeddings`) may be present — the code skips them on purpose.


Step 3 — Download the weights and the extra packages
-----------------------------------------------------
YOLOE is not gated, so no HuggingFace token is needed. Compute nodes have no internet, so
the checkpoint AND the YOLOE text encoder must be fetched on a login node. If you already ran
the download of one of the `1_exemplar_text_*_yoloe` experiments, everything is there — this
only confirms it:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    ./text_only_no_tiling_run_yoloe.sh download

This does four things:
- downloads `yoloe-11l-seg.pt` into `$SCRATCH/yoloe_weights/`
  (from the Ultralytics GitHub release; if `curl` fails it falls back to Ultralytics' own
  downloader)
- downloads the YOLOE text encoder `mobileclip_blt.ts` into the same folder and installs the
  small CLIP tokenizer package into `$SCRATCH/pyextra`. Both are needed only ONCE, to compute
  the text embedding in the run mode. This part is best effort: if it prints a WARNING, use
  **Problem C** (compute the text embedding in Colab) — nothing else is affected
- installs the `supervision` package into `$SCRATCH/pyextra`
  (`supervision` computes AP50 / AP50:95; the code refuses to run without it)
- downloads wheels for the Phase 2 packages (`pandas`, `matplotlib`) into `$SCRATCH/wheels`,
  in case the container does not ship them — see Problem B

`$SCRATCH/pyextra` and `$SCRATCH/wheels` are shared with the SAM3 experiments, so steps 3/4
of the download just confirm what is already there.

You should see:

    --- 1/4 : YOLOE weights -> /scratch/.../yoloe_weights/yoloe-11l-seg.pt ---
    -rw-r--r-- ... yoloe-11l-seg.pt
    --- 2/4 : YOLOE text encoder + CLIP tokenizer (for the text prompt) ---
    -rw-r--r-- ... mobileclip_blt.ts
    Successfully installed clip-... ftfy-... wcwidth-...
    --- 3/4 : supervision -> /scratch/.../pyextra ---
    supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
    --- 4/4 : wheels for the PHASE 2 packages -> /scratch/.../wheels ---
    Done. Compute nodes can now run offline.

If it says `ERROR: ... is missing` → download
`https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt` on your own
machine and copy it to `$SCRATCH/yoloe_weights/yoloe-11l-seg.pt`.


Step 4 — Open an interactive session inside the container
----------------------------------------------------------
Steps 5 and 6 must run inside the container, not on the login node — otherwise you are
checking the login node's Python, which is not the one the job will use.

    srun --account=go077 --time=00:30:00 \
         --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
         --environment=yolo26 --pty bash

You should see: a new shell prompt, running on a compute node.


Step 5 — Check the container has what it needs
-----------------------------------------------
Inside the interactive shell from Step 4:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    ./text_only_no_tiling_run_yoloe.sh dryrun

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

### 5a. Look for these lines in the debug block

    torch: 2.x.x | cuda available: True | device count: 1
    ultralytics: 8.x.x ...
    YOLOE import: OK
    YOLOEVPSegPredictor.get_vpe: OK      (not used here: text prompt only)
    precision argument: quantize=16 (as in the notebook)      (or: half=True ...)
    weights file: /scratch/.../yoloe-11l-seg.pt -> FOUND
    text encoder file: /scratch/.../yoloe_weights/mobileclip_blt.ts -> FOUND
    CLIP tokenizer import: OK
    text embedding cache: /scratch/.../text_pe/yoloe-11l-seg__rumex_obtusifolius.pt -> not yet (the run mode prepares it)
    MeanAveragePrecision import: OK
    pandas: 2.x.x -> PHASE 2 import: OK
    matplotlib: 3.x.x -> confusion-matrix + qualitative PNGs: OK

| If you see                          | Meaning                                                           | Fix                                           |
| ----------------------------------- | ----------------------------------------------------------------- | --------------------------------------------- |
| YOLOE import FAILED                 | the `yolo26` image's ultralytics is too old for YOLOE (get_vpe MISSING alone does not matter here) | see **Problem A** at the bottom |
| precision argument: half=True       | older ultralytics than the notebook's 8.4.163; the code uses `half=True` instead of `quantize=16` automatically | nothing to do |
| weights file ... MISSING            | Step 3 did not finish                                             | rerun Step 3                                  |
| text encoder MISSING / CLIP tokenizer FAILED | the text embedding cannot be computed on the cluster      | fine if the cache already exists; otherwise **Problem C** |
| supervision import FAILED           | `$SCRATCH/pyextra` not found or wrong Python version              | rerun Step 3; if it still fails, see **Problem B** |
| pandas import FAILED                | Phase 1 still works, but Phase 2 cannot run                       | see **Problem B**                             |
| matplotlib import FAILED            | only the PNGs are lost, every CSV is still written                | see **Problem B**, or ignore it               |
| cuda available: False               | no GPU in this allocation                                         | add `--gpus-per-task=1` to Step 4             |

Do not continue until the YOLOE, weights, text-encoder (or cache) and supervision lines are correct.

### 5b. Check the dataset report

    Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
      AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
      AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
    Total images: NNN | this shard: NNN

It also prints a cost estimate:

      sampled NNN image(s) of this shard -> NNN image runs ({...})
      forward passes per image run: 1 (whole image, no tiling, longest side -> 1024 px)
      => ~NNN YOLOE forward passes for those NNN images (+ the text embedding ONCE per job)
      weights file: /scratch/.../yoloe-11l-seg.pt -> FOUND
      text prompt : 'Rumex obtusifolius' (text only)
      text embedding cache: /scratch/.../text_pe/yoloe-11l-seg__rumex_obtusifolius.pt -> FOUND / not yet
      NPZ files that will be written by this shard: ~NNN (one per image)

If it says `No images with labels found` → `$SCRATCH/overney/dataset` is wrong. Go back to Step 2.


Step 6 — Smoke test on 2 images
--------------------------------
Still inside the interactive shell. This is the first time the model actually loads.
It writes to a throwaway folder so it cannot pollute the real results.

    EXPERIMENT_NAME=text_only_no_tiling_yoloe_smoke \
    OUTPUT_DIR=$SCRATCH/experiments/yoloe/_smoke_text_only_no_tiling_yoloe \
    NUM_GPUS=1 \
    ./text_only_no_tiling_run_yoloe.sh run --limit-images 2

You should first see STEP 0 (the text embedding, computed once and cached — or just verified
if another text experiment already made it), then the model load with the text prompt
installed, then one line per image, then the whole Phase 2:

    ===== STEP 0 : TEXT EMBEDDING (computed once, cached) =====
    Text embedding already cached: /scratch/.../text_pe/yoloe-11l-seg__rumex_obtusifolius.pt (1, 1, 512) ('Rumex obtusifolius', yoloe-11l-seg.pt)
       (or: Computing the text embedding ... / Text embedding cached: ...)
    ===== PHASE 1 : SHARDED INFERENCE =====
    Loading YOLOE from '/scratch/.../yoloe-11l-seg.pt' onto cuda (half=True, {'quantize': 16}) ...
    YOLOE loaded.
    Text embedding loaded from cache: /scratch/.../text_pe/yoloe-11l-seg__rumex_obtusifolius.pt
    Text prompt 'Rumex obtusifolius' installed -> e_text (1, 1, 512); classes = {0: 'Rumex obtusifolius'}
      [text_only_no_tiling_yoloe_smoke] shard0 run #1 | AGS_Multi_Rumex/2022.../DJI_0001 | 57 GT box(es) | pre-NMS detections=6 | 0.3s
    ...
    ===== PHASE 2 : OFFLINE EVALUATION =====
    --- CELL 18: image-level metrics ---
    Images with at least one prediction: 2/2
    ...

Things to check in this output:

- `classes = {0: 'Rumex obtusifolius'}` confirms the text prompt replaced YOLOE's default
  vocabulary. It is installed once per shard; the images themselves need no prompt work, so
  one image is just one 1024 px forward pass (the fastest experiment of all).
- `pre-NMS detections=` should not be 0 on every image. All zeros means the text prompt found
  nothing at 0.30 — possible for text-only on ~14 px plants, but check a qualitative figure.
- `Images with at least one prediction` tells you how often text alone finds anything.
- There is no `held_out` line: text-only has no prompt plant to hold out.
- STEP 0 fails with `the text embedding could not be computed` → **Problem C**.
- A dtype error while the text embedding is installed → rerun with `USE_FP16=0`.

Then leave the interactive session:

    exit


PHASE B — RUN
=============

Step 7 — Submit the job
------------------------
On a login node:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    sbatch text_only_no_tiling_submit_yoloe.sh

You should see: `Submitted batch job 1234567`

No arguments needed. The job asks for 1 node, 4 GPUs, 64 CPUs, 24 hours, prepares (or just
verifies) the cached text embedding, splits the images across the 4 GPUs for Phase 1, and then
runs Phase 2 once. With one pass per image this is by far the shortest job.


Step 8 — Watch it
------------------
    squeue -u $USER
    tail -f $SCRATCH/experiments/yoloe/text_only_no_tiling_yoloe_1234567.out
    tail -f $SCRATCH/experiments/yoloe/text_only_no_tiling_yoloe/shard0.log

Count how many images are finished so far:

    cat $SCRATCH/experiments/yoloe/text_only_no_tiling_yoloe/raw_detections/runs_manifest_text_only_no_tiling_yoloe_shard*.csv \
      | grep -c text_only_no_tiling_yoloe
    ls $SCRATCH/experiments/yoloe/text_only_no_tiling_yoloe/raw_detections/*.npz | wc -l

You will also get an email at BEGIN / END / FAIL.


Step 9 — Collect the results
-----------------------------
Everything lands in `$SCRATCH/experiments/yoloe/text_only_no_tiling_yoloe/`.

The file you actually quote in the thesis:

| File                           | What it is                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| metrics/experiment_summary.csv | the headline numbers — ONE row (`all_gt`), with mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU1 / IoU2 |

The mean is taken over the image-level values, so an image with 40 plants does not outweigh
one with 2. The std is the variation between UAV images. The column layout matches the SAM3
`text_only_no_tiling` output (plus `model`, `imgsz` and `prompt_mode` columns, and without
`mask_threshold`), so the tables can be concatenated directly.

Everything else in that folder:

| File                                                  | What it is |
| ----------------------------------------------------- | ---------- |
| metrics/experiment_summary_per_archive.csv            | the same table split by archive |
| metrics/image_level_metrics.csv                       | one row per image (no anchors in text-only) — incl. TP/FP/FN, `text_prompt`, `gt_median_side_px_at_model_input` (how big the plants are in the 1024 px input), `archive` and `flight` |
| metrics/dataset_ap_metrics.csv                        | pooled AP50 / AP50:95 — all images ranked in ONE precision-recall curve. **Not** the mean of the image-level AP |
| confusion_matrices/confusion_matrix_all_gt.csv + .png | pooled TP / FP / FN, plus micro precision/recall/F1 in `confusion_matrix_summary.csv` |
| plots/best_image_*.png                                | QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right) — one per archive + one global |
| raw_detections/*.npz                                  | the PRE-NMS detections of every image (boxes, scores, GT boxes, text prompt, archive, flight, resize scale). This is what Phase 2 reads |
| raw_detections/runs_manifest_text_only_no_tiling_yoloe_shard*.csv | which images are finished; this is what resume reads |
| run_config_text_only_no_tiling_yoloe.json             | every parameter used, for the thesis appendix |
| shard0..3.log                                         | per-GPU logs |
| `$SCRATCH/yoloe_weights/text_pe/yoloe-11l-seg__rumex_obtusifolius.pt` | the cached text embedding (outside the results folder; shared with the `1_exemplar_text_*_yoloe` experiments) |

Which mode to quote: ONLY `all_gt`. Text-only prompting uses no GT box as input, so every GT
box of the image is evaluated. The `held_out` mode of the exemplar experiments does not apply
here: there is no prompt plant to remove.


Step 10 — Resubmit if it hit the 24 h limit
--------------------------------------------
Unlikely with one pass per image, but harmless. Just run the same command again:

    sbatch text_only_no_tiling_submit_yoloe.sh

Every finished image is listed in the shard manifests and is skipped before any GPU work, so
the job picks up exactly where it stopped:

    Resuming: 4821 image(s) already finished for text_only_no_tiling_yoloe; skipped.


Step 11 — Re-run only the evaluation (optional, cheap)
-------------------------------------------------------
Phase 2 reads only the NPZ files, so you never need a GPU for it:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    ./text_only_no_tiling_run_yoloe.sh evaluate

Use this to change the operating point without re-running YOLOE:

    OPERATING_CONFIDENCE=0.40 NMS_IOU_THRESHOLD=0.50 \
    OUTPUT_DIR=$SCRATCH/experiments/yoloe/text_only_no_tiling_yoloe \
    ./text_only_no_tiling_run_yoloe.sh evaluate

Careful: this overwrites the CSVs in `metrics/`, `confusion_matrices/` and `plots/`.

`evaluate` needs `DATASET_ROOT` to be reachable, because the qualitative figures reopen the
selected images. If the dataset has been purged, add `--no-plots`:

    ./text_only_no_tiling_run_yoloe.sh evaluate --no-plots


Notes and problems
==================

Problem A — YOLOE import FAILED
--------------------------------------------------
The `yolo26` container's ultralytics is too old for YOLOE (the other YOLOE notebooks used
ultralytics 8.4.163). Build a dedicated image:

    # Dockerfile
    FROM <whatever image ~/.edf/yolo26.toml points at>
    RUN pip install --no-cache-dir "ultralytics>=8.4.163" supervision pandas matplotlib
    podman build -t yoloe .
    enroot import -o $SCRATCH/images/yoloe.sqsh podman://yoloe:latest
    cp ~/.edf/yolo26.toml ~/.edf/yoloe.toml
    # edit ~/.edf/yoloe.toml so `image = ` points at $SCRATCH/images/yoloe.sqsh

then change the one line at the bottom of `text_only_no_tiling_submit_yoloe.sh`:

    srun --environment=yoloe ...

and use `--environment=yoloe` in Step 4 as well.


Problem C — the text embedding cannot be computed on the cluster
------------------------------------------------------------------
STEP 0 needs the YOLOE text encoder once. If the download mode could not fetch it, or the
container cannot import the CLIP tokenizer, compute the SAME file in Colab (it has internet)
and copy it over. The embedding depends only on the text and the checkpoint, so it is
identical wherever it is computed. In a Colab cell:

    !pip install -q -U ultralytics
    import torch
    from ultralytics import YOLOE
    model = YOLOE("yoloe-11l-seg.pt")
    text = "Rumex obtusifolius"
    torch.save({"text_prompt": text, "weights": "yoloe-11l-seg.pt",
                "text_pe": model.get_text_pe([text]).detach().float().cpu()},
               "yoloe-11l-seg__rumex_obtusifolius.pt")

Then copy the file to the cluster:

    scp yoloe-11l-seg__rumex_obtusifolius.pt <user>@<cscs login>:$SCRATCH/yoloe_weights/text_pe/

The file name, `text_prompt` and `weights` must match exactly — the code refuses a cache made
for another text or checkpoint. STEP 0 then prints `Text embedding already cached` and never
needs the text encoder.


Problem B — `supervision` / `pandas` / `matplotlib` import FAILED inside the container
---------------------------------------------------------------------------------------
Step 3 installed with the login node's Python. If the container uses a different Python
version, the install is invisible to it. Install from inside the container instead:

    # inside the container (Step 4 shell) — pick whichever package failed
    pip install --target $SCRATCH/pyextra --no-deps --no-index \
        --find-links $SCRATCH/wheels supervision
    pip install --target $SCRATCH/pyextra --no-deps --no-index \
        --find-links $SCRATCH/wheels pandas pytz tzdata python-dateutil six
    pip install --target $SCRATCH/pyextra --no-deps --no-index \
        --find-links $SCRATCH/wheels matplotlib contourpy cycler fonttools kiwisolver pyparsing packaging

Always `--no-deps`: otherwise those packages would install their own numpy / pillow into
`$SCRATCH/pyextra`, which is searched first and would break torch.


What the fixed operating point means for your numbers
------------------------------------------------------
`OPERATING_CONFIDENCE = 0.30` and `NMS_IOU_THRESHOLD = 0.40` are fixed from the start, exactly
as in the notebook (`BEST_CONFIDENCE`, `BEST_NMS_IOU`). No sweep is performed.

- **precision / recall / F1 / IoU1 / IoU2** describe ONE operating point: only detections
  scoring ≥ 0.30.
- **AP50 / AP50:95** always use every post-NMS detection ≥ 0.30 (the YOLOE inference
  threshold), because AP is the area under the precision-recall curve.

If you want a lower floor for the AP curve you must re-run Phase 1 with a lower `THRESHOLD`:

    THRESHOLD=0.05 EXPERIMENT_NAME=text_only_no_tiling_yoloe_lowconf \
    OUTPUT_DIR=$SCRATCH/experiments/yoloe/text_only_no_tiling_yoloe_lowconf sbatch text_only_no_tiling_submit_yoloe.sh


Other troubleshooting
----------------------
| Symptom                         | Fix |
| ------------------------------- | --- |
| SCRATCH: unbound variable       | you are not on the cluster, or the module environment is not loaded |
| `... does not exist and the compute node is offline` | the weights are missing — redo Step 3 |
| hangs / fails at STEP 0         | the text encoder is missing on the offline node — see **Problem C** |
| dtype error in set_classes      | `USE_FP16=0 sbatch text_only_no_tiling_submit_yoloe.sh` |
| CUDA out of memory              | unlikely (one 1024 px image per pass); lower `NUM_GPUS` only if other jobs share the GPUs |
| Nothing to plot in Phase 2      | no image had a valid AP50, or `DATASET_ROOT` is unreachable. Add `--no-plots` |
| Everything reports F1=0.0000    | wrong class id or wrong labels (recheck Step 5b) — or the text prompt finds nothing at 0.30 |
| `... already holds runs made with another text prompt` | the results folder contains runs of another `TEXT_PROMPT` — use a new `EXPERIMENT_NAME` / `OUTPUT_DIR` |
| `... holds the text embedding of '...'` | the cache file belongs to another text or checkpoint — delete it or set `TEXT_PE_CACHE` |


Settings you might change
--------------------------
Set them before `sbatch`; they are forwarded into the job.

    USE_FP16=0                sbatch text_only_no_tiling_submit_yoloe.sh   # full precision
    THRESHOLD=0.05            sbatch text_only_no_tiling_submit_yoloe.sh   # save more detections for the AP curve
    OPERATING_CONFIDENCE=0.40 sbatch text_only_no_tiling_submit_yoloe.sh   # different operating point
    NMS_IOU_THRESHOLD=0.50    sbatch text_only_no_tiling_submit_yoloe.sh   # different NMS
    MAX_DIM=2048              sbatch text_only_no_tiling_submit_yoloe.sh   # larger whole-image input (IMGSZ follows)
    TEXT_PROMPT="dock weed"   sbatch text_only_no_tiling_submit_yoloe.sh   # another phrase (new text embedding)
    YOLOE_WEIGHTS_NAME=yoloe-11m-seg.pt  (download + sbatch)       # a smaller YOLOE
    NUM_GPUS=2                sbatch text_only_no_tiling_submit_yoloe.sh   # fewer GPUs

Changing `TEXT_PROMPT`, `MAX_DIM`, `IMGSZ`, `THRESHOLD` or the weights changes what is stored
in the NPZ files, so give those runs their own `EXPERIMENT_NAME` and `OUTPUT_DIR` (for
`TEXT_PROMPT` the job refuses to mix them anyway; a new phrase gets its own cached text
embedding automatically). Changing only `OPERATING_CONFIDENCE` / `NMS_IOU_THRESHOLD` does not —
use Step 11 instead, it is free.
