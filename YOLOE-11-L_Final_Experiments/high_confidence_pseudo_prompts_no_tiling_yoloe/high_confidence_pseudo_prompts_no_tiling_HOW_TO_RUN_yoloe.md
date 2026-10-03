HOW TO RUN — Experiment `high_confidence_pseudo_prompts_no_tiling_yoloe`
============================================================================

What this experiment is: YOLOE (`yoloe-11l-seg.pt`, imgsz 1024), TWO rounds per image, NO tiling
(the whole image is resized to 1024 px on its longest side). Round 1 is prompted with 3
SIZE-BASED GT exemplars — the Smallest, the Medium (closest to the median area) and the Largest
GT box. From the round-1 detections (after NMS, score ≥ 0.30, not on a prompt plant) the 2
highest-confidence boxes are taken as SELF-PROMPTS — YOLOE's own predictions, not ground truth —
and round 2 is prompted with S, M, L + those 2 on the same resized image. The round-2 output is
final (round 1 is reported for comparison); if round 1 leaves no eligible box, round 2 is
skipped. Run over both `AGS_Multi_Rumex` and
`AgsSpringRumex` pooled as one dataset. The S/M/L choice is deterministic (ties -> lower GT
index; images with fewer than 3 GT boxes get [S] or [S, L]), so there is exactly ONE run per
image — no anchors, no random sampling, no seeds.
This is the cluster port of `E01_single_image_pseudo_prompts_YOLOE_11_no_tiling.ipynb`
(the YOLOE version of the SAM3 notebook `high_confidence_pseudo_prompts_no_tiling`). It is a
two-phase pipeline:

   PHASE 1  INFERENCE   (GPU)      per image: pick S/M/L by area -> resize ONCE to 1024 px
                                   -> ROUND 1 (VPE from S/M/L on that image, one YOLOE
                                   pass) -> NMS -> 2 self-prompts -> ROUND 2 (VPE from
                                   S/M/L + self-prompts on the same image, one pass) ->
                                   NPZ with round 1, the self-prompts and the final output
   PHASE 2  EVALUATION  (no GPU)   NMS 0.40 -> metrics at confidence 0.30
                                   -> per-image / experiment / pooled-AP CSVs, each
                                      with the round-1 numbers and the deltas
                                   -> recall per size group + prompt re-detection
                                   -> confusion matrices (CSV + PNG)
                                   -> qualitative GT-vs-prediction figures

Phase 2 never touches YOLOE, so once Phase 1 is done you can rebuild every number in minutes.

Same as the SAM3 `high_confidence_pseudo_prompts_no_tiling` pipeline: whole image at 1024 px,
S/M/L (and in round 2 also the self-prompts) as prompts on the same image, the same
self-prompt rule, NO plausibility filter, plain offline NMS (no tile provenance).
Differences (all from the notebook):
- YOLOE first encodes the scaled prompt boxes (3 in round 1, 5 in round 2) into one VPE from
  the SAME resized image (`same_image_vpe`), then runs a plain `predict()` on it — exemplars
  and targets are seen at the same scale (0.125× for an 8192 px image, so a 110 px plant is
  ~14 px)
- YOLOE's own NMS is set permissive (0.90), so the offline NMS (0.40) decides

   PHASE A — SETUP            do once, ~20 minutes
   ├─ Step 1   copy the files onto the cluster
   ├─ Step 2   put the dataset on $SCRATCH
   ├─ Step 3   download the weights and the extra packages
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
    ├── high_confidence_pseudo_prompts_no_tiling_infer_yoloe.py
    ├── high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh
    └── high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh

Then, on a login node:

    mkdir -p $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    chmod +x high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh
    ls -l

You should see: three files, with the two `.sh` marked executable (`-rwxr-xr-x`).

If you edited the `.sh` files on Windows, strip the Windows line endings once:

    sed -i 's/\r$//' high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh


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
the checkpoint must be fetched on a login node:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    ./high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh download

This does three things:
- downloads `yoloe-11l-seg.pt` into `$SCRATCH/yoloe_weights/`
  (from the Ultralytics GitHub release; if `curl` fails it falls back to Ultralytics' own
  downloader)
- installs the `supervision` package into `$SCRATCH/pyextra`
  (`supervision` computes AP50 / AP50:95; the code refuses to run without it)
- downloads wheels for the Phase 2 packages (`pandas`, `matplotlib`) into `$SCRATCH/wheels`,
  in case the container does not ship them — see Problem B

`$SCRATCH/pyextra` and `$SCRATCH/wheels` are shared with the SAM3 experiments, so steps 2/3
of the download just confirm what is already there.

You should see:

    --- 1/3 : YOLOE weights -> /scratch/.../yoloe_weights/yoloe-11l-seg.pt ---
    -rw-r--r-- ... yoloe-11l-seg.pt
    --- 2/3 : supervision -> /scratch/.../pyextra ---
    supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
    --- 3/3 : wheels for the PHASE 2 packages -> /scratch/.../wheels ---
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
    ./high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh dryrun

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

### 5a. Look for these lines in the debug block

    torch: 2.x.x | cuda available: True | device count: 1
    ultralytics: 8.x.x ...
    YOLOE import: OK
    YOLOEVPSegPredictor.get_vpe: OK
    precision argument: quantize=16 (as in the notebook)      (or: half=True ...)
    weights file: /scratch/.../yoloe-11l-seg.pt -> FOUND
    MeanAveragePrecision import: OK
    pandas: 2.x.x -> PHASE 2 import: OK
    matplotlib: 3.x.x -> confusion-matrix + qualitative PNGs: OK

| If you see                          | Meaning                                                           | Fix                                           |
| ----------------------------------- | ----------------------------------------------------------------- | --------------------------------------------- |
| YOLOE import FAILED / get_vpe MISSING | the `yolo26` image's ultralytics is too old for YOLOE visual prompts | see **Problem A** at the bottom            |
| precision argument: half=True       | older ultralytics than the notebook's 8.4.163; the code uses `half=True` instead of `quantize=16` automatically | nothing to do |
| weights file ... MISSING            | Step 3 did not finish                                             | rerun Step 3                                  |
| supervision import FAILED           | `$SCRATCH/pyextra` not found or wrong Python version              | rerun Step 3; if it still fails, see **Problem B** |
| pandas import FAILED                | Phase 1 still works, but Phase 2 cannot run                       | see **Problem B**                             |
| matplotlib import FAILED            | only the PNGs are lost, every CSV is still written                | see **Problem B**, or ignore it               |
| cuda available: False               | no GPU in this allocation                                         | add `--gpus-per-task=1` to Step 4             |

Do not continue until the YOLOE, get_vpe, weights and supervision lines are correct.

### 5b. Check the dataset report

    Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
      AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
      AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
    Total images: NNN | this shard: NNN

It also prints a cost estimate:

      sampled NNN image(s) of this shard -> NNN run(s) (one per image with GT; {...})
      images with fewer than 3 GT boxes (fewer prompts): N
      no tiling: an 8192x5460 image is resized to 1024x682 (scale 0.125)
      => up to ~NNN YOLOE forward passes (+ up to NNN VPE encodings) for those NNN images - two rounds per image; ...
      NPZ files that will be written by this shard: ~NNN (one per image)

If it says `No images with labels found` → `$SCRATCH/overney/dataset` is wrong. Go back to Step 2.


Step 6 — Smoke test on 2 images
--------------------------------
Still inside the interactive shell. This is the first time the model actually loads.
It writes to a throwaway folder so it cannot pollute the real results.

    EXPERIMENT_NAME=high_confidence_pseudo_prompts_no_tiling_yoloe_smoke \
    OUTPUT_DIR=$SCRATCH/experiments/yoloe/_smoke_high_confidence_pseudo_prompts_no_tiling_yoloe \
    NUM_GPUS=1 \
    ./high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh run --limit-images 2

You should see the model load, then one line per image, then the whole Phase 2:

    Loading YOLOE from '/scratch/.../yoloe-11l-seg.pt' onto cuda (half=True, {'quantize': 16}) ...
    YOLOE loaded.
      [high_confidence_pseudo_prompts_no_tiling_yoloe_smoke] shard0 (1/2) AGS_Multi_Rumex/2022.../DJI_0001 | 57 GT box(es) | prompts=S31+M8+L19 | input=1024x682 | round1=4 det -> self-prompts=2 (0.63, 0.31) -> final=19 det | 1.6s | ...
    ...
    ===== PHASE 2 : OFFLINE EVALUATION =====
    --- per-image metrics ---
      all_gt   : NN images valid for macro averaging, AP50_mean=0.xxxx, F1_mean=0.xxxx
      held_out : NN images valid for macro averaging, AP50_mean=0.xxxx, F1_mean=0.xxxx

Things to check in this output:

- `1.6s` (or whatever you get) is the time for ONE image: up to two VPE encodings + two
  1024 px forward passes. Multiply by the number of images from Step 5b to estimate the whole
  job — still cheap, since there is no tiling.
- Few detections per run is expected: plants are only ~14 px at this scale. Images where
  round 1 finds no eligible box print `NOTE: no eligible round-1 box ... round 2 skipped` —
  expect this more often than with tiling.
- On the notebook's test image (run at confidence 0.10) one self-prompt was a real plant and
  one a false positive; round 2 changed F1 0.095 -> 0.113 and AP50 0.055 -> 0.048.
  `delta_AP50` / `delta_F1` and `n_self_prompts_on_gt` in the CSVs show the dataset-wide effect.
- `pre-NMS detections=` should not be 0 on every run. All zeros means the visual prompt did
  not take — check the `get_vpe` line of Step 5a.
- `F1_mean=` should be a plausible number in `all_gt`, not `0.0000` everywhere. All zeros
  means something is wrong with the labels or the class id.
- `held_out` legitimately shows fewer valid runs than `all_gt`: on an image where every plant
  was used as a prompt, there is no GT left to evaluate (`valid_for_macro = False`).
- The `prompts=` field shows the roles and GT indices (e.g. `S12+M5+L40`); images with
  fewer than 3 GT boxes print a `NOTE: only N GT box(es)` line.
- A dtype error while the visual prompt is applied → rerun with `USE_FP16=0`.

Then leave the interactive session:

    exit


PHASE B — RUN
=============

Step 7 — Submit the job
------------------------
On a login node:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh

You should see: `Submitted batch job 1234567`

No arguments needed. The job asks for 1 node, 4 GPUs, 64 CPUs, 24 hours, splits the images
across the 4 GPUs for Phase 1, and then runs Phase 2 once.


Step 8 — Watch it
------------------
    squeue -u $USER
    tail -f $SCRATCH/experiments/yoloe/high_confidence_pseudo_prompts_no_tiling_yoloe_1234567.out
    tail -f $SCRATCH/experiments/yoloe/high_confidence_pseudo_prompts_no_tiling_yoloe/shard0.log

Count how many runs are finished so far:

    cat $SCRATCH/experiments/yoloe/high_confidence_pseudo_prompts_no_tiling_yoloe/raw_detections/runs_manifest_high_confidence_pseudo_prompts_no_tiling_yoloe_shard*.csv \
      | grep -c high_confidence_pseudo_prompts_no_tiling_yoloe
    ls $SCRATCH/experiments/yoloe/high_confidence_pseudo_prompts_no_tiling_yoloe/raw_detections/*.npz | wc -l

You will also get an email at BEGIN / END / FAIL.


Step 9 — Collect the results
-----------------------------
Everything lands in `$SCRATCH/experiments/yoloe/high_confidence_pseudo_prompts_no_tiling_yoloe/`.

The file you actually quote in the thesis:

| File                           | What it is                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| metrics/experiment_summary.csv | the headline numbers — one row per evaluation mode (`all_gt`, `held_out`), with mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU1 / IoU2 |

The mean is taken over the image-level values, so an image with 40 plants does not outweigh
one with 2. The std is the variation between UAV images. The column layout matches the
SAM3 `high_confidence_pseudo_prompts_no_tiling` output (plus `model`, `max_dim`, `imgsz` and
`prompt_mode` columns). It also carries the round-1 means (`*_round1_mean`), `delta_AP50_mean` /
`delta_F1_mean`, `n_images_with_second_round` and `self_prompts_on_gt_share` (the share of
self-prompts that sit on a real plant), so the
tables can be concatenated directly.

Everything else in that folder:

| File                                                  | What it is |
| ----------------------------------------------------- | ---------- |
| metrics/experiment_summary_per_archive.csv            | the same table split by archive |
| metrics/size_group_recall.csv                         | per image: recall of the NON-prompt GT boxes in the smaller half (area ≤ median) and the larger half |
| metrics/prompt_plants_redetected.csv                  | per image and role (S/M/L): was the prompt plant itself detected again? |
| metrics/image_level_metrics.csv                       | one row per (image × mode) — one run per image, so this replaces the run-level table; incl. TP/FP/FN, `Prompt_ID`, the median GT area, the self-prompts (`n_self_prompts`, their scores, `n_self_prompts_on_gt`, `second_run`) and the same metrics for round 1 (`*_round1`, `delta_AP50`, `delta_F1`) |
| metrics/dataset_ap_metrics.csv                        | pooled AP50 / AP50:95 — all runs ranked in ONE precision-recall curve (final and round 1). **Not** the mean of the image-level AP |
| confusion_matrices/confusion_matrix_{all_gt,held_out}.csv + .png | pooled TP / FP / FN, plus micro precision/recall/F1 in `confusion_matrix_summary.csv` |
| plots/best_image_*.png                                | QUALITATIVE PLOT: BEST IMAGE, GT (left) vs PREDICTIONS (right) — one per archive + one global |
| raw_detections/*.npz                                  | the PRE-NMS detections of every image (round 1 AND final boxes + scores, the self-prompt boxes + scores, `second_run`, GT boxes, S/M/L indices + roles + areas, median GT area, resize scale). This is what Phase 2 reads |
| raw_detections/runs_manifest_high_confidence_pseudo_prompts_no_tiling_yoloe_shard*.csv | which images are finished (incl. the S/M/L areas); this is what resume reads |
| run_config_high_confidence_pseudo_prompts_no_tiling_yoloe.json               | every parameter used, for the thesis appendix |
| shard0..3.log                                         | per-GPU logs |

Which mode to quote: both, and say what they mean.

- `all_gt` = classical evaluation, every GT box counts.
- `held_out` = the plants shown to YOLOE as prompts are removed from the GT and the
  predictions that land on them are ignored — "after being shown a few examples, how well
  does it find the remaining plants?".


Step 10 — Resubmit if it hit the 24 h limit
--------------------------------------------
Completely normal, and harmless. Just run the same command again:

    sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh

Every finished image is listed in the shard manifests and is skipped before
any GPU work, so the job picks up exactly where it stopped:

    Resuming: 4821 run(s) already finished for high_confidence_pseudo_prompts_no_tiling_yoloe; skipped.


Step 11 — Re-run only the evaluation (optional, cheap)
-------------------------------------------------------
Phase 2 reads only the NPZ files, so you never need a GPU for it:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/yoloe
    ./high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh evaluate

Use this to change the operating point without re-running YOLOE:

    OPERATING_CONFIDENCE=0.40 NMS_IOU_THRESHOLD=0.50 \
    OUTPUT_DIR=$SCRATCH/experiments/yoloe/high_confidence_pseudo_prompts_no_tiling_yoloe \
    ./high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh evaluate

Careful: this overwrites the CSVs in `metrics/`, `confusion_matrices/` and `plots/`.

`evaluate` needs `DATASET_ROOT` to be reachable, because the qualitative figures reopen the
selected images. If the dataset has been purged, add `--no-plots`:

    ./high_confidence_pseudo_prompts_no_tiling_run_yoloe.sh evaluate --no-plots


Notes and problems
==================

If it will not fit in 24 hours
-------------------------------
1. **Do nothing.** Resubmit as many times as needed (Step 10).
2. More GPUs are not needed: with one run per image this job is short.


Problem A — YOLOE import FAILED / get_vpe MISSING
--------------------------------------------------
The `yolo26` container's ultralytics is too old for YOLOE visual prompts (the notebook used
ultralytics 8.4.163). Build a dedicated image:

    # Dockerfile
    FROM <whatever image ~/.edf/yolo26.toml points at>
    RUN pip install --no-cache-dir "ultralytics>=8.4.163" supervision pandas matplotlib
    podman build -t yoloe .
    enroot import -o $SCRATCH/images/yoloe.sqsh podman://yoloe:latest
    cp ~/.edf/yolo26.toml ~/.edf/yoloe.toml
    # edit ~/.edf/yoloe.toml so `image = ` points at $SCRATCH/images/yoloe.sqsh

then change the one line at the bottom of `high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh`:

    srun --environment=yoloe ...

and use `--environment=yoloe` in Step 4 as well.


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

    THRESHOLD=0.05 EXPERIMENT_NAME=high_confidence_pseudo_prompts_no_tiling_yoloe_lowconf \
    OUTPUT_DIR=$SCRATCH/experiments/yoloe/high_confidence_pseudo_prompts_no_tiling_yoloe_lowconf sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh


Other troubleshooting
----------------------
| Symptom                         | Fix |
| ------------------------------- | --- |
| SCRATCH: unbound variable       | you are not on the cluster, or the module environment is not loaded |
| `... does not exist and the compute node is offline` | the weights are missing — redo Step 3 |
| dtype error in set_classes / get_vpe | `USE_FP16=0 sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh` |
| CUDA out of memory              | unlikely (one 1024 px image per pass); lower `NUM_GPUS` only if other jobs share the GPUs |
| Nothing to plot in Phase 2      | no image had a valid AP50, or `DATASET_ROOT` is unreachable. Add `--no-plots` |
| Everything reports F1=0.0000    | wrong class id or wrong labels — recheck Step 5b |
| held_out is all NaN             | every plant of every image was used as a prompt — expected only on images with ≤ 3 GT boxes |
| `... holds results for another prompt setting` | the output folder already contains another experiment's runs — use a new `EXPERIMENT_NAME` / `OUTPUT_DIR` |


Settings you might change
--------------------------
Set them before `sbatch`; they are forwarded into the job.

    USE_FP16=0                sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh   # full precision
    THRESHOLD=0.05            sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh   # save more detections for the AP curve
    OPERATING_CONFIDENCE=0.40 sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh   # different operating point
    NMS_IOU_THRESHOLD=0.50    sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh   # different NMS
    MAX_DIM=2048              sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh   # larger whole-image input (IMGSZ follows)
    N_SELF_PROMPTS=1          sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh   # one self-prompt instead of two
    THRESHOLD=0.10 OPERATING_CONFIDENCE=0.10 EXPERIMENT_NAME=high_confidence_pseudo_prompts_no_tiling_yoloe_conf010 \
      OUTPUT_DIR=$SCRATCH/experiments/yoloe/high_confidence_pseudo_prompts_no_tiling_yoloe_conf010 \
                              sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh   # the 0.10 setting of the notebook
    YOLOE_WEIGHTS_NAME=yoloe-11m-seg.pt  (download + sbatch)             # a smaller YOLOE
    NUM_GPUS=2                sbatch high_confidence_pseudo_prompts_no_tiling_submit_yoloe.sh   # fewer GPUs

Changing `N_SELF_PROMPTS`, `SELF_PROMPT_EXCLUDE_IOU`, `MAX_DIM`, `IMGSZ`, `THRESHOLD` or the
weights changes what is stored in the NPZ files, so give those runs their own
`EXPERIMENT_NAME` and `OUTPUT_DIR`. Changing only `OPERATING_CONFIDENCE` /
`NMS_IOU_THRESHOLD` does not — use Step 11 instead, it is free.
