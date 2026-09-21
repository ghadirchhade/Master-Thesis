HOW TO RUN — Experiment `pos_neg_pseudo_prompts_tiling`

What this experiment is: SAM3 run TWICE per image, tiling ON (1000px tiles, 150px overlap).
Round 1 uses 3 size-based ground-truth plants as exemplars; round 2 adds SAM3's own round-1
predictions as extra POSITIVE and NEGATIVE exemplars. Run over both `AGS_Multi_Rumex` and
`AgsSpringRumex` pooled as one dataset.

How it works in one line: the UAV image is split into overlapping tiles and an exemplar strip
is composed above each tile (local background and feathering); the tile + strip is sent to
SAM3 with the strip boxes as box prompts. Detections are filtered by target-region (>= 50% area
in tile) and plausibility (fill ratio, max area, edge margin), then mapped back to full image
coords.

    GT boxes of the image
      -> S = smallest area, L = largest, M = closest to the median area       (CELL 14)
      -> ROUND 1: strip = crops of S, M, L (label 1), SAM3 over every tile     (CELL 12, 13, 15)
      -> NMS 0.40 on the round-1 detections of the whole image                  (CELL 16)
      -> drop every detection with IoU >= 0.50 to S, M or L (the prompt plants again)
      -> POSITIVE self-prompts: the 2 HIGHEST-confidence remaining detections (picked first)
      -> NEGATIVE self-prompts: of the rest, the ones below 0.50, the 2 LOWEST   (CELL 17)
      -> ROUND 2: strip = S, M, L, 2 positive crops (label 1), 2 negative crops (label 0),
                  SAM3 over every tile again
      -> FINAL = round-2 output (round 1 is NOT merged in)
    If no self-prompt exists at all, round 2 would repeat round 1 exactly, so it is skipped
    and round 1 is the final output (second_run = False).

The S/M/L choice is deterministic (ties -> lower GT index), so there is exactly ONE run per
image — no anchors. The only random ingredient is the strip background, seeded with SHA-256 of
(experiment name, image_ID), so every run reproduces itself exactly.

The self-prompts are SAM3's own predictions, not ground truth: a false positive fed back as a
positive (or a real plant fed back as a negative) can pull round 2 off. That is why every metric
is ALSO computed for the round-1 output ( `*_round1`  columns,  `delta_AP50` ,  `delta_F1` ), so the
effect of the self-prompts is visible per image.

This is the cluster port of  `pos_neg_pseudo_prompts_tiling.ipynb` . Like the notebook, it is a
two-phase pipeline:

PHASE 1  INFERENCE   (GPU)      per image: round 1 over all tiles, self-prompts, round 2 over
                                all tiles, in batches of 4 tiles at score 0.30
                                -> round-1 + self-prompts + final detections saved as NPZ

PHASE 2  EVALUATION  (no GPU)   NMS 0.40 (with cross-tile provenance) -> metrics at
                                confidence 0.30 (final AND round 1)
                                -> image / experiment / per-archive / pooled-AP CSVs
                                -> confusion matrices (CSV + PNG)
                                -> recall per size group, prompt plants re-detected
                                -> qualitative GT-vs-prediction figures

Phase 2 never touches SAM3, so once Phase 1 is done you can rebuild every number in minutes.

What differs from the notebook (the pipeline, the functions and the parameters are the same):
- both archives instead of  `AGS_Multi_Rumex`  only, each with its own Rumex class id;
- `archive`  and  `flight`  columns in every CSV ( `ALL`  in the pooled tables);
- `metrics/experiment_summary_per_archive.csv`  and one qualitative figure per archive
  (+ one over ALL images);
- SAM3 in bfloat16 (the notebook used float16 on a Colab T4,  `USE_FP16 = True` );
- one manifest per GPU shard instead of the single  `runs_manifest.csv` .


PHASE A — SETUP            do once, ~30 minutes
├─ Step 1   copy the files onto the cluster
├─ Step 2   put the dataset on $SCRATCH
├─ Step 3   HuggingFace licence + token
├─ Step 4   download the model and the extra packages
├─ Step 5   open an interactive session inside the container
├─ Step 6   check the container has what it needs
└─ Step 7   smoke test on 2 images

PHASE B — RUN              every time
├─ Step 8   submit the job
├─ Step 9   watch it
├─ Step 10  collect the results
├─ Step 11  resubmit if it hit the walltime
└─ Step 12  re-run only the evaluation (optional, cheap)


================================================================================
PHASE A — SETUP (once)
================================================================================

Step 1 — Copy the files onto the cluster
----------------------------------------
Put the three scripts here, next to the other experiments:

    $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3/
    ├── pos_neg_pseudo_prompts_tiling_infer_sam3.py
    ├── pos_neg_pseudo_prompts_tiling_run_sam3.sh
    └── pos_neg_pseudo_prompts_tiling_submit_sam3.sh

Then, on a login node:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
    chmod +x pos_neg_pseudo_prompts_tiling_run_sam3.sh pos_neg_pseudo_prompts_tiling_submit_sam3.sh
    ls -l

You should see: the two `.sh` marked executable (`-rwxr-xr-x`).

Important: the files use Unix (LF) line endings. If you edit them on Windows, make sure your
editor keeps LF — a `.sh` with CRLF endings fails with `syntax error near unexpected token` or
`$'\r': command not found`. Fix with:

    sed -i 's/\r$//' pos_neg_pseudo_prompts_tiling_*


Step 2 — Put the dataset on  `$SCRATCH`
---------------------------------------
If you already did this for another experiment, skip this step. All the experiments read
the same  `$SCRATCH/overney/dataset` .

Extract the two archives side by side under one folder:

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

Check it:

    ls $SCRATCH/overney/dataset
    ls $SCRATCH/overney/dataset/AGS_Multi_Rumex
    ls $SCRATCH/overney/dataset/AGS_Multi_Rumex/images | head

Important: the folder names are case-sensitive and must match exactly. The other two
archives ( `AGS_Multiple_Fields` ,  `AGS_Multiple_Fields_Embeddings` ) may be present — the code
detects them and skips them on purpose. The labels are found exactly like in the notebook
( `find_label_path` :  `annotations_yolo/<flight>/<name>.txt` , then  `annotations_yolo/<name>.txt` ),
with a one-time index of every  `.txt`  under  `annotations_yolo`  as a last fallback.

Note on the class ids. The notebook ran on  `AGS_Multi_Rumex`  alone with
 `RUMEX_CLASS_ID = 0` . Both archives are used here and they label Rumex with different class
ids, so the id is a property of the archive:

| Archive          | Rumex class id in its YOLO files |
|------------------|----------------------------------|
| AGS_Multi_Rumex  | 0                                |
| AgsSpringRumex   | 2                                |

Every other class in those files is ignored.

Why  `$SCRATCH`  and not  `$HOME` :  `$HOME`  is small and slow;  `$SCRATCH`  is the large fast
filesystem. (It is also periodically purged, so keep the original archives somewhere safe.)


Step 3 — HuggingFace licence + token
------------------------------------
`facebook/sam3` is a gated model. Two things are needed:

1. Go to https://huggingface.co/facebook/sam3 while logged in and accept the licence.
2. Create a token at https://huggingface.co/settings/tokens (read access is enough).

Then, on a login node:

    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

Note: the token must belong to the same account that accepted the licence, otherwise
Step 4 fails with a 401.


Step 4 — Download the model and the extra packages
--------------------------------------------------
If you already ran the  `download`  mode of any other SAM3 experiment, skip this step.
They all share  `$SCRATCH/hf_cache` ,  `$SCRATCH/pyextra`  and  `$SCRATCH/wheels` .

Still on a login node (this is the only step that needs internet — compute nodes have none):

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
    ./pos_neg_pseudo_prompts_tiling_run_sam3.sh download

This does three things:
- downloads the SAM3 weights (and processor) into  `$SCRATCH/hf_cache`
- installs the  `supervision`  package into  `$SCRATCH/pyextra`
  ( `supervision`  computes AP50 / AP50:95; the code refuses to run without it)
- downloads wheels for the Phase 2 packages ( `pandas` ,  `matplotlib` ) into  `$SCRATCH/wheels` ,
  in case the container does not ship them — see Problem B

You should see:

    --- 1/3 : model weights -> /scratch/.../hf_cache ---
    Snapshot cached at: /scratch/.../models--facebook--sam3/snapshots/...
    --- 2/3 : supervision -> /scratch/.../pyextra ---
    Successfully installed supervision-...
    supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
    --- 3/3 : wheels for the PHASE 2 packages -> /scratch/.../wheels ---
    Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1).

If it says  `ERROR: HF_TOKEN is not set`  → go back to Step 3.
If it says  `401`  or  `gated`  → the licence was not accepted with that token's account.


Step 5 — Open an interactive session inside the container
---------------------------------------------------------
Steps 6 and 7 must run inside the container, not on the login node — otherwise you are
checking the login node's Python, which is not the one the job will use.

    srun --account=go077 --time=00:30:00 \
         --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
         --environment=yolo26 --pty bash

You should see: a new shell prompt, running on a compute node. Everything in Steps 6 and 7
happens in this shell.


Step 6 — Check the container has what it needs
----------------------------------------------
Inside the interactive shell from Step 5:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
    ./pos_neg_pseudo_prompts_tiling_run_sam3.sh dryrun

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

6a. Look for these lines in the debug block

    torch: 2.x.x | cuda available: True | device count: 1
    bfloat16 supported: True
    Sam3Model import: OK
    MeanAveragePrecision import: OK
    pandas: 2.x.x -> PHASE 2 import: OK
    matplotlib: 3.x.x -> confusion-matrix + qualitative PNGs: OK

| If you see                     | Meaning                                                | Fix                                  |
|--------------------------------|--------------------------------------------------------|--------------------------------------|
| Sam3Model import FAILED        | the yolo26 image's transformers is too old for SAM3    | see Problem A at the bottom          |
| supervision import FAILED      | $SCRATCH/pyextra not found or wrong Python version     | rerun Step 4; if it still fails, see Problem B |
| pandas import FAILED           | Phase 1 still works, but Phase 2 cannot run            | see Problem B                        |
| matplotlib import FAILED       | only the PNGs are lost (confusion matrices + qualitative figures); every CSV is still written | see Problem B, or ignore it |
| bfloat16 supported: False      | old GPU                                                | use DTYPE=float16                    |
| cuda available: False          | no GPU in this allocation                              | add --gpus-per-task=1 to Step 5      |

Do not continue until the first four lines are correct. Everything after this depends on them.

6b. Check the dataset report

    Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
      AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
      AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
    Discovered NNN images in N folders.
    With labels: NNN | without labels (skipped): N
    Total images: NNN | this shard: NNN

Confirm the image counts match what you expect, and that  `class_id`  is 0 for
 `AGS_Multi_Rumex`  and 2 for  `AgsSpringRumex` .

It then opens every image header and label file of the shard (CPU, no model) and prints the
real cost, computed from the real image sizes:

    --- DRY RUN: counting the work without loading SAM3 ---
      this shard: NNN image(s) -> NNN with GT boxes ({'AGS_Multi_Rumex': NNN, 'AgsSpringRumex': NNN}), N without (skipped)
      GT boxes in those images: NNNN; images with < 3 GT boxes (fewer S/M/L prompts): NN
      ONE run per image (S/M/L prompts, no anchors)
      tiles per image: 70.0 on average (tiling=True, tile=1000, overlap=150)
      SAM3 forward passes (batch=4): NNNN (round 1) + up to NNNN (round 2) = at most NNNN
      NPZ files that will be written by this shard: NNN (one per image)

An 8192 × 5460 image gives 70 tiles at 1000 / 150, i.e. 18 batched forward passes per round.
Round 2 depends on the round-1 output, so the dry run can only give the upper bound. Images
with 1 GT box get only  `S` , with 2 GT boxes  `S + L`  — exactly like the notebook.

If it says  `No images with labels found`  →  `$SCRATCH/overney/dataset`  is wrong, or the
folder names do not match. Go back to Step 2.
If  `without labels (skipped)`  is large → those images are skipped. A handful is normal; if it
is every image, the label basenames do not match the image basenames.


Step 7 — Smoke test on 2 images
-------------------------------
Still inside the interactive shell. This is the first time the model actually loads.
It writes to a throwaway folder so it cannot pollute the real results.

    EXPERIMENT_NAME=pos_neg_pseudo_prompts_tiling_smoke \
    OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke_pos_neg_pseudo_prompts_tiling \
    NUM_GPUS=1 \
    ./pos_neg_pseudo_prompts_tiling_run_sam3.sh run --limit-images 2

You should see the model load, then one line per image, then the whole Phase 2:

    Loading SAM3 from 'facebook/sam3' onto cuda (bfloat16) ...
    SAM3 loaded.
    [pos_neg_pseudo_prompts_tiling_smoke] shard0 (1/2) AGS_Multi_Rumex/2022.../DJI_0001 | 14 GT box(es) | prompts=S3+M9+L0 | tiles=70 | round1=41 det -> self-prompts: +2 pos (0.91, 0.88) / -2 neg (0.31, 0.36) -> final=38 det | 45.2s | avg/image=45.2s | ETA=0.8 min (0.01 h)
    ...
    ===== PHASE 2 : OFFLINE EVALUATION =====
    --- CELL 25: per-image metrics (final vs round 1) ---
      all_gt   : AP50 round1=0.xxxx -> final=0.xxxx (mean delta +0.xxxx) | images better/worse/unchanged: N/N/N
      held_out : AP50 round1=0.xxxx -> final=0.xxxx (mean delta +0.xxxx) | images better/worse/unchanged: N/N/N
    ...
    EXPERIMENT pos_neg_pseudo_prompts_tiling_smoke - FINAL SUMMARY

Things to check in this output:
- `45.2s`  (or whatever you get) is the time for ONE image — both rounds over all its tiles.
  Multiply by the image count from Step 6b to estimate the whole job.
- `prompts=S3+M9+L0`  — three different GT indices, in the order S, M, L.
- `self-prompts: +2 pos (...) / -2 neg (...)`  — the positive scores should be high, the
  negative ones between 0.30 and 0.50.  `-0 neg`  is normal when every remaining round-1
  detection is ≥ 0.50. A  `NOTE: no self-prompt at all ... round 2 skipped`  line means round 1
  found nothing outside the three prompt plants; that image keeps its round-1 output.
- `round1=`  /  `final=`  should not be 0 on every image. All zeros means SAM3 found nothing
  above 0.30, or the prompts are wrong.
- The AP50 values should be plausible in  `all_gt` , not  `0.0000`  everywhere. All zeros means
  something is wrong with the labels or the class id.

Then leave the interactive session:

    exit

Setup is done. You never have to repeat Phase A.


================================================================================
PHASE B — RUN
================================================================================

Step 8 — Submit the job
-----------------------
On a login node:

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
    sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh

You should see: `Submitted batch job 1234567`

That is it. No arguments needed. The job asks for 1 node, 4 GPUs, 64 CPUs, 24 hours, splits the
images across the 4 GPUs for Phase 1, and then runs Phase 2 once.

Every image is processed tile by tile TWICE (round 1 and round 2), so Phase 1 takes about twice
as long as a single-round S/M/L tiled run. It is still per IMAGE, not per anchor. If it does
not fit in 24 h, just resubmit (Step 11).


Step 9 — Watch it
-----------------
    squeue -u $USER                                                                          # queued / running?
    tail -f pos_neg_pseudo_prompts_tiling_sam3_1234567.out                                   # everything, live
    tail -f $SCRATCH/experiments/sam3/pos_neg_pseudo_prompts_tiling/shard0.log               # just GPU 0

Count how many images are finished so far:

    cat $SCRATCH/experiments/sam3/pos_neg_pseudo_prompts_tiling/raw_detections/runs_manifest_pos_neg_pseudo_prompts_tiling_shard*.csv | grep -c pos_neg_pseudo_prompts_tiling
    ls $SCRATCH/experiments/sam3/pos_neg_pseudo_prompts_tiling/raw_detections/*.npz | wc -l

You will also get an email at BEGIN / END / FAIL.


Step 10 — Collect the results
-----------------------------
Everything lands in  `$SCRATCH/experiments/sam3/pos_neg_pseudo_prompts_tiling/` :

    ls -R $SCRATCH/experiments/sam3/pos_neg_pseudo_prompts_tiling/ | head -40

The file you actually quote in the thesis:

| File                                   | What it is                                                                  |
|----------------------------------------|-----------------------------------------------------------------------------|
| metrics/experiment_summary.csv         | the headline numbers — one row per evaluation mode (all_gt, held_out), with mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU1 / IoU2 of the FINAL output, the round-1 means (`*_round1_mean`), `delta_AP50_mean`, `delta_F1_mean` and `n_images_with_second_round` |

The mean is taken over images, so an image with 40 plants does not outweigh one with 2. The std
is the variation between UAV images.

Everything else in that folder:

| File                                   | What it is                                                                  |
|----------------------------------------|-----------------------------------------------------------------------------|
| metrics/experiment_summary_per_archive.csv | the same table split by archive, plus that archive's pooled AP (dataset_AP50*) and pooled TP / FP / FN. Not in the notebook (it ran on one archive); the pooled table above is the notebook's definition, unchanged |
| metrics/image_level_metrics.csv        | one row per (image × mode): archive, flight, S/M/L prompt ids, the self-prompt scores, second_run, the tile / cross-tile-NMS counters, every metric for the final output AND for round 1, delta_AP50, delta_F1 |
| metrics/dataset_ap_metrics.csv         | pooled AP50 / AP50:95 — all images ranked in ONE precision-recall curve, final and round 1. Not the mean of the image-level AP |
| metrics/size_group_recall.csv          | recall of the NON-prompt GT boxes in the smaller / larger half of each image (split at that image's median area) |
| metrics/prompt_plants_redetected.csv   | for every S / M / L prompt plant: was it re-detected in the final output?   |
| confusion_matrices/confusion_matrix_{all_gt,held_out}.csv + .png | pooled TP / FP / FN, plus micro precision/recall/F1 in confusion_matrix_summary.csv |
| plots/best_image_*.png                 | the qualitative GT-vs-prediction figures (notebook CELL 31) — see below     |
| raw_detections/*.npz                   | per image: the round-1 PRE-NMS detections, the positive and negative self-prompt boxes + scores, second_run, the FINAL pre-NMS detections, the tile provenance, the GT boxes, the S/M/L indices, archive and flight. This is what Phase 2 reads |
| raw_detections/runs_manifest_pos_neg_pseudo_prompts_tiling_shard*.csv | which images are finished (with n_tiles, the S/M/L areas, the self-prompt counts and scores, second_run); this is what resume reads |
| run_config_pos_neg_pseudo_prompts_tiling.json | every parameter used, for the thesis appendix                        |
| shard0..3.log                          | per-GPU logs                                                                |

Look at the headline table:

    cd $SCRATCH/experiments/sam3/pos_neg_pseudo_prompts_tiling/metrics
    column -s, -t < experiment_summary.csv | less -S

Which mode to quote: both, and say what they mean.
- `all_gt`  = the final predictions against ALL original human GT annotations — the main number.
- `held_out`  = the 3 S/M/L prompt plants are removed from the GT and the predictions that land
  on them are ignored — "after being shown a small, a medium and a large example, how well does
  it find the remaining plants?". The self-prompts are NOT GT boxes, so they never change the
  GT set in either mode.

The effect of the self-prompts: compare  `AP50_round1_mean`  with  `AP50_mean`  (and
`delta_AP50_mean` ) in the summary, and  `dataset_AP50_round1`  with  `dataset_AP50`  in the pooled
table. Phase 2 also prints how many images got better / worse / unchanged.

The qualitative figures. The notebook produced one. Both archives are pooled here, so the
same selection runs three times and you get three files in  `plots/` :

    best_image_AGS_Multi_Rumex_<image>_all_gt.png
    best_image_AgsSpringRumex_<image>_all_gt.png
    best_image_ALL_<image>_all_gt.png

Each is GT (yellow, left) next to the FINAL predictions (red, right, with the confidence written
next to each box), with exactly the notebook's colours: the S / M / L prompts dashed cyan / lime /
orange, the positive self-prompts dashed magenta and the negative self-prompts dashed black. The
image chosen is the one with the highest image-level AP50 among those with ≥ 7 GT boxes (with the
notebook's fallback when none qualifies). The selection rule is printed in the figure title, so
the figure is self-documenting.

Reading the raw detections in Python:

    import numpy as np
    z = np.load("raw_detections/AGS_Multi_Rumex__20220518_Eschikon__DJI_0001.npz")
    print(z["boxes_round1"].shape, z["boxes"].shape, bool(z["second_run"]))
    print(z["prompt_roles"], z["prompt_indices"])          # ['S' 'M' 'L'] [ 3  9  0]
    print(z["self_pos_scores"], z["self_neg_scores"])
    print(z["tile_id"][:10], str(z["archive"]), str(z["flight"]))


Step 11 — Resubmit if it hit the walltime
-----------------------------------------
Completely normal, and harmless. Just run the same command again:

    sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh

Every finished image is listed in the shard manifests and is skipped before any GPU work, so the
job picks up exactly where it stopped. You will see this near the top of each shard log:

    Resuming: 812 images already finished for pos_neg_pseudo_prompts_tiling.

Each shard reads all the manifests, not just its own, so resuming still works if you change
 `NUM_GPUS`  between submissions. The strip background of an image depends only on
(experiment name, image_ID, tile), never on which shard or submission made it, so a resumed image
is identical to an uninterrupted one.

The manifests also carry the notebook's guard: if they already hold results produced with a
different prompt setting (another  `N_SELF_POSITIVES`  /  `N_SELF_NEGATIVES` , i.e. another
`PROMPT_TYPE` ), the run aborts before loading SAM3 instead of silently mixing two settings in one
results folder. Give such a run its own  `EXPERIMENT_NAME`  and  `OUTPUT_DIR` .

If a shard crashed but the others finished: Phase 2 still runs on whatever NPZ files reached
disk, and the job exits non-zero so you get the FAIL email. Just resubmit — it fills the gaps.


Step 12 — Re-run only the evaluation (optional, cheap)
------------------------------------------------------
Phase 2 reads only the NPZ files, so you never need a GPU for it:
on a login node, or in any small allocation

    cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
    ./pos_neg_pseudo_prompts_tiling_run_sam3.sh evaluate

Use this to change the evaluation NMS or operating point without re-running SAM3:

    THRESHOLD=0.40 NMS_IOU_THRESHOLD=0.50 \
    OUTPUT_DIR=$SCRATCH/experiments/sam3/pos_neg_pseudo_prompts_tiling \
    ./pos_neg_pseudo_prompts_tiling_run_sam3.sh evaluate

In  `evaluate`  mode  `THRESHOLD`  only moves the operating point (and the AP floor) — it cannot
go below the value Phase 1 used, because lower detections were never saved. What it can NOT
change is the self-prompt selection: the self-prompts were chosen in Phase 1 with the Phase 1
values of  `THRESHOLD`  /  `NMS_IOU_THRESHOLD`  and are stored in the NPZ files.

Careful: this overwrites the CSVs in  `metrics/` ,  `confusion_matrices/`  and  `plots/` . To keep
both, copy the folder first, or point  `OUTPUT_DIR`  at a copy of the experiment.

The evaluation only loads the runs made with the current  `PROMPT_TYPE`  (exactly like the
notebook's CELL 20), so keep  `N_SELF_POSITIVES`  /  `N_SELF_NEGATIVES`  the same as in Phase 1.

Note that  `evaluate`  needs  `DATASET_ROOT`  to be reachable, because the qualitative figures
reopen the selected images. If the dataset has been purged from  `$SCRATCH` , add  `--no-plots` :

    ./pos_neg_pseudo_prompts_tiling_run_sam3.sh evaluate --no-plots

Every CSV and confusion matrix is still written; only the figures are skipped.


================================================================================
Notes and problems
================================================================================

If it will not fit in the walltime
----------------------------------
Tiling multiplies the number of forward passes, and this experiment does it twice. In order of
preference:

1. Do nothing. Resubmit as many times as needed (Step 11). Two 24 h jobs = one 48 h job.
2. More GPUs — needs a job array; ask and I will write one.
3. `TILE_SIZE` / `BATCH_SIZE` are the other levers (larger tiles = fewer passes but more GPU
   memory), but changing the tiles changes the experiment, so it is not a way to save time on
   this run.

Three things to say about this in the thesis:
- One confidence threshold. Like the notebook,  `CONFIDENCE_THRESHOLD = 0.30`  is used inside
  SAM3 in both rounds and every tile, for picking the self-prompts, and as the operating point.
  E01_2 / E02_2 freeze their operating point at 0.40 instead; replay this one at 0.40 with
  Step 12 (free, no GPU) if you want the operating-point columns to be comparable. The AP
  columns are unaffected.
- The self-prompts are the model's own output. They cost no annotation, but a wrong positive or
  negative self-prompt is fed straight back into round 2. The  `*_round1`  columns are round 1
  of exactly this pipeline, i.e. the plain S/M/L tiled experiment, so  `delta_AP50`  measures
  what the self-prompts add or remove.
- The self-prompts enter round 2 as CROPS in the strip, not as boxes at their real location
  (a tile may not contain that plant at all), so they act as APPEARANCE examples. In the
  no-tiling version they are boxes at their real location — worth one sentence when you compare
  the two.


Problem A —  `Sam3Model import FAILED`
--------------------------------------
The  `yolo26`  container was built for ultralytics and its  `transformers`  is too old for SAM3.
Build a dedicated image:

    # Dockerfile
    FROM <whatever image ~/.edf/yolo26.toml points at>
    RUN pip install --no-cache-dir "transformers >= <version with SAM3>" \
        supervision accelerate pandas matplotlib

    podman build -t sam3 .
    enroot import -o $SCRATCH/images/sam3.sqsh podman://sam3:latest
    cp ~/.edf/yolo26.toml ~/.edf/sam3.toml
    # edit ~/.edf/sam3.toml so `image =` points at $SCRATCH/images/sam3.sqsh

then change the one line at the bottom of  `pos_neg_pseudo_prompts_tiling_submit_sam3.sh` :

    srun --environment=sam3 ...

and use  `--environment=sam3`  in Step 5 as well.


Problem B —  `supervision`  /  `pandas`  /  `matplotlib`  import FAILED inside the container
--------------------------------------------------------------------------------------------
Step 4 installed with the login node's Python. If the container uses a different Python
version, the install is invisible to it. Install from inside the container instead — the
compute node has no internet, which is why Step 4 already put the wheels on  `$SCRATCH` :

inside the container (Step 5 shell) — pick whichever package failed

    pip install --target $SCRATCH/pyextra --no-deps --no-index \
        --find-links $SCRATCH/wheels supervision

    pip install --target $SCRATCH/pyextra --no-deps --no-index \
        --find-links $SCRATCH/wheels pandas pytz tzdata python-dateutil six

    pip install --target $SCRATCH/pyextra --no-deps --no-index \
        --find-links $SCRATCH/wheels matplotlib contourpy cycler fonttools kiwisolver pyparsing packaging

If Step 4 could not fetch a wheel, get it on the login node first:

    pip download --no-deps -d $SCRATCH/wheels <package>

Always  `--no-deps` . Those packages would otherwise install their own numpy / pillow into
 `$SCRATCH/pyextra` , and because that directory is searched before the container's own
packages, those copies would shadow the ones  `torch`  was compiled against and break torch.

 `matplotlib`  is the only optional one: without it every CSV is still written and only the PNGs
are skipped. Phase 2 does not need  `torch`  at all.


What the fixed operating point means for your numbers
-----------------------------------------------------
`CONFIDENCE_THRESHOLD = 0.30` and `NMS_IOU_THRESHOLD = 0.40` are fixed from the start, exactly
as in the notebook. No confidence × NMS sweep is performed, so nothing is tuned on the test
data.

- precision / recall / F1 / IoU1 / IoU2 describe ONE operating point: detections scoring
  ≥ 0.30.
- AP50 / AP50:95 always use every post-NMS detection ≥ 0.30, because AP is the area
  under the precision-recall curve and truncating the detection list would just cut the tail
  off that curve.

Because the two thresholds are the same number here, both use the same prediction set.
They still answer different questions — ranking quality vs deployed behaviour — and they are
both in the same CSV row on purpose.

If you want a lower floor for the AP curve you must re-run Phase 1 with a lower `THRESHOLD`,
since detections below it were never saved. That also changes the self-prompt pool (more
low-confidence negatives), so it is a different experiment:

    THRESHOLD=0.05 EXPERIMENT_NAME=pos_neg_pseudo_prompts_tiling_lowconf \
    OUTPUT_DIR=$SCRATCH/experiments/sam3/pos_neg_pseudo_prompts_tiling_lowconf \
    sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh


Other troubleshooting
---------------------
| Symptom                                      | Fix                                                                 |
|----------------------------------------------|---------------------------------------------------------------------|
| SCRATCH: unbound variable                    | you are not on the cluster, or the module environment is not loaded |
| syntax error near unexpected token in the .sh | Windows (CRLF) line endings — see Step 1                           |
| Job hangs at "Loading SAM3"                  | $SCRATCH/hf_cache is empty and the node is offline — redo Step 4    |
| CUDA out of memory                           | lower BATCH_SIZE (4 -> 2), or NUM_GPUS=2. Round 2 has up to 7 crops in the strip, so it is the heavier round |
| Everything reports F1=0.0000                 | wrong class id or wrong labels — recheck Step 6b                    |
| Many "NOTE: no self-prompt at all ... round 2 skipped" | round 1 found nothing outside the three prompt plants on those images. Expected on images with few plants; they keep their round-1 output (second_run = False) |
| The run aborts with "already holds results for another prompt setting" | the results folder was reused with another N_SELF_POSITIVES / N_SELF_NEGATIVES. Use a fresh EXPERIMENT_NAME and OUTPUT_DIR |
| held_out is NaN for some images              | every plant of that image was a prompt — expected on images with <= 3 GT boxes (valid_for_macro = False) |
| Nothing to plot in Phase 2                   | no image had a valid AP50, or DATASET_ROOT is unreachable. Every CSV is still written; add --no-plots to silence it |
| Phase 2 is slow / heavy                      | it holds every image's detections in memory at once, exactly like the notebook (needed for the pooled AP of CELL 27). Give it a node with more RAM if it gets killed |


Settings you might change
-------------------------
Set them before  `sbatch` ; they are forwarded into the job.

    DTYPE=float16               sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # what the notebook's T4 used
    BATCH_SIZE=2                sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # smaller batch for GPU memory
    TILE_SIZE=1500              sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # larger tiles, fewer passes
    OVERLAP=200                 sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # more overlap
    MIN_FILL_RATIO=0.20         sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # stricter plausibility filter
    THRESHOLD=0.05              sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # more detections (and another self-prompt pool)
    NMS_IOU_THRESHOLD=0.50      sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # different NMS (round-1 selection too)
    N_SELF_POSITIVES=3          sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # three positive self-prompts
    N_SELF_NEGATIVES=0          sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # positives only
    UNRELIABLE_MAX_SCORE=0.40   sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # stricter negative pool
    SELF_PROMPT_EXCLUDE_IOU=0.3 sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # stricter "not a prompt plant" rule
    NUM_GPUS=2                  sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # fewer GPUs
    DATASET_ROOT=/some/path     sbatch pos_neg_pseudo_prompts_tiling_submit_sam3.sh  # dataset elsewhere

Every setting above except  `DTYPE` ,  `BATCH_SIZE` ,  `NUM_GPUS`  and  `DATASET_ROOT`  changes what
is stored in the NPZ files (the self-prompts are chosen in Phase 1), so give those runs their own
 `EXPERIMENT_NAME`  and  `OUTPUT_DIR` . The manifest guard enforces it for  `N_SELF_POSITIVES`  /
 `N_SELF_NEGATIVES` ; for the others it is up to you. To change only the evaluation NMS /
operating point of an existing run, use Step 12 instead — it is free.

Note that changing  `EXPERIMENT_NAME`  also changes the strip-background seeds, so the composed
tiles (and therefore the detections) change slightly with it.

To run this pipeline on one archive only:

    ./pos_neg_pseudo_prompts_tiling_run_sam3.sh run --archives AGS_Multi_Rumex

(which is what the notebook did — same images, same pipeline). The numbers will be close to the
notebook's but not bit-identical: the cluster  `image_ID`  carries the archive prefix
( `AGS_Multi_Rumex/<flight>/<name>`  instead of the notebook's  `<flight>/<name>` ), and that ID is
part of the SHA-256 strip-background seed, so the background texture behind the exemplar crops
is sampled differently (and SAM3 runs in bfloat16 instead of float16). Every cluster run
reproduces itself exactly.