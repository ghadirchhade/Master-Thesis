=HOW TO RUN — Experiment  `pos_neg_exemplars_tiling_extra_path`

What this experiment is: SAM3 prompted with 2 positive exemplar boxes + 3 negative exemplar
boxes (NO text prompt), tiling ON (1536px tiles, 384px overlap), PLUS an
extra downsampled global-context pass (downscale=2), run over both  `AGS_Multi_Rumex`  and
`AgsSpringRumex`  pooled as one dataset.

How it works in one line: the UAV image is split into overlapping tiles. For each
(image × anchor), an exemplar strip is composed above each tile (2 positives, then 3
negatives, with local background and feathering). The tile + strip is sent to SAM3
(NO text prompt). Additionally, an extra downsampled global-context pass is run over the
whole image and its detections are merged with the tiled detections before NMS.
Detections are filtered by target-region ( >=50% area in tile) and plausibility (fill ratio,
max area, edge margin), then mapped back to full image coords.

The negative exemplars are generated anew for every run (seeded, so  reproducible):
GT Rumex boxes
-> 2000 plant-sized candidate boxes (sizes drawn from this image's own GT boxes)
-> drop every candidate that overlaps Rumex (0 overlap, gap  >= 10 px)
-> keep the ones close to Rumex (gap  <= 1 x median plant size, relaxed x2 up to 3 times)
so the negatives are vegetation, not bare soil
-> pick 3 spatially diverse ones (farthest-point sampling)
-> crop them from the full image -> exemplar strip -> SAM3 (label 0)

The generator uses all GT boxes of the image (that is what guarantees zero overlap), i.e.
label information of the evaluated plants — worth one sentence in the thesis.

This is the cluster port of  `pos_neg_exemplars_tiling_extra_path.ipynb` . Like the notebook, it is
a two-phase pipeline:

PHASE 1  INFERENCE   (GPU)      SAM3 runs in batches of 4 tiles per (image x anchor) at
                                score 0.30 with 2 pos crops + 3 neg crops (NO text), PLUS the
                                extra downsampled global-context pass.
                                -> pre-NMS detections + prompts saved as NPZ

PHASE 2  EVALUATION  (no GPU)   Offline NMS 0.40 (with cross-tile provenance) -> metrics
                                at confidence 0.30
                                -> run / image / experiment / pooled-AP CSVs
                                -> confusion matrices (CSV + PNG)
                                -> qualitative GT-vs-prediction figures

Phase 2 never touches SAM3, so once Phase 1 is done you can rebuild every number in minutes.

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

PHASE A — SETUP (once)

Step 1 — Copy the files onto the cluster
Put the three scripts here, next to the other experiments:
$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3/
├── pos_neg_exemplars_tiling_extra_path_infer_sam3.py
├── pos_neg_exemplars_tiling_extra_path_run_sam3.sh
└── pos_neg_exemplars_tiling_extra_path_submit_sam3.sh

Then, on a login node:
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
chmod +x pos_neg_exemplars_tiling_extra_path_run_sam3.sh pos_neg_exemplars_tiling_extra_path_submit_sam3.sh
ls -l
You should see: the two `.sh` marked executable (`-rwxr-xr-x`).

Step 2 — Put the dataset on  `$SCRATCH`
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
detects them and skips them on purpose.

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
`facebook/sam3` is a gated model. Two things are needed:
1. Go to https://huggingface.co/facebook/sam3 while logged in and accept the licence.
2. Create a token at https://huggingface.co/settings/tokens (read access is enough).

Then, on a login node:
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

Note: the token must belong to the same account that accepted the licence, otherwise
Step 4 fails with a 401.

Step 4 — Download the model and the extra packages
If you already ran the  `download`  mode of any other SAM3 experiment, skip this step.
They all share  `$SCRATCH/hf_cache` ,  `$SCRATCH/pyextra`  and  `$SCRATCH/wheels` .

Still on a login node (this is the only step that needs internet — compute nodes have none):
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./pos_neg_exemplars_tiling_extra_path_run_sam3.sh download

This does three things:
- downloads the SAM3 weights (and processor) into  `$SCRATCH/hf_cache`
- installs the  `supervision`  and  `psutil`  packages into  `$SCRATCH/pyextra`
  ( `supervision`  computes AP50 / AP50:95; `psutil` powers the RAM guard of Phase 1)
- downloads wheels for the Phase 2 packages ( `pandas` ,  `matplotlib` ) into  `$SCRATCH/wheels` ,
  in case the container does not ship them — see Problem B

You should see:
--- 1/3 : model weights -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--facebook--sam3/snapshots/...
--- 2/3 : supervision + psutil -> /scratch/.../pyextra ---
Successfully installed psutil-... supervision-...
supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
psutil x.x.x -> /scratch/.../pyextra/psutil/__init__.py
--- 3/3 : wheels for the PHASE 2 packages -> /scratch/.../wheels ---
Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1).

If it says  `ERROR: HF_TOKEN is not set`  → go back to Step 3.
If it says  `401`  or  `gated`  → the licence was not accepted with that token's account.

Step 5 — Open an interactive session inside the container
Steps 6 and 7 must run inside the container, not on the login node — otherwise you are
checking the login node's Python, which is not the one the job will use.

srun --account=go077 --time=00:30:00 \
     --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
     --environment=yolo26 --pty bash

You should see: a new shell prompt, running on a compute node. Everything in Steps 6 and 7
happens in this shell.

Step 6 — Check the container has what it needs
Inside the interactive shell from Step 5:
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./pos_neg_exemplars_tiling_extra_path_run_sam3.sh dryrun

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

6a. Look for these lines in the debug block
torch: 2.x.x | cuda available: True | device count: 1
bfloat16 supported: True
Sam3Model import: OK
MeanAveragePrecision import: OK
pandas: 2.x.x -> PHASE 2 import: OK
matplotlib: 3.x.x -> confusion-matrix + qualitative PNGs: OK
psutil: x.x.x -> RAM guard (MEM_STOP_THRESHOLD_PCT): OK

| If you see                           | Meaning                                                | Fix                                            |
|--------------------------------------|--------------------------------------------------------|------------------------------------------------|
| Sam3Model import FAILED              | the yolo26 image's transformers is too old for SAM3    | see Problem A at the bottom                    |
| supervision import FAILED            | $SCRATCH/pyextra not found or wrong Python version     | rerun Step 4; if it still fails, see Problem B |
| pandas import FAILED                 | Phase 1 still works, but Phase 2 cannot run            | see Problem B                                  |
| matplotlib import FAILED             | only the PNGs are lost (confusion matrices + qualitative figures); every CSV is still written | see Problem B, or ignore it |
| bfloat16 supported: False            | old GPU                                                | use DTYPE=float16                              |
| cuda available: False                | no GPU in this allocation                              | add --gpus-per-task=1 to Step 5                |
| psutil import FAILED                 | Phase 1 still runs, but the RAM guard is switched off  | rerun Step 4, or see Problem B                 |

Do not continue until the first four lines are correct. Everything after this depends on them.

6b. Check the dataset report
Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
Total images: NNN | this shard: NNN

Confirm the image counts match what you expect, and that  `class_id`  is 0 for
`AGS_Multi_Rumex`  and 2 for  `AgsSpringRumex` .

It also prints a cost estimate and probes the negative generator (CPU, no model):
sampled NNN image(s) of this shard -> NNNN anchor runs ({...})
prompts per run: 2 positive + 3 negative (no text) (2pos+3neg)
negative generator probed on NNN image(s): 0 could not place all 3 negatives
tiles per image: ~35 (1536px, 384px overlap; 8192x5460 image)
forward passes per anchor run: ~9 (batched, 4 tiles per pass) + 1 global-context pass (downscale=2)
=> ~NNNN SAM3 forward passes for those NNN images
(scale by len(shard)/sampled for the full estimate)
NPZ files that will be written by this shard: ~NNNN (one per image x anchor)

If the negative-generator line says many images could not place all 3 negatives, those
images are extremely dense (almost no free vegetation between plants). The run still works —
it uses as many negatives as it could place and logs a  WARNING — but mention it if it is
frequent. Loosen it with  `NEG_MAX_GAP_FACTOR=2.0`  or  `NEG_MIN_GAP_PX=5`  (in a new
`EXPERIMENT_NAME` ).

If it says  `No images with labels found`  →  `$SCRATCH/overney/dataset`  is wrong, or the
folder names do not match. Go back to Step 2.
If it warns  `N image(s) have no matching label file`  → those are skipped. A handful is
normal; if it is every image, the label basenames do not match the image basenames.

Step 7 — Smoke test on 2 images
Still inside the interactive shell. This is the first time the model actually loads.
It writes to a throwaway folder so it cannot pollute the real results.

EXPERIMENT_NAME=pos_neg_exemplars_tiling_extra_path_smoke \
OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke_pos_neg_exemplars_tiling_extra_path \
NUM_GPUS=1 \
./pos_neg_exemplars_tiling_extra_path_run_sam3.sh run --limit-images 2

You should see the model load, then one line per anchor run, then the whole Phase 2:
Loading SAM3 from 'facebook/sam3' onto cuda (bfloat16) ...
SAM3 loaded.
[pos_neg_exemplars_tiling_extra_path_smoke] shard0 run #1 | AGS_Multi_Rumex/2022.../DJI_0001 | anchor=0 (1/12) | prompt=POS 0+7 + 3 NEG (no text) | tiles=NN | global_pass=True | pre-NMS detections=9 | 1.6s
...
===== PHASE 2 : OFFLINE EVALUATION =====
--- CELL 26: run-level metrics ---
--- CELL 27: image-level metrics ---
...

Four things to check in this output:
- `1.6s`   (or whatever you get) is the time for ONE anchor run — a batched forward pass over
  all tiles + the global pass (the negative generator and strip composition are a few milliseconds
  of numpy), so expect seconds. Multiply by the total anchor count from Step 6b to estimate the
  whole job.
- `prompt=POS 0+7 + 3 NEG (no text)`   — two different GT indices (the anchor first), 3
  negatives.   `2 NEG`   or fewer with a WARNING line means the generator
  could not place all three on that image.
- `global_pass=True`   — the extra downsampled pass ran and found detections.
- `pre-NMS detections=`   should not be 0 on every run. All zeros means SAM3 found nothing
  above 0.30, or the prompts are wrong.
- `F1_mean`  in the final summary table should be a plausible number in  `all_gt` , not
  `0.0000`  everywhere. All zeros means something is wrong with the labels or the class id.
- `held_out`  legitimately has fewer valid runs ( `n_runs_valid_for_macro` ) than  `all_gt` : on an image with 2 or fewer
  plants, every plant is used as a positive prompt, so there is no GT left to evaluate and that
  run is excluded from the means (its false positives are  still counted). This is the notebook's
  `valid_for_macro = False`   case.

Then leave the interactive session:
exit

Setup is done. You never have to repeat Phase A.

PHASE B — RUN

Step 8 — Submit the job
On a login node:
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh

You should see: `Submitted batch job 1234567`
That is it. No arguments needed. The job asks for 1 node, 4 GPUs, 64 CPUs, 24 hours, splits the
images across the 4 GPUs for Phase 1, and then runs Phase 2 once.

The 24 h walltime is deliberately generous for this experiment — tiling + the global pass
means more forward passes per image than the no-tiling version. Once you have seen the real
timing, lowering `#SBATCH --time` will get you through the queue faster.

Step 9 — Watch it
squeue -u $USER                                                                      # queued / running?
tail -f pos_neg_exemplars_tiling_extra_path_sam3_1234567.out                         # everything, live
tail -f $SCRATCH/experiments/sam3/pos_neg_exemplars_tiling_extra_path/shard0.log     # just GPU 0

Count how many runs are finished so far:
cat $SCRATCH/experiments/sam3/pos_neg_exemplars_tiling_extra_path/raw_detections/runs_manifest_pos_neg_exemplars_tiling_extra_path_shard*.csv | grep -c pos_neg_exemplars_tiling_extra_path
ls $SCRATCH/experiments/sam3/pos_neg_exemplars_tiling_extra_path/raw_detections/*.npz | wc -l

You will also get an email at BEGIN / END / FAIL.

Step 10 — Collect the results
Everything lands in  `$SCRATCH/experiments/sam3/pos_neg_exemplars_tiling_extra_path/` :
ls -R $SCRATCH/experiments/sam3/pos_neg_exemplars_tiling_extra_path/ | head -40

The file you actually quote in the thesis:
| File                                   | What it is                                                                  |
|----------------------------------------|----------------------------------------------------------------------------|
|  metrics/experiment_summary.csv         | the headline numbers — one row per evaluation mode (all_gt, held_out), with mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU 1 / IoU2, plus the prompt type and the number of negatives |

The mean is taken over the image-level values, so an image with 40 plants does not outweigh
one with 2. The std is the variation between UAV images.

Everything else in that folder:
| File                                   | What it is                                                                  |
|----------------------------------------|----------------------------------------------------------------------------|
| metrics/experiment_summary_per_archive.csv | the same table split by archive. Not in the notebook (it ran on one archive); the pooled table above is the notebook's definition, unchanged |
| metrics/run_level_metrics.csv          | one row per (image × anchor × mode) — the raw data, incl. the archive, the flight, the positive prompt ids, the number of negatives actually placed, and TP/FP/FN |
| metrics/image_level_metrics.csv        | one row per (image × mode); the std here is the spread between the different anchors of the SAME image |
| metrics/dataset_ap_metrics.csv         | pooled AP50 / AP50:95 — all runs ranked in ONE precision-recall curve. Not the mean of the image-level AP |
| confusion_matrices/confusion_matrix_{all_gt,held_out}.csv + .png | pooled TP / FP / FN, plus micro precision/recall/F1 in confusion_matrix_summary.csv |
| plots/best_image_{archive}_..._anchorNNN_all_gt.png | the qualitative GT-vs-prediction figures (notebook CELL 31) — see below. One per archive + one for ALL pooled |
| raw_detections/*.npz                   | the PRE-NMS detections of every run (boxes, scores), the GT boxes, the positive prompt indices, the negative prompt boxes, archive, flight and the tile provenance. This is what Phase 2 reads |
| raw_detections/runs_manifest_pos_neg_exemplars_tiling_extra_path_shard*.csv | which runs are finished (with n_negatives and neg_max_gap_px per run ); this is what resume reads |
| shard0..3.log                          | per-GPU logs |

Look at the headline table:
cd $SCRATCH/experiments/sam3/pos_neg_exemplars_tiling_extra_path/metrics
column -s, -t  < experiment_summary.csv | less -S

Which mode to quote: both, and say what they mean.
`all_gt`  = classical evaluation, every GT box counts.
`held_out`  = the two plants shown to SAM3 as positive prompts are removed from the GT and the
predictions that land on them are ignored —  "after being shown two examples (plus what is not
Rumex), how well does it find the remaining plants? ". The negatives are not
GT boxes, so they play no role in either mode. The exemplars sit inside the very image being
evaluated, so SAM3 almost always re-detects them and  `all_gt`  gets up to two free true positives
per run.

The qualitative figures. The notebook produced one. Both archives are pooled here, so the
same selection runs three times and you get three files in  `plots/` :
best_image_AGS_Multi_Rumex_...anchorNNN_all_gt.png
best_image_AgsSpringRumex...anchorNNN_all_gt.png
best_image_ALL..._anchorNNN_all_gt.png

Each is GT (yellow, left) next to the predictions (red, right), with the positive prompts
dashed lime, the negative prompts dashed magenta, and the confidence written next to
each prediction. The image chosen is the one with the highest image-level AP50 among those with
≥ 7 GT boxes (with the notebook's fallback when none qualifies), and the anchor shown is that
image's best run. The selection rule is printed in the figure title, so the figure is
self-documenting.

Reading the raw detections in Python:
import numpy as np
z = np.load("raw_detections/AGS_Multi_Rumex__20220518_Eschikon__DJI_0001__anchor000.npz")
print(z["boxes"].shape, z["scores"].min(), z["gt_boxes"].shape, z["prompt_indices"])
print(z["neg_boxes"])
print(str(z["archive"]), str(z["flight"]))

Step 11 — Resubmit if it hit the walltime
Completely normal, and harmless. Just run the same command again:
sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh

Every finished (image, anchor) pair is listed in the shard manifests and is skipped before
any GPU work, so the job picks up exactly where it stopped. You will see this near the top of
each shard log:
Resuming: 4821 run(s) already finished for pos_neg_exemplars_tiling_extra_path; skipped.

Each shard reads all the manifests, not just its own, so resuming still works if you change
`NUM_GPUS`  between submissions. The negatives of a run depend only on
(experiment name, image, anchor), never on which shard or submission made them, so a resumed
run is identical to an uninterrupted one.

The manifests also carry the notebook's guard: if they already hold results produced with a
different prompt type (e.g. another  `K_POSITIVES`  /  `J_NEGATIVES` ), the run aborts instead of silently mixing two settings in one results folder.
Give such a run its own  `EXPERIMENT_NAME`  and  `OUTPUT_DIR` .

If a shard crashed but the others finished: Phase 2 still runs on whatever NPZ files
reached disk, and the job exits non-zero so you get the FAIL email. Just resubmit — it fills
the gaps.

The RAM guard (`MEM_STOP_THRESHOLD_PCT`, default 70 %) works the same way. If the node's RAM
goes above the threshold, the shard stops cleanly (everything finished so far is on disk),
prints `STOPPED: shard N hit the RAM threshold`, and exits with code 3. The job log then shows
`shard N STOPPED EARLY by the RAM guard (exit 3)` and you get the FAIL email instead of
"All shards completed". Just resubmit. Note that psutil measures the RAM of the whole node,
which the 4 shards share; if it stops too often, lower `BATCH_SIZE` or use `NUM_GPUS=2`.

Step 12 — Re-run only the evaluation (optional, cheap)
Phase 2 reads only the NPZ files, so you never need a GPU for it:
# on a login node, or in any small allocation
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./pos_neg_exemplars_tiling_extra_path_run_sam3.sh evaluate

Use this to change the operating point without re-running SAM3 — the pre-NMS detections on
disk are exactly what is needed to replay any (confidence, NMS IoU) pair:
OPERATING_CONFIDENCE=0.40 NMS_IOU_THRESHOLD=0.50 \
OUTPUT_DIR=$SCRATCH/experiments/sam3/pos_neg_exemplars_tiling_extra_path \
./pos_neg_exemplars_tiling_extra_path_run_sam3.sh evaluate

Careful: this overwrites the CSVs in  `metrics/` ,  `confusion_matrices/`  and  `plots/` . To
keep both, copy the folder first, or point  `OUTPUT_DIR`  at a copy of the experiment.

The evaluation only loads the runs made with the current  `PROMPT_TYPE`  (exactly like the
notebook's CELL 17), so keep it the same as in Phase 1.

Note that  `evaluate`  needs  `DATASET_ROOT`  to be reachable, because the qualitative figures
reopen the selected images. If the dataset has been purged from  `$SCRATCH` , add  `--no-plots` :
./pos_neg_exemplars_tiling_extra_path_run_sam3.sh evaluate --no-plots

Every CSV and confusion matrix is still written; only the figures are skipped.

Notes and problems

If it will not fit in the walltime
Tiling + the global pass multiplies the number of forward passes. In order of preference:
1. Do nothing. Resubmit as many times as needed (Step 11). Two 24 h jobs = one 48 h job.
2. `--max-anchors-per-image 5` — instead of every ground-truth box taking a turn as the
   anchor, use only the first 5 per image. This changes what you are measuring, so mention it
   in the thesis. Pass it after the mode:
   `./pos_neg_exemplars_tiling_extra_path_run_sam3.sh run --max-anchors-per-image 5`.
3. More GPUs — needs a job array; ask and I will write one.
4. `TILE_SIZE` is the other lever (larger tiles = fewer passes but more GPU memory), but
   changing it changes the experiment, so it is not a way to save time on this run.

Three things to say about this in the thesis:
1. The operating points differ. E01_2 / E02_2 freeze the operating point at confidence
   0.40 while inferring at 0.30. This experiment follows its notebook and keeps the
   operating point equal to the inference threshold, 0.30. Either say so explicitly, or
   replay one of them at the other's threshold with Step 12 (free, no GPU). The AP columns are
   unaffected.
2. What is being ablated. Against 1_exemplar_no_tiling (1 positive box, nothing else) this
   experiment adds a second positive, three negatives, tiling, the
   exemplar strip composition, AND the extra downsampled global-context pass. A difference in
   the numbers is the joint effect of all these additions.
3. The negatives use label information. They are placed with knowledge of every GT box of the
   image (that is how zero overlap is guaranteed), so this is an oracle-assisted prompt, not
   something a user could reproduce without annotations.

Problem A —  `Sam3Model import FAILED`
The  `yolo26`  container was built for ultralytics and its  `transformers`  is too old for SAM3.
Build a dedicated image:
# Dockerfile
FROM <whatever image ~/.edf/yolo26.toml points at>
RUN pip install --no-cache-dir "transformers >= <version with SAM3>" \
    supervision accelerate pandas matplotlib psutil

# then, on the login node:
podman build -t sam3 .
enroot import -o $SCRATCH/images/sam3.sqsh podman://sam3:latest
cp ~/.edf/yolo26.toml ~/.edf/sam3.toml
# edit ~/.edf/sam3.toml so `image =` points at $SCRATCH/images/sam3.sqsh
then change the one line at the bottom of  `pos_neg_exemplars_tiling_extra_path_submit_sam3.sh` :
`srun --environment=sam3 ...`
and use  `--environment=sam3`  in Step 5 as well.

Problem B —  `supervision`  /  `pandas`  /  `matplotlib`  import FAILED inside the container
Step 4 installed with the login node's Python. If the container uses a different Python
version, the install is invisible to it. Install from inside the container instead — the
compute node has no internet, which is why Step 4 already put the wheels on  `$SCRATCH` :

# inside the container (Step 5 shell) — pick whichever package failed
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels supervision
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels pandas pytz tzdata python-dateutil six
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels matplotlib contourpy cycler fonttools kiwisolver pyparsing packaging
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels psutil

If Step 4 could not fetch a wheel, get it on the login node first:
pip download --no-deps -d $SCRATCH/wheels <package>

Always  `--no-deps` . Those packages would otherwise install their own numpy / pillow into
`$SCRATCH/pyextra` , and because that directory is searched before the container's own
packages, those copies would shadow the ones  `torch`  was compiled against and break torch.

`matplotlib`  is the only optional one: without it every CSV is still written and only the PNGs
are skipped.

What the fixed operating point means for your numbers
`OPERATING_CONFIDENCE = 0.30` and `NMS_IOU_THRESHOLD = 0.40` are fixed from the start, exactly
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
since detections below it were never saved:
THRESHOLD=0.05 EXPERIMENT_NAME=pos_neg_exemplars_tiling_extra_path_lowconf \
OUTPUT_DIR=$SCRATCH/experiments/sam3/pos_neg_exemplars_tiling_extra_path_lowconf \
sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh

Note that this also moves the operating point unless you pin `OPERATING_CONFIDENCE=0.30`
separately — they are two independent flags here, even though the notebook ties them together.

Other troubleshooting
| Symptom                                      | Fix                                                                 |
|----------------------------------------------|---------------------------------------------------------------------|
| SCRATCH: unbound variable                    | you are not on the cluster, or the module environment is not loaded |
| Job hangs at "Loading SAM3"                  | $SCRATCH/hf_cache is empty and the node is offline — redo Step 4    |
| CUDA out of memory                           | lower BATCH_SIZE (4 -> 2), or increase TILE_SIZE, or NUM_GPUS=2     |
| Everything reports F1=0.0000                 | wrong class id or wrong labels — recheck Step 6b                    |
| WARNING: only N of 3 negatives could be placed | a very dense image with almost no free vegetation; the run still uses the ones it could place. Frequent -> loosen the generator in a new EXPERIMENT_NAME |
| AssertionError inside generate_negative_exemplars | a negative overlapped a GT box — the zero-overlap guarantee was broken by an edit to the generator. Do not run the job; fix the code first |
| The run aborts with "already holds results for another prompt setting" | the results folder was reused with another prompt type. Use a fresh EXPERIMENT_NAME and OUTPUT_DIR |
| held_out is all NaN                          | every plant of every image was used as a prompt — expected only on images with <= 2 GT boxes |
| Nothing to plot in Phase 2                   | no image had a valid AP50, or DATASET_ROOT is unreachable. Every CSV is still written; add --no-plots to silence it |
| Phase 2 is slow / heavy                      | it holds every run in memory at once, exactly like the notebook (needed for the pooled AP of CELL 27). Give it a node with more RAM if it gets killed |

Settings you might change
Set them before  `sbatch` ; they are forwarded into the job.
DTYPE=float32              sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # what the notebook's T4 used
TILE_SIZE=2048             sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # larger tiles, fewer passes
OVERLAP=512                sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # more overlap
BATCH_SIZE=2               sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # smaller batch for GPU memory
MIN_FILL_RATIO=0.20        sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # stricter plausibility filter
THRESHOLD=0.05             sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # save more detections for the AP curve
OPERATING_CONFIDENCE=0.40  sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # match E01_2 / E02_2's operating point
NMS_IOU_THRESHOLD=0.50     sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # different NMS
K_POSITIVES=3              sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # three positive boxes
J_NEGATIVES=0              sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # no negatives (pos only)
NEG_MAX_GAP_FACTOR=2.0     sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # negatives may lie further from Rumex
ADD_GLOBAL_CONTEXT_PASS=false sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh # disable the extra path
GLOBAL_DOWNSCALE=4         sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # more aggressive downscale for global pass
NUM_GPUS=2                 sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # fewer GPUs
DATASET_ROOT=/some/path    sbatch pos_neg_exemplars_tiling_extra_path_submit_sam3.sh  # dataset elsewhere

Changing   `K_POSITIVES`  ,   `J_NEGATIVES`  ,   `NEG_*`  ,   `TILE_SIZE`  ,   `OVERLAP`  ,
`THRESHOLD`  ,   `ADD_GLOBAL_CONTEXT_PASS`   or   `GLOBAL_DOWNSCALE`   changes what is stored in the NPZ
files, so give those runs their own   `EXPERIMENT_NAME`   and   `OUTPUT_DIR`   (the manifest guard
enforces it for the prompt type).

Changing only   `OPERATING_CONFIDENCE`   /   `NMS_IOU_THRESHOLD`   does not — use Step 12 instead, it is free.

Note that changing  `EXPERIMENT_NAME`  also changes the seeds, so the random second positive and
the negatives of every run change with it. Two experiments that should see the same prompts
must share the experiment name — or be compared knowing that the prompts differ.

To run this pipeline on one archive only:
./pos_neg_exemplars_tiling_extra_path_run_sam3.sh run --archives AGS_Multi_Rumex
(which is what the notebook did — same images, same pipeline). The numbers will be close to the
notebook's but not bit-identical: the cluster  `image_ID`  carries the archive prefix
( `AGS_Multi_Rumex/<flight>/<name>`  instead of the notebook's  `<flight>/<name>` ), and that ID is
part of the SHA-256 seed, so the random second positive and the negatives of each run are drawn
differently. They are just as deterministic — every cluster run reproduces itself exactly.