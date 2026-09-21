# HOW TO RUN — Experiment `1_exemplar_with_tiling`

What this experiment is: SAM3 prompted with **1 exemplar crop**, tiling ON (1000 px tiles,
150 px overlap), run over both `AGS_Multi_Rumex` and `AgsSpringRumex` pooled as one dataset.


This is the cluster port of `1_exemplar_with_tiling.ipynb`. Like the notebook, it is a
**two-phase** pipeline:

```
   PHASE 1  INFERENCE   (GPU)      SAM3 runs ONCE per (image x anchor) at score 0.30
                                   -> pre-NMS detections saved as NPZ
   PHASE 2  EVALUATION  (no GPU)   NMS 0.40 -> metrics at confidence 0.30
                                   -> run / image / experiment / pooled-AP CSVs
                                   -> confusion matrices (CSV + PNG)
                                   -> qualitative GT-vs-prediction figures
```

Phase 2 never touches SAM3, so once Phase 1 is done you can rebuild every number in minutes.

```
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
   ├─ Step 11  resubmit if it hit the 24 h limit
   └─ Step 12  re-run only the evaluation (optional, cheap)
```

---
---

# PHASE A — SETUP (once)

## Step 1 — Copy the files onto the cluster

Put the three scripts here:

```
$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3/
├── 1_exemplar_with_tiling_infer_sam3.py
├── 1_exemplar_with_tiling_run_sam3.sh
└── 1_exemplar_with_tiling_submit_sam3.sh
```

Then, **on a login node**:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
chmod +x 1_exemplar_with_tiling_run_sam3.sh 1_exemplar_with_tiling_submit_sam3.sh
ls -l
```

**You should see:** three files, with the two `.sh` marked executable (`-rwxr-xr-x`).

---

## Step 2 — Put the dataset on `$SCRATCH`

Extract the two archives side by side under one folder:

```bash
mkdir -p $SCRATCH/overney/dataset
cd $SCRATCH/overney/dataset
tar -xzf /path/to/AGS_Multi_Rumex.tar.gz
tar -xzf /path/to/AgsSpringRumex.tar.gz
```

**The result must look exactly like this:**

```
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
```

Check it:

```bash
ls $SCRATCH/overney/dataset
ls $SCRATCH/overney/dataset/AGS_Multi_Rumex
ls $SCRATCH/overney/dataset/AGS_Multi_Rumex/images | head
```

**Important:** the folder names are case-sensitive and must match exactly. The other two
archives (`AGS_Multiple_Fields`, `AGS_Multiple_Fields_Embeddings`) may be present — the code
detects them and skips them on purpose.

**Why `$SCRATCH` and not `$HOME`:** `$HOME` is small and slow; `$SCRATCH` is the large fast
filesystem. (It is also periodically purged, so keep the original archives somewhere safe.)

---

## Step 3 — HuggingFace licence + token

`facebook/sam3` is a **gated** model. Two things are needed:

1. Go to <https://huggingface.co/facebook/sam3> while logged in and **accept the licence**.
2. Create a token at <https://huggingface.co/settings/tokens> (read access is enough).

Then, **on a login node**:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

**Note:** the token must belong to the same account that accepted the licence, otherwise
Step 4 fails with a 401.

---

## Step 4 — Download the model and the extra packages

> **If you already ran the `download` mode of any other SAM3 experiment, skip this step.**
> All the experiments share `$SCRATCH/hf_cache`, `$SCRATCH/pyextra` and `$SCRATCH/wheels`.

**Still on a login node** (this is the only step that needs internet — compute nodes have none):

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./1_exemplar_with_tiling_run_sam3.sh download
```

This does three things:

1. downloads the SAM3 weights into `$SCRATCH/hf_cache`
2. installs the `supervision` package into `$SCRATCH/pyextra`
   (`supervision` computes AP50 / AP50:95; the code refuses to run without it)
3. downloads wheels for the Phase 2 packages (`pandas`, `matplotlib`) into `$SCRATCH/wheels`,
   in case the container does not ship them — see **Problem B**

**You should see:**

```
--- 1/3 : model weights -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--facebook--sam3/snapshots/...
--- 2/3 : supervision -> /scratch/.../pyextra ---
Successfully installed supervision-...
supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
--- 3/3 : wheels for the PHASE 2 packages -> /scratch/.../wheels ---
Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1).
```

**If it says `ERROR: HF_TOKEN is not set`** → go back to Step 3.
**If it says `401` or `gated`** → the licence was not accepted with that token's account.

---

## Step 5 — Open an interactive session inside the container

Steps 6 and 7 must run **inside the container**, not on the login node — otherwise you are
checking the login node's Python, which is not the one the job will use.

```bash
srun --account=go077 --time=00:30:00 \
     --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
     --environment=yolo26 --pty bash
```

**You should see:** a new shell prompt, running on a compute node. Everything in Steps 6 and 7
happens in this shell.

*(If your site needs a partition flag, add `--partition=...`. If the allocation takes a while,
that is just the queue.)*

---

## Step 6 — Check the container has what it needs

**Inside the interactive shell from Step 5:**

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./1_exemplar_with_tiling_run_sam3.sh dryrun
```

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

### 6a. Look for these lines in the debug block

```
torch: 2.x.x | cuda available: True | device count: 1
bfloat16 supported: True
Sam3Model import: OK
MeanAveragePrecision import: OK
pandas: 2.x.x -> PHASE 2 import: OK
matplotlib: 3.x.x -> confusion-matrix PNGs: OK
```

| If you see | Meaning | Fix |
|---|---|---|
| `Sam3Model import FAILED` | the `yolo26` image's `transformers` is too old for SAM3 | see **Problem A** at the bottom |
| `supervision import FAILED` | `$SCRATCH/pyextra` not found or wrong Python version | rerun Step 4; if it still fails, see **Problem B** |
| `pandas import FAILED` | Phase 1 still works, but Phase 2 cannot run | see **Problem B** |
| `matplotlib import FAILED` | only the confusion-matrix **PNGs** are lost, every CSV is still written | see **Problem B**, or ignore it |
| `bfloat16 supported: False` | old GPU | use `DTYPE=float16` (what the notebook used) |
| `cuda available: False` | no GPU in this allocation | add `--gpus-per-task=1` to Step 5 |

**Do not continue until the first four lines are correct.** Everything after this depends on them.

### 6b. Check the dataset report

```
Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
  AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
  AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
Total images: NNN | this shard: NNN
```

Confirm the image counts match what you expect, and that `class_id` is **0** for
`AGS_Multi_Rumex` and **2** for `AgsSpringRumex`.

It also prints a cost estimate:

```
  sampled NNN image(s) of this shard -> NNNN anchor runs ({...})
  tiles per anchor run at 8192x5460: 70
  => ~NNNNN SAM3 forward passes for those NNN images
  NPZ files that will be written by this shard: ~NNNN (one per image x anchor)
```

**If it says `No images with labels found`** → `$SCRATCH/overney/dataset` is wrong, or the
folder names do not match. Go back to Step 2.

**If it warns `N image(s) have no matching label file`** → those are skipped. A handful is
normal; if it is *every* image, the label basenames do not match the image basenames.

---

## Step 7 — Smoke test on 2 images

**Still inside the interactive shell.** This is the first time the model actually loads.
It writes to a throwaway folder so it cannot pollute the real results.

```bash
EXPERIMENT_NAME=1_exemplar_with_tiling_smoke \
OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke_1_exemplar_with_tiling \
NUM_GPUS=1 \
./1_exemplar_with_tiling_run_sam3.sh run --limit-images 2
```

**You should see** the model load, then one line per anchor run, then the whole Phase 2:

```
Loading SAM3 from 'facebook/sam3' onto cuda (bfloat16) ...
SAM3 ready. model device: cuda:0
  [1_exemplar_with_tiling_smoke] shard0 run #1 | AGS_Multi_Rumex/2022.../DJI_0001 | anchor=0 (1/12) | prompt=0 | tiles=70 | pre-NMS detections=41 | 58.3s
...
===== PHASE 2 : OFFLINE EVALUATION =====
--- CELL 24: run-level metrics ---
  all_gt   : NN runs, NN valid for macro averaging, F1_mean=0.xxxx
  held_out : NN runs, NN valid for macro averaging, F1_mean=0.xxxx
```

**Three things to check in this output:**

- **`58.3s`** (or whatever you get) is the time for ONE anchor run. Multiply by the total
  anchor count from Step 6b to estimate the whole job. If that exceeds 24 h, see
  **"If it will not fit in 24 hours"** below — the short answer is that it is fine, you just
  resubmit and it resumes.
- **`pre-NMS detections=`** should not be 0 on every run. All zeros means SAM3 found nothing
  above 0.30, or the prompts are wrong.
- **`F1_mean=`** should be a plausible number in `all_gt`, not `0.0000` everywhere. All zeros
  means something is wrong with the labels or the class id.

`held_out` legitimately shows fewer valid runs than `all_gt`: on an image where every plant
was used as a prompt, there is no GT left to evaluate, so that run is excluded from the means
(its false positives are still counted). This is the notebook's `valid_for_macro = False` case.

Then leave the interactive session:

```bash
exit
```

Setup is done. You never have to repeat Phase A.

---
---

# PHASE B — RUN

## Step 8 — Submit the job

**On a login node:**

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
sbatch 1_exemplar_with_tiling_submit_sam3.sh
```

**You should see:** `Submitted batch job 1234567`

That is it. No arguments needed. The job asks for 1 node, 4 GPUs,
64 CPUs, 24 hours, splits the images across the 4 GPUs for Phase 1, and then runs Phase 2 once.

---

## Step 9 — Watch it

```bash
squeue -u $USER                                     # queued / running?
tail -f 1_exemplar_with_tiling_sam3_1234567.out                      # everything, live
tail -f $SCRATCH/experiments/sam3/1_exemplar_with_tiling/shard0.log  # just GPU 0
```

Count how many runs are finished so far:

```bash
cat $SCRATCH/experiments/sam3/1_exemplar_with_tiling/raw_detections/runs_manifest_1_exemplar_with_tiling_shard*.csv \
  | grep -c 1_exemplar_with_tiling
ls $SCRATCH/experiments/sam3/1_exemplar_with_tiling/raw_detections/*.npz | wc -l
```

You will also get an email at BEGIN / END / FAIL.

---

## Step 10 — Collect the results

Everything lands in `$SCRATCH/experiments/sam3/1_exemplar_with_tiling/`:

```bash
ls -R $SCRATCH/experiments/sam3/1_exemplar_with_tiling/ | head -40
```

**The file you actually quote in the thesis:**

| File | What it is |
|---|---|
| `metrics/experiment_summary.csv` | **the headline numbers** — one row per evaluation mode (`all_gt`, `held_out`), with mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU1 / IoU2 |

The mean is taken over the **image-level** values, so an image with 40 plants does not outweigh
one with 2. The std is the variation **between UAV images**.

**Everything else in that folder:**

| File | What it is |
|---|---|
| `metrics/experiment_summary_per_archive.csv` | the same table split by archive. Not in the notebook (it ran on one archive); the pooled table above is the notebook's definition, unchanged |
| `metrics/run_level_metrics.csv` | one row per (image × anchor × mode) — the raw data, incl. TP/FP/FN and the NMS provenance counters (`n_suppressed_cross_tile` tells you how much of the duplication came from the tile overlap) |
| `metrics/image_level_metrics.csv` | one row per (image × mode); the std here is the spread between the different prompt selections of the SAME image |
| `metrics/dataset_ap_metrics.csv` | pooled AP50 / AP50:95 — all runs ranked in ONE precision-recall curve. **Not** the mean of the image-level AP |
| `confusion_matrices/confusion_matrix_{all_gt,held_out}.csv` + `.png` | pooled TP / FP / FN, plus micro precision/recall/F1 in `confusion_matrix_summary.csv` |
| `plots/best_image_*.png` | the qualitative GT-vs-prediction figures (notebook CELL 31) — see below |
| `raw_detections/*.npz` | the PRE-NMS detections of every run (boxes, scores, fill ratios, tile ids, GT boxes, prompt indices). This is what Phase 2 reads |
| `raw_detections/runs_manifest_1_exemplar_with_tiling_shard*.csv` | which runs are finished; this is what resume reads |
| `run_config_1_exemplar_with_tiling.json` | every parameter used, for the thesis appendix |
| `shard0..3.log` | per-GPU logs |

Look at the headline table:

```bash
cd $SCRATCH/experiments/sam3/1_exemplar_with_tiling/metrics
column -s, -t < experiment_summary.csv | less -S
```

**Which mode to quote:** both, and say what they mean.
`all_gt` = classical evaluation, every GT box counts.
`held_out` = the plants shown to SAM3 as prompts are removed from the GT and the predictions
that land on them are ignored — "after being shown a few examples, how well does it find the
**remaining** plants?".

**Reading the raw detections in Python:**

```python
import numpy as np
z = np.load("raw_detections/AGS_Multi_Rumex__20220518_Eschikon__DJI_0001__anchor000.npz")
print(z["boxes"].shape, z["scores"].min(), z["gt_boxes"].shape, z["prompt_indices"])
```

---

## Step 11 — Resubmit if it hit the 24 h limit

Completely normal, and harmless. Just run the same command again:

```bash
sbatch 1_exemplar_with_tiling_submit_sam3.sh
```

Every finished (image, anchor) pair is listed in the shard manifests and is skipped **before**
any GPU work, so the job picks up exactly where it stopped. Repeat until it finishes inside
the walltime. You will see this near the top of each shard log:

```
Resuming: 4821 run(s) already finished for 1_exemplar_with_tiling; skipped.
```

Each shard reads **all** the manifests, not just its own, so resuming still works if you change
`NUM_GPUS` between submissions.

**If a shard crashed but the others finished:** Phase 2 still runs on whatever NPZ files
reached disk, and the job exits non-zero so you get the FAIL email. Just resubmit — it fills
the gaps.

---

## Step 12 — Re-run only the evaluation (optional, cheap)

Phase 2 reads only the NPZ files, so you never need a GPU for it:

```bash
# on a login node, or in any small allocation
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./1_exemplar_with_tiling_run_sam3.sh evaluate
```

Use this to change the operating point without re-running SAM3 — the pre-NMS detections on
disk are exactly what is needed to replay any (confidence, NMS IoU) pair:

```bash
OPERATING_CONFIDENCE=0.40 NMS_IOU_THRESHOLD=0.50 \
OUTPUT_DIR=$SCRATCH/experiments/sam3/1_exemplar_with_tiling \
./1_exemplar_with_tiling_run_sam3.sh evaluate
```

> **Careful:** this overwrites the CSVs in `metrics/`, `confusion_matrices/` and `plots/`. To
> keep both, copy the folder first, or point `OUTPUT_DIR` at a copy of the experiment.

Note that `evaluate` needs `DATASET_ROOT` to be reachable, because the qualitative figures
reopen the selected images. If the dataset has been purged from `$SCRATCH`, add `--no-plots`:

```bash
./1_exemplar_with_tiling_run_sam3.sh evaluate --no-plots
```

Every CSV and confusion matrix is still written; only the figures are skipped.

## Getting these numbers from an existing E01_2 run

Because the two experiments differ **only** in the operating point, an existing E01_2 results
folder already contains everything Phase 1 would produce. Copy it and re-evaluate at 0.30:

```bash
cp -r $SCRATCH/experiments/sam3/E01_2 \
      $SCRATCH/experiments/sam3/1_exemplar_with_tiling_fromE01_2

# the NPZ and manifest rows still carry experiment_name=E01_2, so keep that name
EXPERIMENT_NAME=E01_2 \
OUTPUT_DIR=$SCRATCH/experiments/sam3/1_exemplar_with_tiling_fromE01_2 \
OPERATING_CONFIDENCE=0.30 NMS_IOU_THRESHOLD=0.40 \
./1_exemplar_with_tiling_run_sam3.sh evaluate
```

That takes minutes on a login node and needs no GPU. The AP columns will be **identical** to
E01_2's (AP never depends on the operating point); precision / recall / F1 / IoU1 / IoU2 will
be this experiment's. The only caveat is cosmetic: the `experiment_name` column and the NPZ
files still say `E01_2`. If you want a clean, self-contained artifact for the thesis, run the
job properly instead (Step 8).

---
---

# Notes and problems

## If it will not fit in 24 hours

You have four options, in order of preference:

1. **Do nothing.** Resubmit as many times as needed (Step 11). Three 24 h jobs = one 72 h job.
2. **`BATCH_SIZE=8 sbatch 1_exemplar_with_tiling_submit_sam3.sh`** — the notebook used 4 because a T4 has 16 GB;
   a GH200 has far more, so more tiles per forward pass is usually a straight win.
3. **`--max-anchors-per-image 5`** — instead of every ground-truth box taking a turn as the
   anchor, use only the first 5 per image. This changes what you are measuring, so mention it
   in the thesis. Pass it after the mode: `./1_exemplar_with_tiling_run_sam3.sh run --max-anchors-per-image 5`.
4. **More GPUs** — needs a job array; ask and I will write one.


## Problem A — `Sam3Model import FAILED`

The `yolo26` container was built for ultralytics and its `transformers` is too old for SAM3.
Build a dedicated image:

```dockerfile
# Dockerfile
FROM <whatever image ~/.edf/yolo26.toml points at>
RUN pip install --no-cache-dir "transformers>=<version with SAM3>" \
        supervision accelerate pandas matplotlib
```

```bash
podman build -t sam3 .
enroot import -o $SCRATCH/images/sam3.sqsh podman://sam3:latest
cp ~/.edf/yolo26.toml ~/.edf/sam3.toml
# edit ~/.edf/sam3.toml so `image = ` points at $SCRATCH/images/sam3.sqsh
```

then change the one line at the bottom of `1_exemplar_with_tiling_submit_sam3.sh`:

```bash
srun --environment=sam3 ...
```

and use `--environment=sam3` in Step 5 as well.

## Problem B — `supervision` / `pandas` / `matplotlib` import FAILED inside the container

Step 4 installed with the **login node's** Python. If the container uses a different Python
version, the install is invisible to it. Install from inside the container instead — the
compute node has no internet, which is why Step 4 already put the wheels on `$SCRATCH`:

```bash
# inside the container (Step 5 shell) — pick whichever package failed
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels supervision
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels pandas pytz tzdata python-dateutil six
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels matplotlib contourpy cycler fonttools kiwisolver pyparsing packaging
```

If Step 4 could not fetch a wheel, get it on the login node first:

```bash
pip download --no-deps -d $SCRATCH/wheels <package>
```

> **Always `--no-deps`.** Those packages would otherwise install their own numpy / pillow into
> `$SCRATCH/pyextra`, and because that directory is searched **before** the container's own
> packages, those copies would shadow the ones `torch` was compiled against and break torch.

`matplotlib` is the only optional one: without it every CSV is still written and only the
confusion-matrix PNGs are skipped.

## What the fixed operating point means for your numbers

`OPERATING_CONFIDENCE = 0.30` and `NMS_IOU_THRESHOLD = 0.40` are fixed from the start, exactly
as in the notebook. No confidence × NMS sweep is performed, so nothing is tuned on the test
data.

The notebook deliberately sets the operating confidence **equal to** the SAM3 inference
threshold, so one value (0.30) governs both the forward pass and every operating-point metric.
Two different numbers come out of this, and they answer different questions:

- **precision / recall / F1 / IoU1 / IoU2** describe ONE operating point: only detections
  scoring ≥ 0.30 are used. "If I deploy the detector at confidence 0.30, what happens?"
- **AP50 / AP50:95** always use **every** post-NMS detection ≥ 0.30 (the SAM3 inference
  threshold), because AP is the area under the precision-recall curve and truncating the
  detection list would just cut the tail off that curve.

Both appear in the same CSV row on purpose. If you want a lower floor for the AP curve you
must re-run Phase 1 with a lower `THRESHOLD`, since detections below it were never saved:

```bash
THRESHOLD=0.05 EXPERIMENT_NAME=1_exemplar_with_tiling_lowconf \
OUTPUT_DIR=$SCRATCH/experiments/sam3/1_exemplar_with_tiling_lowconf sbatch 1_exemplar_with_tiling_submit_sam3.sh
```

## Other troubleshooting

| Symptom | Fix |
|---|---|
| `SCRATCH: unbound variable` | you are not on the cluster, or the module environment is not loaded |
| Job hangs at "Loading SAM3" | `$SCRATCH/hf_cache` is empty and the node is offline — redo Step 4 |
| CUDA out of memory | lower `BATCH_SIZE` (4 → 2 → 1), or lower `TILE_SIZE` (e.g. 800), or `NUM_GPUS=2` |
| Host RAM fills up | `./1_exemplar_with_tiling_run_sam3.sh run --no-cache-tiles` — tiles are then cropped on demand instead of all being held in RAM (`CACHE_TILES_IN_MEMORY = False` in the notebook) |
| `Nothing to plot` in Phase 2 | no image had a valid AP50, or `DATASET_ROOT` is unreachable. Every CSV is still written; add `--no-plots` to silence it |
| Everything reports `F1=0.0000` | wrong class id or wrong labels — recheck Step 6b |
| `held_out` is all NaN | every plant of every image was used as a prompt — expected only on images with ≤ 3 GT boxes |
| Phase 2 is slow / heavy | it holds every run in memory at once, exactly like the notebook (needed for the pooled AP of CELL 27). Give it a node with more RAM if it gets killed |

## Settings you might change

Set them before `sbatch`; they are forwarded into the job.

```bash
DTYPE=float16           sbatch 1_exemplar_with_tiling_submit_sam3.sh   # exactly what the notebook used
BATCH_SIZE=8            sbatch 1_exemplar_with_tiling_submit_sam3.sh   # faster on a big GPU
THRESHOLD=0.05          sbatch 1_exemplar_with_tiling_submit_sam3.sh   # save more detections for the AP curve
OPERATING_CONFIDENCE=0.40 sbatch 1_exemplar_with_tiling_submit_sam3.sh # match E01_2's operating point
NMS_IOU_THRESHOLD=0.50  sbatch 1_exemplar_with_tiling_submit_sam3.sh   # different NMS
N_EXEMPLARS=3           sbatch 1_exemplar_with_tiling_submit_sam3.sh   # three visual prompts (= E02_2's scheme)
USE_TILING=0            sbatch 1_exemplar_with_tiling_submit_sam3.sh   # whole image as one tile
NUM_GPUS=2              sbatch 1_exemplar_with_tiling_submit_sam3.sh   # fewer GPUs
DATASET_ROOT=/some/path sbatch 1_exemplar_with_tiling_submit_sam3.sh   # dataset elsewhere
```

Changing `N_EXEMPLARS`, `USE_TILING`, `TILE_SIZE`, `OVERLAP` or `THRESHOLD` changes what is
stored in the NPZ files, so give those runs their own `EXPERIMENT_NAME` **and** `OUTPUT_DIR`.
Changing only `OPERATING_CONFIDENCE` / `NMS_IOU_THRESHOLD` does not — use Step 12 instead, it
is free.
