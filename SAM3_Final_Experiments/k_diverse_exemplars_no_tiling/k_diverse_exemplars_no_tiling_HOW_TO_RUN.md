# HOW TO RUN — Experiment `k_diverse_exemplars_no_tiling`

What this experiment is: SAM3 prompted with **3 exemplar boxes chosen by DINOv2 embedding
diversity**, **tiling OFF**, run over both `AGS_Multi_Rumex` and `AgsSpringRumex` pooled as one
dataset.

**How it works in one line:** every ground-truth box of the image is cropped at full resolution
(+10 % context → square padding → 224 × 224), embedded with DINOv2-large (CLS token,
L2-normalised), and the **triple whose closest pair is as far apart as possible** (max-min cosine
dispersion) is handed to SAM3 as three positive box prompts **at their real locations**, on the
whole image downscaled so its longest side is **1024 px** — no tiles, no exemplar strip, no
plausibility filter. The predicted boxes are scaled back to full resolution immediately.

**The selection is deterministic** — no anchors, no random sampling, no seeds — so there is
exactly **ONE run per image**. That is the main structural difference from `1_exemplar_no_tiling`
and `E01_2` / `E02_2`: there is no `anchor_idx` anywhere, and a single per-image metric table
replaces their run-level + image-level pair.

This is the cluster port of `k_diverse_exemplars_no_tiling.ipynb`. Like the notebook, it is a
**two-phase** pipeline:

```
   PHASE 1  INFERENCE   (GPU)      per image: DINOv2 embeds every GT crop
                                   -> max-min diversity selection (D1, D2, D3)
                                   -> SAM3 runs ONCE at score 0.30
                                   -> pre-NMS detections + the GxG DINOv2 distance
                                      matrix saved as NPZ
   PHASE 2  EVALUATION  (no GPU)   NMS 0.40 -> metrics at confidence 0.30
                                   -> image / experiment / pooled-AP CSVs
                                   -> size-group recall + prompt re-detection CSVs
                                   -> confusion matrices (CSV + PNG)
                                   -> qualitative GT-vs-prediction figures
```

Phase 2 never touches SAM3 **or DINOv2** (the distance matrices are already inside the NPZs), so
once Phase 1 is done you can rebuild every number in minutes.

```
   PHASE A — SETUP            do once, ~30 minutes
   ├─ Step 1   copy the files onto the cluster
   ├─ Step 2   put the dataset on $SCRATCH
   ├─ Step 3   HuggingFace licence + token
   ├─ Step 4   download the models and the extra packages
   ├─ Step 5   open an interactive session inside the container
   ├─ Step 6   check the container has what it needs
   └─ Step 7   smoke test on 2 images

   PHASE B — RUN              every time
   ├─ Step 8   submit the job
   ├─ Step 9   watch it
   ├─ Step 10  collect the results
   ├─ Step 11  resubmit if it hit the walltime
   └─ Step 12  re-run only the evaluation (optional, cheap)
```

---
---

# PHASE A — SETUP (once)

## Step 1 — Copy the files onto the cluster

Put the three scripts here, next to the other experiments:

```
$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3/
├── k_diverse_exemplars_no_tiling_infer_sam3.py
├── k_diverse_exemplars_no_tiling_run_sam3.sh
└── k_diverse_exemplars_no_tiling_submit_sam3.sh
```

Then, **on a login node**:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
chmod +x k_diverse_exemplars_no_tiling_run_sam3.sh k_diverse_exemplars_no_tiling_submit_sam3.sh
ls -l
```

**You should see:** the two `.sh` marked executable (`-rwxr-xr-x`).

---

## Step 2 — Put the dataset on `$SCRATCH`

> **If you already did this for E01_2 / E02_2 / `1_exemplar_no_tiling`, skip this step.** All the
> experiments read the same `$SCRATCH/overney/dataset`.

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

**Note on the class ids.** The notebook ran on `AGS_Multi_Rumex` alone with
`RUMEX_CLASS_ID = 0`. Both archives are used here and they label Rumex with **different class
ids**, so the id is a property of the archive:

| Archive | Rumex class id in its YOLO files |
|---|---|
| `AGS_Multi_Rumex` | **0** |
| `AgsSpringRumex` | **2** |

Every other class in those files is ignored.

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

`facebook/dinov2-large` (the exemplar embeddings) is **not** gated, so it needs no licence — but
it still has to be pre-downloaded in Step 4, because compute nodes have no internet.

---

## Step 4 — Download the models and the extra packages

> **If you already ran the `download` mode of another SAM3 experiment, you still have to run
> this one**, unless that experiment also used DINOv2 — the SAM3-only experiments never cached
> `facebook/dinov2-large`. Everything is shared (`$SCRATCH/hf_cache`, `$SCRATCH/pyextra`,
> `$SCRATCH/wheels`), so re-running it is cheap and idempotent.

**Still on a login node** (this is the only step that needs internet — compute nodes have none):

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./k_diverse_exemplars_no_tiling_run_sam3.sh download
```

This does four things:

1. downloads the SAM3 weights into `$SCRATCH/hf_cache`
2. downloads the **DINOv2-large** weights into the same cache (~1.2 GB in fp32)
3. installs the `supervision` package into `$SCRATCH/pyextra`
   (`supervision` computes AP50 / AP50:95; the code refuses to run without it)
4. downloads wheels for the Phase 2 packages (`pandas`, `matplotlib`) into `$SCRATCH/wheels`,
   in case the container does not ship them — see **Problem B**

**You should see:**

```
--- 1/4 : SAM3 weights -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--facebook--sam3/snapshots/...
--- 2/4 : DINOv2 weights -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--facebook--dinov2-large/snapshots/...
--- 3/4 : supervision -> /scratch/.../pyextra ---
Successfully installed supervision-...
supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
--- 4/4 : wheels for the PHASE 2 packages -> /scratch/.../wheels ---
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
./k_diverse_exemplars_no_tiling_run_sam3.sh dryrun
```

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, the self-test of the
diversity selection, then a dataset report.

### 6a. Look for these lines in the debug block

```
torch: 2.x.x | cuda available: True | device count: 1
bfloat16 supported: True
Sam3Model import: OK
DINOv2 (AutoModel) import: OK
MeanAveragePrecision import: OK
pandas: 2.x.x -> PHASE 2 import: OK
matplotlib: 3.x.x -> confusion-matrix + qualitative PNGs: OK
```

and, just after it:

```
Self-test OK: diversity selection (max_min, cosine) -> D1:0+D2:1+D3:2 | min pairwise distance = 1.000 | search = exhaustive | crop geometry OK
```

| If you see | Meaning | Fix |
|---|---|---|
| `Sam3Model import FAILED` | the `yolo26` image's `transformers` is too old for SAM3 | see **Problem A** at the bottom |
| `DINOv2 (AutoModel) import FAILED` | `transformers` is broken or far too old | same fix as **Problem A** |
| `supervision import FAILED` | `$SCRATCH/pyextra` not found or wrong Python version | rerun Step 4; if it still fails, see **Problem B** |
| `pandas import FAILED` | Phase 1 still works, but Phase 2 cannot run | see **Problem B** |
| `matplotlib import FAILED` | only the **PNGs** are lost (confusion matrices + qualitative figures); every CSV is still written | see **Problem B**, or ignore it |
| `bfloat16 supported: False` | old GPU | use `DTYPE=float16` (leave `DINOV2_DTYPE=float32`) |
| `cuda available: False` | no GPU in this allocation | add `--gpus-per-task=1` to Step 5 |
| the self-test raises `AssertionError` | the selection or the crop geometry was edited and is now wrong | do not run the job; fix the code first |

**Do not continue until the import lines and the self-test are correct.** Everything after this
depends on them.

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
  sampled NNN image(s) of this shard -> NNN runs (ONE per image: the diversity selection is deterministic)
  GT boxes in those images: NNNN ({...})
  DINOv2 crops to embed: NNNN (~NNN forward passes at batch size 16)
  forward passes per image: 1 SAM3 (whole image, no tiling, longest side -> 1024 px) + the DINOv2 crop batches
  max-min search: exhaustive up to 300 GT boxes (C(G,3) triples, vectorised)
  image(s) with fewer than K=3 GT boxes -> fewer prompts: N
  NPZ files that will be written by this shard: ~NNN (one per image)
```

**If it says `No images with labels found`** → `$SCRATCH/overney/dataset` is wrong, or the
folder names do not match. Go back to Step 2.

**If it warns `N image(s) have no matching label file`** → those are skipped. A handful is
normal; if it is *every* image, the label basenames do not match the image basenames.

---

## Step 7 — Smoke test on 2 images

**Still inside the interactive shell.** This is the first time the models actually load.
It writes to a throwaway folder so it cannot pollute the real results.

```bash
EXPERIMENT_NAME=k_diverse_exemplars_no_tiling_smoke \
OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke_k_diverse_exemplars_no_tiling \
NUM_GPUS=1 \
./k_diverse_exemplars_no_tiling_run_sam3.sh run --limit-images 2
```

**You should see** both models load, then one line per image, then the whole Phase 2:

```
Loading DINOv2 from 'facebook/dinov2-large' onto cuda (float32) ...
DINOv2 loaded (facebook/dinov2-large).
  embedding   : cls (1024-d)
Loading SAM3 from 'facebook/sam3' onto cuda (bfloat16) ...
SAM3 loaded.
  [k_diverse_exemplars_no_tiling_smoke] shard0 run #1 (1/2) AGS_Multi_Rumex/2022.../DJI_0001 | 12 GT box(es) | prompts=D1:7+D2:0+D3:4 | min_dist=0.412 (mean over all GT pairs 0.287) | pre-NMS detections=9 | DINOv2 2.1s + SAM3 1.4s = 3.6s | ...
...
===== PHASE 2 : OFFLINE EVALUATION =====
--- CELL 20: per-image metrics ---
  all_gt   : NN images valid for macro averaging, AP50_mean=0.xxxx, F1_mean=0.xxxx
  held_out : NN images valid for macro averaging, AP50_mean=0.xxxx, F1_mean=0.xxxx
```

**Four things to check in this output:**

- **`prompts=D1:7+D2:0+D3:4`** — three *different* GT indices. If you always get `D1:0+D2:1+D3:2`
  on images with many plants, the embeddings are not varying and something is wrong with the
  crops.
- **`min_dist=`** should be clearly **larger** than the `mean over all GT pairs` value printed
  next to it. That is the whole point of max-min selection: the chosen triple is more spread out
  than an average triple. If the two numbers are equal, the selection is not doing anything.
- **`pre-NMS detections=`** should not be 0 on every image. All zeros means SAM3 found nothing
  above 0.30, or the prompts are wrong.
- **`F1_mean=`** should be a plausible number in `all_gt`, not `0.0000` everywhere. All zeros
  means something is wrong with the labels or the class id.

`held_out` legitimately shows fewer valid images than `all_gt`: on an image with 3 or fewer
plants, all of them are used as prompts, so there is no GT left to evaluate and that image is
excluded from the means (its false positives are still counted). This is the notebook's
`valid_for_macro = False` case.

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
sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh
```

**You should see:** `Submitted batch job 1234567`

That is it. No arguments needed. The job asks for 1 node, 4 GPUs, 64 CPUs, 24 hours, splits the
images across the 4 GPUs for Phase 1, and then runs Phase 2 once.

The 24 h walltime is deliberately generous. This experiment does **one** SAM3 forward pass per
*image* (not per anchor), so the SAM3 cost is tiny; the DINOv2 embedding of every GT crop is the
new cost, and it is proportional to the number of plants in the image, not to its size. Once you
have seen the real timing, lowering `#SBATCH --time` will get you through the queue faster.

Both models are resident on each GPU at the same time (DINOv2-large is ~1.2 GB in fp32 next to
SAM3), which fits comfortably.

---

## Step 9 — Watch it

```bash
squeue -u $USER                                                                    # queued / running?
tail -f k_diverse_exemplars_no_tiling_sam3_1234567.out                             # everything, live
tail -f $SCRATCH/experiments/sam3/k_diverse_exemplars_no_tiling/shard0.log          # just GPU 0
```

Count how many images are finished so far:

```bash
cat $SCRATCH/experiments/sam3/k_diverse_exemplars_no_tiling/raw_detections/runs_manifest_k_diverse_exemplars_no_tiling_shard*.csv \
  | grep -c k_diverse_exemplars_no_tiling
ls $SCRATCH/experiments/sam3/k_diverse_exemplars_no_tiling/raw_detections/*.npz | wc -l
```

You will also get an email at BEGIN / END / FAIL.

---

## Step 10 — Collect the results

Everything lands in `$SCRATCH/experiments/sam3/k_diverse_exemplars_no_tiling/`:

```bash
ls -R $SCRATCH/experiments/sam3/k_diverse_exemplars_no_tiling/ | head -40
```

**The file you actually quote in the thesis:**

| File | What it is |
|---|---|
| `metrics/experiment_summary.csv` | **the headline numbers** — one row per evaluation mode (`all_gt`, `held_out`), with mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU1 / IoU2, plus the embedding model, the selection rule and the crop settings |

The mean is taken over the **per-image** values, so an image with 40 plants does not outweigh
one with 2. The std is the variation **between UAV images**.

**Everything else in that folder:**

| File | What it is |
|---|---|
| `metrics/experiment_summary_per_archive.csv` | the same table split by archive. Not in the notebook (it ran on one archive); the pooled table above is the notebook's definition, unchanged |
| `metrics/image_level_metrics.csv` | one row per (image × mode) — the raw data: the prompt ids `D1:i+D2:j+D3:k`, the diversity numbers (`min_pairwise_distance`, `mean_pairwise_distance`, `gt_mean_distance`, `search_mode`), TP/FP/FN and every metric. **This is the notebook's CELL 20 table** — because the selection is deterministic there is exactly one run per image, so there is no separate run-level table |
| `metrics/dataset_ap_metrics.csv` | pooled AP50 / AP50:95 — all images ranked in ONE precision-recall curve. **Not** the mean of the per-image AP |
| `metrics/size_group_recall.csv` | notebook CELL 24: recall of the non-prompt GT boxes, split at each image's median plant area. Size is *not* the selection criterion here — this table shows whether three visually diverse prompts happen to cover both size halves anyway, and is directly comparable with `k_size_exemplars_no_tiling` |
| `metrics/prompt_plants_redetected.csv` | whether SAM3 re-detected each D1 / D2 / D3 prompt plant, per role |
| `confusion_matrices/confusion_matrix_{all_gt,held_out}.csv` + `.png` | pooled TP / FP / FN, plus micro precision/recall/F1 in `confusion_matrix_summary.csv` |
| `plots/best_image_*.png` | the qualitative GT-vs-prediction figures (notebook CELL 26) — see below |
| `raw_detections/*.npz` | one file per image: the PRE-NMS detections (boxes, scores), the GT boxes, the D1/D2/D3 prompt indices and roles, the prompt areas, **the full G×G DINOv2 cosine distance matrix**, the diversity statistics, the archive, the flight and the resize scale. This is what Phase 2 reads |
| `raw_detections/runs_manifest_k_diverse_exemplars_no_tiling_shard*.csv` | which images are finished; this is what resume reads |
| `run_config_k_diverse_exemplars_no_tiling.json` | every parameter used, for the thesis appendix |
| `shard0..3.log` | per-GPU logs |

Look at the headline table:

```bash
cd $SCRATCH/experiments/sam3/k_diverse_exemplars_no_tiling/metrics
column -s, -t < experiment_summary.csv | less -S
```

**Which mode to quote:** both, and say what they mean.
`all_gt` = classical evaluation, every GT box counts.
`held_out` = the three plants shown to SAM3 as prompts are removed from the GT and the
predictions that land on them are ignored — "after being shown three visually **different**
examples, how well does it find the **remaining** plants?". This matters more here than in the
tiled experiments: the exemplars sit inside the very image being evaluated, at their real
locations, so SAM3 almost always re-detects them and `all_gt` gets up to three free true
positives per image. `metrics/prompt_plants_redetected.csv` tells you exactly how often that
happened, per role.

**The qualitative figures.** The notebook produced one. Both archives are pooled here, so the
same selection runs three times and you get three files in `plots/`:

```
best_image_AGS_Multi_Rumex_..._all_gt.png
best_image_AgsSpringRumex_..._all_gt.png
best_image_ALL_..._all_gt.png
```

Each is GT (yellow, left) next to the predictions (red, right), with the three exemplar prompts
drawn as dashed boxes in their role colours — **cyan = D1** (the most isolated of the three in
embedding space), **lime = D2**, **orange = D3** — and the confidence written next to each
prediction. The image chosen is the one with the highest per-image AP50 among those with ≥ 7 GT
boxes (with the notebook's fallback when none qualifies). The selection rule and the minimum
pairwise cosine distance of the chosen triple are printed in the figure title, so the figure is
self-documenting.

**Reading the raw detections in Python:**

```python
import numpy as np
z = np.load("raw_detections/AGS_Multi_Rumex__20220518_Eschikon__DJI_0001.npz")
print(z["boxes"].shape, z["scores"].min(), z["gt_boxes"].shape)
print(z["prompt_indices"], [str(r) for r in z["prompt_roles"]])
print(z["distance_matrix"].shape, float(z["min_pairwise_distance"]), str(z["search_mode"]))
print(str(z["archive"]), float(z["sam_scale"]))   # 1024 / 8192 = 0.125
```

The distance matrix is the reason you can re-analyse the selection — for example re-do it with a
different objective, or check how far the chosen triple was from the best possible one — without
ever re-running DINOv2.

---

## Step 11 — Resubmit if it hit the walltime

Completely normal, and harmless. Just run the same command again:

```bash
sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh
```

Every finished image is listed in the shard manifests and is skipped **before** any GPU work, so
the job picks up exactly where it stopped. You will see this near the top of each shard log:

```
Resuming: 431 image(s) already finished for k_diverse_exemplars_no_tiling; skipped.
```

Each shard reads **all** the manifests, not just its own, so resuming still works if you change
`NUM_GPUS` between submissions.

The manifests also carry a guard: if they already hold results produced with a **different
prompt setting** (`Prompt_Type`), the run aborts instead of silently mixing two methods in one
results folder. Give such a run its own `EXPERIMENT_NAME` **and** `OUTPUT_DIR`.

**If a shard crashed but the others finished:** Phase 2 still runs on whatever NPZ files
reached disk, and the job exits non-zero so you get the FAIL email. Just resubmit — it fills
the gaps.

---

## Step 12 — Re-run only the evaluation (optional, cheap)

Phase 2 reads only the NPZ files, so you never need a GPU for it — and DINOv2 is not loaded
either, because the distance matrices are already stored:

```bash
# on a login node, or in any small allocation
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./k_diverse_exemplars_no_tiling_run_sam3.sh evaluate
```

Use this to change the operating point without re-running SAM3 — the pre-NMS detections on
disk are exactly what is needed to replay any (confidence, NMS IoU) pair:

```bash
OPERATING_CONFIDENCE=0.40 NMS_IOU_THRESHOLD=0.50 \
OUTPUT_DIR=$SCRATCH/experiments/sam3/k_diverse_exemplars_no_tiling \
./k_diverse_exemplars_no_tiling_run_sam3.sh evaluate
```

> **Careful:** this overwrites the CSVs in `metrics/`, `confusion_matrices/` and `plots/`. To
> keep both, copy the folder first, or point `OUTPUT_DIR` at a copy of the experiment.

Note that `evaluate` needs `DATASET_ROOT` to be reachable, because the qualitative figures
reopen the selected images. If the dataset has been purged from `$SCRATCH`, add `--no-plots`:

```bash
./k_diverse_exemplars_no_tiling_run_sam3.sh evaluate --no-plots
```

Every CSV and confusion matrix is still written; only the figures are skipped.

---
---

# Notes and problems

## If it will not fit in the walltime

Unlikely for this experiment, but in order of preference:

1. **Do nothing.** Resubmit as many times as needed (Step 11). Two 24 h jobs = one 48 h job.
2. **`DINOV2_BATCH_SIZE=32`** — fewer, larger DINOv2 forward passes. This does not change any
   result, only the GPU memory used by the embedding step.
3. **More GPUs** — needs a job array; ask and I will write one.

`MAX_DIM`, `CROP_CONTEXT` and `K_EXEMPLARS` are the other levers, but changing any of them
changes the experiment (see the last section), so they are not a way to save time on *this* run.


Three things to say about this in the thesis:

- **The operating points differ.** E01_2 / E02_2 freeze the operating point at confidence
  **0.40** while inferring at 0.30. This experiment follows its notebook and keeps the
  operating point **equal to the inference threshold, 0.30**. Precision / recall / F1 / IoU are
  therefore *not* measured at the same confidence across the two families. Either say so
  explicitly, or replay one of them at the other's threshold with Step 12 (free, no GPU) so the
  comparison is like-for-like. The AP columns are unaffected — they always use every post-NMS
  detection ≥ 0.30 in all experiments.
- **What is being ablated.** Against `1_exemplar_no_tiling` (1 prompt) and
  `k_size_exemplars_no_tiling` (3 prompts chosen by *size*), this experiment isolates the effect
  of choosing the 3 prompts by **visual diversity**. Everything downstream of SAM3 — NMS,
  matcher, evaluation, AP, confusion matrices, tables, plots — is identical, so a difference in
  the numbers is a difference in the prompt selection, nothing else. Against the tiled
  experiments the prompt *mechanism* also differs (box at its real location vs. cropped and
  pasted into a strip above each tile), so that comparison is not tiling alone.
- **Resolution.** Whole-image mode downscales an 8192 px image to 1024 px, so a Rumex plant
  that was ~60 px across becomes ~7 px. Expect recall to suffer on small plants; that is the
  finding, not a bug. Note that the DINOv2 crops are taken from the **full-resolution** image,
  not from the 1024 px copy, so the selection sees the sharpest available pixels even though
  SAM3 does not.

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

then change the one line at the bottom of `k_diverse_exemplars_no_tiling_submit_sam3.sh`:

```bash
srun --environment=sam3 ...
```

and use `--environment=sam3` in Step 5 as well.

The same image fixes a failing DINOv2 import — `AutoModel` / `AutoImageProcessor` come from the
same `transformers` package.

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

`matplotlib` is the only optional one: without it every CSV is still written and only the PNGs
are skipped.

## What the fixed operating point means for your numbers

`OPERATING_CONFIDENCE = 0.30` and `NMS_IOU_THRESHOLD = 0.40` are fixed from the start, exactly
as in the notebook. No confidence × NMS sweep is performed, so nothing is tuned on the test
data. The prompt selection is not tuned either: `max_min` cosine dispersion is a fixed rule
evaluated exhaustively, with ties broken by the lowest GT indices, so it is deterministic and
depends on no seed.

The notebook deliberately sets the operating confidence **equal to** the SAM3 inference
threshold. One consequence worth knowing:

- **precision / recall / F1 / IoU1 / IoU2** describe ONE operating point: detections scoring
  ≥ 0.30.
- **AP50 / AP50:95** always use **every** post-NMS detection ≥ 0.30, because AP is the area
  under the precision-recall curve and truncating the detection list would just cut the tail
  off that curve.
- Because the two thresholds are the same number here, both use the *same* prediction set.
  They still answer different questions — ranking quality vs deployed behaviour — and they are
  both in the same CSV row on purpose.

If you want a lower floor for the AP curve you must re-run Phase 1 with a lower `THRESHOLD`,
since detections below it were never saved:

```bash
THRESHOLD=0.05 EXPERIMENT_NAME=k_diverse_exemplars_no_tiling_lowconf \
OUTPUT_DIR=$SCRATCH/experiments/sam3/k_diverse_exemplars_no_tiling_lowconf \
sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh
```

Note that this also moves the operating point unless you pin `OPERATING_CONFIDENCE=0.30`
separately — they are two independent flags here, even though the notebook ties them together.

## Other troubleshooting

| Symptom | Fix |
|---|---|
| `SCRATCH: unbound variable` | you are not on the cluster, or the module environment is not loaded |
| Job hangs at "Loading SAM3" or "Loading DINOv2" | `$SCRATCH/hf_cache` is missing that model and the node is offline — redo Step 4 |
| CUDA out of memory | lower `DINOV2_BATCH_SIZE` (16 → 8) first, then `MAX_DIM` (1024 → 768), or `NUM_GPUS=2`. Memory is dominated by the two resident models plus the single resized image |
| Host RAM fills up | each process holds one full-resolution image plus all its GT crops; lower `DINOV2_BATCH_SIZE` or use fewer GPUs per node if the images are unusually large |
| Everything reports `F1=0.0000` | wrong class id or wrong labels — recheck Step 6b |
| `search_mode` is `greedy` in the CSVs | an image has more than `DIVERSITY_EXACT_MAX_GT` GT boxes, so the exact max-min search was replaced by the farthest-point fallback. Raise `DIVERSITY_EXACT_MAX_GT` if you want it exact everywhere, and mention it if it happened |
| `search_mode` is `all_gt_used` | that image has ≤ 3 GT boxes, so all of them were used as prompts. Expected; those images are the ones that go NaN in `held_out` |
| `min_pairwise_distance` is NaN | the image has exactly 1 GT box, so there is no pair. Expected |
| Recall is low but precision is fine | expected consequence of the 1024 px downscale on small plants — that is the tiling ablation's whole point |
| `held_out` is all NaN | every plant of every image was used as a prompt — expected only on images with ≤ 3 GT boxes |
| `Nothing to plot` in Phase 2 | no image had a valid AP50, or `DATASET_ROOT` is unreachable. Every CSV is still written; add `--no-plots` to silence it |
| Phase 2 is slow / heavy | it holds every run in memory at once, exactly like the notebook (needed for the pooled AP of CELL 22). Give it a node with more RAM if it gets killed |
| The run aborts with "already holds results for another prompt setting" | the results folder was reused for a different method. Use a fresh `EXPERIMENT_NAME` **and** `OUTPUT_DIR` |

## Settings you might change

Set them before `sbatch`; they are forwarded into the job.

```bash
DTYPE=float32              sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # what the notebook's T4 used for SAM3
MAX_DIM=2048               sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # less downscaling, slower
THRESHOLD=0.05             sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # save more detections for the AP curve
OPERATING_CONFIDENCE=0.40  sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # match E01_2 / E02_2's operating point
NMS_IOU_THRESHOLD=0.50     sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # different NMS
K_EXEMPLARS=5              sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # five diverse visual prompts
CROP_CONTEXT=0.0           sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # tight crops, no context margin
DINOV2_MODEL_ID=facebook/dinov2-base  sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # smaller embedder (download it first!)
DINOV2_BATCH_SIZE=32       sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # faster embedding, more GPU memory
NUM_GPUS=2                 sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # fewer GPUs
DATASET_ROOT=/some/path    sbatch k_diverse_exemplars_no_tiling_submit_sam3.sh  # dataset elsewhere
```

Changing `K_EXEMPLARS`, `MAX_DIM`, `THRESHOLD`, `CROP_CONTEXT`, `DINOV2_MODEL_ID`,
`DINOV2_INPUT_SIZE` or `DIVERSITY_EXACT_MAX_GT` changes what is stored in the NPZ files, so
give those runs their own `EXPERIMENT_NAME` **and** `OUTPUT_DIR`. Changing only
`OPERATING_CONFIDENCE` / `NMS_IOU_THRESHOLD` does not — use Step 12 instead, it is free.

A new `DINOV2_MODEL_ID` must be cached first, on a login node:

```bash
DINOV2_MODEL_ID=facebook/dinov2-base ./k_diverse_exemplars_no_tiling_run_sam3.sh download
```

To run this pipeline on one archive only:

```bash
./k_diverse_exemplars_no_tiling_run_sam3.sh run --archives AGS_Multi_Rumex
```

(which is what the notebook did, so it is the way to reproduce the notebook's numbers exactly).
