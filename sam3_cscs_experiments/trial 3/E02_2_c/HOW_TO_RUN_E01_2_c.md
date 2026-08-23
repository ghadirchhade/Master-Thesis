# HOW TO RUN — Experiment E02_2_c

**What E02_2_c is:** SAM3 prompted with **3 exemplar crops** (the anchor box plus two other ground-truth boxes from the same image), **tiling ON at 1536 px tiles / 384 px overlap (25%)**, **plus one extra forward pass on the whole image downscaled by 2**, everything merged by a single NMS. Run over both `AGS_Multi_Rumex` and `AgsSpringRumex` pooled as one dataset. Confidence threshold 0.3, NMS IoU 0.5.

It is **E02_2_b on a finer tile grid and nothing else** — same three crops, same global pass at ×2, same thresholds, same exemplar seeding. Only the tile geometry changes. The pair therefore isolates exactly one question: does a smaller tile (less downscaling inside SAM3, so more effective resolution per plant) beat a bigger one (more surrounding context per forward pass)?


```
PHASE A — SETUP            do once, ~30 minutes
├─ Step 1   copy the files onto the cluster
├─ Step 2   put the dataset on $SCRATCH
├─ Step 3   HuggingFace licence + token
├─ Step 4   download the model and supervision
├─ Step 5   open an interactive session inside the container
├─ Step 6   check the container has what it needs
└─ Step 7   smoke test on 2 images

PHASE B — RUN              every time
├─ Step 8   submit the job
├─ Step 9   watch it
├─ Step 10  collect the results
└─ Step 11  resubmit if it hit the 24 h limit
```

> If you have already set up any of the other experiments, Phase A is nearly free: Steps 2, 3 and 4 produce exactly the same dataset, token and `$SCRATCH/hf_cache` / `$SCRATCH/pyextra`. Copy the three new scripts (Step 1), then skip straight to Step 6.

---

## PHASE A — SETUP (once)

### Step 1 — Copy the files onto the cluster

Put the three scripts next to the ones you already have, in the same folder:

```
$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3/
├── infer_sam3.py             ← E02_2, untouched
├── run_sam3.sh               ← E02_2, untouched
├── submit_sam3.sh            ← E02_2, untouched
├── infer_sam3_single.py      ← E01_2, untouched
├── run_sam3_single.sh        ← E01_2, untouched
├── submit_sam3_single.sh     ← E01_2, untouched
├── infer_sam3_2.py           ← E02_2_b, untouched
├── run_sam3_2.sh             ← E02_2_b, untouched
├── submit_sam3_2.sh          ← E02_2_b, untouched
├── infer_sam3_single_2.py    ← E01_2_b, untouched
├── run_sam3_single_2.sh      ← E01_2_b, untouched
├── submit_sam3_single_2.sh   ← E01_2_b, untouched
├── infer_sam3_3.py           ← NEW  (E02_2_c)
├── run_sam3_3.sh             ← NEW
├── submit_sam3_3.sh          ← NEW
└── HOW_TO_RUN_3.md           ← this file
```

Then, on a **login node**:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
chmod +x run_sam3_3.sh submit_sam3_3.sh
ls -l
```

**You should see:** `run_sam3_3.sh` and `submit_sam3_3.sh` marked executable (`-rwxr-xr-x`).

---

### Step 2 — Put the dataset on `$SCRATCH`

*(Identical to the other experiments. If the dataset is already there, skip to Step 3.)*

Extract the two archives side by side under one folder:

```bash
mkdir -p $SCRATCH/overney/dataset
cd $SCRATCH/overney/dataset
tar -xzf /path/to/AGS_Multi_Rumex.tar.gz
tar -xzf /path/to/AgsSpringRumex.tar.gz
```

The result must look exactly like this:

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

**Important:** the folder names are case-sensitive and must match exactly. The other two archives (`AGS_Multiple_Fields`, `AGS_Multiple_Fields_Embeddings`) may be present — the code detects them and skips them on purpose.

**Note on class ids:** `AGS_Multi_Rumex` labels rumex as class **0**, `AgsSpringRumex` as class **2**. The script knows this per archive; you do not have to do anything. Your notebook hard-coded `RUMEX_CLASS_ID = 0`, which is why it could only read the first archive.

**Why `$SCRATCH` and not `$HOME`:** `$HOME` is small and slow; `$SCRATCH` is the large fast filesystem. (It is also periodically purged, so keep the original archives somewhere safe.)

---

### Step 3 — HuggingFace licence + token

*(Identical to the other experiments. If `$SCRATCH/hf_cache` is already populated, skip to Step 5.)*

`facebook/sam3` is a **gated** model. Two things are needed:

1. Go to https://huggingface.co/facebook/sam3 while logged in and **accept the licence**.
2. Create a token at https://huggingface.co/settings/tokens (read access is enough).

Then, on a **login node**:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

**Note:** the token must belong to the *same account* that accepted the licence, otherwise Step 4 fails with a 401.

*(This replaces the notebook's `userdata.get("HF_TOKEN")` and `login()` cells — there is no interactive prompt in a batch job.)*

---

### Step 4 — Download the model and `supervision`

Still on a **login node** (this is the only step that needs internet — compute nodes have none):

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./run_sam3_3.sh download
```

This does two things:

- downloads the SAM3 weights into `$SCRATCH/hf_cache`
- installs the `supervision` package into `$SCRATCH/pyextra` (supervision computes mAP50; the code refuses to run without it)

**You should see:**

```
--- 1/2 : model weights -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--facebook--sam3/snapshots/...
--- 2/2 : supervision -> /scratch/.../pyextra ---
Successfully installed supervision-...
supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1).
```

If it says `ERROR: HF_TOKEN is not set` → go back to Step 3. If it says `401` or `gated` → the licence was not accepted with that token's account.

If you already ran the `download` mode of any of the other `run_sam3*.sh` scripts, this is a no-op — same cache, same `$PYEXTRA`. Running it again is harmless.

---

### Step 5 — Open an interactive session inside the container

Steps 6 and 7 must run **inside the container**, not on the login node — otherwise you are checking the login node's Python, which is not the one the job will use.

```bash
srun --account=go077 --time=00:30:00 \
     --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
     --environment=yolo26 --pty bash
```

**You should see:** a new shell prompt, running on a compute node. Everything in Steps 6 and 7 happens in this shell.

*(If your site needs a partition flag, add `--partition=...`. If the allocation takes a while, that is just the queue.)*

---

### Step 6 — Check the container has what it needs

Inside the interactive shell from Step 5:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./run_sam3_3.sh dryrun
```

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

#### 6a. Look for these three lines in the debug block

```
torch: 2.x.x | cuda available: True | device count: 1
Sam3Model import: OK
MeanAveragePrecision import: OK
```

| If you see | Meaning | Fix |
|---|---|---|
| `Sam3Model import FAILED` | the `yolo26` image's `transformers` is too old for SAM3 | see **Problem A** at the bottom |
| `supervision import FAILED` | `$SCRATCH/pyextra` not found or wrong Python version | rerun Step 4; if it still fails, see **Problem B** |
| `cuda available: False` | no GPU in this allocation | add `--gpus-per-task=1` to Step 5 |

**Do not continue until all three lines are correct.** Everything after this depends on them.

#### 6b. Check the configuration banner

```
 n_exemplars    : 3  (multiple prompt)
 prompting      : exemplar crop(s) in a strip above each tile
 tiling         : ON  (tile=1536, overlap=384, cache_tiles=True)
 global pass    : ON  (downscale=2, +1 forward pass per anchor)
 batching       : OFF (one tile per forward pass)
 thresholds     : conf=0.3  mask=0.4  match_iou=0.5  nms_iou=0.5
```

These six lines **are** E02_2_c. If `tile=2000, overlap=800`, you are running E02_2_b's geometry — `TILE_SIZE`/`OVERLAP` leaked in from your shell. If `n_exemplars` says 1, set `PRESET=multi`.

#### 6c. Check the dataset report

```
Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
  AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
  AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
Total images: NNN | this shard: NNN
```

Confirm the image counts match what you expect, and that `class_id` is **0** for `AGS_Multi_Rumex` and **2** for `AgsSpringRumex`.

It also prints a cost estimate:

```
  sampled NNN image(s) of this shard -> NNNN anchor runs ({'AGS_Multi_Rumex': ..., 'AgsSpringRumex': ...})
  tiles per anchor run at 8192x5460: 35 (+1 global pass = 36)
  => ~NNNNN SAM3 forward passes for those NNN images
```

**35** is the right number for tile=1536 / overlap=384 on a 8192×5460 image. If you see 28, the geometry has been overridden to E02_2_b's values.

- `No images with labels found` → `$SCRATCH/overney/dataset` is wrong, or the folder names do not match. Go back to Step 2.
- `N image(s) have no matching label file` → those are skipped. A handful is normal; if it is *every* image, the label basenames do not match the image basenames.

---

### Step 7 — Smoke test on 2 images

Still inside the interactive shell. This is the first time the model actually loads. It writes to a throwaway folder so it cannot pollute the real E02_2_c results.

```bash
EXPERIMENT_NAME=E02_2_c_smoke \
OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke_3 \
NUM_GPUS=1 \
./run_sam3_3.sh run --limit-images 2
```

**You should see** the model load, then one line per anchor run:

```
Loading SAM3 from 'facebook/sam3' onto cuda (float32) ...
SAM3 ready. Parameters live on: cuda:0
      NOTE: 2/12 GT box(es) in AGS_Multi_Rumex/2022.../DJI_0001 are larger than the tile overlap (384px; widest 706px). Those may be cut at every tile border they touch.
    [E02_2_c_smoke] run #1 | AGS_Multi_Rumex/2022.../DJI_0001 | anchor=0 (1/12) | prompt=0+7+3 | tiles=35 | preds=10 (tiled=17, global=2) | mAP50=0.352 | 34.8s
```

Five things to note from this output:

1. **The `NOTE:` line** — how many ground-truth boxes in this image are bigger than the 384 px overlap. This is the E02_2_c-specific risk (see the header). It is printed once per image, log only, and changes nothing. If it fires on most images, that is a real argument for a larger `OVERLAP`.
2. **`prompt=0+7+3`** — three numbers joined by `+`. The first is always the anchor, the other two are the sampled companions. If you see a bare `prompt=0`, you are running the `single` preset (or that image only has one plant, which is legitimate — check `n_gt`).
3. **`tiles=35`** — the finer grid. If it says 28, the geometry is wrong.
4. **`tiled=17, global=2`** — how many raw detections each source contributed *before* the merge. If `global=` is 0 on every single run, the global pass is finding nothing and you may as well turn it off.
5. **`34.8s`** (or whatever you get) is the time for **ONE** anchor run. Multiply by the total anchor count from Step 6c to estimate the whole job. If that exceeds 24 h, see *"If it will not fit in 24 hours"* below — the short answer is that it is fine, you just resubmit and it resumes.

`mAP50=` should be a plausible number, not `0.000` on every single run. All zeros means something is wrong with the prompts or the labels.

Then leave the interactive session:

```bash
exit
```

**Setup is done. You never have to repeat Phase A.**

---

## PHASE B — RUN

### Step 8 — Submit the job

On a **login node**:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
sbatch submit_sam3_3.sh
```

**You should see:** `Submitted batch job 1234567`

That is it. No arguments needed — E02_2_c is the default. The job asks for 1 node, 4 GPUs, 64 CPUs, 24 hours, and internally splits the images across the 4 GPUs.

All the experiments can queue at the same time: they write to different folders (`.../sam3/E01_2`, `.../sam3/E02_1`, `.../sam3/E02_2`, `.../sam3/E02_2_b`, `.../sam3/E01_2_b`, `.../sam3/E02_2_c`) and different log files (`sam3_single_%j.out`, `sam3_whole_%j.out`, `sam3_%j.out`, `sam3_2_%j.out`, `sam3_single_2_%j.out`, `sam3_3_%j.out`).

---

### Step 9 — Watch it

```bash
squeue -u $USER                                       # queued / running?
tail -f sam3_3_1234567.out                            # everything, live
tail -f $SCRATCH/experiments/sam3/E02_2_c/shard0.log  # just GPU 0
```

Count how many runs are finished so far:

```bash
cat $SCRATCH/experiments/sam3/E02_2_c/runs/results_E02_2_c_shard*.csv | grep -c E02_2_c
```

Count how often the oversized-plant note fired:

```bash
grep -c "larger than the tile overlap" sam3_3_1234567.out
```

You will also get an email at BEGIN / END / FAIL.

---

### Step 10 — Collect the results

Everything lands in `$SCRATCH/experiments/sam3/E02_2_c/`:

```bash
ls $SCRATCH/experiments/sam3/E02_2_c/
```

**The three files you actually want:**

| File | What it is |
|---|---|
| `summary_E02_2_c_all.csv` | the headline numbers — mean and std of mAP50 / precision / recall / IoU1 / IoU2, averaged over images |
| `summary_E02_2_c_AGS_Multi_Rumex.csv` | same, for that archive only |
| `summary_E02_2_c_AgsSpringRumex.csv` | same, for that archive only |

Look at them:

```bash
cd $SCRATCH/experiments/sam3/E02_2_c
column -s, -t < summary_E02_2_c_all.csv
```

**Everything else in that folder:**

| File | What it is |
|---|---|
| `results_E02_2_c_all.csv` | one row per (image × anchor) — the raw data |
| `results_E02_2_c_<archive>.csv` | the same rows split by archive |
| `image_level_E02_2_c_*.csv` | one row per image (the prompt runs averaged together) |
| `raw_summary_E02_2_c_*.csv` | mean/std treating every run as an independent sample |
| `runs/results_E02_2_c_shard*.csv` | the per-GPU raw output; this is what resume reads |
| `runs/detections_E02_2_c_shard*.jsonl` | the predicted boxes + scores for every run |
| `runs/masks_E02_2_c_shard*.jsonl` | only with `SAVE_MASKS=1` |
| `run_config_E02_2_c.json` | every parameter used, for the thesis appendix |
| `shard0..3.log` | per-GPU logs |



### Step 11 — Resubmit if it hit the 24 h limit

Completely normal, and harmless. Just run the same command again:

```bash
sbatch submit_sam3_3.sh
```

Every finished `(image, anchor)` pair is already in the shard CSVs and is skipped **before any GPU work**, so the job picks up exactly where it stopped. Repeat until it finishes inside the walltime. You will see this near the top of the log:

```
Resume: 4821 run(s) already recorded for E02_2_c; they will be skipped.
```

If a shard crashed but the others finished: the summaries are still built from whatever completed, and the job exits non-zero so you get the FAIL email. Just resubmit — it fills the gaps.

---


### Notebook bugs fixed in the port

1. **`TypeError` on the first anchor.** The notebook calls `run_sam3_pipeline(..., nms_iou_thresh=NMS_IOU_THRESHOLD)` alongside `cached_tiles` and `batch_size`; combined with the presence-gate block being commented out mid-function, the loop as written could not complete a single run. Cleaned up here.
2. **Non-reproducible exemplar seeding.** `select_exemplar_indices` seeded with `hash((image_id, anchor_idx))`, and Python randomises `hash()` per process unless `PYTHONHASHSEED` is set. At 3 exemplars this genuinely matters: the two companion crops differed between runs, so the notebook's numbers were not reproducible — and no two experiments could be paired. Replaced with a blake2b-derived seed, identical on every machine and every run. (`run_sam3_3.sh` also exports `PYTHONHASHSEED=0` for good measure.)
3. **Masks from the global pass were unusable.** The notebook appends them as `(mask, 0, 0)` with the mask still at half resolution and no record of that, so anything reading `masks_*.jsonl` would place them wrong. Here every mask record carries an explicit `"scale"`.
4. **GT boxes were not clipped.** `load_yolo_boxes` now clips to the image and drops degenerate (≤1 px) boxes, so a slightly out-of-frame annotation cannot produce a negative-area GT.

Also: the notebook's main loop writes `results4_E02_2.csv` while the config cell still says `EXPERIMENT_NAME = "E02_2"` — the same tag it used for the 2000/800 run that wrote `results3_E02_2.csv`. Two different tile geometries under one experiment name is exactly the trap `EXPERIMENT_NAME` is supposed to prevent; that is why this port is tagged **E02_2_c**, in its own folder.

---

### If it will not fit in 24 hours

You have four options, in order of preference:

1. **Do nothing.** Resubmit as many times as needed (Step 11). Three 24 h jobs = one 72 h job.
2. **`DTYPE=bfloat16 sbatch submit_sam3_3.sh`** — roughly 2× faster on GH200/H100, negligible accuracy change.
3. **`--max-anchors-per-image 5`** — instead of every ground-truth box taking a turn as the anchor, use only the first 5 per image. This changes *what you are measuring*, so mention it in the thesis. Apply it to E02_2_b as well, or the comparison stops being paired.
4. **More GPUs** — needs a job array.

---

### Problem A — `Sam3Model import FAILED`

The `yolo26` container was built for ultralytics and its `transformers` is too old for SAM3. Build a dedicated image:

```dockerfile
# Dockerfile
FROM <whatever image ~/.edf/yolo26.toml points at>
RUN pip install --no-cache-dir "transformers>=<version with SAM3>" supervision accelerate
```

```bash
podman build -t sam3 .
enroot import -o $SCRATCH/images/sam3.sqsh podman://sam3:latest
cp ~/.edf/yolo26.toml ~/.edf/sam3.toml
# edit ~/.edf/sam3.toml so `image = ` points at $SCRATCH/images/sam3.sqsh
```

then change the one line at the bottom of `submit_sam3_3.sh`:

```bash
srun --environment=sam3 ...
```

and use `--environment=sam3` in Step 5 as well.

---

### Problem B — `supervision import FAILED` inside the container

Step 4 installed it with the *login node's* Python. If the container uses a different Python version, the install is invisible to it. Install from inside the container instead — but the compute node has no internet, so download the wheel on the login node first:

```bash
# login node
pip download --no-deps -d $SCRATCH/wheels supervision

# inside the container (Step 5 shell)
pip install --target $SCRATCH/pyextra --no-deps --no-index --find-links $SCRATCH/wheels supervision
```

**Always `--no-deps`.** `supervision` would otherwise install its own numpy / opencv into `$SCRATCH/pyextra`, and because that directory is searched *before* the container's own packages, those copies would shadow the ones torch was compiled against and break torch.

---

### One thing to decide before quoting numbers

The default confidence threshold is **0.3**, which means detections scoring below 0.3 never reach the metric — so the reported mAP50 is a **lower bound**, not a true mAP.

For the numbers that go in the thesis, run a second pass with a low threshold:

```bash
THRESHOLD=0.001 EXPERIMENT_NAME=E02_2_c_lowconf sbatch submit_sam3_3.sh
```

It writes to its own folder, so you can keep both and compare. If you do this for E02_2_c, **do it for E02_2_b too** — comparing a low-threshold run against a 0.3 run measures the threshold, not the tile geometry.

---

### Other troubleshooting

| Symptom | Fix |
|---|---|
| `SCRATCH: unbound variable` | you are not on the cluster, or the module environment is not loaded |
| Job hangs at "Loading SAM3" | `$SCRATCH/hf_cache` is empty and the node is offline — redo Step 4 |
| `CUDA out of memory` | smaller composites than E02_2_b, so this is less likely here. `DTYPE=bfloat16` first, then `TILE_SIZE=1280 OVERLAP=320`, then `NUM_GPUS=2`. Note there is no batch auto-split any more — batching was removed |
| Host RAM pressure with 4 shards | `CACHE_TILES=0` — trades ~250 MB per shard for slower cropping |
| `NOTE: N/M GT box(es) ... larger than the tile overlap` fires constantly | the 384 px overlap is too small for this dataset's big plants. Run `OVERLAP=768` with a new `EXPERIMENT_NAME` and compare |
| `exemplar strip is wider than the tile` fires often | three large crops do not fit across a 1536 px tile, so tiles get downscaled harder — which undercuts the point of the finer grid. Count how often before interpreting the result |
| `n_pred_global` is 0 everywhere | the global pass is finding nothing. Verify with the pandas snippet in Step 10, then consider `GLOBAL_PASS=0` (with a new `EXPERIMENT_NAME`) |
| `tiles=28` instead of `tiles=35` | `TILE_SIZE`/`OVERLAP` have been overridden to E02_2_b's values |
| `prompt=0` instead of `prompt=0+7+3` | either `PRESET=single` is active, or that image has only one GT box (check `n_gt` in the CSV) |
| Everything reports `mAP50=0.000` | wrong class id or wrong labels — recheck Step 6c |
| merge with E02_2_b drops rows | the two runs did not cover the same images — check `--archives` / `--limit-images` on both |

---

### Settings you might change

Set them **before** `sbatch`; they are forwarded into the job.

```bash
DTYPE=bfloat16          sbatch submit_sam3_3.sh   # faster
THRESHOLD=0.001         sbatch submit_sam3_3.sh   # proper mAP curve
OVERLAP=768             sbatch submit_sam3_3.sh   # overlap clears the widest plant (706px)
GLOBAL_PASS=0           sbatch submit_sam3_3.sh   # tiles only (ablate the global pass)
GLOBAL_DOWNSCALE=4      sbatch submit_sam3_3.sh   # coarser global pass
PRESET=single           sbatch submit_sam3_3.sh   # 1 exemplar crop instead of 3
NMS_IOU=0.3             sbatch submit_sam3_3.sh   # stricter duplicate merging
TILE_SIZE=1280 OVERLAP=320 sbatch submit_sam3_3.sh  # even finer grid
CACHE_TILES=0           sbatch submit_sam3_3.sh   # lower host RAM
NUM_GPUS=2              sbatch submit_sam3_3.sh   # fewer GPUs
SAVE_MASKS=1            sbatch submit_sam3_3.sh   # also store masks (large!)
DATASET_ROOT=/some/path sbatch submit_sam3_3.sh   # dataset elsewhere
```

Anything after the mode is forwarded straight to Python, so one-off flags work too:

```bash
./run_sam3_3.sh run --limit-images 20 --max-anchors-per-image 3
./run_sam3_3.sh run --archives AGS_Multi_Rumex        # one archive only
./run_sam3_3.sh run --no-global-pass                  # same as GLOBAL_PASS=0
```
