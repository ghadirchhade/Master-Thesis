# HOW TO RUN — Experiment E02_2


What E02_2 is: SAM3 prompted with 3 exemplar crops, tiling ON (1000 px tiles, 150 px overlap),
run over both `AGS_Multi_Rumex` and `AgsSpringRumex` pooled as one dataset.

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



# PHASE A — SETUP (once)

## Step 1 — Copy the files onto the cluster

Put the three scripts here:

```
$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3/
├── infer_sam3.py
├── run_sam3.sh
└── submit_sam3.sh
```

Then, **on a login node**:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
chmod +x run_sam3.sh submit_sam3.sh
ls -l
```

**You should see:** three files, with `run_sam3.sh` and `submit_sam3.sh` marked executable (`-rwxr-xr-x`).

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

## Step 4 — Download the model and `supervision`

**Still on a login node** (this is the only step that needs internet — compute nodes have none):

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./run_sam3.sh download
```

This does two things:

1. downloads the SAM3 weights into `$SCRATCH/hf_cache`
2. installs the `supervision` package into `$SCRATCH/pyextra`
   (`supervision` computes mAP50; the code refuses to run without it)

**You should see:**

```
--- 1/2 : model weights -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--facebook--sam3/snapshots/...
--- 2/2 : supervision -> /scratch/.../pyextra ---
Successfully installed supervision-...
supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
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
./run_sam3.sh dryrun
```

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

### 6a. Look for these three lines in the debug block

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
  tiles per anchor run at 8192x5460: 70
  => ~NNNNN SAM3 forward passes for those NNN images
```

**If it says `Discovered 0 images`** → `$SCRATCH/overney/dataset` is wrong, or the folder
names do not match. Go back to Step 2.

**If it warns `N image(s) have no matching label file`** → those are skipped. A handful is
normal; if it is *every* image, the label basenames do not match the image basenames.

---

## Step 7 — Smoke test on 2 images

**Still inside the interactive shell.** This is the first time the model actually loads.
It writes to a throwaway folder so it cannot pollute the real E02_2 results.

```bash
EXPERIMENT_NAME=E02_2_smoke \
OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke \
NUM_GPUS=1 \
./run_sam3.sh run --limit-images 2
```

**You should see** the model load, then one line per anchor run:

```
Loading SAM3 from 'facebook/sam3' onto cuda (float32) ...
SAM3 ready. Parameters live on: cuda:0
    [E02_2_smoke] run #1 | AGS_Multi_Rumex/2022.../DJI_0001 | anchor=0 (1/12) | prompt=0+7+3 | preds=9 | mAP50=0.412 | 58.3s
```

**Two things to note from this output:**

- **`58.3s`** (or whatever you get) is the time for ONE anchor run. Multiply by the total
  anchor count from Step 6b to estimate the whole job. If that exceeds 24 h, see
  **"If it will not fit in 24 hours"** below — the short answer is that it is fine, you just
  resubmit and it resumes.
- **`mAP50=`** should be a plausible number, not `0.000` on every single run. All zeros means
  something is wrong with the prompts or the labels.

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
sbatch submit_sam3.sh
```

**You should see:** `Submitted batch job 1234567`

That is it. No arguments needed — E02_2 is the default. The job asks for 1 node, 4 GPUs,
64 CPUs, 24 hours, and internally splits the images across the 4 GPUs.

---

## Step 9 — Watch it

```bash
squeue -u $USER                                     # queued / running?
tail -f sam3_1234567.out                            # everything, live
tail -f $SCRATCH/experiments/sam3/E02_2/shard0.log  # just GPU 0
```

Count how many runs are finished so far:

```bash
cat $SCRATCH/experiments/sam3/E02_2/runs/results_E02_2_shard*.csv | grep -c E02_2
```

You will also get an email at BEGIN / END / FAIL.

---

## Step 10 — Collect the results

Everything lands in `$SCRATCH/experiments/sam3/E02_2/`:

```bash
ls $SCRATCH/experiments/sam3/E02_2/
```

**The three files you actually want:**

| File | What it is |
|---|---|
| `summary_E02_2_all.csv` | **the headline numbers** — mean and std of mAP50 / precision / recall / IoU1 / IoU2, averaged over images |
| `summary_E02_2_AGS_Multi_Rumex.csv` | same, for that folder only |
| `summary_E02_2_AgsSpringRumex.csv` | same, for that folder only |

Look at them:

```bash
cd $SCRATCH/experiments/sam3/E02_2
column -s, -t < summary_E02_2_all.csv
```

**Everything else in that folder:**

| File | What it is |
|---|---|
| `results_E02_2_all.csv` | one row per (image × anchor) — the raw data |
| `results_E02_2_<archive>.csv` | the same rows split by folder |
| `image_level_E02_2_*.csv` | one row per image (the prompt runs averaged together) |
| `raw_summary_E02_2_*.csv` | mean/std treating every run as an independent sample |
| `runs/results_E02_2_shard*.csv` | the per-GPU raw output; this is what resume reads |
| `runs/detections_E02_2_shard*.jsonl` | the predicted boxes + scores for every run |
| `run_config_E02_2.json` | every parameter used, for the thesis appendix |
| `shard0..3.log` | per-GPU logs |

**Which number to quote:** `summary_E02_2_all.csv`. It averages the per-image means, so an
image with 40 plants does not outweigh one with 2.

**Reading the detections in Python:**

```python
import json
rows = [json.loads(l) for l in open("runs/detections_E02_2_shard0.jsonl")]
# each row: {"image_ID", "anchor_idx", "boxes": [[x1,y1,x2,y2], ...], "scores": [...], ...}
```

---

## Step 11 — Resubmit if it hit the 24 h limit

Completely normal, and harmless. Just run the same command again:

```bash
sbatch submit_sam3.sh
```

Every finished (image, anchor) pair is already in the shard CSVs and is skipped **before**
any GPU work, so the job picks up exactly where it stopped. Repeat until it finishes inside
the walltime. You will see this near the top of the log:

```
Resume: 4821 run(s) already recorded for E02_2; they will be skipped.
```

**If a shard crashed but the others finished:** the summaries are still built from whatever
completed, and the job exits non-zero so you get the FAIL email. Just resubmit — it fills
the gaps.

---
---

# Notes and problems

## If it will not fit in 24 hours

You have four options, in order of preference:

1. **Do nothing.** Resubmit as many times as needed (Step 11). Three 24 h jobs = one 72 h job.
2. **`DTYPE=bfloat16 sbatch submit_sam3.sh`** — roughly 2× faster on GH200/H100, negligible
   accuracy change.
3. **`--max-anchors-per-image 5`** — instead of every ground-truth box taking a turn as the
   anchor, use only the first 5 per image. This changes what you are measuring, so mention it
   in the thesis.
4. **More GPUs** — needs a job array; ask and I will write one.

## Problem A — `Sam3Model import FAILED`

The `yolo26` container was built for ultralytics and its `transformers` is too old for SAM3.
Build a dedicated image:

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

then change the one line at the bottom of `submit_sam3.sh`:

```bash
srun --environment=sam3 ...
```

and use `--environment=sam3` in Step 5 as well.

## Problem B — `supervision import FAILED` inside the container

Step 4 installed it with the **login node's** Python. If the container uses a different Python
version, the install is invisible to it. Install from inside the container instead — but the
compute node has no internet, so download the wheel on the login node first:

```bash
# login node
pip download --no-deps -d $SCRATCH/wheels supervision

# inside the container (Step 5 shell)
pip install --target $SCRATCH/pyextra --no-deps --no-index --find-links $SCRATCH/wheels supervision
```

> **Always `--no-deps`.** `supervision` would otherwise install its own numpy / opencv into
> `$SCRATCH/pyextra`, and because that directory is searched **before** the container's own
> packages, those copies would shadow the ones `torch` was compiled against and break torch.

## One thing to decide before quoting numbers

The default confidence threshold is `0.3`, which means detections scoring below 0.3 never
reach the metric — so the reported mAP50 is a **lower bound**, not a true mAP.

For the numbers that go in the thesis, run a second pass with a low threshold:

```bash
THRESHOLD=0.001 EXPERIMENT_NAME=E02_2_lowconf sbatch submit_sam3.sh
```

It writes to its own folder, so you can keep both and compare.

## Other troubleshooting

| Symptom | Fix |
|---|---|
| `SCRATCH: unbound variable` | you are not on the cluster, or the module environment is not loaded |
| Job hangs at "Loading SAM3" | `$SCRATCH/hf_cache` is empty and the node is offline — redo Step 4 |
| CUDA out of memory | `DTYPE=bfloat16`, or lower `TILE_SIZE` (e.g. 800), or `NUM_GPUS=2` |
| `exemplar strip is wider than the tile` | one of the exemplar crops is very large, so the tile gets downscaled harder than usual and small plants may be missed. Rare; note it if it appears often. |
| Everything reports `mAP50=0.000` | wrong class id or wrong labels — recheck Step 6b |

## Settings you might change

Set them before `sbatch`; they are forwarded into the job.

```bash
DTYPE=bfloat16          sbatch submit_sam3.sh     # faster
THRESHOLD=0.001         sbatch submit_sam3.sh     # proper mAP curve
NUM_GPUS=2              sbatch submit_sam3.sh     # fewer GPUs
SAVE_MASKS=1            sbatch submit_sam3.sh     # also store masks (large!)
DATASET_ROOT=/some/path sbatch submit_sam3.sh     # dataset elsewhere
```
