# HOW TO RUN — Experiment E02_1_b

**What E02_1_b is:** SAM3 prompted with **3 exemplar boxes** (the anchor box plus two other ground-truth boxes from the same image), **no tiling** — the whole drone photo is resized so its long side is 1024 px, the exemplar boxes are scaled into that space and passed straight through as prompts, and **one forward pass** produces every detection. No exemplar strip, no plausibility filter, no NMS, no global pass. Run over both `AGS_Multi_Rumex` and `AgsSpringRumex` pooled as one dataset. Confidence threshold **0.4**, mask threshold 0.5, TP-matching IoU 0.5.



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
└─ Step 11  resubmit if it hit the walltime
```

> If you have already set up any of the other experiments, Phase A is nearly free: Steps 2, 3 and 4 produce exactly the same dataset, token and `$SCRATCH/hf_cache` / `$SCRATCH/pyextra`. Copy the three new scripts (Step 1), then skip straight to Step 6.

---

## PHASE A — SETUP (once)

### Step 1 — Copy the files onto the cluster

Put the three scripts next to the ones you already have, in the same folder:

```
$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3/
├── infer_sam3.py             ← E02_2, untouched
├── infer_sam3_single.py      ← E01_2, untouched
├── infer_sam3_2.py           ← E02_2_b, untouched
├── infer_sam3_single_2.py    ← E01_2_b, untouched
├── infer_sam3_3.py           ← E02_2_c, untouched
├── infer_sam3_single_3.py    ← E01_2_c, untouched
├── (their run_*.sh / submit_*.sh)
├── infer_sam3_whole_2.py     ← NEW  (E02_1_b)
├── run_sam3_whole_2.sh       ← NEW
├── submit_sam3_whole_2.sh    ← NEW
└── HOW_TO_RUN_whole_2.md     ← this file
```

Then, on a **login node**:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
chmod +x run_sam3_whole_2.sh submit_sam3_whole_2.sh
ls -l
```

**You should see:** `run_sam3_whole_2.sh` and `submit_sam3_whole_2.sh` marked executable (`-rwxr-xr-x`).

---

### Step 2 — Put the dataset on `$SCRATCH`

*(Identical to the other experiments. If the dataset is already there, skip to Step 3.)*

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
./run_sam3_whole_2.sh download
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
Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1).
```

If it says `ERROR: HF_TOKEN is not set` → go back to Step 3. If it says `401` or `gated` → the licence was not accepted with that token's account.

> **`cv2` is no longer needed.** The notebook used OpenCV to upsample every predicted mask to 8192×5460 — and then discarded the result. That is removed here (see *"Notebook bugs fixed"*), so the container does not need `opencv`.

---

### Step 5 — Open an interactive session inside the container

Steps 6 and 7 must run **inside the container**, not on the login node — otherwise you are checking the login node's Python, which is not the one the job will use.

```bash
srun --account=go077 --time=00:30:00 \
     --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
     --environment=yolo26 --pty bash
```

**You should see:** a new shell prompt, running on a compute node.

---

### Step 6 — Check the container has what it needs

Inside the interactive shell from Step 5:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./run_sam3_whole_2.sh dryrun
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

**Do not continue until all three lines are correct.**

#### 6b. Check the configuration banner

```
 n_exemplars    : 3  (multiple prompt)
 prompting      : exemplar BOXES scaled into the resized image (no strip)
 tiling         : OFF (whole image resized to max_dim=1024, cache_resized=True)
 global pass    : n/a (this IS a whole-image pass)
 NMS            : OFF (single pass, no tile duplicates)
 thresholds     : conf=0.4  mask=0.5  match_iou=0.5
```

followed by:

```
 NOTE: conf=0.4. The tiled experiments (E01_2*/E02_2*) use 0.3, so a
       direct comparison against them measures the threshold too. ...
```

That NOTE is expected — it fires whenever the threshold is not 0.3, and it is there so nobody quotes a cross-experiment difference without noticing.

These lines **are** E02_1_b. If `tiling` says anything other than `OFF`, you are running the wrong script.

#### 6c. Check the dataset report

```
Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
  AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
  AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
Total images: NNN | this shard: NNN
```

Confirm `class_id` is **0** for `AGS_Multi_Rumex` and **2** for `AgsSpringRumex`.

The cost estimate looks different from the tiled experiments':

```
  sampled NNN image(s) of this shard -> NNNN anchor runs (...)
  forward passes per anchor run: 1 (no tiling)
  => ~NNNN SAM3 forward passes for those NNN images
  (for scale: the tiled experiments need 29-71 passes per anchor run)
```

One pass per anchor. That is the whole point — this run is roughly **30× cheaper** than a tiled one.

- `No images with labels found` → `$SCRATCH/overney/dataset` is wrong, or the folder names do not match. Go back to Step 2.
- `N image(s) have no matching label file` → those are skipped. A handful is normal; if it is *every* image, the label basenames do not match the image basenames.

---

### Step 7 — Smoke test on 2 images

Still inside the interactive shell. This is the first time the model actually loads. It writes to a throwaway folder so it cannot pollute the real E02_1_b results.

```bash
EXPERIMENT_NAME=E02_1_b_smoke \
OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke_whole_2 \
NUM_GPUS=1 \
./run_sam3_whole_2.sh run --limit-images 2
```

**You should see** the model load, then one line per anchor run:

```
Loading SAM3 from 'facebook/sam3' onto cuda (float32) ...
SAM3 ready. Parameters live on: cuda:0
    [E02_1_b_smoke] run #1 | AGS_Multi_Rumex/2022.../DJI_0001 | anchor=0 (1/12) | prompt=0+7+3 | sam_input=1024x683 | preds=2 | mAP50=0.088 | 1.4s
```



**Setup is done. You never have to repeat Phase A.**

---

## PHASE B — RUN

### Step 8 — Submit the job

On a **login node**:

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
sbatch submit_sam3_whole_2.sh
```

**You should see:** `Submitted batch job 1234567`

No arguments needed — E02_1_b is the default. The job asks for 1 node, 4 GPUs, 64 CPUs, and **6 hours** (not 24 — one forward pass per anchor makes this far cheaper than the tiled runs). Check the dryrun estimate; raise `--time` if your anchor count is much bigger than expected.

All the experiments can queue at the same time: different folders, different log files (`sam3_whole_2_%j.out` here).

---

### Step 9 — Watch it

```bash
squeue -u $USER                                       # queued / running?
tail -f sam3_whole_2_1234567.out                      # everything, live
tail -f $SCRATCH/experiments/sam3/E02_1_b/shard0.log  # just GPU 0
```

Count how many runs are finished so far:

```bash
cat $SCRATCH/experiments/sam3/E02_1_b/runs/results_E02_1_b_shard*.csv | grep -c E02_1_b
```

You will also get an email at BEGIN / END / FAIL.

---

### Step 10 — Collect the results

Everything lands in `$SCRATCH/experiments/sam3/E02_1_b/`:

**The three files you actually want:**

| File | What it is |
|---|---|
| `summary_E02_1_b_all.csv` | the headline numbers — mean and std of mAP50 / precision / recall / IoU1 / IoU2, averaged over images |
| `summary_E02_1_b_AGS_Multi_Rumex.csv` | same, for that archive only |
| `summary_E02_1_b_AgsSpringRumex.csv` | same, for that archive only |

```bash
cd $SCRATCH/experiments/sam3/E02_1_b
column -s, -t < summary_E02_1_b_all.csv
```

**Everything else in that folder:**

| File | What it is |
|---|---|
| `results_E02_1_b_all.csv` | one row per (image × anchor) — the raw data |
| `results_E02_1_b_<archive>.csv` | the same rows split by archive |
| `image_level_E02_1_b_*.csv` | one row per image (the prompt runs averaged together) |
| `raw_summary_E02_1_b_*.csv` | mean/std treating every run as an independent sample |
| `runs/results_E02_1_b_shard*.csv` | the per-GPU raw output; this is what resume reads |
| `runs/detections_E02_1_b_shard*.jsonl` | the predicted boxes + scores for every run |
| `runs/masks_E02_1_b_shard*.jsonl` | only with `SAVE_MASKS=1` |
| `run_config_E02_1_b.json` | every parameter used, for the thesis appendix |
| `shard0..3.log` | per-GPU logs |


### Step 11 — Resubmit if it hit the walltime

Completely normal, and harmless. Just run the same command again:

```bash
sbatch submit_sam3_whole_2.sh
```

Every finished `(image, anchor)` pair is already in the shard CSVs and is skipped **before any GPU work**. You will see this near the top of the log:

```
Resume: 4821 run(s) already recorded for E02_1_b; they will be skipped.
```

If a shard crashed but the others finished: the summaries are still built from whatever completed, and the job exits non-zero so you get the FAIL email. Just resubmit — it fills the gaps.

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

then change the one line at the bottom of `submit_sam3_whole_2.sh`:

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

The threshold here is **0.4**, so detections below 0.4 never reach the metric and the reported mAP50 is a **lower bound**, not a true mAP. For the thesis, run a low-threshold pass:

```bash
THRESHOLD=0.001 EXPERIMENT_NAME=E02_1_b_lowconf sbatch submit_sam3_whole_2.sh
```

It writes to its own folder. Do the same for anything you compare it against — a low-threshold run against a 0.4 run measures the threshold, not the pipeline.

---

### Other troubleshooting

| Symptom | Fix |
|---|---|
| `SCRATCH: unbound variable` | you are not on the cluster, or the module environment is not loaded |
| Job hangs at "Loading SAM3" | `$SCRATCH/hf_cache` is empty and the node is offline — redo Step 4 |
| `CUDA out of memory` | very unlikely here — the input is ~1024 px. If it happens, `DTYPE=bfloat16`, then `MAX_DIM=768`, then `NUM_GPUS=2` |
| `preds=0` on every run | check Step 6c (class ids / labels). Also expected to be *common* at conf 0.4 on downscaled 8192px imagery — worry only if it is literally every run |
| very low recall overall | expected for this configuration, see "What the resize actually costs". Compare against the tiled runs at a matched threshold before drawing conclusions |
| `prompt=0` instead of `prompt=0+7+3` | either `PRESET=single`, or that image has only one GT box (check `n_gt`) |
| merge with a tiled experiment drops rows | the two runs did not cover the same images — check `--archives` / `--limit-images` on both |
| results look mixed with another experiment | you overrode `EXPERIMENT_NAME` to a name already in use — see the warning at the top of this file |

---

### Settings you might change

Set them **before** `sbatch`; they are forwarded into the job.

```bash
THRESHOLD=0.3           sbatch submit_sam3_whole_2.sh   # match the tiled experiments
THRESHOLD=0.001         sbatch submit_sam3_whole_2.sh   # proper mAP curve
MAX_DIM=1536            sbatch submit_sam3_whole_2.sh   # bigger input (SAM3 still resizes to ~1008)
PRESET=single           sbatch submit_sam3_whole_2.sh   # 1 exemplar box instead of 3
DTYPE=bfloat16          sbatch submit_sam3_whole_2.sh   # faster
NMS_IOU=0.5             sbatch submit_sam3_whole_2.sh   # ablation only — off by design
CACHE_RESIZED=0         sbatch submit_sam3_whole_2.sh   # re-resize per anchor
NUM_GPUS=2              sbatch submit_sam3_whole_2.sh   # fewer GPUs
SAVE_MASKS=1            sbatch submit_sam3_whole_2.sh   # also store masks (RLE, resized res)
DATASET_ROOT=/some/path sbatch submit_sam3_whole_2.sh   # dataset elsewhere
```

Anything after the mode is forwarded straight to Python, so one-off flags work too:

```bash
./run_sam3_whole_2.sh run --limit-images 20 --max-anchors-per-image 3
./run_sam3_whole_2.sh run --archives AGS_Multi_Rumex        # one archive only
```