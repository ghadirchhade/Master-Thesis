# HOW TO RUN — Experiment `1_exemplars_tiling_extra_path_countGDpp`

What this experiment is: **CountGD++** prompted with **1 exemplar crop** (no text prompt), tiling
ON with **1536 px tiles and 384 px overlap**, PLUS the **"extra path"**: one downsampled
whole-image **global-context pass** whose detections are merged into the tiled detections before
NMS. Run over both `AGS_Multi_Rumex` and `AgsSpringRumex` pooled as one dataset. Every GT box of
every image is used as the exemplar (the "anchor") once.

It is the CountGD++ counterpart of the SAM3 experiment `1_exemplar_with_tiling_extra_path`.
Compared with `1_exemplars_tiling_countGDpp` it changes **two** things at once (tile size
1000 → 1536 / overlap 150 → 384, and the global pass), exactly like the SAM3 notebook. It is
`1_exemplars_text_tiling_extra_path_countGDpp` without the text prompt.

 it is a **two-phase** pipeline:

```
   PHASE 1  INFERENCE   (GPU)      CountGD++ runs ONCE per (image x anchor) at score 0.30,
                                   exemplar only, one tile per forward pass,
                                   35 tiles + 1 global-context pass per anchor
                                   -> pre-NMS detections saved as NPZ
   PHASE 2  EVALUATION  (no GPU)   NMS 0.40 -> metrics at confidence 0.30
                                   -> run / image / experiment / pooled-AP CSVs
                                   -> confusion matrices (CSV + PNG)
                                   -> qualitative GT-vs-prediction figures + exemplar previews
```

Phase 2 never touches CountGD++, so once Phase 1 is done you can rebuild every number in minutes.

```
   PHASE A — SETUP            do once, ~45 minutes (~10 if 1_exemplars_tiling_countGDpp is set up)
   ├─ Step 1   copy the files onto the cluster
   ├─ Step 2   put the dataset on $SCRATCH            (skip if done for SAM3)
   ├─ Step 3   download repo, checkpoint, BERT, wheels (skip if done for 1_exemplars_tiling)
   ├─ Step 4   open an interactive session inside the container
   ├─ Step 5   build: packages, BERT folder, CUDA op  (skip if done for 1_exemplars_tiling)
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

# The extra path: 1536 px tiles + the global-context pass

**Bigger tiles.** `TILE_SIZE = 1536`, `OVERLAP = 384`: every plant whose width **and** height are
≤ 384 px lies completely inside at least one tile. An 8192 × 5460 image gives **35 tiles**
(instead of 70 with 1000 / 150). CountGD++ resizes every tile to 800 × 800, so the tile scale is
800 / 1536 = 0.52 (instead of 0.8). With `match_tile`, the exemplar crop is resized by that same
factor, so a 256 × 163 crop is fed at 133 × 85.

**The global-context pass** (`ADD_GLOBAL_CONTEXT_PASS = 1`, notebook CELL 16–17). For every anchor,
one extra pass looks at the **whole image**:

```
   8192 x 5460 image  --(/ GLOBAL_DOWNSCALE = 2)-->  4096 x 2730  --(CountGD++ resize)-->  1200 x 800
```

It goes through exactly the same pipeline as a tile: exemplar, threshold 0.30, clipping,
plausibility filter (in global-image pixels, like the notebook). Its boxes are scaled back × 2 to
full resolution and **appended to the same pre-NMS pool as the tiles**. The offline NMS then
merges the tiled and global detections, so nothing downstream needs special-casing. Plants larger
than the 384 px overlap, which every tile cuts, are seen whole here.

- **Exemplar scale in the global pass.** The crop is resized by the **total** factor applied to
  the global image (½ × 1200/4096 ≈ 0.146), so the exemplar plant has the same apparent size as
  the plants in the global image (same idea as `match_tile`). A 256 × 163 crop is fed at about
  38 × 24 px. The exemplar preview figure shows both sizes (tiles and global pass).
- **How to recognise them.** Global-pass detections have `tile_id = -1` and `tile_boxes` = the
  whole image in the NPZ files. The manifest column `used_global_pass` is `True` when the global
  pass contributed at least one detection to that run (same definition as the notebook). A
  global-pass box that NMS removes, or that removes a tile box, is counted in
  `n_suppressed_cross_tile`.
- **Note on `GLOBAL_DOWNSCALE`.** CountGD++ shrinks every input to 1200 × 800 anyway (short side
  800, long side ≤ 1333), so the factor 2 changes almost nothing for CountGD++. It is kept
  because the notebook uses it.

**The RAM guard** (`MEM_STOP_THRESHOLD_PCT = 70`, notebook CELL 19). Before every image and every
anchor, each shard checks the system RAM (`psutil`). Above 70 % it **stops cleanly**: completed
runs are saved, the manifest is closed, and the shard exits with **code 3**. The job log then says
`shard N STOPPED by the RAM guard` and you get the FAIL email. Resubmit (Step 11) and it resumes
where it stopped. If `psutil` is missing the guard is disabled with a warning; inference still runs.

---
---

# Installations and libraries (read once)

> **If you already set up `1_exemplars_tiling_countGDpp`, everything in this section is already
> installed.** All CountGD++ experiments share the repo, the checkpoint, BERT, the wheels and
> `$SCRATCH/pyextra_countgdpp`.

CountGD++ is **not** a HuggingFace model like SAM3. It is a GitHub repository with its own code,
a checkpoint on Google Drive, a BERT text encoder, a CUDA extension to compile, and a few pinned
packages. Here is where every piece comes from:

| What | Needed for | Where it comes from | Where it lives |
|---|---|---|---|
| `torch`, `torchvision`, `numpy`, `pillow`, `scipy`, `opencv` | the model itself | the **`yolo26` container** (never re-installed) | inside the container |
| `pandas`, `matplotlib` | Phase 2 tables / PNGs | the `yolo26` container (Step 6 checks them) | inside the container |
| CountGD++ repository (`cfg_app.py`, `models/`, `util/`, `datasets/`) | model code | `git clone` (Step 3) | `$SCRATCH/CountGDPlusPlus` |
| `countgd_plusplus.pth` (1.25 GB) | model weights | Google Drive link of the README, via `gdown` (Step 3) | `$SCRATCH/CountGDPlusPlus/checkpoints/` |
| `bert-base-uncased` | CountGD++'s text encoder (used even with an empty text prompt) | HuggingFace, public, **no token** (Step 3), written by the repo's `download_bert.py` (Step 5) | `$SCRATCH/hf_cache` → `$SCRATCH/CountGDPlusPlus/checkpoints/bert-base-uncased` |
| `transformers<5` (+ `huggingface_hub<1.0`, `tokenizers`, `safetensors`, `regex`) | CountGD++'s BERT wrapper calls `BertModel.get_extended_attention_mask`, **removed in transformers 5** | wheels (Step 3) → installed in the container (Step 5) | `$SCRATCH/pyextra_countgdpp` |
| `addict`, `yapf==0.40.1` (+ `platformdirs`, `tomli`, `importlib_metadata`, `zipp`) | the repo's config loader `SLConfig`; it **breaks with newer yapf** | same | `$SCRATCH/pyextra_countgdpp` |
| `timm`, `pycocotools`, `termcolor` | imported by the repo | same | `$SCRATCH/pyextra_countgdpp` |
| `supervision` (+ `defusedxml`) | AP50 / AP50:95 | same | `$SCRATCH/pyextra_countgdpp` |
| `MultiScaleDeformableAttention` | deformable attention, compiled CUDA op (faster) | compiled from the repo for sm_90 (Step 5) | `$SCRATCH/CountGDPlusPlus/models/GroundingDINO/ops/build/` |
| `gdown`, `huggingface_hub` (login-node copies) | only to download in Step 3 | `pip install` on the login node | `$SCRATCH/pytools_countgdpp` (never used by the job) |

These are the same packages the notebook installed in its CELL 1
(`"transformers<5" addict yapf==0.40.1 timm pycocotools termcolor supervision gdown`). The others
in the table are the dependencies those need at import time.

**Four rules that make this work:**

1. **Its own package folder.** CountGD++ needs `transformers<5`; SAM3 needs a recent
   `transformers`. If they shared `$SCRATCH/pyextra`, one of them would break, so CountGD++ gets
   `$SCRATCH/pyextra_countgdpp`. The run script puts it at the front of `PYTHONPATH`, only for
   CountGD++ jobs.
2. **Always `--no-deps`.** Packages are installed one by one from a fixed list. Otherwise pip
   would also install its own `torch` / `numpy` / `pillow` into the folder, and because that
   folder is searched **before** the container's packages, those copies would shadow the ones
   the container's `torch` was built against and break it.
3. **Download on the login node, install inside the container.** Compute nodes have no internet.
   The container's Python version can differ from the login node's, so Step 3 only *downloads*
   the wheels (for Python 3.10–3.13, `aarch64`, because a GH200 is an ARM machine). Step 5
   *installs* them with the container's own `pip`, which picks the matching ones.
4. **The repo's `torch<2.6` pin is ignored**, exactly as in the notebook. The container's torch is
   kept, and the one incompatibility is handled in the code: `torch.load(..., weights_only=False)`
   (since torch 2.6 the default refuses this checkpoint because it stores an `argparse.Namespace`).

## Why batch size 1

CountGD++ cannot run several tiles in one forward pass. Its `forward()` inserts the exemplar
token by calling `add_exemplar_tokens(..., [tensor([0])])`, which handles **sample 0 only**: with
4 tiles the text features shrink to batch 1 while the image features stay at batch 4. So
`BATCH_SIZE` is fixed at **1**, exactly as in the notebook, and the script refuses any other
value. To use the 4 GPUs anyway, the images are split across 4 shard processes (one per GPU), as
in the SAM3 job.

## Deformable attention: compiled op or fallback

CountGD++ uses multi-scale deformable attention. The repo ships two implementations:

- the **compiled CUDA op** (`MultiScaleDeformableAttention`), which is fast;
- a **pure-PyTorch** version (`multi_scale_deformable_attn_pytorch`), which does the same maths
  but is slower.

---
---

# PHASE A — SETUP (once)

## Step 1 — Copy the files onto the cluster

Put the three scripts in the same folder as the scripts of `1_exemplars_tiling_countGDpp`
(a sibling of the `sam3/` folder); the names do not clash:

```
$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp/
├── 1_exemplars_tiling_*                        (exemplar only, if present)
├── 1_exemplars_text_tiling_*                   (exemplar + text, if present)
├── 1_exemplars_text_tiling_extra_path_*        (exemplar + text + extra path, if present)
├── 1_exemplars_tiling_extra_path_infer_countGDpp.py
├── 1_exemplars_tiling_extra_path_run_countGDpp.sh
└── 1_exemplars_tiling_extra_path_submit_countGDpp.sh
```

Then, **on a login node**:

```bash
mkdir -p $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp
chmod +x 1_exemplars_tiling_extra_path_run_countGDpp.sh 1_exemplars_tiling_extra_path_submit_countGDpp.sh
ls -l
```

**You should see:** the three files, with the two `.sh` marked executable (`-rwxr-xr-x`).

> If the files were edited on Windows, make sure the `.sh` files have Unix line endings,
> otherwise bash fails with `$'\r': command not found`:
> `sed -i 's/\r$//' 1_exemplars_tiling_extra_path_run_countGDpp.sh 1_exemplars_tiling_extra_path_submit_countGDpp.sh`

---

## Step 2 — Put the dataset on `$SCRATCH`

> **If you already did this for the SAM3 experiments, skip this step.** The same
> `$SCRATCH/overney/dataset` is used.

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
│   │   └── 20230426_Wallenwil/       (one folder per flight)
│   │       └── DJI_..._0035.JPG      (8192 × 5460)
│   └── annotations_yolo/             (FLAT: DJI_..._0035.txt, ... + darknet.labels)
└── AgsSpringRumex/
    ├── images/
    └── annotations_yolo/             (FLAT)
```

The folder names are case-sensitive. `AGS_Multi_Rumex` uses class id **0** and
`AgsSpringRumex` class id **2** in their YOLO files; the code knows this.

---

## Step 3 — Download the repo, the checkpoint, BERT and the wheels

> **If you already ran `download` for `1_exemplars_tiling_countGDpp`, skip this step.** It fetches
> exactly the same files into the same places (`$SCRATCH/CountGDPlusPlus`, `$SCRATCH/hf_cache`,
> `$SCRATCH/wheels_countgdpp`). Running it again is harmless: finished parts are skipped.

**On a login node** (this is the only step that needs internet):

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp
./1_exemplars_tiling_extra_path_run_countGDpp.sh download
```

No HuggingFace token is needed: `bert-base-uncased` is public, and the checkpoint comes from
Google Drive. This does five things:

1. `git clone` of <https://github.com/niki-amini-naieni/CountGDPlusPlus> into
   `$SCRATCH/CountGDPlusPlus` (the commit hash is printed and also saved in `run_config_*.json`)
2. installs `gdown` + `huggingface_hub` into `$SCRATCH/pytools_countgdpp` (login node only)
3. downloads `countgd_plusplus.pth` (1.25 GB) into `$SCRATCH/CountGDPlusPlus/checkpoints/`
4. downloads `bert-base-uncased` (config, tokenizer, `model.safetensors`) into `$SCRATCH/hf_cache`
5. downloads the wheels of the extra packages into `$SCRATCH/wheels_countgdpp`, for Python
   3.10 / 3.11 / 3.12 / 3.13 on `aarch64`

**You should see:**

```
--- 1/5 : CountGD++ repository -> /scratch/.../CountGDPlusPlus ---
commit: 1a2b3c...
--- 2/5 : login-node tools (gdown, huggingface_hub) -> /scratch/.../pytools_countgdpp ---
--- 3/5 : CountGD++ checkpoint -> /scratch/.../CountGDPlusPlus/checkpoints/countgd_plusplus.pth ---
-rw-r--r-- 1 you  ...  1.2G ... countgd_plusplus.pth
--- 4/5 : bert-base-uncased (text encoder) -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--bert-base-uncased/snapshots/...
--- 5/5 : wheels of the extra packages -> /scratch/.../wheels_countgdpp ---
  python 3.10 / aarch64
  ...
transformers-4.xx.x-py3-none-any.whl
tokenizers-0.22.x-cp39-abi3-manylinux_2_17_aarch64...whl
...
Done. Compute nodes can now run offline.
```

A few `(warning: no wheel for '...' on python 3.10)` lines are fine. What matters is that
the container's Python version gets every package, which Step 5 checks.

**If the checkpoint download fails** (Google Drive sometimes answers "Too many users have viewed
or downloaded this file recently") → see **Problem C**.

---

## Step 4 — Open an interactive session inside the container

Steps 5, 6 and 7 must run **inside the container on a GPU node**, not on the login node. The
packages have to match the container's Python, and the CUDA op can only be compiled where a GPU
is visible.

```bash
srun --account=go077 --time=01:00:00 \
     --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
     --environment=yolo26 --pty bash
```

**You should see:** a new shell prompt, running on a compute node. Everything in Steps 5–7
happens in this shell.

---

## Step 5 — Build: packages, BERT folder, CUDA op

> **If you already ran `build` for `1_exemplars_tiling_countGDpp` (with the same `yolo26`
> container), skip this step** and go to Step 6. The packages, the BERT folder and the compiled op
> are shared.

**Inside the interactive shell from Step 4:**

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp
./1_exemplars_tiling_extra_path_run_countGDpp.sh build
```

This does three things:

1. installs the wheels from `$SCRATCH/wheels_countgdpp` into `$SCRATCH/pyextra_countgdpp`
   (`--no-deps --no-index`), then checks they import with the **container's** Python
2. runs the repo's `download_bert.py` **offline** (from the cache of Step 3), which writes
   `$SCRATCH/CountGDPlusPlus/checkpoints/bert-base-uncased`, exactly as the notebook did
3. applies the `value.type()` → `value.scalar_type()` fix and compiles
   `MultiScaleDeformableAttention` for this GPU (a few minutes)

**You should see:**

```
--- 1/3 : extra packages -> /scratch/.../pyextra_countgdpp ---
Successfully installed addict-... supervision-... timm-... transformers-4.xx.x yapf-0.40.1 ...
transformers 4.xx.x | timm 1.x.x | supervision 0.xx.x -> OK
--- 2/3 : BERT text encoder -> /scratch/.../CountGDPlusPlus/checkpoints/bert-base-uncased ---
config.json  model.safetensors  special_tokens_map.json  tokenizer.json  tokenizer_config.json  vocab.txt
--- 3/3 : MultiScaleDeformableAttention CUDA op ---
55:    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda", ([&] {
105:    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda", ([&] {
Compiled for sm_90 (log: /scratch/.../ops/build_countgdpp.log)
/scratch/.../ops/build/lib.linux-aarch64-cpython-3xx/MultiScaleDeformableAttention.cpython-3xx-....so
```

| If you see | Meaning | Fix |
|---|---|---|
| `ERROR: Could not find a version that satisfies ...` in 1/3 | no wheel for the container's Python | see **Problem B** |
| `AssertionError` / `ImportError` in the check of 1/3 | version clash between `transformers` and a container package | see **Problem A** |
| `COMPILATION FAILED -> ... pure-PyTorch fallback` | the op could not be built | **not fatal**; see **Problem D** if you want the speed |
| `No GPU visible -> cannot compile` | the session has no GPU | add `--gpus-per-task=1` to Step 4 |

---

## Step 6 — Check the container has what it needs

**Still inside the interactive shell:**

```bash
./1_exemplars_tiling_extra_path_run_countGDpp.sh dryrun
```

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

### 6a. Look for these lines in the debug block

```
torch: 2.x.x | cuda available: True | device count: 1
transformers: 4.xx.x /scratch/.../pyextra_countgdpp/transformers/__init__.py
transformers<5 for CountGD++: OK
timm: 1.x.x -> OK
addict: ... -> OK
yapf: 0.40.1 -> OK
pycocotools: ... -> OK
termcolor: ... -> OK
scipy: ... -> OK
cv2: ... -> OK
psutil: 5.x.x -> RAM guard: OK (system RAM in use now: NN%)
supervision: 0.xx.x -> MeanAveragePrecision import: OK
pandas: 2.x.x -> PHASE 2 import: OK
matplotlib: 3.x.x -> confusion-matrix + qualitative PNGs: OK
CountGD++ repo: /scratch/.../CountGDPlusPlus -> OK
checkpoint: /scratch/.../countgd_plusplus.pth (1.25 GB) -> OK
BERT folder: /scratch/.../bert-base-uncased -> OK
deformable attention: compiled CUDA op -> OK        (or: ... pure-PyTorch fallback)
```

Check that `transformers` is loaded **from `pyextra_countgdpp`** (the path is printed) and is a
4.x version.

| If you see | Meaning | Fix |
|---|---|---|
| `transformers<5 for CountGD++: FAILED` | the container's transformers 5 is used instead of ours | `$SCRATCH/pyextra_countgdpp` missing or not on `PYTHONPATH`; rerun Step 5 |
| any `import FAILED` for timm / addict / yapf / pycocotools / termcolor / supervision | the package is missing for this Python | rerun Step 5; if it still fails, **Problem B** |
| `scipy` or `cv2 import FAILED` | the container lacks them (unexpected for `yolo26`) | **Problem B** |
| `pandas import FAILED` | Phase 1 still works, but Phase 2 cannot run | **Problem B** |
| `matplotlib import FAILED` | only the PNGs are lost, every CSV is still written | **Problem B**, or ignore it |
| `... -> MISSING (run 'download')` | Step 3 incomplete | rerun Step 3 |
| `BERT folder ... MISSING (run 'build')` | Step 5 part 2 did not run | rerun Step 5 |
| `cuda available: False` | no GPU in this allocation | add `--gpus-per-task=1` to Step 4 |

**Do not continue until everything except the deformable-attention line says OK.**

### 6b. Check the dataset report

```
Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
  AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
  AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
Total images: NNN | this shard: NNN

--- DRY RUN: counting the work without loading CountGD++ ---
  sampled NNN image(s) of this shard -> NNNN anchor runs ({...})
  tiles per anchor run at 8192x5460: 35 + 1 global-context pass
  => ~NNNNN CountGD++ forward passes (1 image each) for those NNN images
  NPZ files that will be written by this shard: ~NNNN (one per image x anchor)
```

Confirm the image counts, and that `class_id` is **0** for `AGS_Multi_Rumex` and **2** for
`AgsSpringRumex`.

**If it says `No images with labels found`**: `$SCRATCH/overney/dataset` is wrong; go back to Step 2.

---

## Step 7 — Smoke test on 2 images

**Still inside the interactive shell.** This is the first time the model actually loads. It
writes to a throwaway folder so it cannot pollute the real results.

```bash
EXPERIMENT_NAME=1_exemplars_tiling_extra_path_countGDpp_smoke \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/_smoke_1_exemplars_tiling_extra_path \
NUM_GPUS=1 \
./1_exemplars_tiling_extra_path_run_countGDpp.sh run --limit-images 2
```

**You should see** the model load, then one line per anchor run, then the whole Phase 2:

```
final text_encoder_type: checkpoints/bert-base-uncased
load tokenizer done.
Loading CountGD++ from '/scratch/.../countgd_plusplus.pth' onto cuda (float32) ...
CountGD++ ready.
  model device    : cuda:0
  deformable attn : compiled CUDA op
  fp16 autocast   : False
  missing keys    : 0 | unexpected keys: 38
  [1_exemplars_tiling_extra_path_countGDpp_smoke] shard0 run #1 | AGS_Multi_Rumex/.../DJI_... | anchor=0 (1/12) | prompt=0 | tiles=35 | global_pass=True | pre-NMS detections=41 | 7.1s
...
===== PHASE 2 : OFFLINE EVALUATION =====
--- run-level metrics ---
  all_gt   : NN runs, NN valid for macro averaging, F1_mean=0.xxxx, count_abs_error_mean=N.NN
  held_out : NN runs, NN valid for macro averaging, F1_mean=0.xxxx, count_abs_error_mean=N.NN
  exemplar plant re-detected in NN.N% of the runs (all_gt matching at conf=0.30)
```

**Things to check in this output:**

- **`missing keys : 0 | unexpected keys: 38`**: the same numbers as the notebook. Any other
  number means a different repo version or checkpoint.
- Messages like `Some weights of BertModel were not initialized ... pooler` and a `timm`
  `FutureWarning` are **normal**; the notebook shows them too.
- **`tiles=35 | global_pass=True`**: 1536 px tiles, and the global pass found something.
  `global_pass=False` on some runs is normal; it only means the global pass found nothing there.
- **`7.1s`** (or whatever you get) is the time for ONE anchor run (35 tiles + 1 global pass).
  Multiply it by the total anchor count from Step 6b and divide by 4 GPUs to estimate the job.
  With 36 forward passes per run instead of 70, expect about half the time of
  `1_exemplars_tiling_countGDpp`.
- The per-image lines end with `[MEM] RSS=… GB` (memory of that shard), as in the notebook.
- **`pre-NMS detections=`** should not be 0 on every run.
- **`F1_mean=`** should be a plausible number in `all_gt`, not `0.0000` everywhere.

Then leave the interactive session:

```bash
exit
```

Setup is done. You never have to repeat Phase A, unless the `yolo26` container changes
(then rerun Step 5).

---
---

# PHASE B — RUN

## Step 8 — Submit the job

**On a login node:**

```bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp
sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh
```

**You should see:** `Submitted batch job 1234567`

No arguments needed. The job asks for 1 node, 4 GPUs, 64 CPUs and 24 hours. It splits the images
across the 4 GPUs for Phase 1 (one tile per forward pass on each GPU), then runs Phase 2 once.

---

## Step 9 — Watch it

```bash
squeue -u $USER                                                            # queued / running?
tail -f 1_exemplars_tiling_extra_path_countGDpp_1234567.out                           # everything, live
tail -f $SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp/shard0.log   # just GPU 0
```

(The `.out` / `.err` files are written in the folder you ran `sbatch` from.)

Count how many runs are finished so far:

```bash
cat $SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp/raw_detections/runs_manifest_1_exemplars_tiling_extra_path_countGDpp_shard*.csv \
  | grep -c 1_exemplars_tiling_extra_path_countGDpp
ls $SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp/raw_detections/*.npz | wc -l
```

You will also get an email at BEGIN / END / FAIL.

---

## Step 10 — Collect the results

Everything lands in `$SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp/`:

```bash
ls -R $SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp/ | head -40
```

**The file you actually quote in the thesis:**

| File | What it is |
|---|---|
| `metrics/experiment_summary.csv` | **the headline numbers**: one row per evaluation mode (`all_gt`, `held_out`), with `tile_size`, `overlap`, `add_global_context_pass` and the mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU1 / IoU2 / **count_abs_error** |

The mean is taken over the **image-level** values, so an image with 40 plants does not outweigh
one with 2. The std is the variation **between UAV images**.

**Everything else in that folder:**

| File | What it is |
|---|---|
| `metrics/experiment_summary_per_archive.csv` | the same table split by archive |
| `metrics/run_level_metrics.csv` | one row per (image × anchor × mode), the raw data, incl. TP/FP/FN, `count_abs_error`, `exemplar_redetected` and the NMS provenance counters (`n_suppressed_cross_tile`) |
| `metrics/image_level_metrics.csv` | one row per (image × mode); the std here is the spread between the different anchors of the SAME image |
| `metrics/dataset_ap_metrics.csv` | pooled AP50 / AP50:95, all runs ranked in ONE precision-recall curve. **Not** the mean of the image-level AP |
| `confusion_matrices/confusion_matrix_{all_gt,held_out}.csv` + `.png` | pooled TP / FP / FN, plus micro precision/recall/F1 in `confusion_matrix_summary.csv` |
| `plots/best_image_*.png` | the qualitative GT (left) vs predictions (right) figures: one per archive + one global |
| `plots/exemplar_preview_*.png` | notebook CELL 18 for the same image and anchor: indexed GT boxes, the exemplar crop, and the crop exactly as fed to CountGD++ for the tiles **and** for the global pass |
| `raw_detections/*.npz` | the PRE-NMS detections of every run (boxes, scores, tile ids, GT boxes, prompt indices), tiles **and** global pass (`tile_id = -1`). This is what Phase 2 reads |
| `raw_detections/runs_manifest_*_shard*.csv` | which runs are finished (with `used_global_pass`); this is what resume reads |
| `run_config_1_exemplars_tiling_extra_path_countGDpp.json` | every parameter used + the CountGD++ repo commit, for the thesis appendix |
| `shard0..3.log` | per-GPU logs |

**The two notebook-only columns:**

- `count_abs_error` = |evaluated predictions − evaluated GT| at the operating point (0.30, after
  NMS). In `held_out` the ignored predictions are not counted. Like every other metric, it is NaN
  on runs with no evaluable GT (`valid_for_macro = False`).
- `exemplar_redetected` = was the anchor plant itself found (all_gt matching at 0.30)? The same
  value appears in both mode rows of a run. The log prints the overall rate.

**Comparing with the other CountGD++ experiments.** Same images, anchors, exemplar crops and
thresholds as `1_exemplars_tiling_countGDpp`. The difference between the two is the effect of
the **extra path** (1536 / 384 tiles + the global pass, taken together, as in the SAM3
notebook). `1_exemplars_text_tiling_extra_path_countGDpp` has the same extra path plus the text:

```bash
cd $SCRATCH/experiments/countgdpp
column -s, -t < 1_exemplars_tiling_countGDpp/metrics/experiment_summary.csv                 | less -S   # exemplar only
column -s, -t < 1_exemplars_tiling_extra_path_countGDpp/metrics/experiment_summary.csv                 | less -S   # + extra path
column -s, -t < 1_exemplars_text_tiling_extra_path_countGDpp/metrics/experiment_summary.csv | less -S   # + extra path + text
```

The run-level CSVs of all of them have the same `(image_ID, anchor_idx)` rows, so they can also
be compared run by run. To separate the two parts of the extra path, run it once more with
`ADD_GLOBAL_CONTEXT_PASS=0` (tiles 1536 only; see "Settings you might change").

How much did the global pass contribute? Count its detections in the raw files:

```bash
cd $SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp
python -c "import glob,numpy as np; f=glob.glob('raw_detections/*.npz'); t=sum(len(np.load(p)['tile_id']) for p in f); g=sum(int((np.load(p)['tile_id']==-1).sum()) for p in f); print(f'{g} of {t} pre-NMS detections ({100*g/max(t,1):.1f}%) come from the global pass')"
```

**Which mode to quote:** both, and say what they mean.
`all_gt` = classical evaluation, every GT box counts.
`held_out` = the plant shown to CountGD++ as the exemplar is removed from the GT and the
predictions that land on it are ignored. It answers "after being shown one example, how well does
it find the **remaining** plants?".

---

## Step 11 — Resubmit if it hit the 24 h limit (or the RAM guard stopped it)

Completely normal, and harmless. If the log says `shard N STOPPED by the RAM guard`, the same
applies. Run the same command again:

```bash
sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh
```

Every finished (image, anchor) pair is listed in the shard manifests and is skipped **before**
any GPU work, so the job picks up exactly where it stopped. You will see this near the top of
each shard log:

```
Resuming: 4821 run(s) already finished for 1_exemplars_tiling_extra_path_countGDpp; skipped.
```

Each shard reads **all** the manifests, so resuming still works if you change `NUM_GPUS`.

**If a shard crashed but the others finished:** Phase 2 still runs on whatever NPZ files reached
disk, and the job exits non-zero so you get the FAIL email. Resubmit and it fills the gaps.

---

## Step 12 — Re-run only the evaluation (optional, cheap)

Phase 2 reads only the NPZ files, so it never needs a GPU or CountGD++:

```bash
# on a login node, or in any small allocation
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp
./1_exemplars_tiling_extra_path_run_countGDpp.sh evaluate
```

Use this to change the operating point without re-running CountGD++:

```bash
OPERATING_CONFIDENCE=0.40 NMS_IOU_THRESHOLD=0.50 \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp \
./1_exemplars_tiling_extra_path_run_countGDpp.sh evaluate
```

> **Careful:** this overwrites the CSVs in `metrics/`, `confusion_matrices/` and `plots/`. To
> keep both, copy the folder first, or point `OUTPUT_DIR` at a copy of the experiment.

On a login node Phase 2 needs `supervision` and `pandas` from the login node's Python. If they are
missing there, run `evaluate` inside the container instead (Step 4 session, no GPU needed).

`evaluate` needs `DATASET_ROOT` to be reachable, because the figures reopen the selected images.
If the dataset has been purged from `$SCRATCH`, add `--no-plots`:

```bash
./1_exemplars_tiling_extra_path_run_countGDpp.sh evaluate --no-plots
```

---
---

# Notes and problems

## If it will not fit in 24 hours

Batch size is fixed at 1, so the levers are:

1. **Do nothing.** Resubmit as many times as needed (Step 11). Three 24 h jobs = one 72 h job.
2. **Make sure the compiled op is used.** The pure-PyTorch fallback is the slowest part; check
   the `deformable attn :` line in `shard0.log`, and see **Problem D**.
3. **`--max-anchors-per-image 5`**: use only the first 5 GT boxes per image as anchors. This
   changes what you are measuring, so mention it in the thesis. Pass it after the mode:
   `./1_exemplars_tiling_extra_path_run_countGDpp.sh run --max-anchors-per-image 5` (or edit the last line
   of the submit script).
4. **More GPUs**: needs a job array; ask and I will write one.

## Problem A — `transformers` imports but the check fails / version clash

`transformers` 4.x checks the versions of `huggingface_hub`, `tokenizers`, `safetensors`,
`regex`, `pyyaml`, `requests`, `packaging`, `filelock`, `tqdm` and `numpy` when it is imported.
The first five come from `$SCRATCH/pyextra_countgdpp`; the rest come from the container. The
error message names the package and the required range, e.g.:

```
ImportError: tokenizers>=0.22.0,<=0.23.0 is required for a normal functioning of this module, but found tokenizers==0.21.0.
```

Fix: download a wheel in that range on the login node and install it in the container
(**Problem B** commands). Alternatively, build a dedicated image:

```dockerfile
# Dockerfile
FROM <whatever image ~/.edf/yolo26.toml points at>
RUN pip install --no-cache-dir "transformers<5" addict yapf==0.40.1 timm pycocotools \
        termcolor supervision pandas matplotlib
```

```bash
podman build -t countgdpp .
enroot import -o $SCRATCH/images/countgdpp.sqsh podman://countgdpp:latest
cp ~/.edf/yolo26.toml ~/.edf/countgdpp.toml
# edit ~/.edf/countgdpp.toml so `image = ` points at $SCRATCH/images/countgdpp.sqsh
```

then use `--environment=countgdpp` in Step 4 and on the `srun` line of the submit script, and
rerun Step 5 (the CUDA op must be compiled against that image's torch).

## Problem B — a package import FAILED inside the container

Step 3 downloads wheels for Python 3.10–3.13 on `aarch64`. If the container uses another version,
or a package has no wheel for it, get it explicitly. First read the container's Python version
from the debug block (`python --version`), e.g. `3.12`. Then, **on the login node**:

```bash
pip download --no-deps --only-binary=:all: --implementation cp --python-version 3.12 \
    --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 \
    -d $SCRATCH/wheels_countgdpp <package>
```

and **inside the container** (Step 4 shell):

```bash
pip install --target $SCRATCH/pyextra_countgdpp --no-deps --no-index \
    --find-links $SCRATCH/wheels_countgdpp <package>
```

For `pandas` or `matplotlib` (Phase 2), the packages are
`pandas pytz tzdata python-dateutil six` and
`matplotlib contourpy cycler fonttools kiwisolver pyparsing packaging`.

If the container's architecture is not `aarch64`, redo Step 3 with
`CONTAINER_ARCH=x86_64 ./1_exemplars_tiling_extra_path_run_countGDpp.sh download`.

> **Always `--no-deps`**, for the reason in rule 2 at the top.

## Problem C — the checkpoint download fails

Google Drive throttles popular files. Either retry later, or download `countgd_plusplus.pth` in
your browser from the README link (or copy the one cached in your Google Drive by the notebook,
`master_thesis/checkpoints/countgd_plusplus.pth`) and copy it over:

```bash
scp countgd_plusplus.pth <you>@<cluster>:$SCRATCH/CountGDPlusPlus/checkpoints/
```

The file is 1.25 GB; Step 3 treats anything above 1 GB as already downloaded and skips it.

## Problem D — the CUDA op does not compile

Not fatal: the pure-PyTorch fallback gives the same results, only slower. To get the speed, read
`$SCRATCH/CountGDPlusPlus/models/GroundingDINO/ops/build_countgdpp.log`. Common causes:

| In the log | Meaning |
|---|---|
| `CUDA_HOME` / `nvcc: not found` | the container has no CUDA toolkit; use an image with `nvcc` (e.g. an NGC PyTorch image, **Problem A** Dockerfile) |
| `no suitable conversion function from "const at::DeprecatedTypeProperties"` | another line still uses `.type()`; replace it with `.scalar_type()` like Step 5 did |
| `undefined symbol` when importing (debug block) | the `.so` was compiled against another torch; delete `ops/build/` and rerun Step 5 |

After fixing, `rm -rf $SCRATCH/CountGDPlusPlus/models/GroundingDINO/ops/build` and rerun Step 5.

## What the fixed operating point means for your numbers

`OPERATING_CONFIDENCE = 0.30` and `NMS_IOU_THRESHOLD = 0.40` are fixed from the start, exactly
as in the notebook. No confidence × NMS sweep is performed, so nothing is tuned on the test data.

The confidence 0.30 is used **inside** CountGD++'s post-processing (instead of the repo default
0.23) **and** as the operating point:

- **precision / recall / F1 / IoU1 / IoU2 / count_abs_error** describe ONE operating point: only
  detections scoring ≥ 0.30.
- **AP50 / AP50:95** use **every** post-NMS detection ≥ 0.30, because AP is the area under the
  precision-recall curve.

To get a lower floor for the AP curve, re-run Phase 1 with a lower `THRESHOLD`, since
detections below it were never saved:

```bash
THRESHOLD=0.05 EXPERIMENT_NAME=1_exemplars_tiling_extra_path_countGDpp_lowconf \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp_lowconf \
sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh
```

## Other troubleshooting

| Symptom | Fix |
|---|---|
| `SCRATCH: unbound variable` | you are not on the cluster |
| `ERROR: no CountGD++ repository at ...` / `checkpoint not found` | Step 3 incomplete |
| `ERROR: .../bert-base-uncased is missing` | Step 5 part 2 did not run |
| `OSError: We couldn't connect to 'https://huggingface.co'` in Step 5 | the BERT cache of Step 3 is missing or `HF_HOME` differs; rerun Step 3 with the same `HF_HOME` |
| `NameError: name '_C' is not defined` | should never happen (the fallback catches it); report the shard log |
| `--batch-size must be 1` | expected; CountGD++ cannot batch tiles (see "Why batch size 1") |
| `shard N STOPPED by the RAM guard (> 70% RAM)` / `RAM at NN% ... stopping cleanly` | system RAM went above `MEM_STOP_THRESHOLD_PCT`; completed runs are saved. Resubmit (Step 11). If it keeps happening: `--no-cache-tiles` (see below) or `NUM_GPUS=2` |
| `WARNING: psutil unavailable -> the RAM guard is DISABLED` | the container has no `psutil` (unexpected for `yolo26`); inference still runs without the guard, or install it per **Problem B** |
| CUDA out of memory | unlikely in fp32 at batch 1 on a GH200; lower `TILE_SIZE` or `NUM_GPUS=2` |
| Host RAM fills up | `./1_exemplars_tiling_extra_path_run_countGDpp.sh run --no-cache-tiles` (tiles cropped on demand) |
| `Nothing to plot` in Phase 2 | no image had a valid AP50, or `DATASET_ROOT` is unreachable. Every CSV is still written |
| Everything reports `F1=0.0000` | wrong class id or wrong labels; recheck Step 6b |
| `held_out` is all NaN | every plant of every image was used as a prompt; expected only on images with 1 GT box |
| Phase 2 is slow / heavy | it holds every run in memory at once (needed for the pooled AP). Give it a node with more RAM if it gets killed |

## Settings you might change

Set them before `sbatch`; they are forwarded into the job.

```bash
DTYPE=float16                     sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # the notebook's USE_FP16=True (not validated)
THRESHOLD=0.05                    sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # save more detections for the AP curve
OPERATING_CONFIDENCE=0.40         sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # different operating point
NMS_IOU_THRESHOLD=0.50            sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # different NMS
EXEMPLAR_SCALE_MODE=model_default sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # standard 800 px resize of the crop
TEXT_PROMPT="rumex"               sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # exemplar + text prompt (needs its own EXPERIMENT_NAME + OUTPUT_DIR)
USE_TILING=0                      sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # whole image as one tile
NUM_GPUS=2                        sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # fewer GPUs
DATASET_ROOT=/some/path           sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # dataset elsewhere
MEM_STOP_THRESHOLD_PCT=85         sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh   # RAM guard threshold (notebook: 70)
```

These change what is stored in the NPZ files, so give each its **own** `EXPERIMENT_NAME` and
`OUTPUT_DIR` (otherwise new runs would be mixed with the old ones on resume):

```bash
# the extra path WITHOUT the global pass (1536 / 384 tiles only) -> separates its two parts
ADD_GLOBAL_CONTEXT_PASS=0 \
EXPERIMENT_NAME=1_exemplars_tiling_extra_path_countGDpp_noglobal \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp_noglobal \
sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh

# another global-pass downscale (almost no effect for CountGD++, see "The extra path")
GLOBAL_DOWNSCALE=4 \
EXPERIMENT_NAME=1_exemplars_tiling_extra_path_countGDpp_gd4 \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/1_exemplars_tiling_extra_path_countGDpp_gd4 \
sbatch 1_exemplars_tiling_extra_path_submit_countGDpp.sh
```

