# HOW TO RUN — Experiment `text_only_tiling_countGDpp`

What this experiment is: **CountGD++** prompted with **text only**: the caption
`" rumex obtusifolius . "` (the same text as `1_exemplars_text_tiling_countGDpp`), **no exemplar**,
tiling ON (1000 px tiles, 150 px overlap), run over both `AGS_Multi_Rumex` and `AgsSpringRumex`
pooled as one dataset. There is no visual prompt, so there are no anchors and no prompt plants:
**one run per image**, evaluated against **all** GT boxes (`all_gt`; `held_out` does not exist).

It is the CountGD++ counterpart of the SAM3 experiment `text_only_with_tiling`: the baseline that
shows what the text alone achieves, without any exemplar.

 it is a **two-phase** pipeline:

```
   PHASE 1  INFERENCE   (GPU)      CountGD++ runs ONCE per image x tile at score 0.30,
                                   text prompt only, one tile per forward pass
                                   -> pre-NMS detections saved as NPZ (one file per image)
   PHASE 2  EVALUATION  (no GPU)   NMS 0.40 -> metrics at confidence 0.30 (all_gt)
                                   -> per-image / experiment / pooled-AP CSVs
                                   -> confusion matrix (CSV + PNG)
                                   -> qualitative GT-vs-prediction figures
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

# The text-only prompt

**What CountGD++ receives.** Every raw tile, resized like in the official code (1000 × 1000 →
800 × 800), with the caption `" rumex obtusifolius . "`: the text, then `.`, the separator that
closes the positive prompt. Nothing is pasted onto the tile and no box is given.

**The exemplar branch.** CountGD++'s `forward()` always runs its exemplar branch on an exemplar
image. For text-only counting, the official `test_dataset.py` and `app.py` pass the **input image
itself** as that exemplar image with an **empty box list** (`{"image": image, "points": []}`).
With zero boxes no exemplar token is created, so that image has no influence on the result. The
script does exactly the same: the tile is passed as its own exemplar image with a `(0, 4)` box
tensor. You will see no exemplar anywhere in the outputs.

**Post-processing**, as in every CountGD++ experiment: the official two-stage filter with the
threshold raised from 0.23 to **0.30**, the boxes clipped to the tile, then the **plausibility
filter of the other CountGD++ experiments** (boxes ≤ 5 px wide or high, or covering > 80 % of the
tile, are removed). The SAM3 text-only notebook has no such filter; it is kept here so that the
CountGD++ experiments differ only in their prompt.

**Evaluation.** The GT boxes are used only to evaluate and to draw the figures, never as model
input. So every GT box counts (`all_gt`): there is no prompt plant to remove, and no `held_out`
row anywhere. Plants larger than the 150 px overlap can be cut by every tile; the per-image table
counts them (`n_gt_larger_than_overlap`).

**Text prompt guard.** The text prompt is written in every manifest row. A run with another
`TEXT_PROMPT` in the same `OUTPUT_DIR` is refused before anything is written, so two prompts are
never mixed.

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

In the exemplar experiments, CountGD++ cannot run several tiles in one forward pass: its
`forward()` adds the exemplar tokens for **sample 0 only**. With text only no exemplar token is
added, so batching would technically work here. `BATCH_SIZE` is still kept at **1**, like every
other CountGD++ experiment, so that all of them run exactly the same way. The script refuses any
other value. To use the 4 GPUs, the images are split across 4 shard processes (one per GPU), as
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
├── text_only_tiling_infer_countGDpp.py
├── text_only_tiling_run_countGDpp.sh
└── text_only_tiling_submit_countGDpp.sh
```

Then, **on a login node**:

```bash
mkdir -p $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/countgdpp
chmod +x text_only_tiling_run_countGDpp.sh text_only_tiling_submit_countGDpp.sh
ls -l
```

**You should see:** the three files, with the two `.sh` marked executable (`-rwxr-xr-x`).

> If the files were edited on Windows, make sure the `.sh` files have Unix line endings,
> otherwise bash fails with `$'\r': command not found`:
> `sed -i 's/\r$//' text_only_tiling_run_countGDpp.sh text_only_tiling_submit_countGDpp.sh`

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
./text_only_tiling_run_countGDpp.sh download
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
./text_only_tiling_run_countGDpp.sh build
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
./text_only_tiling_run_countGDpp.sh dryrun
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
  sampled NNN image(s) of this shard -> NNN runs, ONE per image ({...})
  GT boxes: NNNN (NNN larger than the 150 px overlap)
  tiles per run at 8192x5460: 70
  => ~NNNNN CountGD++ forward passes (1 tile each) for those NNN images
  NPZ files that will be written by this shard: ~NNN (one per image)
```

Confirm the image counts, and that `class_id` is **0** for `AGS_Multi_Rumex` and **2** for
`AgsSpringRumex`.

**If it says `No images with labels found`**: `$SCRATCH/overney/dataset` is wrong; go back to Step 2.

---

## Step 7 — Smoke test on 2 images

**Still inside the interactive shell.** This is the first time the model actually loads. It
writes to a throwaway folder so it cannot pollute the real results.

```bash
EXPERIMENT_NAME=text_only_tiling_countGDpp_smoke \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/_smoke_text_only_tiling \
NUM_GPUS=1 \
./text_only_tiling_run_countGDpp.sh run --limit-images 2
```

**You should see** the model load, then one line per image, then the whole Phase 2:

```
final text_encoder_type: checkpoints/bert-base-uncased
load tokenizer done.
Loading CountGD++ from '/scratch/.../countgd_plusplus.pth' onto cuda (float32) ...
CountGD++ ready.
  model device    : cuda:0
  deformable attn : compiled CUDA op
  fp16 autocast   : False
  missing keys    : 0 | unexpected keys: 38
Resuming: 0 image(s) already finished for text_only_tiling_countGDpp_smoke (text ' rumex obtusifolius '); skipped.
[text_only_tiling_countGDpp_smoke] shard0 (1/2) AGS_Multi_Rumex/.../DJI_... | 12 GT box(es) (1 > 150px) | pre-NMS detections=41 from 23/70 tiles | 9.8s | avg/image=9.8s | ETA=0.2 min (0.00 h)
...
===== PHASE 2 : OFFLINE EVALUATION =====
--- per-image metrics ---
Per-image metrics: 2 images -> .../metrics/image_level_metrics.csv
Images with at least one prediction: 2/2
AP50       0.xxxx
...
```

**Things to check in this output:**

- **`missing keys : 0 | unexpected keys: 38`**: the same numbers as the notebook. Any other
  number means a different repo version or checkpoint.
- Messages like `Some weights of BertModel were not initialized ... pooler` and a `timm`
  `FutureWarning` are **normal**; the notebook shows them too.
- **`9.8s`** (or whatever you get) is the time for ONE image (70 tiles, its only run). Multiply
  it by the image count from Step 6b and divide by 4 GPUs to estimate the job. One run per image:
  roughly the number of images × 70 forward passes in total, so it fits easily in one job.
- **`from 23/70 tiles`**: how many tiles produced at least one detection.
- **`pre-NMS detections=`** should not be 0 on every image. Text-only can find fewer plants than
  the exemplar runs; a low number is a result, not an error, as long as it is not 0 everywhere.

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
sbatch text_only_tiling_submit_countGDpp.sh
```

**You should see:** `Submitted batch job 1234567`

No arguments needed. The job asks for 1 node, 4 GPUs, 64 CPUs and 24 hours. It splits the images
across the 4 GPUs for Phase 1 (one tile per forward pass on each GPU), then runs Phase 2 once.

---

## Step 9 — Watch it

```bash
squeue -u $USER                                                            # queued / running?
tail -f text_only_tiling_countGDpp_1234567.out                           # everything, live
tail -f $SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp/shard0.log   # just GPU 0
```

(The `.out` / `.err` files are written in the folder you ran `sbatch` from.)

Count how many images are finished so far (one run = one image):

```bash
cat $SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp/raw_detections/runs_manifest_text_only_tiling_countGDpp_shard*.csv \
  | grep -c text_only_tiling_countGDpp
ls $SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp/raw_detections/*.npz | wc -l
```

You will also get an email at BEGIN / END / FAIL.

---

## Step 10 — Collect the results

Everything lands in `$SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp/`:

```bash
ls -R $SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp/ | head -40
```

**The file you actually quote in the thesis:**

| File | What it is |
|---|---|
| `metrics/experiment_summary.csv` | **the headline numbers**: one row (`all_gt`), with mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU1 / IoU2 / **count_abs_error** |

There is one row (`all_gt`). The mean is taken over the **per-image** values, so an image with
40 plants does not outweigh one with 2. The std is the variation **between UAV images**.
`n_images_with_predictions` says on how many images CountGD++ found anything at all.

**Everything else in that folder:**

| File | What it is |
|---|---|
| `metrics/experiment_summary_per_archive.csv` | the same table split by archive |
| `metrics/image_level_metrics.csv` | one row per image, the raw data: `text_prompt`, `n_gt`, `n_gt_larger_than_overlap`, `n_tiles`, TP/FP/FN, `count_abs_error` and the NMS provenance counters (`n_suppressed_cross_tile`). One run per image, so this single table replaces the run-level + image-level tables of the anchor experiments |
| `metrics/dataset_ap_metrics.csv` | pooled AP50 / AP50:95, all images ranked in ONE precision-recall curve. **Not** the mean of the per-image AP |
| `confusion_matrices/confusion_matrix_all_gt.csv` + `.png` | pooled TP / FP / FN, plus micro precision/recall/F1 in `confusion_matrix_summary.csv` |
| `plots/best_image_*.png` | the qualitative GT (left) vs predictions (right) figures: one per archive + one global |
| `raw_detections/*.npz` | the PRE-NMS detections of every image (boxes, scores, tile ids, GT boxes, text prompt). This is what Phase 2 reads |
| `raw_detections/runs_manifest_*_shard*.csv` | which images are finished (with the text prompt); this is what resume reads |
| `run_config_text_only_tiling_countGDpp.json` | every parameter used + the CountGD++ repo commit, for the thesis appendix |
| `shard0..3.log` | per-GPU logs |

**`count_abs_error`** = |predictions − GT boxes| at the operating point (0.30, after NMS).

**Comparisons.** Same images, tiles, thresholds and evaluation as the other CountGD++
experiments; only the prompt differs. Compare with the `all_gt` rows of the exemplar experiments:

```bash
cd $SCRATCH/experiments/countgdpp
column -s, -t < text_only_tiling_countGDpp/metrics/experiment_summary.csv            | less -S   # text only
column -s, -t < 1_exemplars_tiling_countGDpp/metrics/experiment_summary.csv      | less -S   # 1 exemplar
column -s, -t < 1_exemplars_text_tiling_countGDpp/metrics/experiment_summary.csv | less -S   # 1 exemplar + the same text
```

All are means over images with the std between images. Use the `all_gt` rows of the exemplar
experiments: `held_out` has no counterpart here.

**Which mode to quote:** only `all_gt` exists here (classical evaluation, every GT box counts).

---

## Step 11 — Resubmit if it hit the 24 h limit

Completely normal, and harmless. Run the same command again:

```bash
sbatch text_only_tiling_submit_countGDpp.sh
```

Every finished image is listed in the shard manifests and is skipped **before**
any GPU work, so the job picks up exactly where it stopped. You will see this near the top of
each shard log:

```
Resuming: 412 image(s) already finished for text_only_tiling_countGDpp (text ' rumex obtusifolius '); skipped.
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
./text_only_tiling_run_countGDpp.sh evaluate
```

Use this to change the operating point without re-running CountGD++:

```bash
OPERATING_CONFIDENCE=0.40 NMS_IOU_THRESHOLD=0.50 \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp \
./text_only_tiling_run_countGDpp.sh evaluate
```

> **Careful:** this overwrites the CSVs in `metrics/`, `confusion_matrices/` and `plots/`. To
> keep both, copy the folder first, or point `OUTPUT_DIR` at a copy of the experiment.

On a login node Phase 2 needs `supervision` and `pandas` from the login node's Python. If they are
missing there, run `evaluate` inside the container instead (Step 4 session, no GPU needed).

`evaluate` needs `DATASET_ROOT` to be reachable, because the figures reopen the selected images.
If the dataset has been purged from `$SCRATCH`, add `--no-plots`:

```bash
./text_only_tiling_run_countGDpp.sh evaluate --no-plots
```

---
---

# Notes and problems

## If it will not fit in 24 hours

Batch size is fixed at 1, so the levers are:

1. **Do nothing.** Resubmit as many times as needed (Step 11). Three 24 h jobs = one 72 h job.
2. **Make sure the compiled op is used.** The pure-PyTorch fallback is the slowest part; check
   the `deformable attn :` line in `shard0.log`, and see **Problem D**.
3. **More GPUs**: needs a job array; ask and I will write one.

With one run per image this experiment needs far fewer forward passes than the anchor ones, so
it should fit in one job.

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
`CONTAINER_ARCH=x86_64 ./text_only_tiling_run_countGDpp.sh download`.

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
THRESHOLD=0.05 EXPERIMENT_NAME=text_only_tiling_countGDpp_lowconf \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp_lowconf \
sbatch text_only_tiling_submit_countGDpp.sh
```

## Other troubleshooting

| Symptom | Fix |
|---|---|
| `SCRATCH: unbound variable` | you are not on the cluster |
| `ERROR: no CountGD++ repository at ...` / `checkpoint not found` | Step 3 incomplete |
| `ERROR: .../bert-base-uncased is missing` | Step 5 part 2 did not run |
| `OSError: We couldn't connect to 'https://huggingface.co'` in Step 5 | the BERT cache of Step 3 is missing or `HF_HOME` differs; rerun Step 3 with the same `HF_HOME` |
| `NameError: name '_C' is not defined` | should never happen (the fallback catches it); report the shard log |
| `--batch-size must be 1` | expected; kept at 1 like every CountGD++ experiment (see "Why batch size 1") |
| CUDA out of memory | unlikely in fp32 at batch 1 on a GH200; lower `TILE_SIZE` or `NUM_GPUS=2` |
| Host RAM fills up | `./text_only_tiling_run_countGDpp.sh run --no-cache-tiles` (tiles cropped on demand) |
| `Nothing to plot` in Phase 2 | no image had a valid AP50, or `DATASET_ROOT` is unreachable. Every CSV is still written |
| Everything reports `F1=0.0000` | wrong class id or wrong labels; recheck Step 6b |
| `RuntimeError: ... already holds results for another text prompt` | `OUTPUT_DIR` already has runs with another `TEXT_PROMPT`; use a new `EXPERIMENT_NAME` + `OUTPUT_DIR` |
| `--text-prompt is empty` | the text is the only prompt here; set `TEXT_PROMPT` |
| Many images with 0 predictions | possible with text only (the text may not be enough for CountGD++ on UAV tiles); check `n_images_with_predictions` before blaming the setup |
| Phase 2 is slow / heavy | it holds every image's detections in memory at once (needed for the pooled AP). Give it a node with more RAM if it gets killed |

## Settings you might change

Set them before `sbatch`; they are forwarded into the job.

```bash
DTYPE=float16                     sbatch text_only_tiling_submit_countGDpp.sh   # the notebook's USE_FP16=True (not validated)
THRESHOLD=0.05                    sbatch text_only_tiling_submit_countGDpp.sh   # save more detections for the AP curve
OPERATING_CONFIDENCE=0.40         sbatch text_only_tiling_submit_countGDpp.sh   # different operating point
NMS_IOU_THRESHOLD=0.50            sbatch text_only_tiling_submit_countGDpp.sh   # different NMS
USE_TILING=0                      sbatch text_only_tiling_submit_countGDpp.sh   # whole image as one tile
NUM_GPUS=2                        sbatch text_only_tiling_submit_countGDpp.sh   # fewer GPUs
DATASET_ROOT=/some/path           sbatch text_only_tiling_submit_countGDpp.sh   # dataset elsewhere
```

Another text prompt, or another filter, is another experiment: give it its **own**
`EXPERIMENT_NAME` and `OUTPUT_DIR` (the script refuses to mix two text prompts in one folder):

```bash
# another text prompt
TEXT_PROMPT="dock plant" \
EXPERIMENT_NAME=text_only_tiling_countGDpp_dock \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp_dock \
sbatch text_only_tiling_submit_countGDpp.sh

# without the plausibility filter, exactly like the SAM3 text-only notebook
MAX_AREA_FRACTION=1.0 EDGE_MARGIN=0 \
EXPERIMENT_NAME=text_only_tiling_countGDpp_nofilter \
OUTPUT_DIR=$SCRATCH/experiments/countgdpp/text_only_tiling_countGDpp_nofilter \
sbatch text_only_tiling_submit_countGDpp.sh
```

