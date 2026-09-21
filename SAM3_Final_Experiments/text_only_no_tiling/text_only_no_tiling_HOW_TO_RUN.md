HOW TO RUN — Experiment `text_only_no_tiling`
What this experiment is: SAM3 prompted with TEXT ONLY, tiling OFF, run over both
`AGS_Multi_Rumex` and `AgsSpringRumex` pooled as one dataset.
How it works in one line: the whole UAV image is downscaled so its longest side is
1024 px, and sent to SAM3 with ONLY a text prompt (e.g. "Rumex obtusifolius") — no
exemplar box, no visual prompts, no tiles, no exemplar strip, no plausibility filter.
This is the cluster port of `text_only_no_tiling.ipynb`. Like the notebook, it is a
two-phase pipeline:

   PHASE 1  INFERENCE   (GPU)      SAM3 runs ONCE per image at score 0.30
                                   -> pre-NMS detections saved as NPZ
   PHASE 2  EVALUATION  (no GPU)   NMS 0.40 -> metrics at confidence 0.30
                                   -> image / experiment / pooled-AP CSVs
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
├── text_only_no_tiling_infer_sam3.py
├── text_only_no_tiling_run_sam3.sh
└── text_only_no_tiling_submit_sam3.sh

Then, on a login node:
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
chmod +x text_only_no_tiling_run_sam3.sh text_only_no_tiling_submit_sam3.sh
ls -l
You should see: the two `.sh` marked executable (`-rwxr-xr-x`).

Step 2 — Put the dataset on `$SCRATCH`
If you already did this for the other experiments, skip this step. All the experiments read the
same `$SCRATCH/overney/dataset`.
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

Important: the folder names are case-sensitive and must match exactly.
Note on the class ids. The notebook ran on `AGS_Multi_Rumex` alone with
`RUMEX_CLASS_ID = 0`. Both archives are used here and they label Rumex with different class
ids, so the id is a property of the archive:
| Archive          | Rumex class id in its YOLO files |
|------------------|----------------------------------|
| AGS_Multi_Rumex  | 0                                |
| AgsSpringRumex   | 2                                |
Every other class in those files is ignored.

Why `$SCRATCH` and not `$HOME`: `$HOME` is small and slow; `$SCRATCH` is the large fast
filesystem. (It is also periodically purged, so keep the original archives somewhere safe.)

Step 3 — HuggingFace licence + token
`facebook/sam3` is a gated model. Two things are needed:
1. Go to [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) while logged in and accept the licence.
2. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough).

Then, on a login node:
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
Note: the token must belong to the same account that accepted the licence, otherwise
Step 4 fails with a 401.

Step 4 — Download the model and the extra packages
If you already ran the `download` mode of any other SAM3 experiment, skip this step.
They all share `$SCRATCH/hf_cache`, `$SCRATCH/pyextra` and `$SCRATCH/wheels`.
Still on a login node (this is the only step that needs internet — compute nodes have none):
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./text_only_no_tiling_run_sam3.sh download

This does three things:
- downloads the SAM3 weights into `$SCRATCH/hf_cache`
- installs the `supervision` package into `$SCRATCH/pyextra`
  (`supervision` computes AP50 / AP50:95; the code refuses to run without it)
- downloads wheels for the Phase 2 packages (`pandas`, `matplotlib`) into `$SCRATCH/wheels`,
  in case the container does not ship them — see Problem B

You should see:
--- 1/3 : model weights -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--facebook--sam3/snapshots/...
--- 2/3 : supervision -> /scratch/.../pyextra ---
Successfully installed supervision-...
supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
--- 3/3 : wheels for the PHASE 2 packages -> /scratch/.../wheels ---
Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1).

If it says `ERROR: HF_TOKEN is not set` → go back to Step 3.
If it says `401` or `gated` → the licence was not accepted with that token's account.

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
./text_only_no_tiling_run_sam3.sh dryrun

This loads no model and uses no GPU. It prints a `PYTHON DEBUG` block, then a dataset report.

6a. Look for these lines in the debug block
torch: 2.x.x | cuda available: True | device count: 1
bfloat16 supported: True
Sam3Model import: OK
MeanAveragePrecision import: OK
pandas: 2.x.x -> PHASE 2 import: OK
matplotlib: 3.x.x -> confusion-matrix + qualitative PNGs: OK

| If you see                     | Meaning                                                | Fix                                |
|--------------------------------|--------------------------------------------------------|------------------------------------|
| Sam3Model import FAILED        | the `yolo26` image's `transformers` is too old for SAM3| see `Problem A` at the bottom      |
| supervision import FAILED      | `$SCRATCH/pyextra` not found or wrong Python version   | rerun Step 4; if it still fails, see `Problem B` |
| pandas import FAILED           | Phase 1 still works, but Phase 2 cannot run            | see `Problem B`                    |
| matplotlib import FAILED       | only the PNGs are lost; every CSV is still written     | see `Problem B`, or ignore it      |
| bfloat16 supported: False      | old GPU                                                | use `DTYPE=float16`                |
| cuda available: False          | no GPU in this allocation                              | add `--gpus-per-task=1` to Step 5  |

Do not continue until the first four lines are correct. Everything after this depends on them.

6b. Check the dataset report
Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
  AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
  AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
Total images: NNN | this shard: NNN

Confirm the image counts match what you expect, and that `class_id` is 0 for
`AGS_Multi_Rumex` and 2 for `AgsSpringRumex`.
It also prints a cost estimate:
  sampled NNN image(s) of this shard -> NNNN forward passes
  forward passes per image: 1 (whole image, no tiling, longest side -> 1024 px)
  => ~NNNN SAM3 forward passes for those NNN images
  NPZ files that will be written by this shard: ~NNNN (one per image)

If it says `No images with labels found` → `$SCRATCH/overney/dataset` is wrong, or the
folder names do not match. Go back to Step 2.

Step 7 — Smoke test on 2 images
Still inside the interactive shell. This is the first time the model actually loads.
It writes to a throwaway folder so it cannot pollute the real results.
EXPERIMENT_NAME=text_only_no_tiling_smoke \
OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke_text_only_no_tiling \
TEXT_PROMPT="Rumex obtusifolius" \
NUM_GPUS=1 \
./text_only_no_tiling_run_sam3.sh run --limit-images 2

You should see the model load, then one line per image, then the whole Phase 2:
Loading SAM3 from 'facebook/sam3' onto cuda (bfloat16) ...
SAM3 loaded.
  [text_only_no_tiling_smoke] shard0 run #1 | AGS_Multi_Rumex/2022.../DJI_0001 | pre-NMS detections=7 | 1.4s
...
===== PHASE 2 : OFFLINE EVALUATION =====
--- CELL 18: image-level metrics ---
  all_gt   : NN images, F1_mean=0.xxxx

Three things to check in this output:
- `1.4s` (or whatever you get) is the time for ONE image — a single forward pass, so
  expect seconds, not the ~60 s the tiled experiments take. Multiply by the total image
  count from Step 6b to estimate the whole job.
- `pre-NMS detections=` should not be 0 on every run. All zeros means SAM3 found nothing
  above 0.30, or the text prompt is wrong.
- `F1_mean=` should be a plausible number in `all_gt`, not `0.0000` everywhere. All zeros
  means something is wrong with the labels or the class id.

Then leave the interactive session:
exit
Setup is done. You never have to repeat Phase A.

PHASE B — RUN

Step 8 — Submit the job
On a login node:
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
sbatch text_only_no_tiling_submit_sam3.sh

You should see: `Submitted batch job 1234567`
That is it. No arguments needed. The job asks for 1 node, 4 GPUs, 64 CPUs, 24 hours, splits the
images across the 4 GPUs for Phase 1, and then runs Phase 2 once.

Step 9 — Watch it
squeue -u $USER                                                        # queued / running?
tail -f text_only_no_tiling_sam3_1234567.out                           # everything, live
tail -f $SCRATCH/experiments/sam3/text_only_no_tiling/shard0.log       # just GPU 0

Count how many images are finished so far:
cat $SCRATCH/experiments/sam3/text_only_no_tiling/raw_detections/runs_manifest_text_only_no_tiling_shard*.csv \
  | grep -c text_only_no_tiling
ls $SCRATCH/experiments/sam3/text_only_no_tiling/raw_detections/*.npz | wc -l

You will also get an email at BEGIN / END / FAIL.

Step 10 — Collect the results
Everything lands in `$SCRATCH/experiments/sam3/text_only_no_tiling/`:
ls -R $SCRATCH/experiments/sam3/text_only_no_tiling/ | head -40

The file you actually quote in the thesis:
| File                                     | What it is                                                                 |
|------------------------------------------|----------------------------------------------------------------------------|
| metrics/experiment_summary.csv           | the headline numbers — one row for `all_gt`, with mean and std of AP50 / AP50_95 / precision / recall / F1 / IoU1 / IoU2 |

The mean is taken over the image-level values, so an image with 40 plants does not outweigh
one with 2. The std is the variation between UAV images.

Everything else in that folder:
| File                                     | What it is                                                                 |
|------------------------------------------|----------------------------------------------------------------------------|
| metrics/experiment_summary_per_archive.csv | the same table split by archive.                                           |
| metrics/image_level_metrics.csv          | one row per image (no anchors in text-only); includes `archive` and `flight` columns |
| metrics/dataset_ap_metrics.csv           | pooled AP50 / AP50:95 — all images ranked in ONE precision-recall curve.     |
| confusion_matrices/confusion_matrix_all_gt.csv + .png | pooled TP / FP / FN, plus micro precision/recall/F1              |
| plots/best_image_*.png                   | the qualitative GT-vs-prediction figures (one per archive + one global)      |
| raw_detections/*.npz                     | the PRE-NMS detections of every image (boxes, scores, GT boxes, archive, flight, the resize scale) |
| raw_detections/runs_manifest_*.csv       | which images are finished; this is what resume reads                       |
| run_config_text_only_no_tiling.json      | every parameter used, for the thesis appendix                              |
| shard0..3.log                            | per-GPU logs                                                               |

Look at the headline table:
cd $SCRATCH/experiments/sam3/text_only_no_tiling/metrics
column -s, -t < experiment_summary.csv | less -S

Which mode to quote: ONLY `all_gt`. Text-only prompting uses no GT box as input, so every GT
box of the image is evaluated. The `held_out` mode of the exemplar experiments does not apply
here: there is no prompt plant to remove.

The qualitative figures. The notebook produced one. Both archives are pooled here, so the
same selection runs three times and you get three files in `plots/`:
best_image_AGS_Multi_Rumex_..._all_gt.png
best_image_AgsSpringRumex_..._all_gt.png
best_image_ALL_..._all_gt.png
Each is GT (yellow, left) next to the predictions (red, right), with the confidence written
next to each prediction. The image chosen is the one with the highest image-level AP50 among
those with ≥ 7 GT boxes.

Reading the raw detections in Python:
import numpy as np
z = np.load("raw_detections/AGS_Multi_Rumex__20220518_Eschikon__DJI_0001.npz")
print(z["boxes"].shape, z["scores"].min(), z["gt_boxes"].shape)
print(str(z["archive"]), float(z["sam_scale"]))   # 1024 / 8192 = 0.125

Step 11 — Resubmit if it hit the walltime
Completely normal, and harmless. Just run the same command again:
sbatch text_only_no_tiling_submit_sam3.sh

Every finished image is listed in the shard manifests and is skipped before
any GPU work, so the job picks up exactly where it stopped.

Step 12 — Re-run only the evaluation (optional, cheap)
Phase 2 reads only the NPZ files, so you never need a GPU for it:
# on a login node, or in any small allocation
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./text_only_no_tiling_run_sam3.sh evaluate

Use this to change the operating point without re-running SAM3:
OPERATING_CONFIDENCE=0.40 NMS_IOU_THRESHOLD=0.50 \
OUTPUT_DIR=$SCRATCH/experiments/sam3/text_only_no_tiling \
./text_only_no_tiling_run_sam3.sh evaluate

Careful: this overwrites the CSVs in `metrics/`, `confusion_matrices/` and `plots/`.

Notes and problems

If it will not fit in the walltime
Unlikely for this experiment, but in order of preference:
1. Do nothing. Resubmit as many times as needed (Step 11).
2. More GPUs — needs a job array; ask and I will write one.
3. `MAX_DIM` is the other lever, but changing it changes the experiment.

Three things to say about this in the thesis:
1. The prompt mechanism differs. In the exemplar experiments the exemplar is cropped and
   pasted into a strip above each tile (or passed as a box at its real location); here it is
   passed as a TEXT prompt only. That is what "text-only" means for this pipeline.
2. Resolution. Whole-image mode downscales an 8192 px image to 1024 px, so a Rumex plant
   that was ~60 px across becomes ~7 px. Expect recall to suffer on small plants; that is
   the finding, not a bug.
3. Evaluation mode. Only `all_gt` is used because there are no prompt plants to hold out.

Problem A — `Sam3Model import FAILED`
The `yolo26` container was built for ultralytics and its `transformers` is too old for SAM3.
Build a dedicated image:
# Dockerfile
FROM <whatever image ~/.edf/yolo26.toml points at>
RUN pip install --no-cache-dir "transformers>=<version with SAM3>" \
        supervision accelerate pandas matplotlib
podman build -t sam3 .
enroot import -o $SCRATCH/images/sam3.sqsh podman://sam3:latest
cp ~/.edf/yolo26.toml ~/.edf/sam3.toml
# edit ~/.edf/sam3.toml so `image = ` points at $SCRATCH/images/sam3.sqsh
then change the one line at the bottom of `text_only_no_tiling_submit_sam3.sh`:
srun --environment=sam3 ...
and use `--environment=sam3` in Step 5 as well.

Problem B — `supervision` / `pandas` / `matplotlib` import FAILED inside the container
Step 4 installed with the login node's Python. If the container uses a different Python
version, the install is invisible to it. Install from inside the container instead:
# inside the container (Step 5 shell) — pick whichever package failed
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels supervision
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels pandas pytz tzdata python-dateutil six
pip install --target $SCRATCH/pyextra --no-deps --no-index \
    --find-links $SCRATCH/wheels matplotlib contourpy cycler fonttools kiwisolver pyparsing packaging

Always `--no-deps`. Those packages would otherwise install their own numpy / pillow into
`$SCRATCH/pyextra`, and because that directory is searched before the container's own
packages, those copies would shadow the ones `torch` was compiled against and break torch.

Settings you might change
Set them before `sbatch`; they are forwarded into the job.
TEXT_PROMPT="Rumex obtusifolius" sbatch text_only_no_tiling_submit_sam3.sh  # change the text prompt
DTYPE=float32              sbatch text_only_no_tiling_submit_sam3.sh        # what the notebook's T4 used
MAX_DIM=2048               sbatch text_only_no_tiling_submit_sam3.sh        # less downscaling, slower
THRESHOLD=0.05             sbatch text_only_no_tiling_submit_sam3.sh        # save more detections for the AP curve
OPERATING_CONFIDENCE=0.40  sbatch text_only_no_tiling_submit_sam3.sh        # match E01_2 / E02_2's operating point
NMS_IOU_THRESHOLD=0.50     sbatch text_only_no_tiling_submit_sam3.sh        # different NMS
NUM_GPUS=2                 sbatch text_only_no_tiling_submit_sam3.sh        # fewer GPUs
DATASET_ROOT=/some/path    sbatch text_only_no_tiling_submit_sam3.sh        # dataset elsewhere

Changing `TEXT_PROMPT`, `MAX_DIM` or `THRESHOLD` changes what is stored in the NPZ files, so
give those runs their own `EXPERIMENT_NAME` and `OUTPUT_DIR`. Changing only
`OPERATING_CONFIDENCE` / `NMS_IOU_THRESHOLD` does not — use Step 12 instead, it is free.
