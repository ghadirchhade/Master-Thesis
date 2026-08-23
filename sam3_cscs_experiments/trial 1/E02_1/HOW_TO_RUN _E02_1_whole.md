HOW TO RUN — Experiment E02_1

What E02_1 is: SAM3 prompted with 3 ground-truth bounding boxes passed directly as box prompts, tiling OFF (the whole image is resized once so its longest side is 1024 px), run over both AGS_Multi_Rumex and AgsSpringRumex pooled as one dataset.


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
If you have already set up E02_2, Phase A is nearly free: Steps 2, 3 and 4 produce exactly the same dataset, token and $SCRATCH/hf_cache / $SCRATCH/pyextra. Copy the three new scripts (Step 1), then skip straight to Step 6.

PHASE A — SETUP (once)
Step 1 — Copy the files onto the cluster
Put the three scripts next to the E02_2 ones, in the same folder:

$HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3/
├── infer_sam3.py            ← E02_2 (tiled), untouched
├── run_sam3.sh              ← E02_2, untouched
├── submit_sam3.sh           ← E02_2, untouched
├── infer_sam3_whole.py      ← NEW  (E02_1)
├── run_sam3_whole.sh        ← NEW
├── submit_sam3_whole.sh     ← NEW
└── HOW_TO_RUN_whole.md      ← this file
Then, on a login node:

bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
chmod +x run_sam3_whole.sh submit_sam3_whole.sh
ls -l
You should see: run_sam3_whole.sh and submit_sam3_whole.sh marked executable (-rwxr-xr-x).

Step 2 — Put the dataset on $SCRATCH
(Identical to E02_2. If the dataset is already there, skip to Step 3.)

Extract the two archives side by side under one folder:

bash
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

bash
ls $SCRATCH/overney/dataset
ls $SCRATCH/overney/dataset/AGS_Multi_Rumex
ls $SCRATCH/overney/dataset/AGS_Multi_Rumex/images | head
Important: the folder names are case-sensitive and must match exactly. The other two archives (AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings) may be present — the code detects them and skips them on purpose.

Note on class ids: AGS_Multi_Rumex labels rumex as class 0, AgsSpringRumex as class 2. The script knows this per archive; you do not have to do anything. Your notebook hard-coded RUMEX_CLASS_ID = 0, which is why it could only read the first archive.

Why $SCRATCH and not $HOME: $HOME is small and slow; $SCRATCH is the large fast filesystem. (It is also periodically purged, so keep the original archives somewhere safe.)

Step 3 — HuggingFace licence + token
(Identical to E02_2. If $SCRATCH/hf_cache is already populated, skip to Step 5.)

facebook/sam3 is a gated model. Two things are needed:

Go to https://huggingface.co/facebook/sam3 while logged in and accept the licence.
Create a token at https://huggingface.co/settings/tokens (read access is enough).
Then, on a login node:

bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
Note: the token must belong to the same account that accepted the licence, otherwise Step 4 fails with a 401.

(This replaces the notebook's userdata.get("HF_TOKEN") and login() cells — there is no interactive prompt in a batch job.)

Step 4 — Download the model and supervision
Still on a login node (this is the only step that needs internet — compute nodes have none):

bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./run_sam3_whole.sh download
This does two things:

downloads the SAM3 weights into $SCRATCH/hf_cache
installs the supervision package into $SCRATCH/pyextra (supervision computes mAP50; the code refuses to run without it)
You should see:

--- 1/2 : model weights -> /scratch/.../hf_cache ---
Snapshot cached at: /scratch/.../models--facebook--sam3/snapshots/...
--- 2/2 : supervision -> /scratch/.../pyextra ---
Successfully installed supervision-...
supervision 0.x.x -> /scratch/.../pyextra/supervision/__init__.py
Done. Compute nodes can now run offline (HF_HUB_OFFLINE=1).
If it says ERROR: HF_TOKEN is not set → go back to Step 3. If it says 401 or gated → the licence was not accepted with that token's account.

If you already ran ./run_sam3.sh download for E02_2, this is a no-op — same cache, same $PYEXTRA. Running it again is harmless.

Step 5 — Open an interactive session inside the container
Steps 6 and 7 must run inside the container, not on the login node — otherwise you are checking the login node's Python, which is not the one the job will use.

bash
srun --account=go077 --time=00:30:00 \
     --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
     --environment=yolo26 --pty bash
You should see: a new shell prompt, running on a compute node. Everything in Steps 6 and 7 happens in this shell.

(If your site needs a partition flag, add --partition=.... If the allocation takes a while, that is just the queue.)

Step 6 — Check the container has what it needs
Inside the interactive shell from Step 5:

bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
./run_sam3_whole.sh dryrun
This loads no model and uses no GPU. It prints a PYTHON DEBUG block, then a dataset report.

6a. Look for these three lines in the debug block
torch: 2.x.x | cuda available: True | device count: 1
Sam3Model import: OK
MeanAveragePrecision import: OK
If you see	Meaning	Fix
Sam3Model import FAILED	the yolo26 image's transformers is too old for SAM3	see Problem A at the bottom
supervision import FAILED	$SCRATCH/pyextra not found or wrong Python version	rerun Step 4; if it still fails, see Problem B
cuda available: False	no GPU in this allocation	add --gpus-per-task=1 to Step 5
Do not continue until all three lines are correct. Everything after this depends on them.

6b. Check the configuration banner
 n_exemplars    : 3  (multiple prompt)
 prompting      : GT boxes passed directly as input_boxes (no strip, no tiling)
 max_dim        : 1024  (longest side after resize; never upscaled)
 thresholds     : conf=0.3  mask=0.5  match_iou=0.5
These four lines are E02_1. If prompting mentions a strip, or max_dim is missing, you are running run_sam3.sh instead of run_sam3_whole.sh.

6c. Check the dataset report
Ignoring archives (by design): AGS_Multiple_Fields, AGS_Multiple_Fields_Embeddings
  AGS_Multi_Rumex        class_id=0  images=NNN   flights=N   labels_indexed=NNN
  AgsSpringRumex         class_id=2  images=NNN   flights=N   labels_indexed=NNN
Total images: NNN | this shard: NNN
Confirm the image counts match what you expect, and that class_id is 0 for AGS_Multi_Rumex and 2 for AgsSpringRumex.

It also prints a cost estimate:

  sampled NNN image(s) of this shard -> NNNN anchor runs ({'AGS_Multi_Rumex': ..., 'AgsSpringRumex': ...})
  forward passes per anchor run: 1 (whole image, no tiling)
  => NNNN SAM3 forward passes for those NNN images
forward passes per anchor run: 1 is the whole point of this experiment — E02_2 needs ~70 per anchor on an 8192×5460 photo, this needs one.

If it says No images with labels found → $SCRATCH/overney/dataset is wrong, or the folder names do not match. Go back to Step 2.

If it warns N image(s) have no matching label file → those are skipped. A handful is normal; if it is every image, the label basenames do not match the image basenames.

Step 7 — Smoke test on 2 images
Still inside the interactive shell. This is the first time the model actually loads. It writes to a throwaway folder so it cannot pollute the real E02_1 results.

bash
EXPERIMENT_NAME=E02_1_smoke \
OUTPUT_DIR=$SCRATCH/experiments/sam3/_smoke_whole \
NUM_GPUS=1 \
./run_sam3_whole.sh run --limit-images 2
You should see the model load, then one line per anchor run:

Loading SAM3 from 'facebook/sam3' onto cuda (float32) ...
SAM3 ready. Parameters live on: cuda:0
    [E02_1_smoke] run #1 | AGS_Multi_Rumex/2022.../DJI_0001 | anchor=0 (1/12) | prompt=0+7+3 | preds=4 | mAP50=0.291 | 1.4s
Three things to note from this output:

1.4s (or whatever you get) is the time for ONE anchor run — expect low single-digit seconds, not the ~60 s of the tiled experiment. Multiply by the total anchor count from Step 6c to estimate the whole job.
prompt=0+7+3 — three boxes: the anchor first, then two others sampled from the same image. If you see prompt=0 only, the preset is single (or the image has one plant).
mAP50= should be a plausible number, not 0.000 on every single run. All zeros means something is wrong with the prompts or the labels.
Expect preds= to be small. Without tiling, a 1024 px view of an 8192 px photo shrinks each plant to a handful of pixels, so SAM3 finds far fewer of them than in E02_2. That is the finding this experiment exists to measure, not a bug — the notebook behaved the same way.

Then leave the interactive session:

bash
exit
Setup is done. You never have to repeat Phase A.

PHASE B — RUN
Step 8 — Submit the job
On a login node:

bash
cd $HOME/2025-OverneyTechnologiesProject/7_cscs_experiments/sam3
sbatch submit_sam3_whole.sh
You should see: Submitted batch job 1234567

That is it. No arguments needed — E02_1 is the default. The job asks for 1 node, 4 GPUs, 64 CPUs, 12 hours (half of E02_2's request, because there is no tiling), and internally splits the images across the 4 GPUs.

Both experiments can queue at the same time: they write to different folders (.../sam3/E02_1 vs .../sam3/E02_2) and different log files (sam3_whole_%j.out).

Step 9 — Watch it
bash
squeue -u $USER                                     # queued / running?
tail -f sam3_whole_1234567.out                      # everything, live
tail -f $SCRATCH/experiments/sam3/E02_1/shard0.log  # just GPU 0
Count how many runs are finished so far:

bash
cat $SCRATCH/experiments/sam3/E02_1/runs/results_E02_1_shard*.csv | grep -c E02_1
You will also get an email at BEGIN / END / FAIL.

Step 10 — Collect the results
Everything lands in $SCRATCH/experiments/sam3/E02_1/:

bash
ls $SCRATCH/experiments/sam3/E02_1/
The three files you actually want:

File	What it is
summary_E02_1_all.csv	the headline numbers — mean and std of mAP50 / precision / recall / IoU1 / IoU2, averaged over images
summary_E02_1_AGS_Multi_Rumex.csv	same, for that archive only
summary_E02_1_AgsSpringRumex.csv	same, for that archive only
Look at them:

bash
cd $SCRATCH/experiments/sam3/E02_1
column -s, -t < summary_E02_1_all.csv
Everything else in that folder:

File	What it is
results_E02_1_all.csv	one row per (image × anchor) — the raw data
results_E02_1_<archive>.csv	the same rows split by archive
image_level_E02_1_*.csv	one row per image (the prompt runs averaged together)
raw_summary_E02_1_*.csv	mean/std treating every run as an independent sample
runs/results_E02_1_shard*.csv	the per-GPU raw output; this is what resume reads
runs/detections_E02_1_shard*.jsonl	the predicted boxes + scores for every run
runs/masks_E02_1_shard*.jsonl	only with SAVE_MASKS=1
run_config_E02_1.json	every parameter used, for the thesis appendix
shard0..3.log	per-GPU logs
Which number to quote: summary_E02_1_all.csv. It averages the per-image means, so an image with 40 plants does not outweigh one with 2.

These three tables are your notebook's last three cells: raw_summary_* is the "every row independent" table, image_level_* is the per-image groupby, summary_* is the mean/std over those image-level means.

One extra column vs E02_2: sam_scale — the resize factor actually applied to that image (1.0 means the image was already ≤ 1024 px and was left alone). Useful for checking how hard each archive was downscaled.

Reading the detections in Python:

python
import json
rows = [json.loads(l) for l in open("runs/detections_E02_1_shard0.jsonl")]
# each row: {"image_ID", "anchor_idx", "boxes": [[x1,y1,x2,y2], ...], "scores": [...],
#            "sam_scale", ...}   -- boxes are in FULL-RESOLUTION pixels
Step 11 — Resubmit if it hit the walltime
Completely normal, and harmless. Just run the same command again:

bash
sbatch submit_sam3_whole.sh
Every finished (image, anchor) pair is already in the shard CSVs and is skipped before any GPU work, so the job picks up exactly where it stopped. Repeat until it finishes inside the walltime. You will see this near the top of the log:

Resume: 4821 run(s) already recorded for E02_1; they will be skipped.
If a shard crashed but the others finished: the summaries are still built from whatever completed, and the job exits non-zero so you get the FAIL email. Just resubmit — it fills the gaps.

Notes and problems
How E02_1 differs from E02_2
E02_2 (infer_sam3.py)	E02_1 (infer_sam3_whole.py)
prompt	exemplar crops pasted into a strip above each tile	exemplar boxes passed straight to SAM3 — the prompt already lives in the image
geometry	overlapping 1000 px tiles, 150 px overlap	whole image resized once to 1024 px
post-processing	strip-detection removal, plausibility filter, global NMS	none — every detection above --threshold is kept
forward passes per anchor	~70 at 8192×5460	1
--mask-threshold	0.4	0.5 (what the notebook used)
walltime request	24 h	12 h
Dataset discovery, exemplar sampling, the five metrics, the CSV schema, sharding, resume and aggregation are identical in both, so rows from the two experiments are directly comparable.

Reproducibility — one deliberate change from the notebook
The notebook drew exemplars from a single np.random.default_rng(42) consumed in loop order. That makes the sampled set depend on how many runs happened before it — which cannot survive sharding, resume, or reordering. This port seeds per run instead:

python
rng = np.random.default_rng(stable_seed(image_id, anchor_idx))
Same anchor, same image → same exemplar set, on any machine, in any order, whether it is run 1 or run 40 000. The exemplar sets therefore differ from a given Colab run, but the distribution is the same and the experiment is now actually reproducible. This is also what lets you verify a result: rerun one image with --limit-images 1 and get identical rows.

If it will not fit in the walltime
You have four options, in order of preference:

Do nothing. Resubmit as many times as needed (Step 11). Two 12 h jobs = one 24 h job.
DTYPE=bfloat16 sbatch submit_sam3_whole.sh — roughly 2× faster on GH200/H100, negligible accuracy change.
--max-anchors-per-image 5 — instead of every ground-truth box taking a turn as the anchor, use only the first 5 per image. This changes what you are measuring, so mention it in the thesis.
Raise --time in submit_sam3_whole.sh back to 24 h. Cheap here, since option 1 makes it mostly unnecessary.
Problem A — Sam3Model import FAILED
The yolo26 container was built for ultralytics and its transformers is too old for SAM3. Build a dedicated image:

dockerfile
# Dockerfile
FROM <whatever image ~/.edf/yolo26.toml points at>
RUN pip install --no-cache-dir "transformers>=<version with SAM3>" supervision accelerate
bash
podman build -t sam3 .
enroot import -o $SCRATCH/images/sam3.sqsh podman://sam3:latest
cp ~/.edf/yolo26.toml ~/.edf/sam3.toml
# edit ~/.edf/sam3.toml so `image = ` points at $SCRATCH/images/sam3.sqsh
then change the one line at the bottom of submit_sam3_whole.sh:

bash
srun --environment=sam3 ...
and use --environment=sam3 in Step 5 as well.

Problem B — supervision import FAILED inside the container
Step 4 installed it with the login node's Python. If the container uses a different Python version, the install is invisible to it. Install from inside the container instead — but the compute node has no internet, so download the wheel on the login node first:

bash
# login node
pip download --no-deps -d $SCRATCH/wheels supervision

# inside the container (Step 5 shell)
pip install --target $SCRATCH/pyextra --no-deps --no-index --find-links $SCRATCH/wheels supervision
Always --no-deps. supervision would otherwise install its own numpy / opencv into $SCRATCH/pyextra, and because that directory is searched before the container's own packages, those copies would shadow the ones torch was compiled against and break torch.

One thing to decide before quoting numbers
The default confidence threshold is 0.3, which means detections scoring below 0.3 never reach the metric — so the reported mAP50 is a lower bound, not a true mAP.

For the numbers that go in the thesis, run a second pass with a low threshold:

bash
THRESHOLD=0.001 EXPERIMENT_NAME=E02_1_lowconf sbatch submit_sam3_whole.sh
It writes to its own folder, so you can keep both and compare. This matters more here than in E02_2: without tiling the plants are tiny, so more of the true detections sit low in the score distribution.

Other troubleshooting
Symptom	Fix
SCRATCH: unbound variable	you are not on the cluster, or the module environment is not loaded
Job hangs at "Loading SAM3"	$SCRATCH/hf_cache is empty and the node is offline — redo Step 4
CUDA out of memory	rare here (one 1024 px pass). DTYPE=bfloat16, then MAX_DIM=768 — but that changes the experiment, so give it a new EXPERIMENT_NAME
preds=0 on most runs	expected at 1024 px if the plants are small; confirm with a low THRESHOLD before concluding the prompt is broken
Everything reports mAP50=0.000	wrong class id or wrong labels — recheck Step 6c
Rows appear under the wrong experiment	two configurations sharing one EXPERIMENT_NAME. The name is the resume key and the file name — one name per configuration, always
Settings you might change
Set them before sbatch; they are forwarded into the job.

bash
DTYPE=bfloat16          sbatch submit_sam3_whole.sh   # faster
THRESHOLD=0.001         sbatch submit_sam3_whole.sh   # proper mAP curve
PRESET=single           sbatch submit_sam3_whole.sh   # 1 exemplar box instead of 3
MAX_DIM=2048            sbatch submit_sam3_whole.sh   # less aggressive downscale
NUM_GPUS=2              sbatch submit_sam3_whole.sh   # fewer GPUs
SAVE_MASKS=1            sbatch submit_sam3_whole.sh   # also store masks
DATASET_ROOT=/some/path sbatch submit_sam3_whole.sh   # dataset elsewhere
Anything after the mode is forwarded straight to Python, so one-off flags work too:

bash
./run_sam3_whole.sh run --limit-images 20 --max-anchors-per-image 3
./run_sam3_whole.sh run --archives AGS_Multi_Rumex        # one archive only
Always change EXPERIMENT_NAME when you change a setting. It is the resume key, the CSV tag and the folder name at once — reusing it silently mixes two configurations into one set of results.

The two ablations this file set gives you for free
bash
# 3 boxes vs 1 box, same geometry
EXPERIMENT_NAME=E02_1        PRESET=multi   sbatch submit_sam3_whole.sh
EXPERIMENT_NAME=E02_1_single PRESET=single  sbatch submit_sam3_whole.sh

# whole-image (this) vs tiled (E02_2), same prompts
sbatch submit_sam3_whole.sh
sbatch submit_sam3.sh
Masks
SAVE_MASKS=1 stores masks in the resized coordinate space, together with sam_scale and the full image size. A full-resolution mask of an 8192×5460 photo is ~45 MB before compression; the resized one is ~1 MB, and nearest-neighbour upsampling by 1/sam_scale recovers it exactly. To map a mask back:

python
import numpy as np
from PIL import Image
h, w = record["full_size"]
m = decode_rle(entry["rle"])                       # bool array, resized space
full = np.array(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST), bool)


