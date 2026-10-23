# IVP Concept Showcase

Twelve standalone scripts, each demonstrating one stage of the real image-processing
pipeline that powers this project's breed identification, run in order on a single sample
photo. This folder exists to show that the pipeline was actually built — not wired together
from a pre-made system — by making every stage's logic visible and runnable on its own.

**This folder is completely independent of `backend/` and `frontend/`.** No script here
imports anything from `backend/`; every algorithm is rewritten standalone (dialed down for
clarity, not for correctness), and the four stages that need a trained neural network load
the exact same weights the live app uses, from local copies physically sitting in
`weights/` — never re-downloaded, never re-trained.

Track C never modifies `backend/`, `frontend/`, or any other existing project file. See
`PROJECT_ROADMAP.md`'s "TRACK C" section for the full specification this was built from.

## Running it

From the repository root, using the same Python environment the rest of the project uses
(all required packages — `torch`, `transformers`, `timm`, `opencv`, `scikit-learn`,
`scikit-image`, `scipy` — are already installed there; nothing new to install):

```powershell
cd ivp_showcase
..\backend\.venv\Scripts\python.exe 01_quality_check.py
..\backend\.venv\Scripts\python.exe 02_detection.py
..\backend\.venv\Scripts\python.exe 03_segmentation.py
..\backend\.venv\Scripts\python.exe 04_subject_quality_recheck.py
..\backend\.venv\Scripts\python.exe 05_pose_estimation.py
..\backend\.venv\Scripts\python.exe 06_view_posture_classification.py
..\backend\.venv\Scripts\python.exe 07_morphometrics.py
..\backend\.venv\Scripts\python.exe 08_appearance_colour.py
..\backend\.venv\Scripts\python.exe 09_appearance_coat_pattern.py
..\backend\.venv\Scripts\python.exe 10_appearance_ear_tail.py
..\backend\.venv\Scripts\python.exe 11_depth_estimation.py
..\backend\.venv\Scripts\python.exe 12_breed_classification.py
```

Run them in order — the numeric prefix **is** the run order, because each script reads the
previous ones' saved output from `outputs/` rather than recomputing the whole pipeline from
scratch every time (that also means only one model is ever loaded into GPU memory at once,
matching the real app's own phase-exclusive GPU lease). A script that's missing its input
says exactly which earlier script to run first and exits, rather than guessing.

Each script also prints a running explanation of what it measured and why to the console —
read that alongside the code; the two are meant to be read together.

## What's in this folder

```
ivp_showcase/
  README.md                  <- this file
  research.md                <- brainstorm menu of IVP research-idea options
  _common.py                 <- shared IO/drawing helpers only - no CV/ML logic
  _quality.py                <- quality metrics, shared by stages 1 and 4 (they are
                                 genuinely the same module in the real pipeline too)
  _pose_constants.py         <- the 17 AP-10K keypoint names/order, shared by stages
                                 5, 6, 7 and 10 so they agree on keypoint layout
  weights/                   <- local copies of every trained model this needs (not
                                 committed to git - see "Weights" below)
  sample_images/
    sample_dog.jpg           <- one real photo from the project's own fixtures
  outputs/                   <- created on run; each script saves its result here
  01_quality_check.py ... 12_breed_classification.py
```

## What each stage does, and what it needs/produces

| # | Script | Needs (from `outputs/`) | Produces |
|---|---|---|---|
| 1 | `01_quality_check.py` | sample image only | `analysis_copy.jpg` (the resized frame every later stage works from), `01_quality_check.json/.jpg` |
| 2 | `02_detection.py` | stage 1 | `02_detection.json/.jpg` — all detections, the ranked/deduped `all_dogs` list, and the selected subject box |
| 3 | `03_segmentation.py` | stages 1, 2 | `03_mask.png`, `03_dog_crop.jpg`, `03_dog_crop_clean.jpg`, `03_segmentation.json/.jpg` for the subject; `03_all_dogs.json` + `03_dog_crop_clean_{i}.jpg` per dog |
| 4 | `04_subject_quality_recheck.py` | stage 3 | `04_subject_quality.json/.jpg` |
| 5 | `05_pose_estimation.py` | stages 1, 2 | `05_pose.json/.jpg` — 17 AP-10K keypoints |
| 6 | `06_view_posture_classification.py` | stage 5 | `06_view_posture.json/.jpg` |
| 7 | `07_morphometrics.py` | stages 5, 6 | `07_morphometrics.json/.jpg` — 4 of 6 ratios (see below) |
| 8 | `08_appearance_colour.py` | stage 3 | `08_colour.json/.jpg`, `08_colour_pixels.npz` |
| 9 | `09_appearance_coat_pattern.py` | stages 3, 8 | `09_coat_pattern.json/.jpg` |
| 10 | `10_appearance_ear_tail.py` | stages 3, 5, 6 | `10_ear_tail.json/.jpg` — the other 2 of 6 ratios |
| 11 | `11_depth_estimation.py` (bonus) | stages 3, 5 | `11_depth.json/.jpg` |
| 12 | `12_breed_classification.py` | stage 3 (stage 4 improves the subject's result if present) | `12_breed_results.json`, `12_annotated.jpg` — the final deliverable |

Stages 2, 3, 5 and 12 load a trained neural network from `weights/`. The other eight are
pure Python/NumPy/OpenCV logic — no model call.

**Stage 12 classifies and labels every dog stage 3 segmented, not only the single ranked
"subject" stages 4-11 deep-dive on** — `12_annotated.jpg` boxes every dog in the photo with
its own breed name and confidence percentage above the box (or "unable to identify" /
"no usable segmentation" when the evidence or the mask doesn't support a call). This is a
genuinely new capability beyond what the live app does today — the real app boxes every dog
but only classifies the one you've selected (PROJECT_STATE.md §2.0's "click another to
re-analyse about it"). Every dog other than the subject uses stage 1's frame-level
`degraded` flag for its quality gate rather than a per-dog version of stage 4's recheck,
which was designed for one subject at a time; each such result says so explicitly in
`12_breed_results.json`.

## Weights

| Folder/file | Model | Used by |
|---|---|---|
| `weights/detection_rtdetr/` | `PekingU/rtdetr_r50vd_coco_o365` (RT-DETR, Apache-2.0) | Stage 2 |
| `weights/segmentation_sam2/` | `facebook/sam2.1-hiera-tiny` (SAM 2.1, Apache-2.0) | Stage 3 |
| `weights/pose_vitpose/` | `usyd-community/vitpose-plus-base` (AP-10K animal expert) | Stages 5, 11 |
| `weights/depth_anything/` | `depth-anything/Depth-Anything-V2-Small-hf` | Stage 11 |
| `weights/breed_classifier_convnext.pt` | Fine-tuned ConvNeXt-Tiny, Part 6's checkpoint | Stage 12 |
| `weights/breed_taxonomy.json` | Breed id -> display name (Stanford Dogs, 120 breeds) | Stage 12 |

These are copies of exactly what the live app uses — copied once from `backend/.cache/`
and `models/breed_classifier/`, not re-downloaded, not re-trained. They total roughly 1 GB
and are **not committed to git**, per this project's standing rule to never commit model
weights — CLAUDE.md's own instruction, unchanged for this folder. If you're reading this on
a fresh checkout, `weights/` will be empty; regenerate it from a working copy of the
project's `backend/.cache/` and `models/breed_classifier/` (the four HF model IDs above, and
`models/breed_classifier/convnext_tiny_finetuned.pt` + `fixtures/part5/taxonomy.json`).

## About the sample photo

`sample_images/sample_dog.jpg` is `fixtures/real/coco/images/000000017029.jpg` — a COCO
image the project's own Part 3 test fixtures already used, of a dog jumping mid-air for a
frisbee. It was picked deliberately over a calmer standing photo: it exercises detection,
segmentation and pose cleanly, but its frontal, mid-air pose correctly **fails** the
view/posture gate that stages 7 and 10 require for most of their measurements. That's not a
bug in this demo — it's the same honest "unmeasured is `None`, never a guess" behaviour the
real pipeline has, and PROJECT_STATE.md §3.2 documents this exact gate passing on 0 of 31
real photos it was measured against. A showcase that only ever ran on a cherry-picked photo
where everything succeeds wouldn't actually be demonstrating what was built. Swap in a
different `sample_images/sample_dog.jpg` any time — nothing here depends on this specific
photo.

## A note on two scripts

- **Stage 9's pattern classifier** uses this script's own illustrative CIELAB reference
  points for 14 named coat colours (see `08_appearance_colour.py`'s docstring) — the
  mechanism (nearest-neighbour match, then merge same-named clusters) is faithfully
  reproduced; the exact numbers are this script's own reasonable approximation.
- **Stage 12's mixed-breed heuristic** uses a small illustrative breed→FCI-group mapping
  (a dozen common breeds) to demonstrate the mechanism. The real project's Part 7 (the
  actual breed feature database, which would supply this mapping for all 120 breeds) was
  never built, so in the live application today this heuristic is always `False` — this
  script says so directly rather than silently working "better" than production does.
