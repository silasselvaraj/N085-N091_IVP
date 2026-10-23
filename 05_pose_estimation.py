"""Stage 5 of 12 — pose estimation (17 dog keypoints).

ViTPose+ base, run with `dataset_index=3` to select its AP-10K (animal pose)
expert head rather than its default human-pose one. This one setting is
worth dwelling on: the checkpoint's own `id2label` metadata still describes
COCO *human* joints even when expert 3 is selected — trusting it would
silently mislabel every keypoint below. The real names, asserted here by
channel position only, are `AP10K_KEYPOINTS`.

`-plus-base` specifically, not `-plus-small`: a bake-off in the real project
found the small variant's AP-10K expert broken (landmarks stacked on top of
each other at 0.85-1.00 confidence — a model can be confident and wrong at
the same time, which is exactly why this pipeline never trusts a model's own
confidence as ground truth). See docs/pose_bakeoff.md.

Weights: ivp_showcase/weights/pose_vitpose/ (local copy of
usyd-community/vitpose-plus-base).

Run: python 05_pose_estimation.py   (needs 02_detection.py to have run first)
"""

from __future__ import annotations

import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

from _common import (
    OUTPUTS_DIR,
    WEIGHTS_DIR,
    Box,
    analysis_copy_path,
    draw_keypoints,
    load_bgr,
    load_json,
    load_rgb,
    require_previous_output,
    save_image,
    save_json,
)
from _pose_constants import AP10K_KEYPOINTS, CONFIDENCE_FLOOR, MIN_CONFIDENT_KEYPOINTS, SKELETON_IDX

POSE_DIR = WEIGHTS_DIR / "pose_vitpose"
AP10K_DATASET_INDEX = 3  # 0=COCO(human) 1=AIC 2=MPII 3=AP-10K(animal) 4=APT-36K 5=COCO-WholeBody


def main() -> None:
    require_previous_output(analysis_copy_path(), "01_quality_check.py")
    det_path = require_previous_output(OUTPUTS_DIR / "02_detection.json", "02_detection.py")
    det = load_json(det_path)
    if not det.get("dog_found"):
        print("No dog was found in stage 2 - nothing to pose-estimate.")
        return
    box = Box.from_dict(det["subject_box"])

    image_rgb = load_rgb(analysis_copy_path())
    image_bgr = load_bgr(analysis_copy_path())

    print(f"Loading ViTPose+ base from {POSE_DIR.relative_to(POSE_DIR.parents[1])} (local weights only)")
    processor = AutoProcessor.from_pretrained(str(POSE_DIR), local_files_only=True)
    model = VitPoseForPoseEstimation.from_pretrained(str(POSE_DIR), local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  device: {device}, dataset_index={AP10K_DATASET_INDEX} (AP-10K animal expert)")

    boxes = [[[box.x1, box.y1, box.width, box.height]]]  # [x, y, w, h] pixel, per-image nesting
    inputs = processor(images=image_rgb, boxes=boxes, return_tensors="pt").to(device)
    dataset_index = torch.tensor([AP10K_DATASET_INDEX], device=device)

    with torch.no_grad():
        outputs = model(**inputs, dataset_index=dataset_index)

    results = processor.post_process_pose_estimation(outputs, boxes=boxes)
    keypoints = results[0][0]["keypoints"].cpu().numpy()
    scores = results[0][0]["scores"].cpu().numpy().clip(0, 1)

    confident = scores >= CONFIDENCE_FLOOR
    print(f"\n{int(confident.sum())}/17 keypoints above confidence floor {CONFIDENCE_FLOOR}:")
    for name, (x, y), s in zip(AP10K_KEYPOINTS, keypoints, scores):
        flag = "OK" if s >= CONFIDENCE_FLOOR else "--"
        print(f"  [{flag}] {name:<16s} ({x:6.1f}, {y:6.1f})  conf={s:.3f}")

    usable = int(confident.sum()) >= MIN_CONFIDENT_KEYPOINTS
    print(f"\nusable for downstream measurement: {usable}  (needs >= {MIN_CONFIDENT_KEYPOINTS} confident keypoints)")

    overlay = draw_keypoints(image_bgr, [tuple(p) for p in keypoints], list(scores), SKELETON_IDX, CONFIDENCE_FLOOR)
    save_image(OUTPUTS_DIR / "05_pose.jpg", overlay)

    save_json(
        OUTPUTS_DIR / "05_pose.json",
        {
            "keypoint_names": AP10K_KEYPOINTS,
            "keypoints": [[float(x), float(y)] for x, y in keypoints],
            "scores": [float(s) for s in scores],
            "confidence_floor": CONFIDENCE_FLOOR,
            "confident_keypoint_count": int(confident.sum()),
            "usable_for_measurement_gate": usable,
            "subject_box": box.to_dict(),
        },
    )
    print("\nWrote outputs/05_pose.json and outputs/05_pose.jpg")


if __name__ == "__main__":
    main()
