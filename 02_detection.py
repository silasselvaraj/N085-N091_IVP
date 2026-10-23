"""Stage 2 of 12 — animal/dog detection.

Runs RT-DETR (Apache-2.0, chosen over Ultralytics YOLO specifically to avoid
an AGPL dependency — see PROJECT_ROADMAP.md Part 3) over the analysis copy
from stage 1. RT-DETR is end-to-end: no separate NMS pass to tune, unlike
the classic Faster-RCNN/YOLO family.

If more than one dog is found, the real pipeline doesn't just take the
highest-confidence box — a small, sharp dog in the foreground should win
over a big, blurry one merely passing through the background. So detections
are ranked by a composite of confidence, how much of the frame they cover,
and how central they are, after deduplicating near-identical boxes. Stages
4-11 deep-dive on that single top-ranked "subject" (matching the real app's
own design: every dog gets boxed, but you inspect one at a time). Stage 12
is different - it classifies and labels *every* deduped dog this script
found, using the full `all_dogs` list saved below, not just the subject.

Weights: ivp_showcase/weights/detection_rtdetr/ (local copy of
PekingU/rtdetr_r50vd_coco_o365, loaded with local_files_only — never re-fetched).

Run: python 02_detection.py   (needs 01_quality_check.py to have run first)
"""

from __future__ import annotations

import math

import torch
from transformers import AutoImageProcessor, RTDetrForObjectDetection

from _common import (
    OUTPUTS_DIR,
    WEIGHTS_DIR,
    Box,
    analysis_copy_path,
    draw_box,
    load_bgr,
    load_rgb,
    require_previous_output,
    save_image,
    save_json,
)

DETECTOR_DIR = WEIGHTS_DIR / "detection_rtdetr"
CONFIDENCE_THRESHOLD = 0.42  # measured against 0.35/0.50 alternatives in Part 3 — see docs/part3_measurements.md

DOG_LABEL = "dog"
ANIMAL_LABELS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
PLUSH_LABEL = "teddy bear"

DUPLICATE_IOU = 0.75


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def centrality(box: Box, width: int, height: int) -> float:
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    dx, dy = (cx - width / 2) / (width / 2), (cy - height / 2) / (height / 2)
    return max(0.0, 1.0 - math.hypot(dx, dy) / math.sqrt(2))


def rank_candidates(boxes: list[Box], scores: list[float], width: int, height: int) -> list[int]:
    """Dedupe near-identical boxes, then rank the rest by
    confidence^1.0 * area_fraction^0.5 * centrality^0.5 — a small, sharp,
    centred dog should beat a big blurry one merely passing through frame."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    for i in order:
        if any(iou(boxes[i], boxes[j]) >= DUPLICATE_IOU for j in kept):
            continue
        kept.append(i)

    def composite(i: int) -> float:
        area_fraction = boxes[i].area / (width * height)
        return (max(scores[i], 1e-6) ** 1.0) * (max(area_fraction, 1e-6) ** 0.5) * (
            max(centrality(boxes[i], width, height), 1e-6) ** 0.5
        )

    return sorted(kept, key=composite, reverse=True)


def main() -> None:
    require_previous_output(analysis_copy_path(), "01_quality_check.py")
    image_rgb = load_rgb(analysis_copy_path())
    image_bgr = load_bgr(analysis_copy_path())
    height, width = image_rgb.shape[:2]

    print(f"Loading RT-DETR r50vd from {DETECTOR_DIR.relative_to(DETECTOR_DIR.parents[1])} (local weights only)")
    processor = AutoImageProcessor.from_pretrained(str(DETECTOR_DIR), local_files_only=True)
    model = RTDetrForObjectDetection.from_pretrained(str(DETECTOR_DIR), local_files_only=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  device: {device}")

    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # target_sizes is (height, width) - transposing this silently produces
    # wrong boxes on non-square images, per the real detector.py's own warning.
    results = processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(height, width)]), threshold=CONFIDENCE_THRESHOLD
    )[0]

    id2label = model.config.id2label
    detections = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = id2label[int(label_id)]
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        b = Box(x1, y1, x2, y2).clipped(width, height)
        detections.append({"label": label, "score": float(score), "box": b})
    detections.sort(key=lambda d: d["score"], reverse=True)

    print(f"\n{len(detections)} detection(s) above confidence {CONFIDENCE_THRESHOLD}:")
    for d in detections:
        print(f"  {d['label']:<20s} score={d['score']:.3f}  box={d['box'].to_dict()}")

    dogs = [d for d in detections if d["label"] == DOG_LABEL]

    if not dogs:
        other_animals = sorted({d["label"] for d in detections if d["label"] in ANIMAL_LABELS})
        if other_animals:
            print(f"\nNo dog found. Animal(s) seen instead: {', '.join(other_animals)}")
        elif detections:
            print(f"\nNo dog found. Seen instead: {', '.join(sorted({d['label'] for d in detections}))}")
        else:
            print("\nNo dog, and nothing else recognisable, found in this photo.")
        annotated = image_bgr
        for d in detections:
            annotated = draw_box(annotated, d["box"], f"{d['label']} {d['score']:.2f}", color=(0, 0, 220))
        save_image(OUTPUTS_DIR / "02_detection.jpg", annotated)
        save_json(
            OUTPUTS_DIR / "02_detection.json",
            {"all_detections": [{"label": d["label"], "score": d["score"], "box": d["box"].to_dict()} for d in detections], "dog_found": False},
        )
        return

    boxes = [d["box"] for d in dogs]
    scores = [d["score"] for d in dogs]
    ranked = rank_candidates(boxes, scores, width, height)
    subject_idx = ranked[0]
    subject_box = boxes[subject_idx]

    print(f"\n{len(dogs)} dog detection(s); {len(ranked)} after dedup; subject: box={subject_box.to_dict()} score={scores[subject_idx]:.3f}")

    annotated = image_bgr
    for i, d in enumerate(dogs):
        colour = (0, 220, 90) if i == subject_idx else (60, 180, 255)
        label = "SUBJECT" if i == subject_idx else f"dog {scores[i]:.2f}"
        annotated = draw_box(annotated, d["box"], label, color=colour)
    save_image(OUTPUTS_DIR / "02_detection.jpg", annotated)

    # Every deduped dog, subject first - stage 3 segments each one and stage 12
    # classifies each one, so the final result can label every dog in the
    # photo, not only the single ranked subject stages 4-11 deep-dive on.
    all_dogs = [{"box": boxes[i].to_dict(), "score": scores[i]} for i in ranked]

    save_json(
        OUTPUTS_DIR / "02_detection.json",
        {
            "dog_found": True,
            "all_detections": [{"label": d["label"], "score": d["score"], "box": d["box"].to_dict()} for d in detections],
            "dog_count": len(dogs),
            "subject_box": subject_box.to_dict(),
            "subject_score": scores[subject_idx],
            "all_dogs": all_dogs,
            "image_width": width,
            "image_height": height,
        },
    )
    print("\nWrote outputs/02_detection.json and outputs/02_detection.jpg")


if __name__ == "__main__":
    main()
