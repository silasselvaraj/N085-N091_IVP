"""Stage 3 of 12 — segmentation and mask quality.

SAM 2.1 tiny (Apache-2.0) traces the subject's outline, prompted with
stage 2's box — not points, since a box prompt is unambiguous for a single
animal and doesn't need a click location that doesn't exist in a batch
script. Of the 3 mask hypotheses SAM2 proposes, the one with the model's own
highest IoU estimate is kept.

That IoU estimate is a *prior*, not a measurement: a model can be confident
about a bad mask. So this script independently measures the mask afterwards
— fill fraction, solidity, boundary smoothness, edge contact — and only
that independent measurement decides whether the mask is usable. Masks are
stored as uint8 0/255; indexing pixels with a raw uint8 array is a classic
NumPy trap (integer fancy-indexing, wrong pixels, no error), so every pixel
lookup below goes through `as_bool()` first.

Also produces `dog_crop_clean` — a tight, background-suppressed crop with a
feathered edge — which is what stage 12's breed classifier actually looks
at, not the full photo.

This runs SAM2 twice over: once for the single ranked "subject" (unchanged
`03_mask.png` / `03_dog_crop_clean.jpg` outputs, feeding stages 4-11 exactly
as before), and again for every other deduped dog stage 2 found, so stage 12
can classify and label all of them, not only the subject.

Weights: ivp_showcase/weights/segmentation_sam2/ (local copy of
facebook/sam2.1-hiera-tiny).

Run: python 03_segmentation.py   (needs 02_detection.py to have run first)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor

from _common import (
    OUTPUTS_DIR,
    WEIGHTS_DIR,
    Box,
    analysis_copy_path,
    draw_mask_overlay,
    load_bgr,
    load_json,
    load_rgb,
    require_previous_output,
    save_image,
    save_json,
)

SEGMENTER_DIR = WEIGHTS_DIR / "segmentation_sam2"

MIN_BOX_FILL, MAX_BOX_FILL = 0.12, 1.05
OCCLUSION_SOLIDITY = 0.72
FRAGMENT_RATIO = 0.10  # if the largest component holds less than 90% of all foreground pixels, call it fragmented

CROP_MARGIN_FRACTION = 0.08
FEATHER_PX = 3
BACKGROUND_FILL = (128, 128, 128)


def as_bool(mask: np.ndarray) -> np.ndarray:
    """A uint8 0/255 mask used directly as a boolean index silently does
    integer fancy-indexing and returns the wrong pixels rather than
    raising - this is the fix, and it caught a real Part 4 bug."""
    return mask if mask.dtype == bool else mask > 0


def postprocess(raw_mask_bool: np.ndarray) -> tuple[np.ndarray, bool]:
    mask_u8 = (raw_mask_bool.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    fragmented = False
    if n_labels <= 1:
        largest = opened
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        total_fg = float(areas.sum())
        largest_idx = 1 + int(np.argmax(areas))
        largest_area = float(areas[largest_idx - 1])
        fragmented = total_fg > 0 and (largest_area / total_fg) < (1.0 - FRAGMENT_RATIO)
        largest = np.where(labels == largest_idx, 255, 0).astype(np.uint8)

    closed = cv2.morphologyEx(largest, cv2.MORPH_CLOSE, kernel)

    # Fill interior holes: external contours, drawn filled.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    return filled, fragmented


def measure_mask_quality(mask_u8: np.ndarray, box: Box, fragmented: bool) -> dict:
    h, w = mask_u8.shape[:2]
    mask_bool = as_bool(mask_u8)
    area_px = int(mask_bool.sum())

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or area_px == 0:
        return {
            "area_px": area_px, "box_fill": 0.0, "solidity": 0.0, "boundary_smoothness": 0.0,
            "edge_contact": 0.0, "plausible": False, "occluded": True, "fragmented": fragmented, "quality": 0.0,
        }
    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(contour, True)
    hull_perimeter = cv2.arcLength(hull, True)

    solidity = contour_area / hull_area if hull_area > 0 else 0.0
    boundary_smoothness = hull_perimeter / perimeter if perimeter > 0 else 0.0

    pts = contour.reshape(-1, 2)
    near_edge = (pts[:, 0] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] <= 1) | (pts[:, 1] >= h - 2)
    edge_contact = float(near_edge.mean())

    box_fill = area_px / box.area if box.area > 0 else 0.0
    plausible = MIN_BOX_FILL <= box_fill <= MAX_BOX_FILL and area_px > 0
    occluded = fragmented or (plausible and solidity < OCCLUSION_SOLIDITY)

    fill_term = 1.0 - min(abs(box_fill - 0.62) / 0.45, 1.0)
    solidity_term = float(np.clip((solidity - 0.45) / 0.35, 0, 1))
    smoothness_term = float(np.clip((boundary_smoothness - 0.70) / 0.25, 0, 1))
    edge_penalty = 1.0 - min(edge_contact, 0.5)
    quality = (0.4 * fill_term + 0.35 * solidity_term + 0.25 * smoothness_term) * edge_penalty if plausible else 0.0

    return {
        "area_px": area_px, "box_fill": round(box_fill, 4), "solidity": round(solidity, 4),
        "boundary_smoothness": round(boundary_smoothness, 4), "edge_contact": round(edge_contact, 4),
        "plausible": plausible, "occluded": occluded, "fragmented": fragmented, "quality": round(float(quality), 4),
    }


def tight_crop(image_bgr: np.ndarray, box: Box, margin_fraction: float) -> tuple[np.ndarray, Box]:
    h, w = image_bgr.shape[:2]
    mx, my = box.width * margin_fraction, box.height * margin_fraction
    crop_box = Box(box.x1 - mx, box.y1 - my, box.x2 + mx, box.y2 + my).clipped(w, h)
    x1, y1, x2, y2 = int(crop_box.x1), int(crop_box.y1), int(crop_box.x2), int(crop_box.y2)
    return image_bgr[y1:y2, x1:x2].copy(), crop_box


def background_suppressed(image_bgr: np.ndarray, mask_u8: np.ndarray, crop_box: Box) -> np.ndarray:
    x1, y1, x2, y2 = int(crop_box.x1), int(crop_box.y1), int(crop_box.x2), int(crop_box.y2)
    crop = image_bgr[y1:y2, x1:x2].astype(np.float32)
    mask_crop = mask_u8[y1:y2, x1:x2].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(mask_crop, (0, 0), sigmaX=FEATHER_PX)[..., None]
    background = np.full_like(crop, BACKGROUND_FILL, dtype=np.float32)
    blended = crop * alpha + background * (1 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def segment_box(model, processor, device: str, image_rgb: np.ndarray, box: Box) -> tuple[np.ndarray, dict, list[float], int]:
    """Run SAM2 on one box, clean the mask, and independently measure it. Shared by
    the single-subject pass and the every-dog loop below - same model, same logic,
    just called once per box."""
    inputs = processor(
        images=image_rgb, input_boxes=[[[box.x1, box.y1, box.x2, box.y2]]], return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    iou_scores = outputs.iou_scores.cpu().numpy().reshape(-1)
    best = int(np.argmax(iou_scores))

    raw_mask_bool = masks[0][best].numpy() > 0
    cleaned_u8, fragmented = postprocess(raw_mask_bool)
    quality = measure_mask_quality(cleaned_u8, box, fragmented)
    return cleaned_u8, quality, [float(s) for s in iou_scores], best


def main() -> None:
    require_previous_output(analysis_copy_path(), "01_quality_check.py")
    det_path = require_previous_output(OUTPUTS_DIR / "02_detection.json", "02_detection.py")
    det = load_json(det_path)
    if not det.get("dog_found"):
        print("No dog was found in stage 2 - nothing to segment.")
        return
    box = Box.from_dict(det["subject_box"])
    all_dogs = det.get("all_dogs") or [{"box": det["subject_box"], "score": det["subject_score"]}]

    image_rgb = load_rgb(analysis_copy_path())
    image_bgr = load_bgr(analysis_copy_path())
    h, w = image_rgb.shape[:2]

    print(f"Loading SAM 2.1 tiny from {SEGMENTER_DIR.relative_to(SEGMENTER_DIR.parents[1])} (local weights only)")
    processor = Sam2Processor.from_pretrained(str(SEGMENTER_DIR), local_files_only=True)
    model = Sam2Model.from_pretrained(str(SEGMENTER_DIR), local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  device: {device}")

    # --- pass 1: the single ranked subject - unchanged outputs, feeds stages 4-11 ---
    print(f"\n[subject] box={box.to_dict()}")
    cleaned_u8, quality, iou_scores, best = segment_box(model, processor, device, image_rgb, box)
    print(f"  SAM2's own IoU estimates: {[round(s, 3) for s in iou_scores]}  (picked hypothesis {best} - a prior, not yet trusted)")
    print("  independently measured (not the model's IoU estimate):")
    for k, v in quality.items():
        print(f"    {k:<20s} = {v}")

    save_image(OUTPUTS_DIR / "03_mask.png", cleaned_u8)
    overlay = draw_mask_overlay(image_bgr, as_bool(cleaned_u8))
    save_image(OUTPUTS_DIR / "03_segmentation.jpg", overlay)

    if quality["plausible"]:
        crop, crop_box = tight_crop(image_bgr, box, CROP_MARGIN_FRACTION)
        save_image(OUTPUTS_DIR / "03_dog_crop.jpg", crop)
        clean = background_suppressed(image_bgr, cleaned_u8, crop_box)
        save_image(OUTPUTS_DIR / "03_dog_crop_clean.jpg", clean)
        print("  wrote 03_dog_crop.jpg and 03_dog_crop_clean.jpg (background suppressed to grey)")
    else:
        crop_box = box
        print("  mask not plausible - no crop produced (mirrors the real pipeline continuing without one).")

    save_json(
        OUTPUTS_DIR / "03_segmentation.json",
        {"iou_hypotheses": iou_scores, "chosen_hypothesis": best, **quality, "crop_box": crop_box.to_dict(), "image_width": w, "image_height": h},
    )

    # --- pass 2: every deduped dog stage 2 found, so stage 12 can label all of them ---
    print(f"\n[all {len(all_dogs)} dog(s)] segmenting each for stage 12's per-dog classification")
    all_results = []
    for i, dog in enumerate(all_dogs):
        dog_box = Box.from_dict(dog["box"])
        cleaned_i, quality_i, _iou_i, _best_i = segment_box(model, processor, device, image_rgb, dog_box)
        entry = {"index": i, "box": dog_box.to_dict(), "detection_score": dog["score"], "mask_quality": quality_i, "crop_path": None}
        if quality_i["plausible"]:
            _crop_i, crop_box_i = tight_crop(image_bgr, dog_box, CROP_MARGIN_FRACTION)
            clean_i = background_suppressed(image_bgr, cleaned_i, crop_box_i)
            crop_path = OUTPUTS_DIR / f"03_dog_crop_clean_{i}.jpg"
            save_image(crop_path, clean_i)
            entry["crop_path"] = crop_path.name
        all_results.append(entry)
        print(f"  dog {i}: mask_quality={quality_i['quality']}  plausible={quality_i['plausible']}  crop={entry['crop_path']}")

    save_json(OUTPUTS_DIR / "03_all_dogs.json", {"dogs": all_results})
    print("\nWrote outputs/03_segmentation.json, outputs/03_mask.png, outputs/03_segmentation.jpg and outputs/03_all_dogs.json")


if __name__ == "__main__":
    main()
