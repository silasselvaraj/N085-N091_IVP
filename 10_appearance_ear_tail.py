"""Stage 10 of 12 — ear shape and tail carriage, from the mask's contour.

AP-10K has no ear or tail-tip keypoints, so these two features - and the
two morphometric ratios stage 7 always refused (ear:head, tail:body) - are
measured a different way: searching the segmentation mask itself in a
window anchored to keypoints that do exist (eyes/nose for ears, neck/tail
base for tail), rather than reading them off a skeleton that was never
built to represent them.

Ears are only attempted in profile or three-quarter views, where an ear's
actual silhouette is visible above the head outline; a frontal or rear view
gives no reliable "rise above the eye line" to measure. Tail carriage has
no such view restriction - it only needs the mask and two keypoints that
are visible from any angle.

Certain enum values genuinely never come out of this algorithm - cropped
ears, docked or bobtail carriages - and that's by design: nothing here
claims a surgical alteration from silhouette geometry alone.

Run: python 10_appearance_ear_tail.py   (needs 03_segmentation.py, 05_pose_estimation.py and 06_view_posture_classification.py)
"""

from __future__ import annotations

import math

import numpy as np

from _common import OUTPUTS_DIR, load_bgr, load_json, put_lines, require_previous_output, save_image, save_json
from _pose_constants import KP

EAR_VIEWS = ("PROFILE_LEFT", "PROFILE_RIGHT", "THREE_QUARTER")
MIN_EAR_SHARE = 0.12

MAX_PLAUSIBLE_TAIL = 0.7
MIN_PLAUSIBLE_TAIL = 0.12


def pt(keypoints, name):
    return keypoints[KP[name]]


def sc(scores, name):
    return scores[KP[name]]


def lateral_mid(keypoints, scores, left, right, floor):
    l_ok, r_ok = sc(scores, left) >= floor, sc(scores, right) >= floor
    if l_ok and r_ok:
        lx, ly = pt(keypoints, left)
        rx, ry = pt(keypoints, right)
        return (lx + rx) / 2, (ly + ry) / 2
    if l_ok:
        return tuple(pt(keypoints, left))
    if r_ok:
        return tuple(pt(keypoints, right))
    return None


def classify_ears(mask: np.ndarray, keypoints, scores, floor, view: str) -> dict:
    if view not in EAR_VIEWS:
        return {"ear_shape": "UNKNOWN", "confidence": 0.0, "ear_to_head_ratio": None, "reason": f"view {view} not eligible (needs profile or three-quarter)"}

    eye_mid = lateral_mid(keypoints, scores, "left_eye", "right_eye", floor)
    if eye_mid is None or sc(scores, "neck") < floor or sc(scores, "nose") < floor:
        return {"ear_shape": "UNKNOWN", "confidence": 0.0, "ear_to_head_ratio": None, "reason": "eyes, nose or neck not confidently located"}

    neck = pt(keypoints, "neck")
    nose = pt(keypoints, "nose")
    head_height = abs(neck[1] - eye_mid[1])
    if head_height < 8:
        return {"ear_shape": "UNKNOWN", "confidence": 0.0, "ear_to_head_ratio": None, "reason": "head too small in frame (<8px)"}

    half_width = max(abs(nose[0] - eye_mid[0]), head_height) * 1.1
    cx = (nose[0] + eye_mid[0]) / 2
    x0, x1 = int(max(0, cx - half_width)), int(min(mask.shape[1], cx + half_width))
    y0, y1 = int(max(0, eye_mid[1] - head_height * 1.6)), int(min(mask.shape[0], eye_mid[1] + head_height * 0.4))
    if x1 <= x0 or y1 <= y0:
        return {"ear_shape": "UNKNOWN", "confidence": 0.0, "ear_to_head_ratio": None, "reason": "search window empty at frame edge"}

    window = mask[y0:y1, x0:x1]
    row_has_fg = window.any(axis=1)
    if not row_has_fg.any():
        return {"ear_shape": "UNKNOWN", "confidence": 0.0, "ear_to_head_ratio": None, "reason": "no mask coverage in ear search window"}
    top_row_local = int(np.argmax(row_has_fg))
    top_y = y0 + top_row_local

    rise = (eye_mid[1] - top_y) / head_height
    if rise < MIN_EAR_SHARE:
        return {"ear_shape": "UNKNOWN", "confidence": 0.0, "ear_to_head_ratio": round(rise, 4), "reason": f"rise {rise:.3f} below minimum {MIN_EAR_SHARE}"}

    band_bottom_local = min(window.shape[0], top_row_local + int(0.25 * head_height))
    band = window[top_row_local:band_bottom_local]
    cols_with_fg = np.where(band.any(axis=0))[0]
    peak_width_px = (cols_with_fg.max() - cols_with_fg.min()) if len(cols_with_fg) else 0
    peak_width = peak_width_px / head_height

    if rise > 0.55 and peak_width < 0.9:
        shape, conf = "ERECT", 0.6
    elif rise > 0.3 and peak_width < 1.2:
        shape, conf = "SEMI_ERECT", 0.45
    elif rise <= 0.3:
        shape, conf = "DROP", 0.55
    else:
        shape, conf = "UNKNOWN", 0.2

    return {"ear_shape": shape, "confidence": conf, "ear_to_head_ratio": round(rise, 4), "peak_width": round(peak_width, 4)}


def classify_tail(mask: np.ndarray, keypoints, scores, floor) -> dict:
    if sc(scores, "neck") < floor or sc(scores, "tail_base") < floor:
        return {"tail_carriage": "UNKNOWN", "confidence": 0.0, "tail_to_body_ratio": None, "reason": "neck or tail_base not confidently located"}

    neck, tail_base = pt(keypoints, "neck"), pt(keypoints, "tail_base")
    body_length = math.hypot(neck[0] - tail_base[0], neck[1] - tail_base[1])
    if body_length < 8:
        return {"tail_carriage": "UNKNOWN", "confidence": 0.0, "tail_to_body_ratio": None, "reason": "body too small in frame (<8px)"}

    direction = 1 if tail_base[0] >= neck[0] else -1
    radius = body_length * 0.8
    h, w = mask.shape
    if direction > 0:
        x0, x1 = int(tail_base[0]), int(min(w, tail_base[0] + radius))
    else:
        x0, x1 = int(max(0, tail_base[0] - radius)), int(tail_base[0])
    y0, y1 = int(max(0, tail_base[1] - radius)), int(min(h, tail_base[1] + radius))

    if x1 <= x0 or y1 <= y0:
        return {"tail_carriage": "UNKNOWN", "confidence": 0.0, "tail_to_body_ratio": None, "reason": "search window empty at frame edge"}

    ys, xs = np.where(mask[y0:y1, x0:x1])
    if len(xs) == 0:
        return {"tail_carriage": "UNKNOWN", "confidence": 0.0, "tail_to_body_ratio": None, "reason": "no mask coverage behind hips"}
    ys, xs = ys + y0, xs + x0
    dists = np.hypot(xs - tail_base[0], ys - tail_base[1])
    best = int(np.argmax(dists))
    tip = (float(xs[best]), float(ys[best]))
    tail_length = float(dists[best])

    if tail_length < MIN_PLAUSIBLE_TAIL * body_length:
        return {"tail_carriage": "UNKNOWN", "confidence": 0.0, "tail_to_body_ratio": None, "reason": "nothing plausible behind hips"}
    if tail_length > MAX_PLAUSIBLE_TAIL * body_length:
        return {"tail_carriage": "UNKNOWN", "confidence": 0.0, "tail_to_body_ratio": None, "reason": "search found something implausibly far (likely a leg, not a tail)"}

    rise = (tail_base[1] - tip[1]) / body_length
    if rise > 0.28:
        carriage, conf = "UPRIGHT", 0.55
    elif rise > 0.10:
        carriage, conf = "SICKLE", 0.5
    elif rise > -0.10:
        carriage, conf = "STRAIGHT_HORIZONTAL", 0.5
    else:
        carriage, conf = "LOW_HANGING", 0.5

    return {"tail_carriage": carriage, "confidence": conf, "tail_to_body_ratio": round(tail_length / body_length, 4), "tip_px": tip}


def main() -> None:
    pose = load_json(require_previous_output(OUTPUTS_DIR / "05_pose.json", "05_pose_estimation.py"))
    vp = load_json(require_previous_output(OUTPUTS_DIR / "06_view_posture.json", "06_view_posture_classification.py"))
    mask = load_bgr(require_previous_output(OUTPUTS_DIR / "03_mask.png", "03_segmentation.py"))[:, :, 0] > 0

    keypoints, scores, floor = pose["keypoints"], pose["scores"], pose["confidence_floor"]

    ears = classify_ears(mask, keypoints, scores, floor, vp["view"])
    tail = classify_tail(mask, keypoints, scores, floor)

    print("Ears:")
    for k, v in ears.items():
        print(f"  {k:<18s} = {v}")
    print("\nTail:")
    for k, v in tail.items():
        print(f"  {k:<18s} = {v}")

    save_json(OUTPUTS_DIR / "10_ear_tail.json", {"ears": ears, "tail": tail})
    annotated = put_lines(load_bgr(OUTPUTS_DIR / "05_pose.jpg"), [f"ear_shape={ears['ear_shape']}", f"tail_carriage={tail['tail_carriage']}"])
    save_image(OUTPUTS_DIR / "10_ear_tail.jpg", annotated)
    print("\nWrote outputs/10_ear_tail.json and outputs/10_ear_tail.jpg")
    print("(these ear_to_head_ratio / tail_to_body_ratio values are the two stage 7 always left as None)")


if __name__ == "__main__":
    main()
