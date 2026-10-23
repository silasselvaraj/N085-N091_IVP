"""Stage 7 of 12 — body-proportion morphometrics.

Four ratios, each computed from stage 5's keypoints or refused with a
recorded reason - never a zero, never a guess, because an unmeasured value
and a measured zero mean different things to anything reasoning over this
data later. Three of the four (height:length, leg:body, chest depth) need
stage 6's gate to have passed (profile view, standing posture) because they
all divide by body length measured along the spine, which only means what
it says when the dog is standing side-on to the camera. The fourth
(muzzle:skull) only uses head keypoints, so it survives views the other
three can't.

Two more ratios exist in the schema - ear:head and tail:body - but AP-10K
has no ear or tail-tip keypoints to measure them from; this script always
refuses both. Stage 10 fills them in later from the mask's contour instead.

Run: python 07_morphometrics.py   (needs 05_pose_estimation.py, 06_view_posture_classification.py and 03_segmentation.py)
"""

from __future__ import annotations

import math

import numpy as np

from _common import OUTPUTS_DIR, load_bgr, load_json, put_lines, require_previous_output, save_image, save_json
from _pose_constants import KP, LATERAL_PAIRS

ASYMMETRY_THRESHOLD = 0.12


def pt(keypoints, name):
    return keypoints[KP[name]]


def sc(scores, name):
    return scores[KP[name]]


def confident(scores, name, floor):
    return sc(scores, name) >= floor


def lateral_mid(keypoints, scores, left, right, floor):
    l_ok, r_ok = confident(scores, left, floor), confident(scores, right, floor)
    if l_ok and r_ok:
        lx, ly = pt(keypoints, left)
        rx, ry = pt(keypoints, right)
        return (lx + rx) / 2, (ly + ry) / 2, (sc(scores, left) + sc(scores, right)) / 2
    if l_ok:
        x, y = pt(keypoints, left)
        return x, y, sc(scores, left)
    if r_ok:
        x, y = pt(keypoints, right)
        return x, y, sc(scores, right)
    return None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main() -> None:
    pose_path = require_previous_output(OUTPUTS_DIR / "05_pose.json", "05_pose_estimation.py")
    vp_path = require_previous_output(OUTPUTS_DIR / "06_view_posture.json", "06_view_posture_classification.py")
    pose = load_json(pose_path)
    vp = load_json(vp_path)
    keypoints = pose["keypoints"]
    scores = pose["scores"]
    floor = pose["confidence_floor"]

    ratios: dict[str, float | None] = {}
    confidences: dict[str, float | None] = {}
    refusals: dict[str, str] = {}

    neck, tail = pt(keypoints, "neck"), pt(keypoints, "tail_base")
    body_length = dist(neck, tail)
    gate_reason = f"view {vp['view']}, posture {vp['posture']}"
    gate_open = vp["usable_for_measurement"]

    if not gate_open:
        for name in ("height_to_length_ratio", "leg_to_body_ratio", "chest_depth_ratio"):
            refusals[name] = gate_reason
    else:
        shoulder = lateral_mid(keypoints, scores, "left_shoulder", "right_shoulder", floor)
        mask_path = OUTPUTS_DIR / "03_mask.png"
        ground_y = None
        if mask_path.exists():
            mask = load_bgr(mask_path)[:, :, 0] > 0
            rows = np.where(mask.any(axis=1))[0]
            if len(rows):
                ground_y = float(rows.max())
        if ground_y is None:
            paw_names = ["left_front_paw", "right_front_paw", "left_back_paw", "right_back_paw"]
            paw_ys = [pt(keypoints, n)[1] for n in paw_names if confident(scores, n, floor)]
            ground_y = max(paw_ys) if paw_ys else None

        if shoulder is None or ground_y is None:
            refusals["height_to_length_ratio"] = "shoulder or ground not confidently located"
        else:
            height = ground_y - shoulder[1]
            if height <= 0:
                refusals["height_to_length_ratio"] = "computed height <= 0 (shoulder below ground?)"
            else:
                ratios["height_to_length_ratio"] = round(height / body_length, 4)
                confidences["height_to_length_ratio"] = round(shoulder[2], 4)

        front_paw = lateral_mid(keypoints, scores, "left_front_paw", "right_front_paw", floor)
        if vp["posture"] != "STANDING" or shoulder is None or front_paw is None:
            refusals["leg_to_body_ratio"] = "not standing, or shoulder/front paw not confidently located"
        else:
            leg = dist(shoulder[:2], front_paw[:2])
            ratios["leg_to_body_ratio"] = round(leg / body_length, 4)
            confidences["leg_to_body_ratio"] = round((shoulder[2] + front_paw[2]) / 2, 4)

        elbow = lateral_mid(keypoints, scores, "left_elbow", "right_elbow", floor)
        if shoulder is None or elbow is None:
            refusals["chest_depth_ratio"] = "shoulder or elbow not confidently located"
        else:
            depth = abs(elbow[1] - shoulder[1])
            ratios["chest_depth_ratio"] = round(depth / body_length, 4)
            confidences["chest_depth_ratio"] = round((shoulder[2] + elbow[2]) / 2, 4)

    # Muzzle:skull only needs head landmarks - it survives views the other three can't.
    eye_mid = lateral_mid(keypoints, scores, "left_eye", "right_eye", floor)
    if not confident(scores, "nose", floor) or eye_mid is None:
        refusals["muzzle_to_skull_ratio"] = "nose or eyes not confidently located"
    else:
        nose = pt(keypoints, "nose")
        muzzle = dist(nose, eye_mid[:2])
        skull = dist(eye_mid[:2], neck)
        if skull <= 0:
            refusals["muzzle_to_skull_ratio"] = "computed skull length <= 0"
        else:
            ratios["muzzle_to_skull_ratio"] = round(muzzle / skull, 4)
            confidences["muzzle_to_skull_ratio"] = round((sc(scores, "nose") + eye_mid[2]) / 2, 4)

    refusals["ear_to_head_ratio"] = "AP-10K has no ear keypoints; measured from contour in stage 10"
    refusals["tail_to_body_ratio"] = "AP-10K has no tail tip; measured from contour in stage 10"

    asymmetry_findings = []
    if vp["view"] in ("PROFILE_LEFT", "PROFILE_RIGHT") and vp["posture"] == "STANDING":
        for left, right in LATERAL_PAIRS:
            if confident(scores, left, floor) and confident(scores, right, floor):
                drop = abs(pt(keypoints, left)[1] - pt(keypoints, right)[1]) / body_length
                if drop > ASYMMETRY_THRESHOLD:
                    asymmetry_findings.append(f"{left}/{right} vertical offset {drop:.3f} (threshold {ASYMMETRY_THRESHOLD})")
    else:
        asymmetry_findings = None  # not computed - not a profile+standing view

    print("Ratios (None = genuinely unmeasured, not zero):")
    for name in ("height_to_length_ratio", "leg_to_body_ratio", "chest_depth_ratio", "muzzle_to_skull_ratio", "ear_to_head_ratio", "tail_to_body_ratio"):
        if name in ratios:
            print(f"  {name:<24s} = {ratios[name]}  (confidence {confidences[name]})")
        else:
            print(f"  {name:<24s} = None  ({refusals[name]})")

    print(f"\nAsymmetry findings: {asymmetry_findings if asymmetry_findings is not None else 'not computed (view is not profile+standing)'}")

    save_json(
        OUTPUTS_DIR / "07_morphometrics.json",
        {"ratios": ratios, "confidences": confidences, "refusals": refusals, "asymmetry_findings": asymmetry_findings, "body_length_px": round(body_length, 2)},
    )

    lines = [f"{k}={v}" for k, v in ratios.items()] or ["no ratios measurable from this view/posture"]
    annotated = put_lines(load_bgr(OUTPUTS_DIR / "05_pose.jpg"), lines[:4])
    save_image(OUTPUTS_DIR / "07_morphometrics.jpg", annotated)
    print("\nWrote outputs/07_morphometrics.json and outputs/07_morphometrics.jpg")


if __name__ == "__main__":
    main()
