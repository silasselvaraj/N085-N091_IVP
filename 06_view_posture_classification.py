"""Stage 6 of 12 — view and posture classification.

Purely geometric: no learned model, just ratios between the 17 keypoints
from stage 5. This classification exists to *gate* what stage 7 is allowed
to measure — a sitting or three-quarter-view dog cannot honestly yield a
body-proportion ratio, so nothing downstream is allowed to try.

This is also the one stage in the whole pipeline documented as genuinely
unreliable on real photos (view 14.3%, posture 25.0% accuracy against a
24-image hand-labelled set - see PROJECT_STATE.md SS3.2), which is why
Part 4 shipped deliberately deferred rather than re-tuned: three independent
geometric signals were tried and all three overlap heavily across true view
classes on real photos. This script reproduces the same rules anyway,
because the point of the showcase is to demonstrate what was actually
built, including the parts still awaiting a better signal - not to hide
them behind a cherry-picked demo photo.

Run: python 06_view_posture_classification.py   (needs 05_pose_estimation.py to have run first)
"""

from __future__ import annotations

import math

from _common import OUTPUTS_DIR, load_bgr, load_json, put_lines, require_previous_output, save_image, save_json
from _pose_constants import CONFIDENCE_FLOOR, KP, MIN_CONFIDENT_KEYPOINTS


def pt(keypoints, name):
    return keypoints[KP[name]]


def conf(scores, name):
    return scores[KP[name]]


def is_confident(scores, name):
    return conf(scores, name) >= CONFIDENCE_FLOOR


def lateral_mid(keypoints, scores, left_name, right_name):
    left_ok, right_ok = is_confident(scores, left_name), is_confident(scores, right_name)
    if left_ok and right_ok:
        lx, ly = pt(keypoints, left_name)
        rx, ry = pt(keypoints, right_name)
        return ((lx + rx) / 2, (ly + ry) / 2)
    if left_ok:
        return tuple(pt(keypoints, left_name))
    if right_ok:
        return tuple(pt(keypoints, right_name))
    return None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify_view(keypoints, scores):
    neck, tail = pt(keypoints, "neck"), pt(keypoints, "tail_base")
    span = dist(neck, tail)
    if span < 1e-6:
        return "UNKNOWN", 0.0

    spreads = []
    if is_confident(scores, "left_shoulder") and is_confident(scores, "right_shoulder"):
        spreads.append(abs(pt(keypoints, "left_shoulder")[0] - pt(keypoints, "right_shoulder")[0]))
    if is_confident(scores, "left_hip") and is_confident(scores, "right_hip"):
        spreads.append(abs(pt(keypoints, "left_hip")[0] - pt(keypoints, "right_hip")[0]))
    if not spreads:
        return "UNKNOWN", 0.0
    spread = (sum(spreads) / len(spreads)) / span

    nose_ok = is_confident(scores, "nose")
    if spread > 0.55:
        return ("FRONTAL", 0.75) if nose_ok else ("REAR", 0.6)
    if spread > 0.25:
        return "THREE_QUARTER", 0.7
    if nose_ok:
        nose = pt(keypoints, "nose")
        return ("PROFILE_LEFT", 0.85) if nose[0] < tail[0] else ("PROFILE_RIGHT", 0.85)
    return "THREE_QUARTER", 0.4


def classify_posture(keypoints, scores):
    shoulder = lateral_mid(keypoints, scores, "left_shoulder", "right_shoulder")
    hip = lateral_mid(keypoints, scores, "left_hip", "right_hip")
    if shoulder is None or hip is None:
        return "UNKNOWN", 0.0

    neck, tail = pt(keypoints, "neck"), pt(keypoints, "tail_base")
    span = dist(neck, tail) if dist(neck, tail) > 1e-6 else dist(shoulder, hip)
    if span < 1e-6:
        return "UNKNOWN", 0.0

    tilt = abs(shoulder[1] - hip[1]) / max(abs(shoulder[0] - hip[0]), 1e-6)

    front_paw = lateral_mid(keypoints, scores, "left_front_paw", "right_front_paw")
    back_paw = lateral_mid(keypoints, scores, "left_back_paw", "right_back_paw")

    if front_paw is not None and back_paw is not None:
        front_drop = (front_paw[1] - shoulder[1]) / span
        rear_drop = (back_paw[1] - hip[1]) / span
        if front_drop < 0.25 and rear_drop < 0.25:
            return "LYING", 0.75
        if rear_drop < 0.45 <= front_drop:
            return "SITTING", 0.7
        if front_drop >= 0.45 and rear_drop >= 0.45:
            return ("RUNNING", 0.45) if tilt > 0.5 else ("STANDING", 0.8)
        # falls between the named bands - treat like the incomplete-legs fallback
        return ("STANDING", 0.5) if tilt < 0.35 else ("UNKNOWN", 0.2)

    return ("STANDING", 0.5) if tilt < 0.35 else ("UNKNOWN", 0.2)


def main() -> None:
    pose_path = require_previous_output(OUTPUTS_DIR / "05_pose.json", "05_pose_estimation.py")
    pose = load_json(pose_path)
    keypoints = pose["keypoints"]
    scores = pose["scores"]

    if pose["confident_keypoint_count"] < MIN_CONFIDENT_KEYPOINTS:
        print(f"Only {pose['confident_keypoint_count']} confident keypoints (need {MIN_CONFIDENT_KEYPOINTS}) - refusing to classify.")
        result = {"view": "UNKNOWN", "view_confidence": 0.0, "posture": "UNKNOWN", "posture_confidence": 0.0,
                   "usable_for_measurement": False, "confidence": 0.0}
    else:
        view, view_conf = classify_view(keypoints, scores)
        posture, posture_conf = classify_posture(keypoints, scores)
        usable = view in ("PROFILE_LEFT", "PROFILE_RIGHT") and posture == "STANDING"
        result = {
            "view": view, "view_confidence": view_conf, "posture": posture, "posture_confidence": posture_conf,
            "usable_for_measurement": usable, "confidence": min(view_conf, posture_conf),
        }
        print(f"view     = {view}  (confidence {view_conf:.2f})")
        print(f"posture  = {posture}  (confidence {posture_conf:.2f})")
        print(f"usable_for_measurement = {usable}  (needs profile view AND standing posture)")
        if not usable:
            print("\n-> Stage 7 will refuse to compute body-proportion ratios because of this gate.")
            print("   This is the documented, deliberate behaviour, not a bug: PROJECT_STATE.md SS3.2")
            print("   measured this gate passing on 0/31 real photos and traced it to a genuine")
            print("   signal problem in view/posture classification, not a threshold to retune.")

    save_json(OUTPUTS_DIR / "06_view_posture.json", result)

    crop = load_bgr(OUTPUTS_DIR / "05_pose.jpg")
    annotated = put_lines(crop, [f"view={result['view']}", f"posture={result['posture']}", f"usable_for_measurement={result['usable_for_measurement']}"])
    save_image(OUTPUTS_DIR / "06_view_posture.jpg", annotated)
    print("\nWrote outputs/06_view_posture.json and outputs/06_view_posture.jpg")


if __name__ == "__main__":
    main()
