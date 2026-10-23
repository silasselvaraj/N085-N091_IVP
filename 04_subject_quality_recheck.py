"""Stage 4 of 12 — subject-region quality re-check.

Stage 1 measured the whole frame, which can be unfair: a dark background
behind a perfectly well-lit dog can make the *frame* look underexposed even
though the *subject* is fine. So the exact same quality metrics from stage 1
are measured again here, on nothing but the cropped dog region from stage 3
- and only this second measurement can actually reject the photo. A marginal
stage-1 reject is "provisional" for exactly this reason: it waits to be
confirmed or overturned here before the pipeline gives up on the photo.

`TOO_SMALL` is dropped from the fatal-reason set here on purpose — on a
crop that's already just the dog, "too small" means something different
(and less actionable) than it does on a full frame.

Run: python 04_subject_quality_recheck.py   (needs 03_segmentation.py to have run first)
"""

from __future__ import annotations

import _quality
from _common import OUTPUTS_DIR, load_bgr, load_json, put_lines, require_previous_output, save_image, save_json


def main() -> None:
    seg_path = require_previous_output(OUTPUTS_DIR / "03_segmentation.json", "03_segmentation.py")
    seg = load_json(seg_path)
    crop_path = OUTPUTS_DIR / "03_dog_crop.jpg"
    if not seg.get("plausible") or not crop_path.exists():
        print("No usable crop from stage 3 - nothing to re-check.")
        return

    frame_verdict = load_json(OUTPUTS_DIR / "01_quality_check.json")["verdict"]
    crop = load_bgr(crop_path)
    print(f"Re-measuring quality on the {crop.shape[1]}x{crop.shape[0]} dog crop (not the full frame)")

    m = _quality.measure(crop)
    verdict = _quality.assess(m, exclude_reasons=("TOO_SMALL",))

    print("\nSubject-region measurements:")
    print(f"  edge_sharpness = {m.edge_sharpness:.3f}")
    print(f"  mean_luminance = {m.mean_luminance:.2f}  contrast = {m.contrast:.2f}")
    print(f"  noise_sigma    = {m.noise_sigma:.3f}  colour_cast = {m.colour_cast:.3f}")

    print("\nSubject verdict:")
    print(f"  usable   = {verdict.usable}")
    print(f"  degraded = {verdict.degraded}")
    print(f"  reasons  = {verdict.reported_reasons or '(none)'}")

    print(f"\nStage 1 (frame) said provisional={frame_verdict['provisional']}, reasons={frame_verdict['reported_reasons']}")
    if frame_verdict["provisional"] and verdict.usable:
        print("-> Subject re-check OVERTURNED the frame-level concern: the dog itself is fine.")
    elif not frame_verdict["usable"] and not verdict.usable:
        print("-> Subject re-check CONFIRMS the frame-level rejection.")

    annotated = put_lines(crop, [f"usable={verdict.usable} degraded={verdict.degraded}", f"reasons={verdict.reported_reasons or 'none'}"])
    save_image(OUTPUTS_DIR / "04_subject_quality.jpg", annotated)
    save_json(
        OUTPUTS_DIR / "04_subject_quality.json",
        {
            "measurements": vars(m),
            "verdict": {
                "usable": verdict.usable,
                "degraded": verdict.degraded,
                "reject_reasons": verdict.reject_reasons,
                "degrade_reasons": verdict.degrade_reasons,
                "reported_reasons": verdict.reported_reasons,
            },
        },
    )
    print("\nWrote outputs/04_subject_quality.json and outputs/04_subject_quality.jpg")


if __name__ == "__main__":
    main()
