"""Stage 11 of 12 (bonus) — monocular depth, built but disabled in production.

This stage runs for real here - Depth Anything V2 Small genuinely loads and
produces a depth map - but `depth_enabled = False` in the live application,
and it's worth explaining exactly why rather than pretending the stage
doesn't exist.

Monocular depth models produce *relative*, scale-invariant depth: nothing
in the output says "55cm", only "nearer than that other pixel". Without
metric scale you cannot recover the one number that would actually
discriminate a Miniature from a Standard Poodle. The classifier's own RGB
features already encode shape cues better than a noisy relative depth map
does. Two narrower uses remain defensible - foreshortening detection as a
trust signal for stage 7's morphometrics (implemented below), and mask-edge
corroboration (not implemented, currently unused) - and the roadmap set an
explicit bar before wiring either into the live classifier: >=1.0 percentage
point of val top-1 improvement. Part 6's fusion experiment measured it
directly and got -0.13% (86.66% vs the non-fusion baseline's 86.79%) - a
real, honestly-measured negative result, so it stayed off.

Weights: ivp_showcase/weights/depth_anything/ (local copy of
depth-anything/Depth-Anything-V2-Small-hf).

Run: python 11_depth_estimation.py   (needs 03_segmentation.py and 05_pose_estimation.py)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from _common import (
    OUTPUTS_DIR,
    WEIGHTS_DIR,
    analysis_copy_path,
    load_bgr,
    load_json,
    load_rgb,
    require_previous_output,
    save_image,
    save_json,
)
from _pose_constants import KP

DEPTH_DIR = WEIGHTS_DIR / "depth_anything"
FORESHORTENING_THRESHOLD = 0.35


def main() -> None:
    image_rgb = load_rgb(require_previous_output(analysis_copy_path(), "01_quality_check.py"))
    mask = load_bgr(require_previous_output(OUTPUTS_DIR / "03_mask.png", "03_segmentation.py"))[:, :, 0] > 0
    pose = load_json(require_previous_output(OUTPUTS_DIR / "05_pose.json", "05_pose_estimation.py"))

    print(f"Loading Depth Anything V2 Small from {DEPTH_DIR.relative_to(DEPTH_DIR.parents[1])} (local weights only)")
    processor = AutoImageProcessor.from_pretrained(str(DEPTH_DIR), local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(str(DEPTH_DIR), local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  device: {device}")

    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    h, w = image_rgb.shape[:2]
    depth = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    )[0, 0].cpu().numpy()

    depth_norm = (depth - depth.min()) / max(1e-6, depth.max() - depth.min())
    colour = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    save_image(OUTPUTS_DIR / "11_depth.jpg", colour)

    keypoints = pose["keypoints"]
    neck, tail = keypoints[KP["neck"]], keypoints[KP["tail_base"]]
    n_samples = 50
    xs = np.linspace(neck[0], tail[0], n_samples)
    ys = np.linspace(neck[1], tail[1], n_samples)
    xs_i, ys_i = np.clip(xs.astype(int), 0, w - 1), np.clip(ys.astype(int), 0, h - 1)
    on_mask = mask[ys_i, xs_i]
    axis_samples = depth[ys_i[on_mask], xs_i[on_mask]]

    mask_depth = depth[mask]
    if len(axis_samples) < 5 or len(mask_depth) < 5:
        print("\nToo few on-mask samples along the neck-tail axis to score foreshortening.")
        result = {"foreshortening_score": None, "measurements_trustworthy": None, "reason": "insufficient on-mask samples"}
    else:
        axis_range = float(np.percentile(axis_samples, 90) - np.percentile(axis_samples, 10))
        body_spread = float(np.percentile(mask_depth, 95) - np.percentile(mask_depth, 5))
        score = float(np.clip(axis_range / max(body_spread, 1e-6), 0, 1))
        trustworthy = score < FORESHORTENING_THRESHOLD
        print(f"\nforeshortening_score = {score:.4f}  (axis_range={axis_range:.3f}, body_spread={body_spread:.3f})")
        print(f"measurements_trustworthy = {trustworthy}  (threshold {FORESHORTENING_THRESHOLD})")
        result = {"foreshortening_score": round(score, 4), "measurements_trustworthy": trustworthy, "axis_range": round(axis_range, 4), "body_spread": round(body_spread, 4)}

    print("\nDisabled in production (depth_enabled=False) - see this script's own docstring for why:")
    print("  relative depth has no metric scale; RGB shape cues already beat it; the Part 6")
    print("  fusion experiment measured -0.13% val top-1 when tried, below the +1.0pp bar to ship.")

    save_json(OUTPUTS_DIR / "11_depth.json", {**result, "depth_enabled_in_production": False})
    print("\nWrote outputs/11_depth.json and outputs/11_depth.jpg")


if __name__ == "__main__":
    main()
