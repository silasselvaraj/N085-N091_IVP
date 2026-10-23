"""Stage 1 of 12 — frame-level quality gate.

Before anything else runs, the real pipeline asks a cheap question first:
is this photo even readable? Blur, exposure, contrast, noise and colour
cast are all measured here on a resized "analysis copy" of the upload, and
a photo that fails badly enough is rejected before a single GPU model is
spent on it.

This script also produces `outputs/analysis_copy.jpg` — the resized frame
every later stage in this showcase works from, exactly like the real
pipeline never re-reads the original upload after ingestion.

Run: python 01_quality_check.py
"""

from __future__ import annotations

import _quality
from _common import (
    OUTPUTS_DIR,
    SAMPLE_IMAGE,
    analysis_copy_path,
    load_bgr,
    put_lines,
    resize_to_analysis_copy,
    save_image,
    save_json,
)


def main() -> None:
    print(f"Loading {SAMPLE_IMAGE.relative_to(SAMPLE_IMAGE.parents[1])}")
    original = load_bgr(SAMPLE_IMAGE)
    print(f"  original size: {original.shape[1]}x{original.shape[0]}")

    analysis = resize_to_analysis_copy(original)
    save_image(analysis_copy_path(), analysis)
    print(f"  analysis copy: {analysis.shape[1]}x{analysis.shape[0]} -> {analysis_copy_path().name}")

    m = _quality.measure(analysis)
    verdict = _quality.assess(m)

    print("\nMeasurements:")
    print(f"  edge_sharpness       = {m.edge_sharpness:.3f}  (decides BLURRY - 99th-pct Sobel gradient / contrast)")
    print(f"  blur_laplacian (norm)= {m.blur_laplacian_normalised:.3f}  (diagnostic only, not used for the verdict)")
    print(f"  tenengrad (norm)     = {m.tenengrad_normalised:.3f}  (diagnostic only, not used for the verdict)")
    print(f"  mean_luminance       = {m.mean_luminance:.2f}")
    print(f"  shadow_clipping      = {m.shadow_clipping:.4f}")
    print(f"  highlight_clipping   = {m.highlight_clipping:.4f}")
    print(f"  contrast (grey std)  = {m.contrast:.2f}")
    print(f"  noise_sigma          = {m.noise_sigma:.3f}")
    print(f"  colour_cast          = {m.colour_cast:.3f}  (saturation={m.saturation:.2f})")

    print("\nVerdict:")
    print(f"  usable     = {verdict.usable}")
    print(f"  degraded   = {verdict.degraded}")
    print(f"  provisional= {verdict.provisional}  (a marginal reject deferred to stage 4's subject re-check)")
    print(f"  reasons    = {verdict.reported_reasons or '(none)'}")

    annotated = put_lines(
        analysis,
        [
            f"usable={verdict.usable} degraded={verdict.degraded}",
            f"edge_sharpness={m.edge_sharpness:.2f}  contrast={m.contrast:.1f}",
            f"reasons={verdict.reported_reasons or 'none'}",
        ],
    )
    save_image(OUTPUTS_DIR / "01_quality_check.jpg", annotated)

    save_json(
        OUTPUTS_DIR / "01_quality_check.json",
        {
            "measurements": vars(m),
            "verdict": {
                "usable": verdict.usable,
                "degraded": verdict.degraded,
                "provisional": verdict.provisional,
                "reject_reasons": verdict.reject_reasons,
                "degrade_reasons": verdict.degrade_reasons,
                "reported_reasons": verdict.reported_reasons,
            },
        },
    )
    print("\nWrote outputs/01_quality_check.json and outputs/01_quality_check.jpg")


if __name__ == "__main__":
    main()
