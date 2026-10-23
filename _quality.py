"""Frame/subject quality metrics — shared by 01_quality_check.py and
04_subject_quality_recheck.py, because in the real pipeline they are
genuinely the same module (app/imaging/metrics.py + thresholds.py) called
twice on two different crops, not two different implementations. Splitting
it in two here would mean maintaining the same thresholds twice and risking
them drifting apart — exactly the kind of duplication CLAUDE.md's own
conventions warn against.

Every threshold below was calibrated (2026-08-28) against a labelled fixture
set in the real project, not guessed — see backend/scripts/calibrate_thresholds.py.

The one metric that decides BLURRY is `edge_sharpness`: the 99th-percentile
Sobel gradient magnitude, contrast-normalised. Not Laplacian variance, and
not mean gradient energy (Tenengrad) — both of those measure *texture
density*, so they call a sharp photo of a smooth-coated dog on a plain wall
"out of focus". Laplacian variance and Tenengrad are still computed below,
but only as diagnostics; they do not feed the verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# --- calibrated thresholds (measured, not guessed) --------------------------

SHORT_EDGE_REJECT, SHORT_EDGE_DEGRADE = 160, 400

SHARPNESS_REJECT, SHARPNESS_DEGRADE, SHARPNESS_HARD_FLOOR = 3.9, 5.7, 2.0

LUM_DARK_REJECT, LUM_DARK_DEGRADE, LUM_DARK_HARD_FLOOR = 22.0, 55.0, 12.0
LUM_BRIGHT_REJECT, LUM_BRIGHT_DEGRADE, LUM_BRIGHT_HARD_FLOOR = 236.0, 205.0, 246.0

SHADOW_CLIP_REJECT, SHADOW_CLIP_DEGRADE, SHADOW_CLIP_HARD_FLOOR = 0.60, 0.28, 0.80
HIGHLIGHT_CLIP_REJECT, HIGHLIGHT_CLIP_DEGRADE, HIGHLIGHT_CLIP_HARD_FLOOR = 0.50, 0.22, 0.70

CONTRAST_REJECT, CONTRAST_DEGRADE, CONTRAST_HARD_FLOOR = 5.0, 22.0, 3.0

NOISE_DEGRADE = 11.0  # never rejects on its own

COLOUR_CAST_DEGRADE = 24.0  # never rejects; also needs SATURATION_GREYSCALE
SATURATION_GREYSCALE = 12.0

# Priority order for which reason(s) to surface, worst-first; at most 2 shown.
REASON_PRIORITY = [
    "TOO_SMALL",
    "UNDEREXPOSED",
    "OVEREXPOSED",
    "LOW_CONTRAST",
    "BLURRY",
    "NOISY",
    "COLOUR_CAST",
]


@dataclass
class QualityMeasurement:
    width: int
    height: int
    short_edge: int
    edge_sharpness: float
    blur_laplacian_normalised: float
    tenengrad_normalised: float
    mean_luminance: float
    shadow_clipping: float
    highlight_clipping: float
    contrast: float
    noise_sigma: float
    colour_cast: float
    saturation: float


@dataclass
class QualityVerdict:
    usable: bool
    degraded: bool
    provisional: bool
    reject_reasons: list[str] = field(default_factory=list)
    degrade_reasons: list[str] = field(default_factory=list)
    reported_reasons: list[str] = field(default_factory=list)


def measure(image_bgr: np.ndarray) -> QualityMeasurement:
    h, w = image_bgr.shape[:2]
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    contrast = float(grey.std())
    edge_sharpness = float(np.percentile(magnitude, 99)) / max(contrast, 3.0)

    laplacian_var = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    tenengrad = float(np.mean(gx**2 + gy**2))
    norm = max(contrast, 3.0) ** 2
    blur_laplacian_normalised = laplacian_var / norm
    tenengrad_normalised = tenengrad / norm

    mean_luminance = float(grey.mean())
    shadow_clipping = float(np.mean(grey <= 8))
    highlight_clipping = float(np.mean(grey >= 247))

    # Immerkaer's fast noise estimator.
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    response = cv2.filter2D(grey, -1, kernel)
    noise_sigma = float(
        np.sum(np.abs(response)) * math.sqrt(0.5 * math.pi) / (6 * max(1, w - 2) * max(1, h - 2))
    )

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    mean_a = float(lab[:, :, 1].mean()) - 128.0
    mean_b = float(lab[:, :, 2].mean()) - 128.0
    colour_cast = math.hypot(mean_a, mean_b)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].mean())

    return QualityMeasurement(
        width=w,
        height=h,
        short_edge=min(w, h),
        edge_sharpness=edge_sharpness,
        blur_laplacian_normalised=blur_laplacian_normalised,
        tenengrad_normalised=tenengrad_normalised,
        mean_luminance=mean_luminance,
        shadow_clipping=shadow_clipping,
        highlight_clipping=highlight_clipping,
        contrast=contrast,
        noise_sigma=noise_sigma,
        colour_cast=colour_cast,
        saturation=saturation,
    )


def _hard_floor_breached(reason: str, m: QualityMeasurement) -> bool:
    if reason == "TOO_SMALL":
        return True  # never provisional, per app/imaging/thresholds.py
    if reason == "BLURRY":
        return m.edge_sharpness < SHARPNESS_HARD_FLOOR
    if reason == "UNDEREXPOSED":
        return m.mean_luminance < LUM_DARK_HARD_FLOOR or m.shadow_clipping > SHADOW_CLIP_HARD_FLOOR
    if reason == "OVEREXPOSED":
        return m.mean_luminance > LUM_BRIGHT_HARD_FLOOR or m.highlight_clipping > HIGHLIGHT_CLIP_HARD_FLOOR
    if reason == "LOW_CONTRAST":
        return m.contrast < CONTRAST_HARD_FLOOR
    return False


def assess(m: QualityMeasurement, *, exclude_reasons: tuple[str, ...] = ()) -> QualityVerdict:
    """`exclude_reasons` lets 04_subject_quality_recheck.py drop TOO_SMALL,
    which means something different on an already-cropped subject than on
    a full frame — see that script for why."""
    rejects: list[str] = []
    degrades: list[str] = []

    if m.short_edge < SHORT_EDGE_REJECT:
        rejects.append("TOO_SMALL")
    elif m.short_edge < SHORT_EDGE_DEGRADE:
        degrades.append("TOO_SMALL")

    if m.edge_sharpness < SHARPNESS_REJECT:
        rejects.append("BLURRY")
    elif m.edge_sharpness < SHARPNESS_DEGRADE:
        degrades.append("BLURRY")

    if m.mean_luminance < LUM_DARK_REJECT or m.shadow_clipping > SHADOW_CLIP_REJECT:
        rejects.append("UNDEREXPOSED")
    elif m.mean_luminance < LUM_DARK_DEGRADE or m.shadow_clipping > SHADOW_CLIP_DEGRADE:
        degrades.append("UNDEREXPOSED")
    elif m.mean_luminance > LUM_BRIGHT_REJECT or m.highlight_clipping > HIGHLIGHT_CLIP_REJECT:
        rejects.append("OVEREXPOSED")
    elif m.mean_luminance > LUM_BRIGHT_DEGRADE or m.highlight_clipping > HIGHLIGHT_CLIP_DEGRADE:
        degrades.append("OVEREXPOSED")

    if m.contrast < CONTRAST_REJECT:
        rejects.append("LOW_CONTRAST")
    elif m.contrast < CONTRAST_DEGRADE:
        degrades.append("LOW_CONTRAST")

    if m.noise_sigma > NOISE_DEGRADE:
        degrades.append("NOISY")

    if m.colour_cast > COLOUR_CAST_DEGRADE and m.saturation > SATURATION_GREYSCALE:
        degrades.append("COLOUR_CAST")

    rejects = [r for r in rejects if r not in exclude_reasons]
    degrades = [r for r in degrades if r not in exclude_reasons]

    usable = not rejects
    degraded = bool(degrades) and usable
    provisional = bool(rejects) and not any(_hard_floor_breached(r, m) for r in rejects)

    all_reasons = rejects + degrades
    reported = [r for r in REASON_PRIORITY if r in all_reasons][:2]

    return QualityVerdict(
        usable=usable,
        degraded=degraded,
        provisional=provisional,
        reject_reasons=rejects,
        degrade_reasons=degrades,
        reported_reasons=reported,
    )
