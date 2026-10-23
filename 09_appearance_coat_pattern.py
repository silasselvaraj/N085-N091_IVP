"""Stage 9 of 12 — coat length, texture and pattern.

Coat length and texture come from the mask's own boundary and interior -
length from how *rough* the silhouette edge is (a long-coated dog's outline
is fuzzy at pixel scale; a short-coated dog's is crisp), texture from a
local binary pattern histogram's flatness/entropy inside the mask, not from
colour.

Pattern is a spatial-layout question over stage 8's *merged* colour
regions, read in a body-relative frame (0=topline, 1=feet down the y-axis;
0..1 across the mask's width): brindle looks for alternating colour runs
across a band of rows, saddle for a dark colour concentrated dorsally,
masked for a dark colour concentrated at the face corners, ticked for a
small, spatially scattered minority colour. First match wins; a single
significant colour falls back to SOLID.

Run: python 09_appearance_coat_pattern.py   (needs 03_segmentation.py and 08_appearance_colour.py)
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from _common import (
    OUTPUTS_DIR,
    analysis_copy_path,
    load_bgr,
    load_json,
    put_lines,
    require_previous_output,
    save_image,
    save_json,
)

MIN_BOUNDARY_PIXELS = 200
MIN_INTERIOR_PIXELS = 100
SHORT_MAX, MEDIUM_MAX = 0.30, 0.55

LBP_RADIUS, LBP_POINTS = 3, 24
MIN_TEXTURE_INTERIOR_PIXELS = 1000
MIN_LBP_VALUES = 500

TRIM_COVERAGE = 0.10
MIN_STRIPE_PX = 4
BRINDLE_RUNS_PER_100PX = 8.0
SADDLE_DORSAL_SHARE = 0.55
SADDLE_CONCENTRATION = 0.6
FACE_MASK_SHARE = 0.45
TICKED_MIN_COVERAGE, TICKED_MAX_COVERAGE = 0.03, 0.2
TICKED_MIN_PIXELS = 200
TICKED_SPREAD = 0.55


def classify_length_and_texture(mask: np.ndarray, grey: np.ndarray) -> dict:
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2) > 0
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=2) > 0
    band = dilated & ~eroded
    interior = eroded

    if int(band.sum()) < MIN_BOUNDARY_PIXELS or int(interior.sum()) < MIN_INTERIOR_PIXELS:
        return {"length": "UNKNOWN", "length_confidence": 0.0, "texture": "UNKNOWN", "texture_confidence": 0.0, "reason": "insufficient boundary/interior pixels"}

    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    reference = np.percentile(magnitude[mask], 99) if mask.any() else 1.0
    band_mean = magnitude[band].mean()
    roughness = float(np.clip(band_mean / max(reference, 1e-6), 0, 4))

    length = "SHORT" if roughness < SHORT_MAX else ("MEDIUM" if roughness < MEDIUM_MAX else "LONG")
    boundary_coverage = int(band.sum())
    overall_confidence = float(np.clip(boundary_coverage / 4000, 0, 1))

    texture, texture_conf = "UNKNOWN", 0.0
    if overall_confidence > 0.4 and int(interior.sum()) >= MIN_TEXTURE_INTERIOR_PIXELS:
        quantised = (grey / 255.0 * 255).astype(np.uint8)
        lbp = local_binary_pattern(quantised, LBP_POINTS, LBP_RADIUS, method="uniform")
        values = lbp[interior]
        if len(values) >= MIN_LBP_VALUES:
            hist, _ = np.histogram(values, bins=26, range=(0, 26), density=True)
            flat_share = float(hist[-1] + hist[0])
            nonzero = hist[hist > 0]
            entropy = float(-np.sum(nonzero * np.log2(nonzero)))
            if flat_share > 0.45:
                texture, texture_conf = "SMOOTH", 0.6
            elif entropy > 3.6:
                texture, texture_conf = "CURLY", 0.45
            elif entropy > 3.2:
                texture, texture_conf = "WAVY", 0.4
            else:
                texture, texture_conf = "UNKNOWN", 0.2

    return {
        "length": length, "length_confidence": round(overall_confidence, 4), "roughness": round(roughness, 4),
        "texture": texture, "texture_confidence": texture_conf,
    }


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _looks_brindle(label_image: np.ndarray, top: int, height: int, left: int, right: int) -> bool:
    """Sample rows across the mid-body band and count alternating colour runs per row."""
    sample_ys = [top + int(f * height) for f in np.linspace(0.3, 0.7, 7)]
    striped_rows, counted_rows = 0, 0
    for y in sample_ys:
        row = label_image[y, left : right + 1]
        if len(row) < 10:
            continue
        runs, prev, run_len = 0, None, 0
        for v in row:
            if v == prev and v != -1:
                run_len += 1
            else:
                if prev is not None and prev != -1 and run_len >= MIN_STRIPE_PX:
                    runs += 1
                prev, run_len = v, 1
        if prev is not None and prev != -1 and run_len >= MIN_STRIPE_PX:
            runs += 1
        counted_rows += 1
        if runs / (len(row) / 100) > BRINDLE_RUNS_PER_100PX:
            striped_rows += 1
    return counted_rows >= 3 and striped_rows >= counted_rows - 1


def classify_pattern(mask: np.ndarray, coords: np.ndarray, labels: np.ndarray, region_names: list[str], region_by_name: dict) -> dict:
    top, bottom, left, right = _mask_bounds(mask)
    height, width = max(1, bottom - top), max(1, right - left)

    label_image = np.full(mask.shape, -1, dtype=np.int32)
    label_image[coords[:, 0], coords[:, 1]] = labels

    coverage_by_label = {name: float(np.mean(labels == i)) for i, name in enumerate(region_names)}
    significant = {i: name for i, name in enumerate(region_names) if coverage_by_label[name] >= TRIM_COVERAGE}

    if _looks_brindle(label_image, top, height, left, right):
        return {"pattern": "BRINDLE", "confidence": 0.6, "significant_colours": list(significant.values())}

    norm_y = (coords[:, 0].astype(np.float64) - top) / height
    norm_x = (coords[:, 1].astype(np.float64) - left) / width

    if significant:
        # "Darkest" = lowest measured CIELAB L from stage 8's own region summary.
        darkest_i = min(significant, key=lambda i: region_by_name[region_names[i]]["lab"][0])
        dark_member = labels == darkest_i
        dark_px = int(dark_member.sum())

        dorsal = norm_y < 0.45
        dorsal_px = int(dorsal.sum())
        dark_and_dorsal = int((dark_member & dorsal).sum())
        share_of_back = dark_and_dorsal / dorsal_px if dorsal_px else 0.0
        concentration = dark_and_dorsal / dark_px if dark_px else 0.0
        if share_of_back > SADDLE_DORSAL_SHARE and concentration > SADDLE_CONCENTRATION:
            return {"pattern": "SADDLE", "confidence": 0.55, "colour": region_names[darkest_i], "share_of_back": round(share_of_back, 3), "concentration": round(concentration, 3)}

        head_band = norm_y < 0.35
        for corner_name, corner in (("left", head_band & (norm_x < 0.25)), ("right", head_band & (norm_x > 0.75))):
            if int(corner.sum()) < 50 or dark_px == 0:
                continue
            share = int((dark_member & corner).sum()) / dark_px
            if share > FACE_MASK_SHARE:
                return {"pattern": "MASKED", "confidence": 0.5, "colour": region_names[darkest_i], "corner": corner_name, "share": round(share, 3)}

        for i, name in significant.items():
            cov = coverage_by_label[name]
            if not (TICKED_MIN_COVERAGE <= cov <= TICKED_MAX_COVERAGE):
                continue
            member = labels == i
            if int(member.sum()) < TICKED_MIN_PIXELS:
                continue
            spread = float(norm_x[member].std() + norm_y[member].std())
            if spread > TICKED_SPREAD:
                return {"pattern": "TICKED", "confidence": 0.45, "colour": name, "spread": round(spread, 3)}

    n_sig = len(significant)
    fallback = {1: "SOLID", 2: "BICOLOUR", 3: "TRICOLOUR"}.get(n_sig, "UNKNOWN")
    return {"pattern": fallback, "confidence": 0.5 if fallback != "UNKNOWN" else 0.0, "significant_colour_count": n_sig}


def main() -> None:
    require_previous_output(OUTPUTS_DIR / "03_mask.png", "03_segmentation.py")
    colour_json = require_previous_output(OUTPUTS_DIR / "08_colour.json", "08_appearance_colour.py")
    colour = load_json(colour_json)

    mask = load_bgr(OUTPUTS_DIR / "03_mask.png")[:, :, 0] > 0
    grey = cv2.cvtColor(load_bgr(analysis_copy_path()), cv2.COLOR_BGR2GRAY)

    coat = classify_length_and_texture(mask, grey)
    print("Coat length/texture:")
    print(f"  length  = {coat['length']}  (confidence {coat['length_confidence']}, roughness {coat.get('roughness')})")
    print(f"  texture = {coat['texture']}  (confidence {coat['texture_confidence']})")

    npz_path = OUTPUTS_DIR / "08_colour_pixels.npz"
    if not npz_path.exists() or colour.get("refused"):
        pattern_result = {"pattern": "UNKNOWN", "confidence": 0.0, "reason": "no colour regions from stage 8"}
    else:
        data = np.load(npz_path, allow_pickle=True)
        coords, labels, region_names = data["coords"], data["labels"], list(data["region_names"])
        region_by_name = {r["name"]: r for r in colour["regions"]}
        pattern_result = classify_pattern(mask, coords, labels, region_names, region_by_name)

    print(f"\nPattern = {pattern_result.get('pattern')}  {pattern_result}")

    save_json(OUTPUTS_DIR / "09_coat_pattern.json", {"coat": coat, "pattern": pattern_result})
    annotated = put_lines(
        load_bgr(analysis_copy_path()),
        [f"length={coat['length']} texture={coat['texture']}", f"pattern={pattern_result.get('pattern')}"],
    )
    save_image(OUTPUTS_DIR / "09_coat_pattern.jpg", annotated)
    print("\nWrote outputs/09_coat_pattern.json and outputs/09_coat_pattern.jpg")


if __name__ == "__main__":
    main()
