"""Stage 8 of 12 — coat colour extraction.

White-balances against near-neutral pixels only (not the whole frame - a
naive grey-world correction on a solid black or red dog would try to
"neutralise" the dog itself), then clusters the masked coat pixels in
CIELAB space, where perceptual colour distance behaves far better than RGB.

k-means alone isn't enough: it happily splits one solid black coat into
three clusters by lighting (shadow/mid-tone/highlight), and every one of
those shades still names to "black". So every cluster is first matched to
its nearest named colour, and clusters sharing a name are merged before
anything is reported - otherwise the packet would claim a solid black dog
is three different colours.

Note: the 14 reference CIELAB points below are this script's own
reasonable approximation of each named colour's centroid (illustrative,
not re-extracted from the private vocabulary table) - the mechanism being
demonstrated is the nearest-neighbour-then-merge logic, not these exact
numbers.

Run: python 08_appearance_colour.py   (needs 03_segmentation.py to have run first)
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

from _common import OUTPUTS_DIR, analysis_copy_path, load_bgr, load_rgb, require_previous_output, save_image, save_json

NEUTRAL_SATURATION = 0.12
NO_REFERENCE_SHIFT = 0.2
MIN_NEUTRAL_FRACTION = 0.02
MAX_WB_GAIN = 1.35

MAX_CLUSTERS = 4
MIN_MASKED_PIXELS = 64
MIN_COVERAGE = 0.06
MAX_VOCABULARY_DISTANCE = 26.0
LIGHTNESS_WEIGHT = 0.6

# L, a, b - illustrative reference centroids (see module docstring).
COLOUR_VOCABULARY = {
    "black": (15, 0, 0), "white": (95, 0, 2), "cream": (90, 2, 18), "fawn": (70, 10, 25),
    "red": (45, 35, 30), "gold": (65, 15, 45), "tan": (60, 18, 30), "chocolate": (35, 18, 20),
    "liver": (32, 20, 15), "grey": (55, 0, 0), "blue": (45, -2, -8), "silver": (75, 0, -2),
    "sable": (40, 8, 15), "brindle_base": (48, 10, 20),
}


def white_balance(rgb_float: np.ndarray) -> tuple[np.ndarray, float]:
    sat = rgb_float.max(axis=2) - rgb_float.min(axis=2)
    neutral = sat <= NEUTRAL_SATURATION
    if neutral.sum() < MIN_NEUTRAL_FRACTION * sat.size:
        return rgb_float, NO_REFERENCE_SHIFT
    means = rgb_float[neutral].mean(axis=0)
    grey_target = means.mean()
    gains = np.clip(grey_target / np.maximum(means, 1e-6), 1 / MAX_WB_GAIN, MAX_WB_GAIN)
    corrected = np.clip(rgb_float * gains, 0, 1)
    shift = float(np.mean(np.abs(np.log(gains))))
    return corrected, shift


def nearest_name(lab_point: np.ndarray) -> tuple[str, float]:
    best_name, best_dist = "unknown", float("inf")
    for name, (L, a, b) in COLOUR_VOCABULARY.items():
        d = np.hypot(np.hypot((lab_point[0] - L) * LIGHTNESS_WEIGHT, lab_point[1] - a), lab_point[2] - b)
        if d < best_dist:
            best_name, best_dist = name, float(d)
    if best_dist > MAX_VOCABULARY_DISTANCE:
        return "unknown", best_dist
    return best_name, best_dist


def main() -> None:
    require_previous_output(OUTPUTS_DIR / "03_mask.png", "03_segmentation.py")
    mask = load_bgr(OUTPUTS_DIR / "03_mask.png")[:, :, 0] > 0
    rgb_float = load_rgb(analysis_copy_path()).astype(np.float64) / 255.0

    coords = np.argwhere(mask)  # (N, 2) row, col
    n_masked = len(coords)
    print(f"{n_masked} masked pixels available for colour extraction")
    if n_masked < MIN_MASKED_PIXELS:
        print(f"Fewer than {MIN_MASKED_PIXELS} masked pixels - refusing (no reliable colour signal).")
        save_json(OUTPUTS_DIR / "08_colour.json", {"regions": [], "confidence": 0.0, "refused": True})
        return

    balanced, shift = white_balance(rgb_float)
    print(f"white-balance shift = {shift:.4f}  ({'no neutral reference found' if shift == NO_REFERENCE_SHIFT else 'corrected against near-neutral pixels'})")

    lab = rgb2lab(balanced)
    pixels_lab = lab[coords[:, 0], coords[:, 1]]

    k = min(MAX_CLUSTERS, max(1, n_masked // 32))
    print(f"k-means with k={k} clusters in CIELAB space")
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42).fit(pixels_lab)
    raw_labels = kmeans.labels_
    centres = kmeans.cluster_centers_

    # Merge clusters that name to the same vocabulary term.
    name_by_cluster = {}
    for i, centre in enumerate(centres):
        name, dist = nearest_name(centre)
        name_by_cluster[i] = name
        print(f"  cluster {i}: L={centre[0]:.1f} a={centre[1]:.1f} b={centre[2]:.1f}  -> '{name}' (distance {dist:.1f})")

    groups: dict[str, list[int]] = {}
    for i, name in name_by_cluster.items():
        groups.setdefault(name, []).append(i)

    regions = []
    remap = {}
    for group_idx, (name, members) in enumerate(groups.items()):
        member_mask = np.isin(raw_labels, members)
        count = int(member_mask.sum())
        share = count / n_masked
        if share < MIN_COVERAGE:
            for m in members:
                remap[m] = -1
            continue
        weighted_centre = np.average(centres[members], axis=0, weights=[np.sum(raw_labels == m) for m in members])
        regions.append({"name": name, "coverage": round(share, 4), "lab": [round(float(x), 2) for x in weighted_centre]})
        for m in members:
            remap[m] = group_idx
    regions.sort(key=lambda r: r["coverage"], reverse=True)

    merged_labels = np.array([remap.get(l, -1) for l in raw_labels])

    if k == 1:
        separation = 1.0
    else:
        dists = [np.linalg.norm(centres[i] - centres[j]) for i in range(k) for j in range(i + 1, k)]
        separation = float(np.clip(min(dists) / 30, 0, 1))
    size_term = float(np.clip(n_masked / 20000, 0, 1))
    wb_term = float(np.clip(1 - shift * 3, 0, 1))
    confidence = round(0.5 * separation + 0.25 * size_term + 0.25 * wb_term, 4)

    print(f"\nMerged into {len(regions)} named region(s) (>= {MIN_COVERAGE:.0%} coverage each):")
    for r in regions:
        print(f"  {r['name']:<14s} coverage={r['coverage']:.1%}  lab={r['lab']}")
    print(f"confidence = {confidence}  (separation={separation:.2f} size={size_term:.2f} wb={wb_term:.2f})")

    swatch = load_bgr(analysis_copy_path()).copy()
    y0 = 10
    for r in regions:
        L, a, b = r["lab"]
        import cv2
        lab_px = np.uint8([[[max(0, min(255, L * 2.55)), a + 128, b + 128]]])
        bgr_px = cv2.cvtColor(lab_px, cv2.COLOR_LAB2BGR)[0, 0].tolist()
        cv2.rectangle(swatch, (10, y0), (60, y0 + 30), bgr_px, -1)
        cv2.putText(swatch, f"{r['name']} {r['coverage']:.0%}", (70, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(swatch, f"{r['name']} {r['coverage']:.0%}", (70, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y0 += 36
    save_image(OUTPUTS_DIR / "08_colour.jpg", swatch)

    np.savez_compressed(OUTPUTS_DIR / "08_colour_pixels.npz", coords=coords, labels=merged_labels, region_names=[r["name"] for r in regions])
    save_json(OUTPUTS_DIR / "08_colour.json", {"regions": regions, "confidence": confidence, "white_balance_shift": round(shift, 4), "refused": False})
    print("\nWrote outputs/08_colour.json, outputs/08_colour_pixels.npz and outputs/08_colour.jpg")


if __name__ == "__main__":
    main()
