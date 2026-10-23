"""Stage 12 of 12 — breed classification: calibration, open-set rejection,
the confidence decision rule, and the final annotated result.

Breed identification reads pixels directly from stage 3's background-
suppressed crop, which is exactly why the real project could ship this
while Part 4 (pose/morphometrics) stayed deferred: nothing about naming a
breed needs a skeleton.

For each dog stage 3 segmented (every deduped dog stage 2 found, not only
the single ranked "subject" stages 4-11 deep-dive on), four things happen,
in order:
1. A fine-tuned ConvNeXt-Tiny scores all 120 breeds.
2. Its raw logits are temperature-scaled (Guo et al. 2017) before being
   turned into probabilities - an uncalibrated softmax is *always*
   overconfident, and this project measured that gap and corrected it
   rather than reporting raw softmax as if it meant something.
3. An energy-based score (not max-softmax - measurably better at this in
   the real project's own OOD experiments) decides whether the input looks
   like any of the 120 trained breeds at all, before a breed name is
   reported.
4. A decision rule turns (probability, margin, image quality, mask quality)
   into CONFIDENT / PROVISIONAL / INSUFFICIENT_EVIDENCE - never a bare
   number pretending to be a verdict.

The real app only has stage 4's subject-specific quality recheck for the
single dog being examined (the UI-switcher pattern from PROJECT_STATE.md
SS2.0: every dog gets boxed, you inspect one at a time). This script scores
every dog at once, so for any dog other than the subject it falls back to
stage 1's frame-level `degraded` flag rather than inventing a per-dog
recheck stage 4 was never designed to produce - noted directly in each
non-subject dog's own result.

Weights: ivp_showcase/weights/breed_classifier_convnext.pt (the exact
fine-tuned checkpoint from Part 6, including its own fitted temperature and
OOD threshold - nothing here is re-fit).

Run: python 12_breed_classification.py   (needs 03_segmentation.py; 04_subject_quality_recheck.py improves the subject's own result if it has run)
"""

from __future__ import annotations

import json

import numpy as np
import timm
import torch
from PIL import Image
from scipy.special import logsumexp
from torchvision import transforms

from _common import (
    OUTPUTS_DIR,
    WEIGHTS_DIR,
    Box,
    analysis_copy_path,
    draw_box,
    load_bgr,
    load_json,
    require_previous_output,
    save_image,
    save_json,
)

CHECKPOINT_PATH = WEIGHTS_DIR / "breed_classifier_convnext.pt"
TAXONOMY_PATH = WEIGHTS_DIR / "breed_taxonomy.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROVISIONAL_PROBABILITY = 0.35
CONFIDENT_PROBABILITY = 0.70
CONFIDENT_MARGIN = 0.15
MIN_MASK_QUALITY_FOR_CONFIDENT = 0.5

MIXED_BREED_MIN_CANDIDATE_PROBABILITY = 0.05
MIXED_BREED_MAX_TOP1_PROBABILITY = 0.60
MIXED_BREED_MIN_GROUPS = 2

# Part 7 (the real breed -> FCI group database) was never built in the live
# project, so BREED_TO_FCI_GROUP is an empty dict there and this heuristic
# is a no-op in production today. This is a small illustrative subset - a
# handful of common breeds only - so the showcase can demonstrate the
# *mechanism* rather than claim a complete mapping that doesn't exist yet.
ILLUSTRATIVE_FCI_GROUPS = {
    "siberian_husky": "spitz_and_primitive", "malamute": "spitz_and_primitive",
    "german_shepherd": "herding", "collie": "herding", "border_collie": "herding",
    "labrador_retriever": "gundog", "golden_retriever": "gundog",
    "beagle": "hound", "afghan_hound": "hound", "basset": "hound",
    "boxer": "working", "rottweiler": "working", "doberman": "working",
    "yorkshire_terrier": "terrier", "west_highland_white_terrier": "terrier",
    "chihuahua": "toy", "pug": "toy", "pomeranian": "toy",
}


def logits_to_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - scaled.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def energy_score(logits: np.ndarray, temperature: float) -> float:
    return float(-temperature * logsumexp(logits / temperature))


def decide(top1_prob: float, margin: float, image_degraded: bool, mask_quality: float | None) -> str:
    quality_allows_confident = (not image_degraded) and not (mask_quality is not None and mask_quality < MIN_MASK_QUALITY_FOR_CONFIDENT)
    meets_confident_bar = top1_prob >= CONFIDENT_PROBABILITY and margin >= CONFIDENT_MARGIN
    if meets_confident_bar and quality_allows_confident:
        return "CONFIDENT"
    return "PROVISIONAL"


def classify_one(model, device, preprocess, crop_path, classes, temperature, ood_threshold, display_name, image_degraded, mask_quality) -> dict:
    pil_image = Image.open(crop_path).convert("RGB")
    tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0].cpu().numpy().astype(np.float64)

    probs = logits_to_probabilities(logits, temperature)
    ood = energy_score(logits, temperature)
    order = np.argsort(probs)[::-1]
    ranked = [(classes[i], float(probs[i])) for i in order[:5]]

    if ood >= ood_threshold:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reason": "does not look like any of the 120 trained breeds", "ood_energy_score": round(ood, 4), "ranked_candidates": _candidates(ranked, display_name)}
    if ranked[0][1] < PROVISIONAL_PROBABILITY:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reason": f"top probability {ranked[0][1]:.3f} below the {PROVISIONAL_PROBABILITY} floor", "ood_energy_score": round(ood, 4), "ranked_candidates": _candidates(ranked, display_name)}

    top1_breed, top1_prob = ranked[0]
    margin = top1_prob - (ranked[1][1] if len(ranked) > 1 else 0.0)
    verdict = decide(top1_prob, margin, image_degraded, mask_quality)

    meaningful = [(b, p) for b, p in ranked if p >= MIXED_BREED_MIN_CANDIDATE_PROBABILITY]
    if top1_prob > MIXED_BREED_MAX_TOP1_PROBABILITY:
        mixed = False
    else:
        groups = {ILLUSTRATIVE_FCI_GROUPS[b] for b, _ in meaningful if b in ILLUSTRATIVE_FCI_GROUPS}
        mixed = len(groups) >= MIXED_BREED_MIN_GROUPS

    return {
        "verdict": verdict, "breed": top1_breed, "display_name": display_name.get(top1_breed, top1_breed),
        "probability": round(top1_prob, 4), "margin": round(margin, 4), "ood_energy_score": round(ood, 4),
        "image_degraded": image_degraded, "mask_quality": mask_quality, "possible_mixed_breed": mixed,
        "ranked_candidates": _candidates(ranked, display_name),
    }


def _candidates(ranked, display_name) -> list[dict]:
    return [{"breed": b, "display_name": display_name.get(b, b), "probability": round(p, 4)} for b, p in ranked]


def _box_label(result: dict) -> str:
    if result["verdict"] == "INSUFFICIENT_EVIDENCE":
        return "unable to identify"
    return f"{result['display_name']} ({result['probability']:.1%})"


def main() -> None:
    all_dogs_path = require_previous_output(OUTPUTS_DIR / "03_all_dogs.json", "03_segmentation.py")
    all_dogs = load_json(all_dogs_path)["dogs"]

    frame_degraded = load_json(OUTPUTS_DIR / "01_quality_check.json")["verdict"]["degraded"]
    subject_quality_path = OUTPUTS_DIR / "04_subject_quality.json"
    subject_degraded = load_json(subject_quality_path)["verdict"]["degraded"] if subject_quality_path.exists() else frame_degraded

    print(f"Loading checkpoint {CHECKPOINT_PATH.relative_to(CHECKPOINT_PATH.parents[1])}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    classes: list[str] = ckpt["classes"]
    temperature: float = float(ckpt["temperature"])
    ood_threshold: float = float(ckpt["ood_energy_threshold"])
    input_size: int = int(ckpt.get("input_size", 224))
    print(f"  arch={ckpt['arch']}  classes={len(classes)}  temperature={temperature:.4f}  ood_threshold={ood_threshold:.4f}")

    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        taxonomy = json.load(f)
    display_name = {k: v["display_name"] for k, v in taxonomy.items()}

    model = timm.create_model(ckpt["arch"], num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  device: {device}")

    preprocess = transforms.Compose(
        [transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )

    print(f"\nClassifying {len(all_dogs)} dog(s):")
    results = []
    for dog in all_dogs:
        i = dog["index"]
        is_subject = i == 0  # 02_detection.py's all_dogs is subject-first by ranking
        image_degraded = subject_degraded if is_subject else frame_degraded
        if dog["crop_path"] is None:
            result = {"verdict": "NO_USABLE_CROP", "reason": "segmentation mask not plausible for this box"}
            print(f"  dog {i} ({'subject' if is_subject else 'other'}): no usable crop - {result['reason']}")
        else:
            result = classify_one(
                model, device, preprocess, OUTPUTS_DIR / dog["crop_path"], classes, temperature, ood_threshold,
                display_name, image_degraded, dog["mask_quality"]["quality"],
            )
            label = _box_label(result)
            note = "" if is_subject else "  [image_degraded uses stage 1's frame-level flag, not a per-dog recheck]"
            print(f"  dog {i} ({'subject' if is_subject else 'other'}): {result['verdict']}  {label}{note}")
        result["index"] = i
        result["box"] = dog["box"]
        result["is_subject"] = is_subject
        results.append(result)

    save_json(OUTPUTS_DIR / "12_breed_results.json", {"dogs": results, "checkpoint_version": ckpt.get("version")})

    frame = load_bgr(analysis_copy_path())
    for result in results:
        box = Box.from_dict(result["box"])
        colour = (0, 220, 90) if result.get("verdict") not in ("INSUFFICIENT_EVIDENCE", "NO_USABLE_CROP") else (60, 180, 255)
        label = _box_label(result) if "ranked_candidates" in result else "no usable segmentation"
        frame = draw_box(frame, box, label, color=colour)
    save_image(OUTPUTS_DIR / "12_annotated.jpg", frame)

    print(f"\nWrote outputs/12_breed_results.json and outputs/12_annotated.jpg ({len(results)} dog(s) labelled)")
    print("\n--- End of the showcase pipeline ---")


if __name__ == "__main__":
    main()
