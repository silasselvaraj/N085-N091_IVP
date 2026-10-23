"""Shared plumbing for the numbered showcase scripts.

This module holds only display/IO boilerplate (paths, JSON/image save-load,
drawing) so each numbered script can stay focused on its one pipeline stage's
actual algorithm. No CV/ML logic lives here — that would defeat the point of
a showcase meant to demonstrate the real logic, script by script. Nothing in
this file, or in any numbered script, imports from `backend/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SHOWCASE_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = SHOWCASE_ROOT / "weights"
SAMPLE_IMAGE = SHOWCASE_ROOT / "sample_images" / "sample_dog.jpg"
OUTPUTS_DIR = SHOWCASE_ROOT / "outputs"

# Same convention the real pipeline uses: every stage after ingestion works on
# a fixed-resolution "analysis copy", not the original upload, so pixel
# coordinates saved by one script mean the same thing when the next script
# loads them back. See app/imaging/metrics.py's own analysis-copy comment.
ANALYSIS_LONG_EDGE = 1024


def analysis_copy_path() -> Path:
    return OUTPUTS_DIR / "analysis_copy.jpg"


def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def resize_to_analysis_copy(image_bgr: np.ndarray, long_edge: int = ANALYSIS_LONG_EDGE) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return image_bgr.copy()
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def load_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(load_bgr(path), cv2.COLOR_BGR2RGB)


def save_image(path: Path, image_bgr: np.ndarray) -> None:
    ensure_outputs_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_outputs_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def require_previous_output(path: Path, produced_by: str) -> Path:
    """Fail loudly, with a next-step hint, if an earlier script hasn't been run yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing '{path.relative_to(SHOWCASE_ROOT)}'. Run `{produced_by}` first — "
            "the scripts are numbered because each one feeds the next."
        )
    return path


@dataclass(frozen=True)
class Box:
    """Pixel-space box, x2/y2 exclusive — mirrors app/cv/geometry.py's Box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def clipped(self, width: int, height: int) -> Box:
        return Box(
            x1=max(0.0, min(self.x1, width)),
            y1=max(0.0, min(self.y1, height)),
            x2=max(0.0, min(self.x2, width)),
            y2=max(0.0, min(self.y2, height)),
        )

    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @staticmethod
    def from_dict(d: dict[str, float]) -> Box:
        return Box(d["x1"], d["y1"], d["x2"], d["y2"])


def draw_box(
    image_bgr: np.ndarray, box: Box, label: str = "", color: tuple[int, int, int] = (0, 220, 90)
) -> np.ndarray:
    out = image_bgr.copy()
    p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
    cv2.rectangle(out, p1, p2, color, 2)
    if label:
        cv2.putText(out, label, (p1[0], max(0, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_mask_overlay(
    image_bgr: np.ndarray, mask_bool: np.ndarray, color: tuple[int, int, int] = (0, 220, 90), alpha: float = 0.45
) -> np.ndarray:
    out = image_bgr.copy()
    overlay = out.copy()
    overlay[mask_bool] = color
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, dst=out)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def draw_keypoints(
    image_bgr: np.ndarray,
    keypoints: list[tuple[float, float]],
    scores: list[float],
    skeleton: list[tuple[int, int]],
    confidence_floor: float,
) -> np.ndarray:
    out = image_bgr.copy()
    confident = [s >= confidence_floor for s in scores]
    for i, j in skeleton:
        if confident[i] and confident[j]:
            p1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            p2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(out, p1, p2, (60, 180, 255), 2)
    for (x, y), s in zip(keypoints, scores):
        color = (0, 220, 90) if s >= confidence_floor else (120, 120, 120)
        cv2.circle(out, (int(x), int(y)), 4, color, -1)
    return out


def put_lines(image_bgr: np.ndarray, lines: list[str], origin: tuple[int, int] = (10, 24)) -> np.ndarray:
    out = image_bgr.copy()
    x, y = origin
    for i, line in enumerate(lines):
        yy = y + i * 22
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
