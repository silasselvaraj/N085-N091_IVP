"""AP-10K keypoint layout, shared by every script downstream of pose estimation
(06 view/posture, 07 morphometrics, 10 ear/tail) so they agree on keypoint
order without each re-declaring it - and so importing them doesn't drag
torch/transformers in just for a list of names.
"""

from __future__ import annotations

AP10K_KEYPOINTS = [
    "left_eye", "right_eye", "nose", "neck", "tail_base",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_back_paw",
    "right_hip", "right_knee", "right_back_paw",
]
KP = {name: i for i, name in enumerate(AP10K_KEYPOINTS)}

CONFIDENCE_FLOOR = 0.35
MIN_CONFIDENT_KEYPOINTS = 6

LATERAL_PAIRS = [
    ("left_eye", "right_eye"), ("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"),
    ("left_front_paw", "right_front_paw"), ("left_hip", "right_hip"), ("left_knee", "right_knee"),
    ("left_back_paw", "right_back_paw"),
]

# For the overlay drawings only - not part of any model's output.
SKELETON = [
    ("neck", "nose"), ("left_eye", "nose"), ("right_eye", "nose"), ("neck", "tail_base"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_front_paw"),
    ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_front_paw"),
    ("tail_base", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_back_paw"),
    ("tail_base", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_back_paw"),
]
SKELETON_IDX = [(KP[a], KP[b]) for a, b in SKELETON]
