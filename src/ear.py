"""Eye Aspect Ratio (EAR) — the core drowsiness signal.

        ||p2 - p6|| + ||p3 - p5||
EAR = ---------------------------
             2 * ||p1 - p4||

Open eye   -> vertical gaps large  -> EAR ~ 0.25 - 0.30
Closed eye -> vertical gaps ~ 0    -> EAR ~ 0.00

Reference: Soukupova & Cech (2016),
"Real-Time Eye Blink Detection using Facial Landmarks".
"""
from __future__ import annotations

import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line distance between two 2D points."""
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """EAR for a single eye.

    `eye` is a (6, 2) array of pixel coordinates in this order:
        eye[0] = outer corner   (p1)
        eye[1] = top    #1       (p2)
        eye[2] = top    #2       (p3)
        eye[3] = inner corner    (p4)
        eye[4] = bottom #2       (p5)   <- pairs with eye[2]
        eye[5] = bottom #1       (p6)   <- pairs with eye[1]
    """
    vertical_1 = euclidean(eye[1], eye[5])   # ||p2 - p6||
    vertical_2 = euclidean(eye[2], eye[4])   # ||p3 - p5||
    horizontal = euclidean(eye[0], eye[3])   # ||p1 - p4||
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
