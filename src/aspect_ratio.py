"""Aspect ratio of six ordered facial landmarks [p1..p6].

        ||p2 - p6|| + ||p3 - p5||
ratio = ---------------------------
             2 * ||p1 - p4||

The SAME computation serves two signals; only the points passed in differ:
  - Eye Aspect Ratio (EAR) — drowsiness. Soukupova & Cech (2016),
    "Real-Time Eye Blink Detection using Facial Landmarks".
      p1 = outer corner, p4 = inner corner; p2,p3 = top lid;  p5,p6 = bottom lid.
  - Mouth Aspect Ratio (MAR) — yawning (EAR analogue).
      p1 = left corner,  p4 = right corner; p2,p3 = upper lip; p5,p6 = lower lip.

Point order (identical for both):
    points[0] = p1  corner         points[3] = p4  opposite corner
    points[1] = p2  upper #1        points[5] = p6  lower #1  (pairs with p2)
    points[2] = p3  upper #2        points[4] = p5  lower #2  (pairs with p3)

Closed (eye/mouth) -> verticals ~ 0     -> ratio small.
Open               -> verticals large   -> ratio large.
"""
from __future__ import annotations

import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Straight-line distance between two 2D points."""
    return float(np.linalg.norm(a - b))


def aspect_ratio(points: np.ndarray) -> float:
    """Normalized aspect ratio from a (6, 2) array ordered [p1..p6].

    Used for BOTH the eye (EAR) and the mouth (MAR); only the landmarks differ.
    """
    vertical_1 = euclidean(points[1], points[5])   # ||p2 - p6||
    vertical_2 = euclidean(points[2], points[4])   # ||p3 - p5||
    horizontal = euclidean(points[0], points[3])   # ||p1 - p4||
    return (vertical_1 + vertical_2) / (2.0 * horizontal)
