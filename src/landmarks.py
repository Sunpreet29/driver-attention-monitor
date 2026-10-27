"""Facial landmark extraction via MediaPipe FaceMesh.

MediaPipe's *legacy* Solutions API (`mp.solutions.face_mesh`) is used because
it needs no separate model download - fastest path to a working demo.
The modern replacement is the Tasks API (`FaceLandmarker`).

FaceMesh returns 468 landmarks, each NORMALISED to [0, 1]. Multiply by the
frame width/height to get pixels. The index map is published here:
https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
"""
from __future__ import annotations

import numpy as np
import mediapipe as mp

# Six landmark indices per eye, ordered for ear.eye_aspect_ratio():
#   [outer corner, top1, top2, inner corner, bottom2, bottom1]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


class FaceLandmarker:
    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def eyes(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (right_eye, left_eye) as (6, 2) pixel arrays, or None if no face."""
        h, w = frame_bgr.shape[:2]
        # MediaPipe wants RGB; OpenCV gives BGR.
        rgb = frame_bgr[:, :, ::-1]
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark

        def to_px(indices: list[int]) -> np.ndarray:
            return np.array([(lm[i].x * w, lm[i].y * h) for i in indices])

        return to_px(RIGHT_EYE), to_px(LEFT_EYE)

    def close(self) -> None:
        self._mesh.close()
