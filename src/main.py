"""Webcam entry point — wires the pipeline together and draws the overlay.

Pipeline (this is the "data path" the BHTC JD talks about):

    webcam frame --> FaceLandmarker --> EAR (per eye, averaged)
                 --> DrowsinessDetector --> on-screen overlay + alarm

Run:  python -m src.main        (press 'q' to quit)
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from collections import deque

from src.ear import eye_aspect_ratio
from src.landmarks import FaceLandmarker
from src.drowsiness import DrowsinessDetector

GREEN = (0, 255, 0)
RED = (0, 0, 255)


def main(camera_index: int = 0) -> None:
    fps_log = deque(maxlen=500)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}")

    landmarker = FaceLandmarker()
    detector = DrowsinessDetector(ear_threshold=0.21, drowsy_seconds=1.0, fps=20.0)

    prev_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, so it feels like a mirror

            eyes = landmarker.eyes(frame)
            if eyes is None:
                cv2.putText(frame, "No face", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)
            else:
                right_eye, left_eye = eyes
                ear = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye)) / 2.0
                is_drowsy, counter = detector.update(ear)

                # draw the eye landmarks so the demo is legible
                for pt in np.vstack([right_eye, left_eye]).astype(int):
                    cv2.circle(frame, tuple(pt), 2, GREEN, -1)

                colour = RED if is_drowsy else GREEN
                cv2.putText(frame, f"EAR: {ear:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                if is_drowsy:
                    cv2.putText(frame, "DROWSY!", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, RED, 3)

            # live FPS in the corner
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            fps_log.append(fps)
            prev_t = now
            arr = np.array(fps_log)
            if arr.size:
                p5, p50, p95 = np.percentile(arr, [5, 50, 95])
                cv2.putText(frame, f"5th percentile: {p5:4.1f} FPS", (frame.shape[1] - 230, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)
                cv2.putText(frame, f"95th percentile: {p95:4.1f} FPS", (frame.shape[1] - 230, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)
                cv2.putText(frame, f"Median: {p50:4.1f} FPS", (frame.shape[1] - 230, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)

            cv2.imshow("Driver Attention Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
