"""Drowsiness decision logic — the temporal state machine.

One low-EAR frame is just a blink. Drowsiness = eyes below threshold for
a SUSTAINED period. We count consecutive low-EAR frames and raise an alarm
once that count crosses a limit derived from the camera's frame rate.

        eyes closed for  >= DROWSY_SECONDS   ->  ALARM

>>> This is the second file worth understanding by heart. The logic is
    trivial but it is the heart of the detector — own it, don't skim it. <<<
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DrowsinessDetector:
    ear_threshold: float = 0.21     # below this = eye considered closed
    drowsy_seconds: float = 1.0     # closed this long = drowsy
    fps: float = 10.0               # measured/assumed camera frame rate

    def __post_init__(self) -> None:
        self._counter: int = 0
        self._frames_needed: int = int(self.drowsy_seconds * self.fps)

    def update(self, ear: float) -> tuple[bool, int]:
        """Feed one frame's EAR. Returns (is_drowsy, consecutive_closed_frames)."""
        if ear < self.ear_threshold:
            self._counter += 1
        else:
            self._counter = 0
        is_drowsy = self._counter >= self._frames_needed
        return is_drowsy, self._counter
