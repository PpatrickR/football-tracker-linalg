"""Frame -> field-coord player positions.

The boundary between vision (varies by era/quality) and football logic
(era-invariant). Anything below uses pixels; anything above uses yards.
"""
from dataclasses import dataclass

import numpy as np

from .detection import Detection, PersonDetector
from .field_coords import FieldPoint
from .registration import FieldRegistration


@dataclass
class TrackedPlayer:
    detection: Detection
    field_pos: FieldPoint


class FieldAnalyzer:
    def __init__(
        self,
        detector: PersonDetector,
        registration: FieldRegistration,
    ):
        self.detector = detector
        self.registration = registration

    def analyze(self, frame: np.ndarray) -> list[TrackedPlayer]:
        detections = self.detector.detect(frame)
        if not detections:
            return []
        foot_points = np.array([d.foot_point for d in detections])
        field_pts = self.registration.pixel_to_field(foot_points)
        return [
            TrackedPlayer(
                detection=d,
                field_pos=FieldPoint(x=float(fp[0]), y=float(fp[1])),
            )
            for d, fp in zip(detections, field_pts)
        ]
