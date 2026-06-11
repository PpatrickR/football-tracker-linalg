"""Field registration: pixel coords <-> field coords (yards).

Uses ONLY yard lines and sidelines as fiducials, never hash marks. NFL hashes
are 18'6" apart and college hashes are 40' apart, but yard-line spacing (5 yd)
and field width (53.333 yd) are identical in both leagues. By refusing to
anchor on hashes, this module produces a homography that works on either.

Initial API: caller supplies >=4 image<->field correspondences. Auto-detection
of yard-line/sideline intersections from imagery is a future iteration; the
manual API ships first so downstream code can develop in parallel.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FieldRegistration:
    H: np.ndarray
    H_inv: np.ndarray

    @classmethod
    def from_correspondences(
        cls,
        pixel_pts: np.ndarray,
        field_pts: np.ndarray,
    ) -> "FieldRegistration":
        pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 2)
        field_pts = np.asarray(field_pts, dtype=np.float64).reshape(-1, 2)
        if len(pixel_pts) < 4 or len(pixel_pts) != len(field_pts):
            raise ValueError(
                "Need >=4 correspondences; pixel and field arrays must match length"
            )
        H, _ = cv2.findHomography(pixel_pts, field_pts, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError("Homography solve failed (degenerate correspondences?)")
        return cls(H=H, H_inv=np.linalg.inv(H))

    def pixel_to_field(self, px: np.ndarray) -> np.ndarray:
        return _apply_h(self.H, px)

    def field_to_pixel(self, fp: np.ndarray) -> np.ndarray:
        return _apply_h(self.H_inv, fp)


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    out = (H @ homog.T).T
    return out[:, :2] / out[:, 2:3]
