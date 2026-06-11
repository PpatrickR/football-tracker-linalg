"""Feature-tracked per-frame field registration.

Approach: anchor a homography on a single hand-picked YAML frame, then track
the camera's motion frame-to-frame via ORB features + RANSAC. The current
frame's homography is the anchor's composed with the accumulated motion.

  H_current_to_field = H_anchor_to_field @ inv(M_anchor_to_current)

Where M_anchor_to_current accumulates frame-to-frame transforms.

If a frame's match fails (low features, large jump), the previous estimate
is reused — better stale than wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .registration import FieldRegistration


@dataclass
class TrackingStats:
    n_frames: int = 0
    n_matched: int = 0
    n_failed: int = 0
    n_low_features: int = 0


class TrackedFieldRegistration:
    def __init__(
        self,
        anchor: FieldRegistration,
        anchor_frame_bgr: np.ndarray,
        n_features: int = 2000,
        ransac_threshold: float = 4.0,
        min_matches: int = 25,
    ):
        self._anchor_to_field = anchor.H
        self._anchor_to_field_inv = anchor.H_inv
        self._motion = np.eye(3, dtype=np.float64)
        self._prev_gray = cv2.cvtColor(anchor_frame_bgr, cv2.COLOR_BGR2GRAY)
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._ransac_threshold = ransac_threshold
        self._min_matches = min_matches
        self.stats = TrackingStats()

    def update(self, frame_bgr: np.ndarray) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.stats.n_frames += 1
        kp_prev, des_prev = self._orb.detectAndCompute(self._prev_gray, None)
        kp_curr, des_curr = self._orb.detectAndCompute(gray, None)
        if (
            des_prev is None
            or des_curr is None
            or len(kp_prev) < self._min_matches
            or len(kp_curr) < self._min_matches
        ):
            self.stats.n_low_features += 1
            self._prev_gray = gray
            return
        matches = self._matcher.match(des_prev, des_curr)
        if len(matches) < self._min_matches:
            self.stats.n_low_features += 1
            self._prev_gray = gray
            return
        src = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, self._ransac_threshold)
        if M is None or (mask is not None and mask.sum() < self._min_matches):
            self.stats.n_failed += 1
            self._prev_gray = gray
            return
        self._motion = M @ self._motion
        self._prev_gray = gray
        self.stats.n_matched += 1

    @property
    def H(self) -> np.ndarray:
        return self._anchor_to_field @ np.linalg.inv(self._motion)

    @property
    def H_inv(self) -> np.ndarray:
        return self._motion @ self._anchor_to_field_inv

    def pixel_to_field(self, px: np.ndarray) -> np.ndarray:
        return _apply_h(self.H, px)

    def field_to_pixel(self, fp: np.ndarray) -> np.ndarray:
        return _apply_h(self.H_inv, fp)


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    out = (H @ homog.T).T
    return out[:, :2] / out[:, 2:3]
