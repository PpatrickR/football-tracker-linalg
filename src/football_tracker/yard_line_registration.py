"""Self-correcting per-frame registration anchored on detected yard lines.

Each frame:
  1. Warp the image to top-down using the prior (YAML-anchored) homography.
     In top-down space, painted yard lines should be perfectly vertical.
  2. Compute the column-summed |Sobel-x| (vertical-edge density) over the
     field, excluding masked player regions.
  3. Find peaks in that 1D signal — those are the actual yard lines.
  4. Snap each detected peak to the nearest 5-yard increment, take the median
     residual, and apply it as a horizontal translation correction in field
     coords. The corrected homography is used for that frame.

No drift: every frame re-anchors from ground truth. No camera-motion bias
from moving players because we mask them out before edge analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks

from .field_coords import FIELD_LENGTH_YD, FIELD_WIDTH_YD
from .registration import FieldRegistration


@dataclass
class YardLineStats:
    n_frames: int = 0
    n_corrected: int = 0
    n_too_few_peaks: int = 0
    last_residual_yd: float = 0.0
    last_n_peaks: int = 0


class YardLineRegistration:
    SCALE_PX_PER_YD = 8

    def __init__(
        self,
        anchor: FieldRegistration,
        peak_distance_yd: float = 4.0,
        peak_prominence_frac: float = 0.04,
        min_peaks_for_correction: int = 3,
        max_translation_yd: float = 6.0,
    ):
        self._H_prior = anchor.H.copy()
        self._H = anchor.H.copy()
        self._H_inv = anchor.H_inv.copy()
        self.TD_W = int(FIELD_LENGTH_YD * self.SCALE_PX_PER_YD)
        self.TD_H = int(FIELD_WIDTH_YD * self.SCALE_PX_PER_YD)
        self._S = np.array(
            [[self.SCALE_PX_PER_YD, 0, 0],
             [0, self.SCALE_PX_PER_YD, 0],
             [0, 0, 1]],
            dtype=np.float64,
        )
        self._peak_distance_px = int(peak_distance_yd * self.SCALE_PX_PER_YD)
        self._peak_prominence_frac = peak_prominence_frac
        self._min_peaks = min_peaks_for_correction
        self._max_translation_yd = max_translation_yd
        self.stats = YardLineStats()

    def update(
        self,
        frame_bgr: np.ndarray,
        player_boxes_xyxy: np.ndarray | None = None,
    ) -> None:
        self.stats.n_frames += 1
        H_pix_to_td = self._S @ self._H_prior
        topdown = cv2.warpPerspective(
            frame_bgr, H_pix_to_td, (self.TD_W, self.TD_H)
        )
        gray = cv2.cvtColor(topdown, cv2.COLOR_BGR2GRAY)

        valid_mask = (gray > 0).astype(np.uint8) * 255

        if player_boxes_xyxy is not None and len(player_boxes_xyxy):
            h, w = frame_bgr.shape[:2]
            player_mask_pix = np.zeros((h, w), dtype=np.uint8)
            for box in player_boxes_xyxy:
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                pad = 6
                cv2.rectangle(
                    player_mask_pix,
                    (max(0, x1 - pad), max(0, y1 - pad)),
                    (min(w, x2 + pad), min(h, y2 + pad)),
                    255, -1,
                )
            player_mask_td = cv2.warpPerspective(
                player_mask_pix, H_pix_to_td, (self.TD_W, self.TD_H)
            )
            valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(player_mask_td))

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        abs_sobel = np.abs(sobelx).astype(np.float32)
        abs_sobel[valid_mask == 0] = 0.0

        col_sum = abs_sobel.sum(axis=0)
        col_smooth = cv2.GaussianBlur(
            col_sum.reshape(1, -1), (1, 11), 0
        ).flatten()
        if col_smooth.max() <= 0:
            self.stats.n_too_few_peaks += 1
            return
        peaks, _ = find_peaks(
            col_smooth,
            distance=self._peak_distance_px,
            prominence=col_smooth.max() * self._peak_prominence_frac,
        )
        self.stats.last_n_peaks = len(peaks)

        if len(peaks) < self._min_peaks:
            self.stats.n_too_few_peaks += 1
            return

        peak_yards = peaks / self.SCALE_PX_PER_YD
        snapped = np.round(peak_yards / 5.0) * 5.0
        residuals = snapped - peak_yards
        med_residual = float(np.median(residuals))
        self.stats.last_residual_yd = med_residual

        if abs(med_residual) > self._max_translation_yd:
            self.stats.n_too_few_peaks += 1
            return

        T = np.eye(3, dtype=np.float64)
        T[0, 2] = med_residual
        self._H = T @ self._H_prior
        self._H_inv = np.linalg.inv(self._H)
        self.stats.n_corrected += 1

    @property
    def H(self) -> np.ndarray:
        return self._H

    @property
    def H_inv(self) -> np.ndarray:
        return self._H_inv

    def pixel_to_field(self, px: np.ndarray) -> np.ndarray:
        return _apply_h(self._H, px)

    def field_to_pixel(self, fp: np.ndarray) -> np.ndarray:
        return _apply_h(self._H_inv, fp)


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    out = (H @ homog.T).T
    return out[:, :2] / out[:, 2:3]
