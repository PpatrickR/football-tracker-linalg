"""Detection filters: drop non-players (refs, sideline officials).

Two layers:
  * `is_in_field`: project the detection's foot point through the homography;
    drop anything that lands outside the [0..120] x [0..53.333] yd field.
    Catches sideline judges and out-of-bounds people.
  * `is_referee`: heuristic on the bbox crop — high horizontal-gradient density
    plus low color saturation in the torso band signals B/W vertical stripes.
    Catches the on-field umpire.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detection import Detection
from .field_coords import FieldPoint
from .registration import FieldRegistration


def is_in_field(detection: Detection, registration: FieldRegistration) -> bool:
    fp_arr = registration.pixel_to_field(np.array([detection.foot_point]))
    return FieldPoint(x=float(fp_arr[0, 0]), y=float(fp_arr[0, 1])).in_field()


def is_referee(
    detection: Detection,
    frame_bgr: np.ndarray,
    saturation_threshold: float = 25.0,
    stripe_periodicity_threshold: int = 5,
) -> bool:
    """Detect zebra-stripe pattern: very low color saturation AND many sign
    changes in the column-mean intensity profile (regular stripes alternate
    bright/dark across columns). Plain white jerseys have low saturation but
    flat profiles; player numbers create local edges but not periodic ones.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox_xyxy)
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 8 or y2 - y1 < 12:
        return False
    crop = frame_bgr[y1:y2, x1:x2]
    h = crop.shape[0]
    torso = crop[h // 4 : 3 * h // 4]
    if torso.size == 0 or torso.shape[1] < 8:
        return False
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    if float(hsv[..., 1].mean()) >= saturation_threshold:
        return False
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY).astype(np.float32)
    col_mean = gray.mean(axis=0)
    detrended = col_mean - np.convolve(
        col_mean, np.ones(7) / 7.0, mode="same"
    )
    zero_crossings = int(np.sum(np.diff(np.sign(detrended)) != 0))
    return zero_crossings >= stripe_periodicity_threshold
