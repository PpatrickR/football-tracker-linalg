"""Top-down 2D field minimap with player dots.

Renders the 120 yd x 53.333 yd field as a clean 2D image — endzones tinted
by team color, painted yard lines every 5 yd, yard numbers at 10/20/30/40/50.
Each player gets plotted as a dot at their (x_field, y_field) yard coords.
"""
from __future__ import annotations

import cv2
import numpy as np

from .field_coords import (
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    PLAYING_FIELD_X_MAX,
    PLAYING_FIELD_X_MIN,
    yard_lines,
)


def render_minimap(
    players,
    width: int = 480,
    height: int | None = None,
) -> np.ndarray:
    """Players is an iterable with .field_pos.x / .field_pos.y (yards) and
    optional .tracker_id (int) attributes."""
    if height is None:
        height = int(width * FIELD_WIDTH_YD / FIELD_LENGTH_YD)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (40, 90, 40)

    def to_px(x_yd: float, y_yd: float) -> tuple[int, int]:
        return (
            int(round(x_yd / FIELD_LENGTH_YD * width)),
            int(round(y_yd / FIELD_WIDTH_YD * height)),
        )

    cv2.rectangle(canvas, to_px(0, 0), to_px(PLAYING_FIELD_X_MIN, FIELD_WIDTH_YD),
                  (40, 30, 80), -1)
    cv2.rectangle(canvas, to_px(PLAYING_FIELD_X_MAX, 0),
                  to_px(FIELD_LENGTH_YD, FIELD_WIDTH_YD), (60, 80, 30), -1)

    for x_yd in yard_lines():
        p1 = to_px(x_yd, 0)
        p2 = to_px(x_yd, FIELD_WIDTH_YD)
        thickness = 2 if x_yd in (PLAYING_FIELD_X_MIN, PLAYING_FIELD_X_MAX, 60.0) else 1
        cv2.line(canvas, p1, p2, (220, 220, 220), thickness)

    for x_yd in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
        label = str(int(min(x_yd - 10, 110 - x_yd)))
        cv2.putText(canvas, label, to_px(x_yd - 1.0, 12.0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, to_px(x_yd - 1.0, 45.0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    for player in players:
        if not (0 <= player.field_pos.x <= FIELD_LENGTH_YD
                and 0 <= player.field_pos.y <= FIELD_WIDTH_YD):
            continue
        center = to_px(player.field_pos.x, player.field_pos.y)
        cv2.circle(canvas, center, 5, (255, 255, 255), -1)
        cv2.circle(canvas, center, 5, (0, 0, 0), 1)
        tid = getattr(player, "tracker_id", None)
        if tid is not None:
            cv2.putText(canvas, str(int(tid)),
                        (center[0] + 6, center[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


def composite_minimap(
    frame_bgr: np.ndarray,
    minimap: np.ndarray,
    margin: int = 10,
    border: int = 2,
) -> np.ndarray:
    """Paste the minimap onto the bottom-right corner of frame_bgr."""
    h, w = frame_bgr.shape[:2]
    mh, mw = minimap.shape[:2]
    y1, x1 = h - mh - margin, w - mw - margin
    y2, x2 = y1 + mh, x1 + mw
    cv2.rectangle(frame_bgr, (x1 - border, y1 - border),
                  (x2 + border, y2 + border), (255, 255, 255), border)
    frame_bgr[y1:y2, x1:x2] = minimap
    return frame_bgr
