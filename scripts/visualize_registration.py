"""Visualize a field registration by re-projecting yard lines onto the frame.

Workflow:
  1. Pull a frame from your video and figure out pixel coords of >=4 known
     points (yard-line/sideline intersections).
  2. Write them into a YAML config (see configs/sample_registration.yaml).
  3. Run this script. It overlays the FULL field grid (all yard lines +
     sidelines) on your frame using your homography. If they line up, your
     registration is good. If not, your correspondences are wrong.

Usage:
    python scripts/visualize_registration.py CONFIG.yaml OUTPUT.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from football_tracker.field_coords import (
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    yard_lines,
)
from football_tracker.registration import FieldRegistration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("output")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    image_path = cfg["image"]
    correspondences = cfg["correspondences"]

    pixel_pts = np.array([c["pixel"] for c in correspondences], dtype=np.float64)
    field_pts = np.array([c["field"] for c in correspondences], dtype=np.float64)
    reg = FieldRegistration.from_correspondences(pixel_pts, field_pts)

    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"Could not read {image_path}")

    for x in yard_lines():
        endpoints_field = np.array([[x, 0.0], [x, FIELD_WIDTH_YD]])
        endpoints_px = reg.field_to_pixel(endpoints_field).astype(int)
        cv2.line(
            img, tuple(endpoints_px[0]), tuple(endpoints_px[1]), (0, 255, 255), 1
        )

    for y in (0.0, FIELD_WIDTH_YD):
        endpoints_field = np.array([[10.0, y], [FIELD_LENGTH_YD - 10.0, y]])
        endpoints_px = reg.field_to_pixel(endpoints_field).astype(int)
        cv2.line(
            img, tuple(endpoints_px[0]), tuple(endpoints_px[1]), (0, 255, 0), 2
        )

    for c in correspondences:
        x, y = c["pixel"]
        cv2.circle(img, (int(x), int(y)), 6, (0, 0, 255), -1)

    cv2.imwrite(args.output, img)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
