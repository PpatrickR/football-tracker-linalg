"""End-to-end demo: frame -> {detections, field-coord positions} overlay.

Reads the same YAML that visualize_registration.py uses, runs detection +
registration, and writes an annotated PNG that shows:
  - boxes around detected people
  - foot-point dots (the contact-with-field point)
  - projected yard lines and sidelines
  - field-coord text on each player (yards-from-left-goal-line, yards-from-sideline)

Usage:
    python scripts/demo_pipeline.py CONFIG.yaml OUTPUT.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from football_tracker.detection import PersonDetector
from football_tracker.field_coords import FIELD_LENGTH_YD, FIELD_WIDTH_YD, yard_lines
from football_tracker.pipeline import FieldAnalyzer
from football_tracker.registration import FieldRegistration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("output")
    p.add_argument("--score-threshold", type=float, default=0.35)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    img = cv2.imread(cfg["image"])
    if img is None:
        raise SystemExit(f"Could not read {cfg['image']}")

    pixel_pts = np.array([c["pixel"] for c in cfg["correspondences"]], dtype=np.float64)
    field_pts = np.array([c["field"] for c in cfg["correspondences"]], dtype=np.float64)
    reg = FieldRegistration.from_correspondences(pixel_pts, field_pts)

    detector = PersonDetector(score_threshold=args.score_threshold)
    analyzer = FieldAnalyzer(detector=detector, registration=reg)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    players = analyzer.analyze(img_rgb)
    print(f"detected {len(players)} people; projecting to field coords")

    for x in yard_lines():
        endpoints_field = np.array([[x, 0.0], [x, FIELD_WIDTH_YD]])
        endpoints_px = reg.field_to_pixel(endpoints_field).astype(int)
        cv2.line(img, tuple(endpoints_px[0]), tuple(endpoints_px[1]), (0, 200, 200), 1)
    for y in (0.0, FIELD_WIDTH_YD):
        endpoints_field = np.array([[10.0, y], [FIELD_LENGTH_YD - 10.0, y]])
        endpoints_px = reg.field_to_pixel(endpoints_field).astype(int)
        cv2.line(img, tuple(endpoints_px[0]), tuple(endpoints_px[1]), (0, 200, 0), 2)

    for player in players:
        x1, y1, x2, y2 = (int(v) for v in player.detection.bbox_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
        fx, fy = player.detection.foot_point
        cv2.circle(img, (int(fx), int(fy)), 4, (0, 0, 255), -1)
        label = f"({player.field_pos.x:.0f},{player.field_pos.y:.0f})"
        cv2.putText(
            img, label, (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
        )
        print(
            f"  bbox=({x1},{y1},{x2},{y2}) "
            f"field=({player.field_pos.x:.1f}yd, {player.field_pos.y:.1f}yd) "
            f"score={player.detection.score:.2f}"
        )

    cv2.imwrite(args.output, img)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
