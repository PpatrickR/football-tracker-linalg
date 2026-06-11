"""Track players across a video segment with feature-tracked registration
and a top-down minimap composite.

The YAML's `image` is the *anchor frame* — the homography is hand-picked
relative to that frame, then ORB feature matching tracks the camera's motion
frame-to-frame so the field grid follows pans/zooms.

Usage:
    python scripts/track_video.py CONFIG.yaml VIDEO.mp4 OUTPUT.mp4
       [--no-ref-filter] [--score-threshold 0.30]
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from tqdm import tqdm

from football_tracker.detection import PersonDetector
from football_tracker.field_coords import (
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    FieldPoint,
    yard_lines,
)
from football_tracker.filters import is_referee
from football_tracker.minimap import composite_minimap, render_minimap
from football_tracker.registration import FieldRegistration
from football_tracker.yard_line_registration import YardLineRegistration


@dataclass
class MinimapPlayer:
    field_pos: FieldPoint
    tracker_id: int


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("video")
    p.add_argument("output")
    p.add_argument("--score-threshold", type=float, default=0.30)
    p.add_argument("--no-ref-filter", action="store_true")
    p.add_argument("--no-minimap", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pixel_pts = np.array([c["pixel"] for c in cfg["correspondences"]], dtype=np.float64)
    field_pts = np.array([c["field"] for c in cfg["correspondences"]], dtype=np.float64)
    static_reg = FieldRegistration.from_correspondences(pixel_pts, field_pts)
    reg = YardLineRegistration(static_reg)

    detector = PersonDetector(score_threshold=args.score_threshold)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.4, text_padding=2)
    trace_annotator = sv.TraceAnnotator(thickness=1, trace_length=20)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    pbar = tqdm(total=total, desc="frames")
    n_frames = 0
    n_kept_total = 0
    n_dropped_ref = 0
    n_dropped_oof = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_dets = detector.detect(frame_rgb)
        det_boxes = np.array([d.bbox_xyxy for d in raw_dets]) if raw_dets else None
        reg.update(frame_bgr, det_boxes)

        kept = []
        for d in raw_dets:
            if not args.no_ref_filter and is_referee(d, frame_bgr):
                n_dropped_ref += 1
                continue
            fp_arr = reg.pixel_to_field(np.array([d.foot_point]))
            if not FieldPoint(
                x=float(fp_arr[0, 0]), y=float(fp_arr[0, 1])
            ).in_field():
                n_dropped_oof += 1
                continue
            kept.append(d)

        minimap_players = []
        if kept:
            xyxy = np.array([d.bbox_xyxy for d in kept], dtype=np.float32)
            scores = np.array([d.score for d in kept], dtype=np.float32)
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=scores,
                class_id=np.zeros(len(kept), dtype=int),
            )
            sv_dets = tracker.update_with_detections(sv_dets)
            n_kept_total += len(sv_dets)

            foot_pts_px = np.column_stack(
                [(sv_dets.xyxy[:, 0] + sv_dets.xyxy[:, 2]) / 2.0, sv_dets.xyxy[:, 3]]
            )
            field_pts_yd = reg.pixel_to_field(foot_pts_px)
            labels = []
            for tid, fp in zip(sv_dets.tracker_id, field_pts_yd):
                labels.append(f"#{int(tid)} ({fp[0]:.0f},{fp[1]:.0f})")
                minimap_players.append(
                    MinimapPlayer(
                        field_pos=FieldPoint(x=float(fp[0]), y=float(fp[1])),
                        tracker_id=int(tid),
                    )
                )
            frame_bgr = trace_annotator.annotate(frame_bgr, sv_dets)
            frame_bgr = box_annotator.annotate(frame_bgr, sv_dets)
            frame_bgr = label_annotator.annotate(frame_bgr, sv_dets, labels=labels)

        for x_yd in yard_lines():
            ends = reg.field_to_pixel(np.array([[x_yd, 0.0], [x_yd, FIELD_WIDTH_YD]]))
            cv2.line(frame_bgr, tuple(ends[0].astype(int)),
                     tuple(ends[1].astype(int)), (0, 200, 200), 1)
        for y_yd in (0.0, FIELD_WIDTH_YD):
            ends = reg.field_to_pixel(
                np.array([[10.0, y_yd], [FIELD_LENGTH_YD - 10.0, y_yd]])
            )
            cv2.line(frame_bgr, tuple(ends[0].astype(int)),
                     tuple(ends[1].astype(int)), (0, 220, 0), 2)

        if not args.no_minimap:
            mini = render_minimap(minimap_players)
            frame_bgr = composite_minimap(frame_bgr, mini)

        writer.write(frame_bgr)
        n_frames += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(
        f"wrote {n_frames} frames to {args.output}\n"
        f"  kept {n_kept_total} tracked detections "
        f"(dropped {n_dropped_ref} refs, {n_dropped_oof} out-of-field)\n"
        f"  yard-line registration: {reg.stats.n_corrected}/{reg.stats.n_frames} "
        f"frames corrected (last residual {reg.stats.last_residual_yd:+.2f} yd, "
        f"last n_peaks {reg.stats.last_n_peaks})"
    )


if __name__ == "__main__":
    main()
