"""Detect players in a video and write an annotated output.

This script does pixel-space detection only. Field registration plugs in
once you've supplied correspondences for the camera; see pipeline.FieldAnalyzer.

Usage:
    python scripts/detect.py INPUT.mp4 OUTPUT.mp4 [--max-frames N]
"""
import argparse

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

from football_tracker.detection import PersonDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--score-threshold", type=float, default=0.5)
    args = p.parse_args()

    detector = PersonDetector(score_threshold=args.score_threshold)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    pbar = tqdm(total=total, desc="frames")
    n = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = detector.detect(frame_rgb)
        if dets:
            sv_dets = sv.Detections(
                xyxy=np.array([d.bbox_xyxy for d in dets], dtype=np.float32),
                confidence=np.array([d.score for d in dets], dtype=np.float32),
                class_id=np.zeros(len(dets), dtype=int),
            )
            labels = [f"{d.score:.2f}" for d in dets]
            frame_bgr = box_annotator.annotate(frame_bgr, sv_dets)
            frame_bgr = label_annotator.annotate(frame_bgr, sv_dets, labels=labels)
        writer.write(frame_bgr)
        n += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()
    print(f"wrote {n} frames to {args.output}")


if __name__ == "__main__":
    main()
