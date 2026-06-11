"""Run the connected-components yard-line detector on every frame of a video
and write an annotated mp4 — so we can see how stable detection is over time.
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
from football_tracker.field_coords import FIELD_LENGTH_YD, FIELD_WIDTH_YD, FieldPoint
from football_tracker.minimap import composite_minimap, render_minimap
from football_tracker.registration import FieldRegistration


@dataclass
class MinimapPlayer:
    field_pos: FieldPoint
    tracker_id: int


def detect_lines_in_frame(img_bgr, field_mask, player_mask):
    H, W = img_bgr.shape[:2]
    mask = cv2.bitwise_and(field_mask, cv2.bitwise_not(player_mask))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, lm = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)
    lm = cv2.bitwise_and(lm, mask)
    lm = cv2.morphologyEx(
        lm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    )
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(lm, connectivity=8)
    lines = []
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] < 80:
            continue
        pixels = np.argwhere(labels == i)
        if len(pixels) < 30:
            continue
        pts = pixels[:, ::-1].astype(np.float32)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        long_axis = eigvecs[:, 1]
        long_var = eigvals[1]
        short_var = eigvals[0]
        if short_var < 1e-3 or long_var < 200:
            continue
        elongation = long_var / short_var
        if elongation < 25:
            continue
        ang = np.arctan2(long_axis[1], long_axis[0]) % np.pi
        norm_ang = (ang + np.pi / 2) % np.pi
        rho = centroid[0] * np.cos(norm_ang) + centroid[1] * np.sin(norm_ang)
        lines.append((ang, rho, len(pts)))
    return lines


def cluster_lines(lines):
    def ad(a, b):
        d = abs(a - b)
        return min(d, np.pi - d)
    out = []
    for ln in sorted(lines, key=lambda x: x[1]):
        merged = False
        for c in out:
            if ad(ln[0], c["ang"]) < np.radians(8) and abs(ln[1] - c["rho"]) < 20:
                c["lines"].append(ln)
                c["ang"] = float(np.mean([l[0] for l in c["lines"]]))
                c["rho"] = float(np.mean([l[1] for l in c["lines"]]))
                merged = True
                break
        if not merged:
            out.append({"ang": ln[0], "rho": ln[1], "lines": [ln]})
    return out


def split_yard_vs_other(clusters):
    if not clusters:
        return [], []
    def ad(a, b):
        d = abs(a - b)
        return min(d, np.pi - d)
    median_ang = float(np.median([c["ang"] for c in clusters]))
    yard = [c for c in clusters if ad(c["ang"], median_ang) < np.radians(10)]
    other = [c for c in clusters if ad(c["ang"], median_ang) >= np.radians(10)]
    return yard, other, median_ang


def draw_clipped(img, ang, rho, mask, color, thickness):
    norm_theta = (ang + np.pi / 2) % np.pi
    a, b = np.cos(norm_theta), np.sin(norm_theta)
    x0, y0 = a * rho, b * rho
    L = 3000
    p1 = (int(x0 + L * (-b)), int(y0 + L * a))
    p2 = (int(x0 - L * (-b)), int(y0 - L * a))
    canvas = np.zeros_like(mask)
    cv2.line(canvas, p1, p2, 255, thickness)
    in_field = cv2.bitwise_and(canvas, mask)
    img[in_field > 0] = color


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("video")
    p.add_argument("output")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pixel_pts = np.array([c["pixel"] for c in cfg["correspondences"]], dtype=np.float64)
    field_pts = np.array([c["field"] for c in cfg["correspondences"]], dtype=np.float64)
    reg = FieldRegistration.from_correspondences(pixel_pts, field_pts)
    m = 0.5
    corners_field = np.array([
        [10 + m, m],
        [FIELD_LENGTH_YD - 10 - m, m],
        [FIELD_LENGTH_YD - 10 - m, FIELD_WIDTH_YD - m],
        [10 + m, FIELD_WIDTH_YD - m],
    ])
    corners_px = reg.field_to_pixel(corners_field).astype(np.int32)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    field_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(field_mask, [corners_px], 255)

    detector = PersonDetector(score_threshold=0.30)
    tracker = sv.ByteTrack()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    n_frames = 0
    n_yard_total = 0
    pbar = tqdm(total=total, desc="frames")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector.detect(rgb)
        player_mask = np.zeros((H, W), dtype=np.uint8)
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
            pad = 12
            cv2.rectangle(player_mask,
                          (max(0, x1 - pad), max(0, y1 - pad)),
                          (min(W, x2 + pad), min(H, y2 + pad)),
                          255, -1)
        lines = detect_lines_in_frame(frame, field_mask, player_mask)
        clusters = cluster_lines(lines)
        if clusters:
            yard, other, mang = split_yard_vs_other(clusters)
            for c in yard:
                draw_clipped(frame, c["ang"], c["rho"], field_mask, (0, 255, 255), 3)
            n_yard_total += len(yard)
            cv2.putText(frame,
                        f"yard:{len(yard)} other:{len(other)} ang:{round(np.degrees(mang))}d",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
        minimap_players = []
        if dets:
            xyxy = np.array([d.bbox_xyxy for d in dets], dtype=np.float32)
            scores = np.array([d.score for d in dets], dtype=np.float32)
            sv_dets = sv.Detections(
                xyxy=xyxy, confidence=scores,
                class_id=np.zeros(len(dets), dtype=int),
            )
            sv_dets = tracker.update_with_detections(sv_dets)
            foot_pts = np.column_stack([
                (sv_dets.xyxy[:, 0] + sv_dets.xyxy[:, 2]) / 2.0,
                sv_dets.xyxy[:, 3],
            ])
            field_pts_yd = reg.pixel_to_field(foot_pts)
            for i, tid in enumerate(sv_dets.tracker_id):
                fx, fy = field_pts_yd[i]
                fp = FieldPoint(x=float(fx), y=float(fy))
                if fp.in_field():
                    minimap_players.append(MinimapPlayer(field_pos=fp, tracker_id=int(tid)))
                box = sv_dets.xyxy[i]
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                              (180, 180, 0), 1)
                cv2.putText(frame, f"#{int(tid)}",
                            (int(box[0]), max(0, int(box[1]) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        mini = render_minimap(minimap_players, width=min(360, W // 2))
        frame = composite_minimap(frame, mini)

        writer.write(frame)
        n_frames += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()
    print(f"wrote {n_frames} frames; avg yard lines/frame = {n_yard_total / max(1, n_frames):.1f}")


if __name__ == "__main__":
    main()
