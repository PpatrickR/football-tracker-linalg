#!/usr/bin/env python
"""Render a clean demo clip: player tracking + top-down minimap.

Primes a single anchor homography via auto-detection, then tracks camera
motion with ORB. Team colors computed once and frozen. No grid overlay.

Usage:
    python scripts/demo_play.py OUTPUT.mp4 [--start FRAME] [--duration SEC]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

sys.path.insert(0, "/home/patrick/football-tracker/src")
from football_tracker.detection import PersonDetector
from football_tracker.field_coords import FIELD_LENGTH_YD, FIELD_WIDTH_YD, yard_lines

VIDEO = "/home/patrick/videoplayback.mp4"
FIELD_W_YD = FIELD_WIDTH_YD


# ── Auto-detection (from overnight/cycles) ───────────────────────────────────

def detect_paint_segments(fr):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    turf = cv2.inRange(hsv, (30, 25, 25), (95, 255, 255))
    near = cv2.dilate(turf, np.ones((15, 15), np.uint8))
    paint = cv2.bitwise_and(cv2.inRange(g, 160, 255), near)
    edges = cv2.Canny(paint, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45,
                            minLineLength=45, maxLineGap=12)
    segs = []
    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, l)
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
            segs.append((x1, y1, x2, y2, ang))
    return segs


def fit_line(pts):
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    return np.array([n[0], n[1], -n @ c])


def ransac_line(segs_pts, n_iter=200, thr=4.0):
    pts = np.asarray(segs_pts, float)
    if len(pts) < 2:
        return None, 0
    best_in, best_n = None, -1
    rng = np.random.default_rng(0)
    for _ in range(n_iter):
        i, j = rng.integers(0, len(pts), 2)
        if i == j:
            continue
        p, q = pts[i], pts[j]
        d = q - p
        nrm = np.hypot(*d)
        if nrm < 1e-6:
            continue
        nvec = np.array([-d[1], d[0]]) / nrm
        dist = np.abs((pts - p) @ nvec)
        inl = dist < thr
        cnt = int(inl.sum())
        if cnt > best_n:
            best_n, best_in = cnt, inl
    if best_in is None or best_n < 2:
        return None, 0
    return fit_line(pts[best_in]), best_n


def line_intersection(L1, L2):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    A = np.array([[a1, b1], [a2, b2]])
    bb = -np.array([c1, c2])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    return np.linalg.solve(A, bb)


def dlt(img_pts, world_pts):
    A = []
    for (u, v), (X, Y) in zip(img_pts, world_pts):
        A.append([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u])
        A.append([0, 0, 0, -X, -Y, -1, v * X, v * Y, v])
    A = np.asarray(A, float)
    _, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2], S


def detect_sidelines(fr):
    h, w = fr.shape[:2]
    segs = detect_paint_segments(fr)
    shallow = [s for s in segs if s[4] < 35 or s[4] > 145]
    def ymid(s): return (s[1] + s[3]) / 2.0
    far_pts, near_pts = [], []
    for s in shallow:
        if abs(s[2] - s[0]) < 80:
            continue
        if ymid(s) < h * 0.45:
            far_pts += [(s[0], s[1]), (s[2], s[3])]
        elif ymid(s) > h * 0.55:
            near_pts += [(s[0], s[1]), (s[2], s[3])]
    L_far, _ = ransac_line(far_pts) if far_pts else (None, 0)
    L_near, _ = ransac_line(near_pts) if near_pts else (None, 0)
    return L_near, L_far


def detect_yard_columns(fr):
    h, w = fr.shape[:2]
    segs = detect_paint_segments(fr)
    steep = [s for s in segs if 35 <= s[4] <= 145]
    def x_at_y(s, yq):
        x1, y1, x2, y2 = s[:4]
        return None if y2 == y1 else x1 + (x2 - x1) * (yq - y1) / (y2 - y1)
    xs = []
    for s in steep:
        xv = x_at_y(s, h * 0.5)
        if xv is not None and -60 < xv < w + 60:
            xs.append((xv, s))
    xs.sort()
    clusters = []
    for xv, s in xs:
        if clusters and xv - clusters[-1][-1][0] < 25:
            clusters[-1].append((xv, s))
        else:
            clusters.append([(xv, s)])
    yl = []
    for c in clusters:
        if len(c) < 2:
            continue
        pts = []
        for _, s in c:
            pts += [(s[0], s[1]), (s[2], s[3])]
        ys = [p[1] for p in pts]
        if (max(ys) - min(ys)) < 25:
            continue
        L = fit_line(pts)
        mx = float(np.mean([a[0] for a in c]))
        yl.append((mx, L))
    yl.sort(key=lambda t: t[0])
    return yl


def assign_world_x(col_xs):
    n = len(col_xs)
    if n < 2:
        return None
    gaps = np.diff(col_xs)
    pitch = np.median(gaps[gaps > 3]) if np.any(gaps > 3) else np.median(gaps)
    if pitch <= 1:
        return None
    steps = np.round(gaps / pitch).clip(1, None).astype(int)
    cum = np.concatenate([[0], np.cumsum(steps)])
    return cum * 5.0


def auto_register(fr):
    """Full-rank auto homography. Returns (H_world_to_img, cond_gap) or (None, 0)."""
    h, w = fr.shape[:2]
    cols = detect_yard_columns(fr)
    col_xs = [c[0] for c in cols]
    L_near, L_far = detect_sidelines(fr)
    Xworld = assign_world_x(col_xs)
    if Xworld is None:
        return None, 0

    img_pts, world_pts = [], []
    for i, (xm, L) in enumerate(cols):
        Xf = Xworld[i]
        for L_side, Yf in [(L_near, 0.0), (L_far, FIELD_W_YD)]:
            if L_side is None:
                continue
            ip = line_intersection(L, L_side)
            if ip is None:
                continue
            u, v = ip
            if -120 < u < w + 120 and -120 < v < h + 120:
                img_pts.append((u, v))
                world_pts.append((Xf, Yf))

    n_near = sum(1 for _, (_, Y) in zip(img_pts, world_pts) if Y < 1)
    n_far = sum(1 for _, (_, Y) in zip(img_pts, world_pts) if Y > FIELD_W_YD - 1)
    if len(img_pts) < 8 or n_near < 2 or n_far < 2:
        return None, 0

    H, S = dlt(img_pts, world_pts)
    cg = S[-2] / S[-1] if S[-1] > 1e-12 else float("inf")
    if not np.isfinite(cg) or cg < 5:
        return None, 0
    return H, cg


# ── ORB Tracking ─────────────────────────────────────────────────────────────

class ORBTracker:
    def __init__(self, n_features=3000, min_matches=30):
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev_gray = None
        self._min_matches = min_matches

    def set_reference(self, frame_bgr):
        self._prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def estimate_motion(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return None
        kp_prev, des_prev = self._orb.detectAndCompute(self._prev_gray, None)
        kp_curr, des_curr = self._orb.detectAndCompute(gray, None)
        self._prev_gray = gray
        if des_prev is None or des_curr is None:
            return None
        if len(kp_prev) < self._min_matches or len(kp_curr) < self._min_matches:
            return None
        matches = self._matcher.match(des_prev, des_curr)
        if len(matches) < self._min_matches:
            return None
        src = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if M is None or (mask is not None and mask.sum() < self._min_matches):
            return None
        return M


# ── Registration (prime once, track with ORB, no re-anchoring) ───────────────

class RegistrationManager:
    def __init__(self):
        self.orb = ORBTracker()
        self.H = None
        self.H_inv = None
        self._anchor_H = None
        self._motion_accum = np.eye(3, dtype=np.float64)
        self.last_motion = None   # this frame's raw ORB motion (consumers: LogoEraser)
        self.stats = {"auto": 0, "orb": 0, "stale": 0}

    def update(self, frame_bgr, try_auto=False):
        motion = self.orb.estimate_motion(frame_bgr)
        self.last_motion = motion

        if try_auto or self.H is None:
            H_auto, cg = auto_register(frame_bgr)
            if H_auto is not None:
                self.H = H_auto
                self.H_inv = np.linalg.inv(H_auto)
                self._anchor_H = H_auto.copy()
                self._motion_accum = np.eye(3, dtype=np.float64)
                self.orb.set_reference(frame_bgr)
                self.stats["auto"] += 1
                return

        if self._anchor_H is not None and motion is not None:
            self._motion_accum = motion @ self._motion_accum
            H_new = self._motion_accum @ self._anchor_H
            self.H = H_new
            self.H_inv = np.linalg.inv(H_new)
            self.stats["orb"] += 1
        else:
            self.stats["stale"] += 1

    def pixel_to_field(self, pts):
        if self.H_inv is None:
            return pts
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        out = (self.H_inv @ homog.T).T
        return out[:, :2] / out[:, 2:3]

    def field_to_pixel(self, pts):
        if self.H is None:
            return pts
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homog = np.hstack([pts, np.ones((len(pts), 1))])
        out = (self.H @ homog.T).T
        return out[:, :2] / out[:, 2:3]

    @property
    def ready(self):
        return self.H is not None


# ── Team assignment via spatial matching on labeled frame ─────────────────────
#
# Match detections to hand-labeled boxes by IoU on the first frame.
# Lock team per tracker ID. No color classification — labels are ground truth.

class SpatialTeamAssigner:
    """On the labeled frame, match each detection to the nearest label by IoU.
    Lock that team assignment to the tracker ID. Unmatched = ref/unknown."""

    def __init__(self):
        self._label_boxes = []   # (x1,y1,x2,y2,team) from JSON
        self._label_frame = -1
        self._tid_team = {}      # tracker_id -> "green"/"white"/None
        self.darker_team = 0
        self._ready = False

    def load_labels(self, label_files, cap):
        import json
        best_count = 0
        best_boxes = []
        best_frame = -1
        for lf in label_files:
            with open(lf) as f:
                ldata = json.load(f)
            boxes = [(b["x1"], b["y1"], b["x2"], b["y2"], b["team"])
                     for b in ldata["boxes"]
                     if b["team"] in ("green", "white", "ref")]
            if len(boxes) > best_count:
                best_count = len(boxes)
                best_boxes = boxes
                best_frame = ldata["frame"]
        if best_count >= 6:
            self._label_boxes = best_boxes
            self._label_frame = best_frame
            self._ready = True
        return self._ready

    @property
    def label_frame(self):
        return self._label_frame

    def get_inject_boxes(self):
        return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in self._label_boxes]

    def get_all_labels(self):
        return list(self._label_boxes)

    def assign_from_detections(self, sv_dets):
        """Match detections to labels by IoU. Call once on the labeled frame."""
        if not self._ready or sv_dets is None or len(sv_dets) == 0:
            return
        for i in range(len(sv_dets)):
            tid = int(sv_dets.tracker_id[i])
            dx1, dy1, dx2, dy2 = sv_dets.xyxy[i]
            best_iou, best_team = 0.0, None
            for lx1, ly1, lx2, ly2, lteam in self._label_boxes:
                ix1 = max(dx1, lx1); iy1 = max(dy1, ly1)
                ix2 = min(dx2, lx2); iy2 = min(dy2, ly2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = ((dx2-dx1)*(dy2-dy1) + (lx2-lx1)*(ly2-ly1) - inter)
                iou = inter / (union + 1e-9)
                if iou > best_iou:
                    best_iou, best_team = iou, lteam
            if best_iou > 0.10:
                self._tid_team[tid] = best_team

    def get_team(self, tracker_id):
        """Returns (team_idx, conf). 0=green, 1=white, 2=ref, -1=unknown."""
        t = self._tid_team.get(tracker_id)
        if t == "green":
            return 0, 1.0
        elif t == "white":
            return 1, 1.0
        elif t == "ref":
            return 2, 1.0
        else:
            return -1, 0.0

    def set_team(self, tracker_id, team):
        """Late assignment (e.g. from the color model) for tracks born mid-play."""
        if team in ("green", "white", "ref") and tracker_id not in self._tid_team:
            self._tid_team[tracker_id] = team

    @property
    def ready(self):
        return self._ready


# ── Color-subspace team model ────────────────────────────────────────────────
#
# Two affine subspaces per team in RGB space — JERSEY (upper half of the box)
# and PANTS (lower half) — plus a shared turf class (so grass pixels can't
# vote; midnight green shares a hue band with turf). Each class is its pixel
# mean + the top-k right singular vectors of the centered pixel matrix; a
# pixel belongs to the class with the smallest projection residual. Each half
# votes independently and the halves are combined weighted by their measured
# separability on the labeled pixels (if both teams wear white pants, the
# pants weight self-calibrates toward zero and the jersey decides).

class TeamColorSubspaces:
    """Nearest-subspace jersey+pants classifier for tracks the labels never
    covered."""

    TEAM_K = 1   # uniform-color shading varies mostly along one direction
    TURF_K = 2   # turf needs more room: lit/shadow/wear

    JERSEY_BAND = (0.15, 0.50)   # fraction of box height
    PANTS_BAND = (0.52, 0.78)    # below 0.78 is socks/feet/grass

    def __init__(self, min_pixels=20, min_score=0.60, lock_votes=5):
        self._min_pixels = min_pixels
        self._min_score = min_score
        self._lock_votes = lock_votes
        self._groups = {}    # "jersey"/"pants" -> [(name, mu, Vk), ...] green,white,turf
        self._weights = {}   # "jersey"/"pants" -> separability weight
        self._votes = {}     # tid -> {"green": n, "white": n}
        self._locked = {}    # tid -> "green"/"white"
        self.ready = False

    @staticmethod
    def _band_pixels(frame_bgr, box, band):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        h_img, w_img = frame_bgr.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        tx1 = max(0, x1 + int(bw * 0.20))
        tx2 = min(w_img, x2 - int(bw * 0.20))
        ty1 = max(0, y1 + int(bh * band[0]))
        ty2 = min(h_img, y1 + int(bh * band[1]))
        if tx2 - tx1 < 3 or ty2 - ty1 < 3:
            return None
        crop = frame_bgr[ty1:ty2, tx1:tx2]
        return crop.reshape(-1, 3)[:, ::-1].astype(np.float32)  # BGR -> RGB

    @staticmethod
    def _fit_basis(pixels, k):
        mu = pixels.mean(axis=0)
        _, S, Vt = np.linalg.svd(pixels - mu, full_matrices=False)
        return mu, Vt[:k].copy(), S

    @staticmethod
    def _residual_sq(pixels, mu, Vk):
        d = pixels - mu
        proj = d @ Vk.T
        return np.maximum((d * d).sum(axis=1) - (proj * proj).sum(axis=1), 0.0)

    def _claim(self, pixels, classes):
        """Index of the nearest subspace per pixel (green=0, white=1, turf=2)."""
        res = np.stack([self._residual_sq(pixels, mu, Vk)
                        for _, mu, Vk in classes], axis=1)
        return res.argmin(axis=1)

    def _fit_group(self, gname, green, white, turf, ref=None):
        """Fit green/white[/ref]/turf subspaces for one body band, with one
        refinement pass, and measure the band's green-vs-white separability.
        The ref class (striped shirt -> flat gray at this resolution) is
        optional: present only when the labels include a 'ref' box."""
        classes = [("green", green, self.TEAM_K), ("white", white, self.TEAM_K)]
        if ref is not None:
            classes.append(("ref", ref, self.TEAM_K))
        classes.append(("turf", turf, self.TURF_K))
        fitted = [(name, *self._fit_basis(px, k)[:2]) for name, px, k in classes]

        # Refinement: refit each team on the pixels its own subspace claims
        refined = []
        for i, (name, px, k) in enumerate(classes):
            if name != "turf":
                own = px[self._claim(px, fitted) == i]
                if len(own) >= self._min_pixels:
                    px = own
            mu, Vk, S = self._fit_basis(px, k)
            refined.append((name, mu, Vk))
            print(f"  {gname}[{name}]: {len(px)} px, mean RGB "
                  f"({mu[0]:.0f},{mu[1]:.0f},{mu[2]:.0f}), sv {S[:3].round(1)}")
        self._groups[gname] = refined

        # Separability: how well does this band classify its own training
        # pixels? 0.5 = chance -> weight 0; 1.0 = perfect -> weight 1.
        accs = []
        for i, px in [(0, green), (1, white)]:
            claim = self._claim(px, refined)
            team_pix = claim < 2
            if team_pix.sum() > 0:
                accs.append((claim[team_pix] == i).mean())
        acc = float(np.mean(accs)) if accs else 0.5
        self._weights[gname] = max(0.0, 2.0 * acc - 1.0)
        print(f"  {gname}: separability acc={acc:.2f} -> weight {self._weights[gname]:.2f}")

    def fit(self, frame_bgr, label_boxes):
        # Turf sample first: outside every label box, inside the canonical
        # turf range — it doubles as the global illumination reference
        h_img, w_img = frame_bgr.shape[:2]
        outside = np.ones((h_img, w_img), np.uint8)
        for x1, y1, x2, y2, _ in label_boxes:
            cv2.rectangle(outside, (int(x1) - 8, int(y1) - 8),
                          (int(x2) + 8, int(y2) + 8), 0, -1)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        turf_mask = cv2.inRange(hsv, (30, 25, 25), (95, 255, 255))
        sel = (outside > 0) & (turf_mask > 0)
        turf = frame_bgr[sel][:, ::-1].astype(np.float32)
        rng = np.random.default_rng(0)
        if len(turf) > 5000:
            turf = turf[rng.choice(len(turf), 5000, replace=False)]
        self._turf_global = turf.mean(axis=0)  # illumination reference

        # Label crops, illumination-normalized like every later query
        bands = {"jersey": self.JERSEY_BAND, "pants": self.PANTS_BAND}
        px = {g: {"green": [], "white": [], "ref": []} for g in bands}
        for x1, y1, x2, y2, team in label_boxes:
            gain = self._local_gain(frame_bgr, (x1, y1, x2, y2))
            for gname, band in bands.items():
                p = self._band_pixels(frame_bgr, (x1, y1, x2, y2), band)
                if p is not None:
                    if gain is not None:
                        p = np.clip(p * gain, 0.0, 255.0)
                    px[gname][team].append(p)
        if any(not px[g][t] for g in bands for t in ("green", "white")):
            return False

        for gname in bands:
            ref = np.vstack(px[gname]["ref"]) if px[gname]["ref"] else None
            self._fit_group(gname, np.vstack(px[gname]["green"]),
                            np.vstack(px[gname]["white"]), turf, ref)

        if sum(self._weights.values()) < 0.05:
            print("  WARNING: neither band separates the teams; color model off")
            return False
        self.ready = True
        return True

    MIN_BOX_H = 30  # px; color evidence is inadmissible below this scale.
                    # Was 24 (just under the smallest hand-labeled box), but
                    # 24-26px boxes proved able to lock WRONG teams late in
                    # the play (the trailing pair at f288 vote-locked green
                    # twice in two runs on 25-26px boxes). Better an honest
                    # gray than a confident wrong color.

    def _local_gain(self, frame_bgr, box):
        """Per-channel illumination gain from the turf at the player's feet
        (shadow scales all channels down; the ground he stands on shares his
        light). Returns None when there's no clean turf patch to measure."""
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        h_img, w_img = frame_bgr.shape[:2]
        bh = max(1, y2 - y1)
        gy1, gy2 = min(h_img, y2), min(h_img, y2 + max(6, bh // 3))
        gx1, gx2 = max(0, x1 - 4), min(w_img, x2 + 4)
        if gy2 - gy1 < 3 or gx2 - gx1 < 3:
            return None
        patch = frame_bgr[gy1:gy2, gx1:gx2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, (30, 25, 25), (95, 255, 255)) > 0).reshape(-1)
        if mask.sum() < 30:
            return None
        local = patch.reshape(-1, 3)[mask][:, ::-1].astype(np.float32).mean(axis=0)
        gain = self._turf_global / np.maximum(local, 1.0)
        return np.clip(gain, 0.5, 2.5)

    def _band_fracs(self, frame_bgr, box, gname):
        """Per-team pixel fraction in one band, turf claims excluded,
        illumination-normalized by the turf at the player's feet.
        Returns {"green": f, "white": f} or None if too few team pixels
        or the box is below the scale the subspaces were trained on."""
        if box[3] - box[1] < self.MIN_BOX_H:
            return None
        band = self.JERSEY_BAND if gname == "jersey" else self.PANTS_BAND
        px = self._band_pixels(frame_bgr, box, band)
        if px is not None:
            gain = self._local_gain(frame_bgr, box)
            if gain is not None:
                px = np.clip(px * gain, 0.0, 255.0)
        if px is None or len(px) < self._min_pixels:
            return None
        groups = self._groups[gname]
        claim = self._claim(px, groups)
        counts = {name: int((claim == i).sum())
                  for i, (name, _, _) in enumerate(groups) if name != "turf"}
        n_team = sum(counts.values())
        if n_team < self._min_pixels:
            return None
        return {name: c / n_team for name, c in counts.items()}

    def check_team(self, frame_bgr, box, team):
        """One-shot appearance score for `team` in this box (0..1), or None
        if there isn't enough pixel evidence. Used as an inheritance veto."""
        score = {"green": 0.0, "white": 0.0}
        wsum = 0.0
        for gname, w in self._weights.items():
            if w <= 0.0:
                continue
            fracs = self._band_fracs(frame_bgr, box, gname)
            if fracs is None:
                continue
            for t in score:
                score[t] += w * fracs[t]
            wsum += w
        if wsum <= 0.0:
            return None
        return score[team] / wsum

    def vote(self, tid, frame_bgr, box):
        """Accumulate a color vote for this track. Returns the locked class
        ("green"/"white"/"ref") once confident, else None."""
        if tid in self._locked:
            return self._locked[tid]
        score = {}
        wsum = 0.0
        band_winners = []
        for gname, w in self._weights.items():
            if w <= 0.0:
                continue
            fracs = self._band_fracs(frame_bgr, box, gname)
            if fracs is None:
                continue
            band_winners.append(max(fracs, key=fracs.get))
            for team, f in fracs.items():
                score[team] = score.get(team, 0.0) + w * f
            wsum += w
        if wsum <= 0.0:
            return None
        if len(set(band_winners)) > 1:
            return None  # jersey and pants disagree -> abstain this frame
        winner = max(score, key=score.get)
        if score[winner] / wsum < self._min_score:
            return None
        v = self._votes.setdefault(tid, {k: 0 for k in score})
        v[winner] = v.get(winner, 0) + 1
        runner_up = max((c for k, c in v.items() if k != winner), default=0)
        if v[winner] - runner_up >= self._lock_votes:
            self._locked[tid] = winner
            return winner
        return None


def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def nms(dets, iou_thresh=0.65):
    """Greedy NMS by score. RT-DETR sometimes emits 2-3 near-identical boxes
    for one player and nothing downstream dedupes them — ByteTrack happily
    births a twin track for each, doubling the boxes on screen."""
    out = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(_box_iou(d.bbox_xyxy, k.bbox_xyxy) < iou_thresh for k in out):
            out.append(d)
    return out


# ── Referee filter ───────────────────────────────────────────────────────────

def is_not_single_player(det):
    """Filter out boxes that can't be a single player — too wide, too small, etc."""
    x1, y1, x2, y2 = det.bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    if bh < 15 or bw < 6:
        return True
    if bw > bh * 1.1:
        return True
    return False


def is_referee(det, frame_bgr):
    x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 8 or y2 - y1 < 12:
        return False
    crop = frame_bgr[y1:y2, x1:x2]
    h = crop.shape[0]
    torso = crop[h // 4: 3 * h // 4]
    if torso.size == 0 or torso.shape[1] < 6:
        return False
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    sat = float(hsv[..., 1].mean())
    if sat >= 55.0:
        return False
    if sat < 35.0:
        return True
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY).astype(np.float32)
    col_mean = gray.mean(axis=0)
    detrended = col_mean - np.convolve(col_mean, np.ones(7) / 7.0, mode="same")
    zero_crossings = int(np.sum(np.diff(np.sign(detrended)) != 0))
    return zero_crossings >= 5


# ── Rescue tracker for players RT-DETR misses ────────────────────────────────

class MotionRescueTracker:
    """Keep lost team-assigned tracks alive briefly: anchor at the track's last
    seen box and warp by camera motion accumulated since the loss frame.
    A rescue entry dies when its track reappears, a detection covers it, or it
    times out."""

    def __init__(self, timeout_frames=60, draw_frames=15, max_rebirth_age=30):
        self._timeout = timeout_frames      # identity handoff window
        self._draw = draw_frames            # visual window: after ~0.5s the
                                            # player has moved; the box is fiction
        self._max_rebirth_age = max_rebirth_age  # only YOUNG tracks can be
                                            # rebirths; an old track standing
                                            # on a ghost is a different person
        self._last_seen = {}  # tid -> (x1, y1, x2, y2, team)
        self._lost = {}       # tid -> [x1, y1, x2, y2, team, inv_motion_at_loss, age]
        self._track_age = {}  # tid -> frames since first seen

    def flush(self):
        """Drop all lost entries (call on a registration break / camera cut —
        warped ghosts are meaningless across a cut)."""
        self._lost.clear()

    def update(self, sv_dets, classifier, motion_accum, frame_bgr=None,
               color_model=None):
        """Call once per frame with the live tracked detections.
        A track reborn under a new id on top of a lost box INHERITS the lost
        track's team (ID churn must not re-decide a known player's team) —
        unless its appearance clearly contradicts that team (color veto).
        Returns rescue boxes (x1, y1, x2, y2, team, tid) for tracks still lost."""
        tid_team = classifier._tid_team
        live = set()
        dets = []  # (tid, x1, y1, x2, y2)
        if sv_dets is not None and len(sv_dets) > 0 and sv_dets.tracker_id is not None:
            for i in range(len(sv_dets)):
                tid = int(sv_dets.tracker_id[i])
                live.add(tid)
                self._track_age[tid] = self._track_age.get(tid, 0) + 1
                box = tuple(float(v) for v in sv_dets.xyxy[i])
                dets.append((tid, *box))
                if tid_team.get(tid) in ("green", "white"):
                    self._last_seen[tid] = (*box, tid_team[tid])

        # Track reappeared under its own id -> kill its rescue entry
        for tid in live:
            self._lost.pop(tid, None)

        # Seed: team-assigned tracks that vanished this frame
        for tid, (x1, y1, x2, y2, team) in self._last_seen.items():
            if tid in live or tid in self._lost:
                continue
            try:
                inv = np.linalg.inv(motion_accum)
            except np.linalg.LinAlgError:
                continue
            self._lost[tid] = [x1, y1, x2, y2, team, inv, 0]

        # Emit: warp lost boxes by camera motion since their loss frame
        results = []
        for tid in list(self._lost):
            x1, y1, x2, y2, team, inv, age = self._lost[tid]
            age += 1
            if age > self._timeout:
                del self._lost[tid]
                del self._last_seen[tid]
                continue
            self._lost[tid][6] = age
            rel = motion_accum @ inv
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            hw, hh = (x2 - x1) / 2.0, (y2 - y1) / 2.0
            warped = rel @ np.array([cx, cy, 1.0])
            wx, wy = warped[0] / warped[2], warped[1] / warped[2]
            rx1, ry1 = int(wx - hw), int(wy - hh)
            rx2, ry2 = int(wx + hw), int(wy + hh)
            ghost_area = max(1.0, float(rx2 - rx1) * float(ry2 - ry1))
            best_unassigned, best_box, best_inter = None, None, 0.0
            covered = False
            for dtid, bx1, by1, bx2, by2 in dets:
                iw = min(rx2, bx2) - max(rx1, bx1)
                ih = min(ry2, by2) - max(ry1, by1)
                if iw <= 0 or ih <= 0:
                    continue
                covered = True
                if tid_team.get(dtid) in ("green", "white", "ref"):
                    # A KNOWN player (or the ref) standing on the ghost is not
                    # a rebirth — hide the ghost this frame but keep it alive
                    continue
                # A graze is not a rebirth: require real coverage, and an
                # established track standing on a ghost is a DIFFERENT
                # person the ghost drifted onto, not a rebirth
                inter = iw * ih
                if (inter > best_inter and inter > 0.3 * ghost_area
                        and self._track_age.get(dtid, 0) <= self._max_rebirth_age):
                    best_inter = inter
                    best_unassigned = dtid
                    best_box = (bx1, by1, bx2, by2)
            if best_unassigned is not None:
                # Appearance gate: hand over ONLY on positive support for
                # the ghost's team. "Can't verify" (None) and lukewarm
                # scores both block — silent wrong inheritance painted a
                # gray track green with no log trail to show for it.
                approved = False
                if color_model is not None and color_model.ready and frame_bgr is not None:
                    s = color_model.check_team(frame_bgr, best_box, team)
                    approved = s is not None and s >= 0.5
                if approved:
                    # Genuine rebirth under a new id: hand the team over
                    print(f"  HANDOVER: track {best_unassigned} inherits "
                          f"{team} from ghost {tid} (s={s:.2f})")
                    classifier.set_team(best_unassigned, team)
                    del self._lost[tid]
                    del self._last_seen[tid]
                    continue
            if covered or age > self._draw:
                continue  # alive for inheritance, but not drawn
            results.append((rx1, ry1, rx2, ry2, team, tid))
        return results


# ── Field-paint eraser (NFL shield etc.) ─────────────────────────────────────

class LogoEraser:
    """Erase static painted logos from the detector's view of the frame.

    Painted logos (the NFL shield near the 30) visually merge with players
    standing on them and RT-DETR drops the player. The logo is static scenery,
    so the frame-to-frame ORB motion chain maps it exactly: capture its
    appearance once from a frame where it's uncovered, accumulate camera
    motion from that frame on, and each frame warp the template into place,
    replacing pixels that still MATCH it (= visible logo) with turf color.
    Player pixels over the logo don't match the template and survive.

    Pure pixel space on purpose: registration isn't ready until late in the
    pre-roll, after players have settled onto the logo, but the ORB chain
    runs from the first primed frame. The match gate makes failure benign:
    if the chain drifts, live pixels stop matching and nothing is erased."""

    MATCH_THRESH = 70      # L1 over BGR; generous for heavy compression

    def __init__(self):
        self.ready = False
        self._canvas = None       # full-frame canvas holding the template
        self._valid = None        # mask of template pixels in the canvas
        self._M = np.eye(3, dtype=np.float64)   # capture frame -> current
        self._turf_ref = None     # median turf BGR near the logo at capture

    @staticmethod
    def _turf_mask(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (30, 25, 25), (95, 255, 255))

    def capture_at(self, frame_bgr, rect):
        """Capture the logo template from a hand-verified pixel rect.

        Auto-detecting the shield failed on this footage: compression crushes
        its blue to nothing and the away jerseys are white+red too, so every
        color heuristic either missed it or grabbed a player. The shield is
        one fixed thing in one fixed place — a hand-picked (frame, rect) is
        the honest solution."""
        x, y, w, h = rect
        h_img, w_img = frame_bgr.shape[:2]
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        self._canvas = np.zeros_like(frame_bgr)
        self._canvas[y:y2, x:x2] = frame_bgr[y:y2, x:x2]
        self._valid = np.zeros((h_img, w_img), np.uint8)
        self._valid[y:y2, x:x2] = 255
        # turf reference: median of turf pixels in a ring around the rect
        gx0, gy0 = max(0, x - 15), max(0, y - 15)
        gx1, gy1 = min(w_img, x2 + 15), min(h_img, y2 + 15)
        patch = frame_bgr[gy0:gy1, gx0:gx1].copy()
        patch[y - gy0:y2 - gy0, x - gx0:x2 - gx0] = 0
        tm = self._turf_mask(patch) > 0
        if tm.sum() < 50:
            self._canvas = None
            return False
        self._turf_ref = np.median(patch[tm].astype(np.float32), axis=0)
        self._M = np.eye(3, dtype=np.float64)
        self.ready = True
        print(f"  Logo template captured: rect {rect}")
        return True

    def advance(self, motion):
        """Accumulate one frame of camera motion (None = ORB failed; skip —
        the match gate in erase() absorbs the resulting drift)."""
        if self.ready and motion is not None:
            self._M = motion @ self._M

    def erase(self, frame_bgr):
        """Return a copy for the DETECTOR with visible logo pixels turfed.
        The display frame keeps the real pixels."""
        if not self.ready:
            return frame_bgr
        h_img, w_img = frame_bgr.shape[:2]
        warped = cv2.warpPerspective(self._canvas, self._M, (w_img, h_img))
        wmask = cv2.warpPerspective(self._valid, self._M, (w_img, h_img))
        if wmask.max() == 0:
            return frame_bgr  # logo out of view
        ys, xs = np.where(wmask > 0)
        bx0, bx1 = xs.min(), xs.max() + 1
        by0, by1 = ys.min(), ys.max() + 1
        # snap to the live frame: the ORB chain drifts ~a pixel over a few
        # hundred frames, which on a high-contrast pattern ruins a naive
        # diff. A masked template-match within +-PAD px re-locks it.
        PAD = 6
        wy0, wx0 = max(0, by0 - PAD), max(0, bx0 - PAD)
        wy1, wx1 = min(h_img, by1 + PAD), min(w_img, bx1 + PAD)
        tmpl_crop = warped[by0:by1, bx0:bx1]
        mask_crop = wmask[by0:by1, bx0:bx1]
        win = frame_bgr[wy0:wy1, wx0:wx1]
        if (win.shape[0] > tmpl_crop.shape[0] and
                win.shape[1] > tmpl_crop.shape[1] and mask_crop.any()):
            r = cv2.matchTemplate(win, tmpl_crop, cv2.TM_CCORR_NORMED,
                                  mask=mask_crop)
            _, _, _, loc = cv2.minMaxLoc(r)
            dx = (wx0 + loc[0]) - bx0
            dy = (wy0 + loc[1]) - by0
            if dx or dy:
                T = np.float64([[1, 0, dx], [0, 1, dy]])
                warped = cv2.warpAffine(warped, T, (w_img, h_img))
                wmask = cv2.warpAffine(wmask, T, (w_img, h_img))
                bx0, bx1, by0, by1 = bx0 + dx, bx1 + dx, by0 + dy, by1 + dy
        # illumination gain from turf around the logo, now vs capture
        gx0, gy0 = max(0, bx0 - 15), max(0, by0 - 15)
        gx1, gy1 = min(w_img, bx1 + 15), min(h_img, by1 + 15)
        around = frame_bgr[gy0:gy1, gx0:gx1]
        tm = self._turf_mask(around) > 0
        tm[by0 - gy0:by1 - gy0, bx0 - gx0:bx1 - gx0] = False
        if tm.sum() < 30:
            return frame_bgr
        turf_now = np.median(around[tm].astype(np.float32), axis=0)
        gain = np.clip(turf_now / np.maximum(self._turf_ref, 1.0), 0.5, 2.0)
        tmpl = np.clip(warped.astype(np.float32) * gain, 0, 255)
        # light blur on both sides absorbs compression ringing
        live = cv2.GaussianBlur(frame_bgr, (3, 3), 0).astype(np.float32)
        tmpl = cv2.GaussianBlur(tmpl.astype(np.uint8), (3, 3), 0).astype(np.float32)
        diff = np.abs(live - tmpl).sum(axis=2)
        erase = ((wmask > 0) & (diff < self.MATCH_THRESH)).astype(np.uint8)
        # 1px dilation (bounded by the logo mask) catches antialiased edges
        erase = cv2.dilate(erase, np.ones((3, 3), np.uint8)) & (wmask > 0)
        if not erase.any():
            return frame_bgr
        out = frame_bgr.copy()
        out[erase > 0] = turf_now.astype(np.uint8)
        return out


# ── Minimap ──────────────────────────────────────────────────────────────────

MINIMAP_W = 140
MINIMAP_H = int(MINIMAP_W * FIELD_W_YD / 50.0)
MINIMAP_ALPHA = 0.75

HOME_BGR = (20, 130, 0)
AWAY_BGR = (230, 230, 245)
UNCERTAIN_BGR = (90, 90, 90)


def render_minimap(players_field, team_labels, darker_team):
    # Fixed cross-field window, shifted DOWN the y-axis (a pure offset, not a
    # zoom) so the near-sideline players, which project to about -9 yd because
    # the width axis is rank-deficient, fit on the map. Span equals the field
    # width, so the scale is unchanged and the window never moves frame to
    # frame. Players outside the window (the far-side crowd at y ~ 47) are
    # cut, not clamped, so fans no longer leak onto the field.
    Y_VIEW_LO = -11.0
    Y_VIEW_HI = Y_VIEW_LO + FIELD_W_YD
    valid = []
    for (fx, fy), t in zip(players_field, team_labels):
        if not (-5 <= fx <= FIELD_LENGTH_YD + 5):
            continue
        if not (Y_VIEW_LO <= fy <= Y_VIEW_HI):
            continue
        valid.append((fx, fy, t))
    cx = np.median([p[0] for p in valid]) if valid else 60.0

    x_lo = max(0, cx - 25)
    x_hi = min(FIELD_LENGTH_YD, x_lo + 50)
    x_lo = x_hi - 50

    canvas = np.zeros((MINIMAP_H, MINIMAP_W, 3), dtype=np.uint8)
    canvas[:] = (45, 105, 45)

    def to_px(x_yd, y_yd):
        return (
            int(round((x_yd - x_lo) / 50.0 * MINIMAP_W)),
            int(round((Y_VIEW_HI - y_yd) / FIELD_W_YD * MINIMAP_H)),
        )

    for x_yd in yard_lines():
        if x_lo <= x_yd <= x_hi:
            p1, p2 = to_px(x_yd, Y_VIEW_LO), to_px(x_yd, Y_VIEW_HI)
            cv2.line(canvas, p1, p2, (100, 100, 100), 1)

    if Y_VIEW_LO <= 0 <= Y_VIEW_HI:
        cv2.line(canvas, to_px(x_lo, 0), to_px(x_hi, 0), (160, 160, 160), 2)
    if Y_VIEW_LO <= FIELD_W_YD <= Y_VIEW_HI:
        cv2.line(canvas, to_px(x_lo, FIELD_W_YD), to_px(x_hi, FIELD_W_YD), (160, 160, 160), 2)

    for fx, fy, team in valid:
        center = to_px(fx, fy)
        if team == -1:
            color = UNCERTAIN_BGR
        elif team == darker_team:
            color = HOME_BGR
        else:
            color = AWAY_BGR
        cv2.circle(canvas, center, 5, color, -1)
        cv2.circle(canvas, center, 5, (0, 0, 0), 1)

    cv2.rectangle(canvas, (0, 0), (MINIMAP_W - 1, MINIMAP_H - 1), (160, 160, 160), 1)
    return canvas


def composite_minimap(frame_bgr, minimap, margin=6):
    h, w = frame_bgr.shape[:2]
    mh, mw = minimap.shape[:2]
    y1 = h - mh - margin
    x1 = w - mw - margin
    if y1 < 0 or x1 < 0:
        return frame_bgr
    roi = frame_bgr[y1:y1 + mh, x1:x1 + mw]
    blended = cv2.addWeighted(minimap, MINIMAP_ALPHA, roi, 1.0 - MINIMAP_ALPHA, 0)
    frame_bgr[y1:y1 + mh, x1:x1 + mw] = blended
    return frame_bgr


# ── Smoothed field positions (EMA to reduce jitter) ─────────────────────────

CONF_THRESHOLD = 0.4
MIN_FRAMES_FOR_TEAM = 3


class PlayerSmoother:
    """EMA on field positions + majority-vote team + uncertainty tracking."""

    def __init__(self, alpha=0.5):
        self._alpha = alpha
        self._positions = {}
        self._team_votes = {}
        self._frame_counts = {}
        self._conf_accum = {}

    def smooth(self, tracker_id, fx, fy):
        if tracker_id in self._positions:
            ox, oy = self._positions[tracker_id]
            fx = self._alpha * fx + (1 - self._alpha) * ox
            fy = self._alpha * fy + (1 - self._alpha) * oy
        self._positions[tracker_id] = (fx, fy)
        self._frame_counts[tracker_id] = self._frame_counts.get(tracker_id, 0) + 1
        return fx, fy

    def vote_team(self, tracker_id, team_raw, conf):
        if tracker_id not in self._team_votes:
            self._team_votes[tracker_id] = [0.0, 0.0]
            self._conf_accum[tracker_id] = 0.0
        self._team_votes[tracker_id][team_raw] += conf
        self._conf_accum[tracker_id] += conf
        counts = self._team_votes[tracker_id]
        return 0 if counts[0] >= counts[1] else 1

    def is_uncertain(self, tracker_id):
        frames = self._frame_counts.get(tracker_id, 0)
        if frames < MIN_FRAMES_FOR_TEAM:
            return True
        counts = self._team_votes.get(tracker_id, [0, 0])
        total = counts[0] + counts[1]
        if total < 0.01:
            return True
        margin = abs(counts[0] - counts[1]) / total
        return margin < 0.3


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--start", type=int, default=72600)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--score-threshold", type=float, default=0.25)
    p.add_argument("--upscale", type=int, default=2)
    args = p.parse_args()

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(args.duration * fps)
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_out, h_out = w_in * args.upscale, h_in * args.upscale

    print(f"Segment: frame {args.start}, {args.duration:.0f}s ({total_frames} frames)")
    print(f"Output: {w_out}x{h_out} -> {args.output}")

    # ── Prime registration ──
    reg = RegistrationManager()
    prime_start = max(0, args.start - 300)
    print(f"Priming from frame {prime_start}...")
    # Hand-verified clean look at the NFL shield during this play's pre-roll:
    # source frame 72510, shield at pixel rect (240, 0, 45, 35). Top edge is
    # frame-cut there; the surviving ~85% is the part players stand on.
    LOGO_CAPTURE_FRAME, LOGO_CAPTURE_RECT = 72510, (240, 0, 45, 35)
    eraser = LogoEraser()
    cap.set(cv2.CAP_PROP_POS_FRAMES, prime_start)
    for pi in range(args.start - prime_start + 1):
        ok, fr = cap.read()
        if not ok:
            break
        reg.update(fr, try_auto=True)
        eraser.advance(reg.last_motion)
        if prime_start + pi == LOGO_CAPTURE_FRAME:
            eraser.capture_at(fr, LOGO_CAPTURE_RECT)
    print(f"  {reg.stats['auto']} anchors found, ready={reg.ready}, "
          f"logo template={'yes' if eraser.ready else 'NO'}")

    # ── Load detector ──
    print("Loading RT-DETR...")
    detector = PersonDetector(score_threshold=args.score_threshold)
    # Crossing/churn tuning, measured on this footage:
    #  - activation 0.2: small/occluded players detect at 0.25-0.40, under
    #    the default ~0.4 start bar (activation+0.1), so dead tracks could
    #    never restart. The field-bounds gate upstream keeps junk out.
    #  - buffer 90 (~3s): players occluded through a crossing reconnect to
    #    their ORIGINAL track instead of being reborn under a new id.
    #  - matching 0.8: accept IoU>0.2 re-associations; these players move
    #    little between frames but their boxes jitter at this resolution.
    tracker = sv.ByteTrack(
        track_activation_threshold=0.2,
        lost_track_buffer=90,
        minimum_matching_threshold=0.8,
        frame_rate=int(fps),
    )

    # ── Load spatial team assigner from hand-labeled data ──
    import glob
    classifier = SpatialTeamAssigner()
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    label_files = sorted(glob.glob(os.path.join(data_dir, "team_labels*.json")))
    if label_files and classifier.load_labels(label_files, cap):
        ng = sum(1 for b in classifier._label_boxes if b[4] == "green")
        nw = sum(1 for b in classifier._label_boxes if b[4] == "white")
        print(f"  Spatial assigner: {ng} green + {nw} white labels on frame {classifier.label_frame}")
    else:
        print("  WARNING: no labels available")

    # Human color corrections from scripts/review_colors.py: at each
    # correction's frame, the stored box is matched to a live track and that
    # track's class is PINNED — immune to votes, audits and handovers.
    # Box+frame is the durable key; track ids renumber between runs.
    corr_by_frame = {}
    corr_path = os.path.join(data_dir, "color_corrections.json")
    if os.path.exists(corr_path):
        with open(corr_path) as f:
            for c in json.load(f).get("corrections", []):
                if not c.get("auto"):
                    corr_by_frame.setdefault(c["frame"], []).append(
                        (c["box"], c["actual"]))
        print(f"  Loaded {sum(map(len, corr_by_frame.values()))} "
              f"manual color corrections")
    pinned = set()

    # Human frame edits from scripts/edit_tracks.py: on an edited frame the
    # human's boxes replace the tracker's DISPLAY output (boxes + minimap)
    # entirely. Processing state (votes/pins/rescue) still runs underneath
    # so unedited frames are unaffected.
    frame_edits = {}
    edits_path = os.path.join(data_dir, "track_edits.json")
    if os.path.exists(edits_path):
        with open(edits_path) as f:
            frame_edits = {int(k): v
                           for k, v in json.load(f).get("edits", {}).items()}
        print(f"  Loaded {len(frame_edits)} human-edited frames (display overrides)")

    # ── Reset and render ──
    # Start from the labeled frame so spatial matching works on frame 0
    render_start = classifier.label_frame if classifier.ready else args.start
    cap.set(cv2.CAP_PROP_POS_FRAMES, render_start)
    total_frames = int(args.duration * fps) + (args.start - render_start)
    skip_frames = args.start - render_start
    reg.stats = {"auto": 0, "orb": 0, "stale": 0}
    assigned = False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w_out, h_out))
    # Sidecar per-frame track dump: one JSON line per frame with every live
    # track's box and team — turns "track N disappeared" reports into a
    # frame number instead of an archaeology session.
    track_log = open(args.output + ".tracks.jsonl", "w")
    smoother = PlayerSmoother(alpha=0.6)
    rescue = MotionRescueTracker()
    color_model = TeamColorSubspaces()
    audit_strikes = {}   # tid -> consecutive uncontested frames contradicting team
    audit_flips = []     # (tid, from_team, to_team) log
    audit_events = []    # (frame_idx, tid, box_h, score) every strike
    lock_events = []     # (frame_idx, tid, team, box_h) every color lock

    t0 = time.time()
    pbar = tqdm(total=total_frames, desc="rendering")

    for fi in range(total_frames):
        ok, frame_bgr = cap.read()
        if not ok:
            break

        stale_before = reg.stats["stale"]
        reg.update(frame_bgr)
        eraser.advance(reg.last_motion)
        # detector sees the frame with static field paint turfed out;
        # vis/color sampling below keep the real pixels
        det_frame = eraser.erase(frame_bgr)
        frame_rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
        if reg.stats["stale"] > stale_before:
            rescue.flush()  # ORB chain broke (cut?) — ghosts are meaningless

        # Fit the color subspaces on the labeled frame (frame 0 of the loop)
        if fi == 0 and classifier.ready:
            print("  Fitting color subspaces from labels...")
            color_model.fit(frame_bgr, classifier.get_all_labels())

        frame_up = cv2.resize(frame_rgb, (w_in * 2, h_in * 2))
        raw_dets_up = detector.detect(frame_up)
        from football_tracker.detection import Detection as _Det
        raw_dets = [_Det(bbox_xyxy=(d.bbox_xyxy[0] / 2, d.bbox_xyxy[1] / 2,
                                     d.bbox_xyxy[2] / 2, d.bbox_xyxy[3] / 2),
                         score=d.score) for d in raw_dets_up]
        kept = nms(list(raw_dets))

        # Field-bounds gate, DOWNFIELD AXIS ONLY. The width axis cannot be
        # gated on this footage (measured at frame 72620): real players at
        # the bottom sideline project to y=-9 while staff is y=-10 — a 1yd
        # gap inside a ~9yd registration error — and the top crowd projects
        # to y=+47.5, INSIDE the field. That's the README's width-axis
        # rank-deficiency in action. A +-3yd width gate ate two real
        # players (tracks 25/18) for the entire play.
        if reg.ready:
            feet = np.array([[(d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2.0,
                              d.bbox_xyxy[3]] for d in kept], dtype=np.float64)
            if len(feet):
                fc = reg.pixel_to_field(feet)
                kept = [d for d, (fx, fy) in zip(kept, fc)
                        if -3 <= fx <= FIELD_LENGTH_YD + 3]

        # On the first frame the hand labels outrank RT-DETR:
        #  1. a detection spanning >=2 labels is a merged pile box (DT+OL
        #     mashed together) — drop it, the labels cover those players;
        #  2. inject labels nothing covers (crouched linemen RT-DETR missed);
        #  3. labels cleanly covered by one detection are NOT injected, else
        #     the twin box births a duplicate track.
        if not assigned and classifier.ready:
            inj = classifier.get_inject_boxes()
            kept = [d for d in kept
                    if sum(_box_iou(d.bbox_xyxy, lb) > 0.30 for lb in inj) <= 1]
            #  4. a label also vouches for ONE weak detection on it (best
            #     IoU only — a wide label can overlap two dets and the
            #     loser is usually a half-grass phantom): the DTs are
            #     detected at 0.29-0.37, under ByteTrack's ~0.4 start bar,
            #     so without this boost they never get a track
            vouched = set()
            for lb in inj:
                cands = [(i, _box_iou(d.bbox_xyxy, lb))
                         for i, d in enumerate(kept) if d.score < 0.50]
                if cands:
                    i, iou = max(cands, key=lambda t: t[1])
                    if iou > 0.40:
                        vouched.add(i)
            kept = [_Det(bbox_xyxy=d.bbox_xyxy, score=0.50)
                    if i in vouched else d for i, d in enumerate(kept)]
            det_boxes = [d.bbox_xyxy for d in kept]
            for bx1, by1, bx2, by2 in inj:
                if all(_box_iou((bx1, by1, bx2, by2), db) < 0.40
                       for db in det_boxes):
                    kept.append(_Det(bbox_xyxy=(bx1, by1, bx2, by2), score=0.50))

        field_positions = []
        team_labels = []
        vis = frame_bgr.copy()
        sv_dets = None

        if kept:
            xyxy = np.array([d.bbox_xyxy for d in kept], dtype=np.float32)
            scores = np.array([d.score for d in kept], dtype=np.float32)
            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=scores,
                class_id=np.zeros(len(kept), dtype=int),
            )
            sv_dets = tracker.update_with_detections(sv_dets)

            if not assigned and classifier.ready:
                classifier.assign_from_detections(sv_dets)
                assigned = True
                matched_labels = []
                for i in range(len(sv_dets)):
                    tid = int(sv_dets.tracker_id[i])
                    t, _ = classifier.get_team(tid)
                    if t >= 0:
                        dx1, dy1, dx2, dy2 = sv_dets.xyxy[i]
                        matched_labels.append((dx1, dy1, dx2, dy2, "green" if t == 0 else "white"))
                n_matched = len(matched_labels)
                print(f"  Matched {n_matched}/{len(sv_dets)} detections to labels")

            # Apply human corrections scheduled for this frame
            for cbox, actual in corr_by_frame.get(fi, []):
                best_tid, best_iou = None, 0.30
                for i in range(len(sv_dets)):
                    iou = _box_iou(tuple(sv_dets.xyxy[i]), tuple(cbox))
                    if iou > best_iou:
                        best_iou, best_tid = iou, int(sv_dets.tracker_id[i])
                if best_tid is not None:
                    # not_a_player hides exactly like ref (no box, no dot)
                    classifier._tid_team[best_tid] = (
                        "ref" if actual == "not_a_player" else actual)
                    pinned.add(best_tid)
                    print(f"  PIN: f{fi} track {best_tid} := {actual} "
                          f"(iou {best_iou:.2f})")

        # Rescue lost tracks FIRST: reborn ids inherit their team here, so
        # the color model below is never asked about a player we already know
        rescue_boxes = rescue.update(sv_dets, classifier, reg._motion_accum,
                                     frame_bgr, color_model)

        # Assigned tracks this frame, for the forced-guess nearest-neighbor
        # fallback below (gray boxes are banned: every player renders green
        # or white; ref stays hidden).
        shown_team = {}   # tid -> team actually rendered this frame
        assigned_pos = []
        if sv_dets is not None and len(sv_dets) > 0 and sv_dets.tracker_id is not None:
            for i in range(len(sv_dets)):
                t, _ = classifier.get_team(int(sv_dets.tracker_id[i]))
                if t in (0, 1):
                    bx = sv_dets.xyxy[i]
                    assigned_pos.append(((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2, t))

        if sv_dets is not None and len(sv_dets) > 0:
            for i in range(len(sv_dets)):
                x1, y1, x2, y2 = sv_dets.xyxy[i].astype(int)
                tid = int(sv_dets.tracker_id[i]) if sv_dets.tracker_id is not None else -1

                foot_pt = np.array([[(x1 + x2) / 2.0, float(y2)]])
                if reg.ready:
                    fp = reg.pixel_to_field(foot_pt)
                    fx, fy = float(fp[0, 0]), float(fp[0, 1])
                    fx, fy = smoother.smooth(tid, fx, fy)
                else:
                    fx, fy = 60.0, 26.6

                team, conf = classifier.get_team(tid)

                # Contested = another box overlaps this one (piles, crossings):
                # those pixels hold the other guy's colors, so they neither
                # vote nor audit
                contested = False
                if color_model.ready:
                    area = max(1.0, float(x2 - x1) * float(y2 - y1))
                    for j in range(len(sv_dets)):
                        if j == i:
                            continue
                        ox1, oy1, ox2, oy2 = sv_dets.xyxy[j]
                        iw = min(x2, ox2) - max(x1, ox1)
                        ih = min(y2, oy2) - max(y1, oy1)
                        if iw > 0 and ih > 0 and iw * ih > 0.15 * area:
                            contested = True
                            break

                if team == -1 and color_model.ready:
                    locked = None if contested else color_model.vote(
                        tid, frame_bgr, (x1, y1, x2, y2))
                    if locked is not None:
                        if tid not in classifier._tid_team:
                            lock_events.append((fi, tid, locked, int(y2 - y1)))
                        classifier.set_team(tid, locked)
                        team, conf = classifier.get_team(tid)
                elif (team in (0, 1) and color_model.ready and not contested
                        and tid not in pinned):
                    # Continuous appearance audit: no team lock is permanent.
                    # Strike only on NEAR-PURE contradiction (s < 0.12): a
                    # mixed box (two players merged into one detection) lands
                    # mid-range and must neither strike nor clear. Only real
                    # support for the assigned team (s >= 0.5) clears strikes.
                    tname = "green" if team == 0 else "white"
                    s = color_model.check_team(frame_bgr, (x1, y1, x2, y2), tname)
                    if s is not None and s < 0.12:
                        audit_strikes[tid] = audit_strikes.get(tid, 0) + 1
                        audit_events.append((fi, tid, int(y2 - y1), round(float(s), 3)))
                        if audit_strikes[tid] >= 15:
                            # Persistent contradiction: do NOT blind-flip to
                            # the other team — "not white" can also mean REF
                            # (track 4 drifted onto the ref and got painted
                            # green this way). Unassign and let the 3-class
                            # vote re-decide from scratch.
                            classifier._tid_team.pop(tid, None)
                            color_model._locked.pop(tid, None)
                            color_model._votes.pop(tid, None)
                            rescue._last_seen.pop(tid, None)
                            audit_strikes[tid] = 0
                            audit_flips.append((tid, tname, "unassigned"))
                            team, conf = classifier.get_team(tid)
                    elif s is not None and s >= 0.5:
                        audit_strikes[tid] = 0

                if team == 2:
                    continue  # referee: no box, no minimap dot

                if team == -1:
                    # Forced binary guess, DISPLAY ONLY (never stored, so it
                    # can't poison locks/handovers): votes if any, else this
                    # frame's pixels, else the nearest assigned player.
                    v = color_model._votes.get(tid) if color_model.ready else None
                    if v and v.get("green", 0) != v.get("white", 0):
                        team = 0 if v.get("green", 0) > v.get("white", 0) else 1
                    else:
                        sg = sw = None
                        if color_model.ready:
                            sg = color_model.check_team(frame_bgr, (x1, y1, x2, y2), "green")
                            sw = color_model.check_team(frame_bgr, (x1, y1, x2, y2), "white")
                        if sg is not None and sw is not None and sg != sw:
                            team = 0 if sg > sw else 1
                        elif assigned_pos:
                            cx_, cy_ = (x1 + x2) / 2, (y1 + y2) / 2
                            team = min(assigned_pos,
                                       key=lambda p: (p[0] - cx_) ** 2 + (p[1] - cy_) ** 2)[2]
                        else:
                            team = 0
                shown_team[tid] = team
                if fi in frame_edits:
                    continue  # human edits own this frame's display
                team_labels.append(team)
                field_positions.append((fx, fy))

                if team == classifier.darker_team:
                    color = HOME_BGR
                else:
                    color = AWAY_BGR
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                cv2.putText(vis, str(tid), (x1, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        # Human-edited frame: draw the human's boxes and project their feet
        # for the minimap; rescue ghosts are suppressed too
        if fi in frame_edits:
            for b in frame_edits[fi]:
                if b["team"] == 2:
                    continue  # ref stays hidden
                ex1, ey1, ex2, ey2 = (int(v) for v in b["box"])
                color = HOME_BGR if b["team"] == classifier.darker_team else AWAY_BGR
                cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), color, 1)
                cv2.putText(vis, str(b["tid"]), (ex1, ey1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
                foot = np.array([[(ex1 + ex2) / 2.0, float(ey2)]])
                if reg.ready:
                    fp = reg.pixel_to_field(foot)
                    fx, fy = float(fp[0, 0]), float(fp[0, 1])
                else:
                    fx, fy = 60.0, 26.6
                team_labels.append(b["team"])
                field_positions.append((fx, fy))

        # Draw boxes for tracks still lost (players RT-DETR drops mid-play)
        for rx1, ry1, rx2, ry2, rteam, rtid in (
                [] if fi in frame_edits else rescue_boxes):
            foot_pt = np.array([[(rx1 + rx2) / 2.0, float(ry2)]])
            if reg.ready:
                fp = reg.pixel_to_field(foot_pt)
                fx, fy = float(fp[0, 0]), float(fp[0, 1])
                fx, fy = smoother.smooth(rtid, fx, fy)
            else:
                fx, fy = 60.0, 26.6
            tidx = 0 if rteam == "green" else 1
            team_labels.append(tidx)
            field_positions.append((fx, fy))
            color = HOME_BGR if rteam == "green" else AWAY_BGR
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), color, 1)
            cv2.putText(vis, str(rtid), (rx1, ry1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        if field_positions:
            mini = render_minimap(field_positions, team_labels, classifier.darker_team)
            vis = composite_minimap(vis, mini)

        rows = []
        if sv_dets is not None and len(sv_dets) > 0 and sv_dets.tracker_id is not None:
            for i in range(len(sv_dets)):
                tid_ = int(sv_dets.tracker_id[i])
                t, _ = classifier.get_team(tid_)
                # row: [tid, box, assigned_team, displayed_team]
                rows.append([tid_, [round(float(v), 1) for v in sv_dets.xyxy[i]],
                             t, shown_team.get(tid_, t)])
        track_log.write(json.dumps({"fi": fi, "tracks": rows}) + "\n")

        if fi < skip_frames:
            pbar.update(1)
            continue

        if args.upscale > 1:
            vis = cv2.resize(vis, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)

        writer.write(vis)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    track_log.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({total_frames / elapsed:.1f} fps)")
    print(f"Registration: auto={reg.stats['auto']} orb={reg.stats['orb']} stale={reg.stats['stale']}")
    if color_model.ready:
        ng = sum(1 for t in color_model._locked.values() if t == "green")
        nw = sum(1 for t in color_model._locked.values() if t == "white")
        print(f"Color model: locked {ng} green + {nw} white late tracks")
        for fi_, tid_, team_, h_ in lock_events:
            print(f"  LOCK: fi={fi_} track {tid_} -> {team_} (box_h={h_})")
        for tid, a, b in audit_flips:
            print(f"  AUDIT: track {tid} flipped {a} -> {b}")
            evs = [e for e in audit_events if e[1] == tid]
            for fi_, _, h_, s_ in evs[-20:]:
                print(f"    strike fi={fi_} box_h={h_} score={s_}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
