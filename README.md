# Football Tracker

Goal: produce a 2D top-down map of player positions tracked through `videoplayback.mp4`
(broadcast-style football footage).

## What genuinely works

The pipeline is solid on **wide / well-marked frames** and follows the camera from there:

1. **Full-rank manual calibration on an anchor frame.** You supply >=4 image<->field
   correspondences (yard-line / sideline intersections) in a YAML config. A DLT/homography
   is solved from those (`src/football_tracker/registration.py`,
   `yard_line_registration.py`). When the anchor frame is wide enough to span markings in
   both axes, the homography is full-rank and accurate.
2. **Camera-motion propagation.** ORB feature matching tracks camera pan/zoom frame-to-frame
   and carries the anchor homography forward, so the field grid follows the motion within a
   shot (`tracking_registration.py`). Validated on the working segments with ~700-1600 ORB
   inliers per frame.
3. **Player detection + foot-point projection.** People are detected, the foot-contact point
   is projected through the homography into field coordinates
   (`detection.py`, `field_coords.py`).
4. **Fisher-discriminant team separation.** Jersey colors are separated by a linear (Fisher)
   discriminant to assign teams (`filters.py`).
5. **Minimap rendering.** Projected foot points + team color are drawn on a top-down field
   (`minimap.py`, `pipeline.py`).

## Honest limits

- **Tight central frames are rank-deficient in the cross-field (width) axis.** When the
  camera is zoomed onto the middle of the field with few markings spanning the width, the
  data needed to place a player *laterally* is simply not in the pixels. No registration
  algorithm manufactures it; the homography's width column is underdetermined. (Confirmed by
  the linear-algebra analysis: a near-singular normal-equation matrix in the width direction.)
- **The L/R hash-mark gap does not save it.** Hashes are 6.17 yd apart = only ~11.6% of the
  53.3-yd width, so anchoring the full width off them is a ~4x extrapolation in exactly the
  bad axis — rejected this cycle.
- **Cuts break the chain.** ORB propagation is intra-shot only. A hard cut requires
  re-anchoring; pixel-space relocalization across cuts has failed.
- **Down-field is the reliable axis.** Yards-from-goal-line is recoverable nearly everywhere;
  full 2-D width is only reliable on wide or panning shots.

## Current best plan (Cycle 1, 2026-05-28)

- **Pursue: multi-frame pan mosaic.** On panning shots, mask players out, chain the working
  intra-shot ORB homographies into one wide field mosaic, and recover a sideline or painted
  yard-number that no single tight frame held — then back-propagate that absolute width datum
  to every frame in the shot. This is the only approach that injects genuinely *new* width
  information, and it rides the ORB chain that already works. Gate it behind a shot-type
  classifier (skip the tight-throughout shots, where it cannot help).
- **Ship: down-field-native minimap (Pivot A).** Report yards-from-goal accurately + Fisher
  team color everywhere; show lateral position best-effort and **explicitly label it
  unreliable** when the frame is width-rank-deficient. True 2-D everywhere would require
  different footage (wide All-22 / Next-Gen tracking), not a new algorithm.
- Deferred: formation-Procrustes cut-bridging (after the mosaic lands).
- Killed: hash-mark ruler, broadcast LOS/1st-down lines, player-height range-finder,
  learned NFL single-image registration.

## How to run

All commands from the repo root (`/home/patrick/football-tracker`). Configs live in `configs/`.

```bash
# 1. Check a manual registration: overlays the full field grid on a frame.
python scripts/visualize_registration.py CONFIG.yaml OUTPUT.png

# 2. End-to-end on one frame: detections + foot points + field-coord text.
python scripts/demo_pipeline.py CONFIG.yaml OUTPUT.png

# 3. Track a video segment: ORB-propagated registration + top-down minimap.
python scripts/track_video.py CONFIG.yaml VIDEO.mp4 OUTPUT.mp4 \
    [--no-ref-filter] [--score-threshold 0.30]

# 4. Pixel-space detection only (no registration).
python scripts/detect.py INPUT.mp4 OUTPUT.mp4 [--max-frames N]
```

The YAML `image` field names the **anchor frame**; the homography is hand-picked relative to
that frame, then ORB tracks the camera from there. See `configs/sample_registration.yaml`.

## Layout

- `src/football_tracker/` — library (registration, tracking, detection, field coords, Fisher
  filters, minimap, pipeline).
- `scripts/` — runnable entrypoints above (`_prototype_*` are scratch, not maintained).
- `configs/` — registration/anchor configs.
- `data/samples/` — sample frames and clips.
- `notebooks/football_tracker_linalg_presentation.ipynb` — the linear algebra behind the
  pipeline, rebuilt from scratch on toy data (homography, SVD/DLT, conditioning, ORB motion
  chain, PCA team color).
- `demo_final.mp4` — a rendered single play (tracking + top-down minimap) on NFL All-22 footage.

Tests: `pytest` (`tests/test_registration.py`, `tests/test_field_coords.py`).
