import cv2
import numpy as np
import yaml
from pathlib import Path
from football_tracker.field_coords import FIELD_LENGTH_YD, FIELD_WIDTH_YD
from football_tracker.registration import FieldRegistration
from football_tracker.detection import PersonDetector

img = cv2.imread('data/samples/u69v2_50.png')
H, W = img.shape[:2]

cfg = yaml.safe_load(Path('configs/umich_1969.yaml').read_text())
pixel_pts = np.array([c['pixel'] for c in cfg['correspondences']], dtype=np.float64)
field_pts = np.array([c['field'] for c in cfg['correspondences']], dtype=np.float64)
reg = FieldRegistration.from_correspondences(pixel_pts, field_pts)
m = 0.5
corners_field = np.array([
    [10 + m, m],
    [FIELD_LENGTH_YD - 10 - m, m],
    [FIELD_LENGTH_YD - 10 - m, FIELD_WIDTH_YD - m],
    [10 + m, FIELD_WIDTH_YD - m],
])
corners_px = reg.field_to_pixel(corners_field).astype(np.int32)
field_mask = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(field_mask, [corners_px], 255)

det = PersonDetector(score_threshold=0.30)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dets = det.detect(img_rgb)
player_mask = np.zeros((H, W), dtype=np.uint8)
for d in dets:
    x1, y1, x2, y2 = (int(v) for v in d.bbox_xyxy)
    pad = 12
    cv2.rectangle(
        player_mask,
        (max(0, x1 - pad), max(0, y1 - pad)),
        (min(W, x2 + pad), min(H, y2 + pad)),
        255, -1,
    )
mask = cv2.bitwise_and(field_mask, cv2.bitwise_not(player_mask))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
_, lm = cv2.threshold(tophat, 18, 255, cv2.THRESH_BINARY)
lm = cv2.bitwise_and(lm, mask)

# Close small gaps so each painted line becomes one connected component
lm_close = cv2.morphologyEx(lm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(lm_close, connectivity=8)
print(f'{n_lab - 1} raw components')

overlay = img.copy()
cv2.polylines(overlay, [corners_px], True, (255, 0, 255), 1)

kept = 0
yard_lines_polar = []  # (theta, rho)
for i in range(1, n_lab):
    x, y, w, h, area = stats[i]
    # Filter by elongation and area
    if area < 80:
        continue
    pixels = np.argwhere(labels == i)  # (N, 2) (row, col) = (y, x)
    if len(pixels) < 30:
        continue
    pts = pixels[:, ::-1].astype(np.float32)  # (x, y)

    # PCA: principal axis = direction of the line
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Largest eigenvalue → direction
    long_axis = eigvecs[:, 1]
    short_axis = eigvecs[:, 0]
    long_var = eigvals[1]
    short_var = eigvals[0]
    if short_var < 1e-3:
        continue
    elongation = long_var / short_var
    if elongation < 25 or long_var < 200:
        continue

    # Project points onto axes to get extents
    long_proj = centered @ long_axis
    p1 = centroid + long_proj.min() * long_axis
    p2 = centroid + long_proj.max() * long_axis
    cv2.line(overlay, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 255), 2)

    # Polar form
    line_ang = np.arctan2(long_axis[1], long_axis[0]) % np.pi
    norm_ang = (line_ang + np.pi / 2) % np.pi
    rho = centroid[0] * np.cos(norm_ang) + centroid[1] * np.sin(norm_ang)
    yard_lines_polar.append((line_ang, rho, len(pts)))
    kept += 1

print(f'{kept} elongated components kept')

# Cluster co-oriented co-positional lines (sometimes a single yard line splits into 2 CCs)
def ang_dist(a, b):
    d = abs(a - b)
    return min(d, np.pi - d)

clustered = []
yard_lines_polar.sort(key=lambda x: x[1])
for ln in yard_lines_polar:
    merged = False
    for c in clustered:
        if ang_dist(ln[0], c['ang']) < np.radians(8) and abs(ln[1] - c['rho']) < 20:
            c['lines'].append(ln)
            c['ang'] = float(np.mean([l[0] for l in c['lines']]))
            c['rho'] = float(np.mean([l[1] for l in c['lines']]))
            merged = True
            break
    if not merged:
        clustered.append({'ang': ln[0], 'rho': ln[1], 'lines': [ln]})

print(f'{len(clustered)} clusters after merging')
# Find median angle (yard-line direction)
all_angs = [c['ang'] for c in clustered]
median_ang = float(np.median(all_angs))
yard = [c for c in clustered if ang_dist(c['ang'], median_ang) < np.radians(10)]
side = [c for c in clustered if ang_dist(c['ang'], median_ang) >= np.radians(10)]
print(f'  -> {len(yard)} yard lines @ {round(np.degrees(median_ang))}deg, {len(side)} other')

overlay2 = img.copy()
cv2.polylines(overlay2, [corners_px], True, (255, 0, 255), 1)

def draw_clipped(img, theta_line, rho, mask, color, thickness=2):
    norm_theta = (theta_line + np.pi / 2) % np.pi
    a, b = np.cos(norm_theta), np.sin(norm_theta)
    x0, y0 = a * rho, b * rho
    L = 3000
    p1 = (int(x0 + L * (-b)), int(y0 + L * a))
    p2 = (int(x0 - L * (-b)), int(y0 - L * a))
    canvas = np.zeros_like(mask)
    cv2.line(canvas, p1, p2, 255, thickness)
    in_field = cv2.bitwise_and(canvas, mask)
    img[in_field > 0] = color


for c in yard:
    draw_clipped(overlay2, c['ang'], c['rho'], field_mask, (0, 255, 255), 3)
for c in side:
    draw_clipped(overlay2, c['ang'], c['rho'], field_mask, (0, 255, 0), 3)

cv2.imwrite('data/samples/u69v2_50_cc.png', overlay2)
print('wrote u69v2_50_cc.png')
