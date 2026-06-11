#!/usr/bin/env python
"""Interactive team labeler: draw boxes on a frame, assign to green or white team.

Controls:
  - Click and drag to draw a box around a player
  - Press 'g' to label the last box as GREEN team (Eagles/dark)
  - Press 'w' to label the last box as WHITE team (visitors)
  - Press 'x' to delete the last box
  - Press 'q' to finish and save

Saves labeled boxes to a JSON file that demo_play.py reads to fit the
Fisher discriminant with your ground-truth labels.
"""
import argparse
import json
import cv2
import numpy as np

VIDEO = "/home/patrick/videoplayback.mp4"

drawing = False
ix, iy = 0, 0
boxes = []       # list of (x1, y1, x2, y2, team)  team: 'green' or 'white' or None
current_box = None
frame_clean = None
SCALE = 2


def draw_overlay(img):
    vis = img.copy()
    for (x1, y1, x2, y2, team) in boxes:
        if team == "green":
            color = (0, 180, 0)
        elif team == "white":
            color = (220, 220, 240)
        elif team == "ref":
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = team[0].upper() if team else "?"
        cv2.putText(vis, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if current_box:
        cv2.rectangle(vis, (current_box[0], current_box[1]),
                      (current_box[2], current_box[3]), (0, 255, 255), 1)
    cv2.putText(vis, "Draw box, then G=green W=white R=ref X=undo Q=done",
                (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, current_box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - ix) > 5 and abs(y - iy) > 5:
            boxes.append((min(ix, x), min(iy, y), max(ix, x), max(iy, y), None))
        current_box = None


def main():
    global frame_clean
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=72700)
    p.add_argument("--output", default="/home/patrick/football-tracker/data/team_labels.json")
    args = p.parse_args()

    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read frame")

    frame_clean = cv2.resize(fr, (fr.shape[1] * SCALE, fr.shape[0] * SCALE))

    cv2.namedWindow("Label Teams")
    cv2.setMouseCallback("Label Teams", mouse_cb)

    while True:
        vis = draw_overlay(frame_clean)
        cv2.imshow("Label Teams", vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("g") and boxes and boxes[-1][4] is None:
            b = boxes[-1]
            boxes[-1] = (b[0], b[1], b[2], b[3], "green")
        elif key == ord("w") and boxes and boxes[-1][4] is None:
            b = boxes[-1]
            boxes[-1] = (b[0], b[1], b[2], b[3], "white")
        elif key == ord("r") and boxes and boxes[-1][4] is None:
            b = boxes[-1]
            boxes[-1] = (b[0], b[1], b[2], b[3], "ref")
        elif key == ord("x") and boxes:
            boxes.pop()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    labeled = [b for b in boxes if b[4] is not None]
    green = [b for b in labeled if b[4] == "green"]
    white = [b for b in labeled if b[4] == "white"]
    print(f"\n{len(labeled)} labeled: {len(green)} green, {len(white)} white")

    out = {
        "frame": args.frame,
        "scale": SCALE,
        "boxes": [{"x1": b[0] // SCALE, "y1": b[1] // SCALE,
                    "x2": b[2] // SCALE, "y2": b[3] // SCALE,
                    "team": b[4]} for b in labeled],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
