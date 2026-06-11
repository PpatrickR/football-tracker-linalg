#!/usr/bin/env python
"""Interactive color-switch reviewer.

Plays a rendered demo video alongside its .tracks.jsonl sidecar. Whenever a
box's DISPLAYED color switches (green<->white), playback pauses, an arrow
points at the offending box, and you say what the player actually is:

  G = green team        W = white team
  R = referee           N = not a player (crowd/staff/phantom)
  SPACE = skip this one Q = quit (saves what you've answered so far)

Each answer is saved to data/color_corrections.json with the track id, the
frame, and the box in SOURCE coordinates — the box+frame pair is the durable
key (track ids renumber between runs), so corrections can be re-matched
spatially when feeding them back into the pipeline.

A track is only asked about once: after you answer, later switches of the
same track are recorded as resolved automatically.

Usage:
    python scripts/review_colors.py VIDEO.mp4 [--out data/color_corrections.json]
    (expects VIDEO.mp4.tracks.jsonl next to the video)
"""
import argparse
import json
import os

import cv2

KEYMAP = {ord("g"): "green", ord("w"): "white", ord("r"): "ref",
          ord("n"): "not_a_player"}
TEAM_NAMES = {0: "green", 1: "white", 2: "ref", -1: "unknown"}


def load_events(jsonl_path):
    """Color-switch events: (fi, tid, box, from_team, to_team)."""
    events = []
    last_shown = {}
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            for row in rec["tracks"]:
                tid, box, _assigned, shown = row
                prev = last_shown.get(tid)
                if prev is not None and prev != shown and 0 <= shown <= 1:
                    events.append((rec["fi"], tid, box, prev, shown))
                last_shown[tid] = shown
    return events


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "color_corrections.json"))
    args = p.parse_args()

    jsonl = args.video + ".tracks.jsonl"
    events = load_events(jsonl)
    by_frame = {}
    for ev in events:
        by_frame.setdefault(ev[0], []).append(ev)
    print(f"{len(events)} color switches across {len(by_frame)} frames")

    cap = cv2.VideoCapture(args.video)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = vid_w / 640.0   # sidecar boxes are in source coords

    corrections = []
    answered = {}           # tid -> user's answer
    win = "Color Review"
    cv2.namedWindow(win)

    fi = 0
    while fi < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        todo = [ev for ev in by_frame.get(fi, [])
                if ev[1] not in answered]
        # auto-resolve switches on tracks already answered
        for ev in by_frame.get(fi, []):
            if ev[1] in answered:
                corrections.append({"frame": fi, "tid": ev[1], "box": ev[2],
                                    "from": TEAM_NAMES[ev[3]],
                                    "to": TEAM_NAMES[ev[4]],
                                    "actual": answered[ev[1]],
                                    "auto": True})
        for fi_, tid, box, t_from, t_to in todo:
            x1, y1, x2, y2 = (int(v * scale) for v in box)
            vis = frame.copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # arrow from upper-left offset toward the box corner
            ax, ay = max(0, x1 - 70), max(0, y1 - 70)
            cv2.arrowedLine(vis, (ax, ay), (x1 - 4, y1 - 4),
                            (0, 0, 255), 3, tipLength=0.35)
            cv2.putText(vis,
                        f"track {tid}: {TEAM_NAMES[t_from]} -> {TEAM_NAMES[t_to]}"
                        f"   actual? G/W/R/N  SPACE=skip  Q=quit",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, vis)
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key in KEYMAP:
                    answered[tid] = KEYMAP[key]
                    corrections.append({"frame": fi, "tid": tid, "box": box,
                                        "from": TEAM_NAMES[t_from],
                                        "to": TEAM_NAMES[t_to],
                                        "actual": KEYMAP[key], "auto": False})
                    break
                if key == ord(" "):
                    break
                if key == ord("q"):
                    _save(args.out, corrections)
                    print(f"quit at frame {fi}; "
                          f"{len(corrections)} corrections saved to {args.out}")
                    return
        cv2.imshow(win, frame)
        if (cv2.waitKey(33) & 0xFF) == ord("q"):
            break
        fi += 1

    _save(args.out, corrections)
    print(f"done; {len(corrections)} corrections saved to {args.out}")


def _save(path, corrections):
    with open(path, "w") as f:
        json.dump({"corrections": corrections}, f, indent=1)


if __name__ == "__main__":
    main()
