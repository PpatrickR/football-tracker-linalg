#!/usr/bin/env python
"""Frame-by-frame track editor.

Opens a rendered demo video with its .tracks.jsonl sidecar and lets you
correct the detections frame by frame. Edited frames are saved to
data/track_edits.json (boxes in SOURCE coordinates, same space as the
sidecar), which downstream tooling can treat as ground truth.

Navigation:
  LEFT / RIGHT (or A / D)   previous / next frame
  Q                         quit (saves)   S  save now

Editing:
  click inside a box        select it (yellow, with corner handles)
  drag inside selected box  move it
  drag a corner handle      resize / transform it
  drag on empty turf        create a new box (selected, green by default)
  G / W / R                 set selected box green / white / ref
  X or DELETE               delete selected box

Usage:
    python scripts/edit_tracks.py VIDEO.mp4 [--out data/track_edits.json]
"""
import argparse
import json
import os

import cv2

TEAM_COLORS = {0: (0, 180, 0), 1: (230, 230, 245), 2: (0, 0, 255)}
TEAM_NAMES = {0: "green", 1: "white", 2: "ref"}
SELECT_COLOR = (0, 220, 255)
HANDLE_PX = 8


class Editor:
    def __init__(self, video, jsonl, out_path):
        self.cap = cv2.VideoCapture(video)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.scale = vid_w / 640.0
        self.out_path = out_path

        # original per-frame state from the sidecar
        self.orig = {}
        with open(jsonl) as f:
            for line in f:
                rec = json.loads(line)
                self.orig[rec["fi"]] = [
                    {"tid": tid, "box": list(box), "team": shown}
                    for tid, box, _assigned, shown in rec["tracks"]
                    if 0 <= shown <= 2]
        # edited frames (overrides), loaded from a previous session if any
        self.edits = {}
        if os.path.exists(out_path):
            with open(out_path) as f:
                prev = json.load(f).get("edits", {})
            self.edits = {int(k): v for k, v in prev.items()}
            print(f"loaded {len(self.edits)} previously edited frames")

        self.fi = 0
        self.frame = None
        self.sel_tid = None
        self.next_new_tid = -1
        # mouse state
        self.mode = None          # 'move' | 'resize' | 'create' | None
        self.anchor = None        # mouse-down point (display px)
        self.box_at_down = None   # selected box copy at mouse-down
        self.resize_corner = None # 0..3 = tl, tr, br, bl
        self.status = "click a box to select"

    # ── state access ─────────────────────────────────────────────────

    def boxes(self):
        """Current editable list for this frame (copy-on-write)."""
        if self.fi in self.edits:
            return self.edits[self.fi]
        return self.orig.get(self.fi, [])

    def boxes_mut(self):
        if self.fi not in self.edits:
            self.edits[self.fi] = [dict(b, box=list(b["box"]))
                                   for b in self.orig.get(self.fi, [])]
        return self.edits[self.fi]

    def selected(self, items=None):
        for b in (items if items is not None else self.boxes()):
            if b["tid"] == self.sel_tid:
                return b
        return None

    # ── geometry (display px <-> source coords) ──────────────────────

    def to_disp(self, box):
        return [v * self.scale for v in box]

    def to_src(self, box):
        return [round(v / self.scale, 1) for v in box]

    def hit_corner(self, box_disp, x, y):
        x1, y1, x2, y2 = box_disp
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for i, (cx, cy) in enumerate(corners):
            if abs(x - cx) <= HANDLE_PX and abs(y - cy) <= HANDLE_PX:
                return i
        return None

    def hit_box(self, x, y, pad=8):
        """Smallest box containing the point (display coords), with a
        tolerance pad — the boxes are small and pixel-perfect interior
        clicks are unreasonable to demand."""
        best, best_area = None, 1e18
        for b in self.boxes():
            x1, y1, x2, y2 = self.to_disp(b["box"])
            if x1 - pad <= x <= x2 + pad and y1 - pad <= y <= y2 + pad:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best, best_area = b, area
        return best

    # ── mouse ────────────────────────────────────────────────────────

    def on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            sel = self.selected()
            if sel is not None:
                corner = self.hit_corner(self.to_disp(sel["box"]), x, y)
                if corner is not None:
                    self.mode = "resize"
                    self.resize_corner = corner
                    self.anchor = (x, y)
                    self.box_at_down = list(sel["box"])
                    self.status = f"resizing track {sel['tid']}"
                    return
            hit = self.hit_box(x, y)
            if hit is not None:
                self.sel_tid = hit["tid"]
                self.mode = "move"
                self.anchor = (x, y)
                self.box_at_down = list(hit["box"])
                self.status = f"selected track {hit['tid']}"
            else:
                self.mode = "create"
                self.anchor = (x, y)
                self.status = "drag to create"
        elif event == cv2.EVENT_MOUSEMOVE and self.mode:
            self.drag(x, y, commit=False)
        elif event == cv2.EVENT_LBUTTONUP and self.mode:
            self.drag(x, y, commit=True)
            self.mode = None

    def drag(self, x, y, commit):
        dx = (x - self.anchor[0]) / self.scale
        dy = (y - self.anchor[1]) / self.scale
        if self.mode == "move":
            items = self.boxes_mut()
            sel = self.selected(items)
            if sel:
                bx = self.box_at_down
                sel["box"] = [bx[0] + dx, bx[1] + dy, bx[2] + dx, bx[3] + dy]
        elif self.mode == "resize":
            items = self.boxes_mut()
            sel = self.selected(items)
            if sel:
                x1, y1, x2, y2 = self.box_at_down
                if self.resize_corner == 0:   x1, y1 = x1 + dx, y1 + dy
                elif self.resize_corner == 1: x2, y1 = x2 + dx, y1 + dy
                elif self.resize_corner == 2: x2, y2 = x2 + dx, y2 + dy
                else:                         x1, y2 = x1 + dx, y2 + dy
                sel["box"] = [min(x1, x2), min(y1, y2),
                              max(x1, x2), max(y1, y2)]
        elif self.mode == "create" and commit:
            if abs(x - self.anchor[0]) > 6 and abs(y - self.anchor[1]) > 6:
                items = self.boxes_mut()
                x1, y1 = self.anchor
                box = self.to_src([min(x1, x), min(y1, y),
                                   max(x1, x), max(y1, y)])
                items.append({"tid": self.next_new_tid, "box": box, "team": 0})
                self.sel_tid = self.next_new_tid
                self.next_new_tid -= 1

    # ── rendering ────────────────────────────────────────────────────

    def draw(self):
        vis = self.frame.copy()
        for b in self.boxes():
            x1, y1, x2, y2 = (int(v) for v in self.to_disp(b["box"]))
            selected = b["tid"] == self.sel_tid
            color = SELECT_COLOR if selected else TEAM_COLORS.get(b["team"], (90, 90, 90))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2 if selected else 1)
            label = f'{b["tid"]}:{TEAM_NAMES.get(b["team"], "?")}'
            cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
            if selected:
                for cx, cy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                    cv2.rectangle(vis, (cx - HANDLE_PX, cy - HANDLE_PX),
                                  (cx + HANDLE_PX, cy + HANDLE_PX),
                                  SELECT_COLOR, 1)
        edited = " [edited]" if self.fi in self.edits else ""
        cv2.putText(vis,
                    f"frame {self.fi}/{self.n_frames - 1}{edited}   "
                    f"arrows=nav G/W/R=color X=del Z=revert frame "
                    f"drag=move/resize/create S=save Q=quit",
                    (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, self.status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 220, 255), 2, cv2.LINE_AA)
        return vis

    def goto(self, fi):
        self.fi = max(0, min(self.n_frames - 1, fi))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fi)
        ok, fr = self.cap.read()
        if ok:
            self.frame = fr

    def save(self):
        with open(self.out_path, "w") as f:
            json.dump({"edits": {str(k): v for k, v in self.edits.items()}},
                      f, indent=1)
        print(f"saved {len(self.edits)} edited frames to {self.out_path}")

    # ── main loop ────────────────────────────────────────────────────

    def run(self):
        win = "Track Editor"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.on_mouse)
        self.goto(0)
        while True:
            cv2.imshow(win, self.draw())
            key = int(cv2.waitKeyEx(30))
            if key in (-1, 0xFFFFFFFF):   # no key (signed or unsigned -1)
                continue
            k = key & 0xFF
            if key in (65361, 2424832) or k == ord("a"):
                self.goto(self.fi - 1)
            elif key in (65363, 2555904) or k == ord("d"):
                self.goto(self.fi + 1)
            elif k in (ord("g"), ord("w"), ord("r")):
                sel = self.selected(self.boxes_mut())
                if sel:
                    sel["team"] = {"g": 0, "w": 1, "r": 2}[chr(k)]
                    self.status = (f"track {sel['tid']} -> "
                                   f"{TEAM_NAMES[sel['team']]}")
            elif k in (ord("x"), 8):       # x or backspace
                if self.sel_tid is not None:
                    items = self.boxes_mut()
                    self.edits[self.fi] = [b for b in items
                                           if b["tid"] != self.sel_tid]
                    self.status = f"deleted track {self.sel_tid}"
                    self.sel_tid = None
                else:
                    self.status = "nothing selected to delete"
            elif k == ord("z"):
                self.edits.pop(self.fi, None)
                self.sel_tid = None
                self.status = "frame reverted to original"
            elif k == ord("s"):
                self.save()
                self.status = f"saved ({len(self.edits)} edited frames)"
            elif k == ord("q"):
                self.save()
                break
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "track_edits.json"))
    args = p.parse_args()
    jsonl = args.video + ".tracks.jsonl"
    if not os.path.exists(jsonl):
        raise SystemExit(f"missing sidecar: {jsonl}")
    Editor(args.video, jsonl, args.out).run()


if __name__ == "__main__":
    main()
