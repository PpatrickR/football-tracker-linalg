#!/usr/bin/env python
"""Score the tracker against human frame edits.

Ground truth: data/track_edits.json (frames Patrick corrected in
scripts/edit_tracks.py). Tracker output: the .tracks.jsonl sidecar of the
SAME render the edits were made on.

Per edited frame, GT boxes and tracker boxes are greedily matched by IoU.
Reports:
  recall      — fraction of GT players the tracker has a box for
  precision   — fraction of tracker boxes that correspond to a GT player
  team acc    — fraction of matched boxes with the right team color
  mean IoU    — localization quality of the matches

Honest caveat: the GT was created by EDITING the tracker's own output, so
boxes the human left untouched agree by construction. The score therefore
measures "how much did a human need to fix", not independent accuracy —
still the right number to drive regressions down.

Usage:
    python scripts/score_tracker.py JSONL [--edits data/track_edits.json]
                                          [--iou 0.4]
"""
import argparse
import json
import os


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--edits", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "track_edits.json"))
    p.add_argument("--iou", type=float, default=0.4)
    args = p.parse_args()

    with open(args.edits) as f:
        edits = {int(k): v for k, v in json.load(f)["edits"].items()}
    tracker = {}
    with open(args.jsonl) as f:
        for line in f:
            rec = json.loads(line)
            tracker[rec["fi"]] = [(box, shown) for _tid, box, _a, shown
                                  in rec["tracks"] if shown in (0, 1)]

    tot_gt = tot_trk = tot_match = tot_team_ok = 0
    iou_sum = 0.0
    per_frame = []
    for fi in sorted(edits):
        gt = [(b["box"], b["team"]) for b in edits[fi] if b["team"] in (0, 1)]
        trk = tracker.get(fi, [])
        pairs = sorted(((iou(g[0], t[0]), gi, ti)
                        for gi, g in enumerate(gt)
                        for ti, t in enumerate(trk)), reverse=True)
        used_g, used_t = set(), set()
        matches = []
        for v, gi, ti in pairs:
            if v < args.iou or gi in used_g or ti in used_t:
                continue
            used_g.add(gi)
            used_t.add(ti)
            matches.append((v, gi, ti))
        team_ok = sum(1 for _v, gi, ti in matches
                      if gt[gi][1] == trk[ti][1])
        tot_gt += len(gt)
        tot_trk += len(trk)
        tot_match += len(matches)
        tot_team_ok += team_ok
        iou_sum += sum(v for v, _g, _t in matches)
        per_frame.append((fi, len(gt), len(trk), len(matches), team_ok))

    rec = tot_match / tot_gt if tot_gt else 0.0
    prec = tot_match / tot_trk if tot_trk else 0.0
    team = tot_team_ok / tot_match if tot_match else 0.0
    miou = iou_sum / tot_match if tot_match else 0.0
    print(f"edited frames : {len(edits)}")
    print(f"GT boxes      : {tot_gt}   tracker boxes: {tot_trk}")
    print(f"recall        : {rec:.1%}   (GT players the tracker found)")
    print(f"precision     : {prec:.1%}   (tracker boxes that are real)")
    print(f"team accuracy : {team:.1%}   (of matched boxes)")
    print(f"mean match IoU: {miou:.3f}")
    worst = sorted(per_frame, key=lambda r: r[3] / max(1, r[1]))[:8]
    print("\nworst frames (fi, GT, trk, matched, team_ok):")
    for row in worst:
        print(f"  f{row[0]:<4d} GT={row[1]:<3d} trk={row[2]:<3d} "
              f"matched={row[3]:<3d} team_ok={row[4]}")


if __name__ == "__main__":
    main()
