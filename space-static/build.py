"""Export the space/ demo index for the static Space: data/emb.bin, data/segments.json, calls/*.mp3.

  python -I space-static/build.py      # from the repo root, needs space/index and space/calls
"""
import json, os, subprocess
import numpy as np
from tonesearch import open_index, tone_score

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "space-static")
ix = open_index(os.path.join(root, "space", "index"))
os.makedirs(os.path.join(out, "data"), exist_ok=True)
os.makedirs(os.path.join(out, "calls"), exist_ok=True)

ix["emb"].astype("<f4").tofile(os.path.join(out, "data", "emb.bin"))  # row-major, n x 768
calls = sorted(set(ix["path"]))
names = [os.path.splitext(os.path.basename(c))[0] for c in calls]
labels = [str(l) for l in ix["labels"]]
json.dump({
    "dim": int(ix["emb"].shape[1]),
    "calls": names,
    "labels": [l for l in labels if l not in ("<unk>", "other")],
    "call": [calls.index(p) for p in ix["path"]],
    "start": ix["start"].round(2).tolist(),
    "end": ix["end"].round(2).tolist(),
    "tone": {l: tone_score(ix, l).round(3).tolist() for l in labels},  # same log-odds as tonesearch.py
}, open(os.path.join(out, "data", "segments.json"), "w"), separators=(",", ":"))

for c, n in zip(calls, names):  # CBR mp3 so seeking to a segment start is exact enough
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", c, "-ac", "1", "-b:a", "48k",
                    os.path.join(out, "calls", n + ".mp3")], check=True)
print(f"{len(ix['path'])} segments, {len(calls)} calls -> {out}")
