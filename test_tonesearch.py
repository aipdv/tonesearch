"""Self-check on the HaessigDB demo index: a tone-only "angry" search must put more irritated speech at the top.
Usage: python -I test_tonesearch.py <haessig dir>   (after make_demo.py and `tonesearch index <dir>/calls <dir>/index`)
"""
import sys, os
import numpy as np, pandas as pd
import tonesearch as ts

root = sys.argv[1]
idx = os.path.join(root, "index")
ix = ts.open_index(idx)
sent = pd.read_csv(os.path.join(root, "calls", "sentences.csv"))

SR = ts.SR
y = np.r_[np.ones(25 * SR), np.zeros(SR), np.ones(SR // 2)].astype(np.float32)
assert all(e - s <= ts.MAX_S * SR for s, e in ts.segments(y)), "long speech is cut to MAX_S"

t = ts.tone_score(ix, "angry")
assert len(np.unique(np.sort(t)[-100:])) == 100, "top tone scores tie"

order, _ = ts.search(idx, "", "angry", 1.0, k=len(ix["path"]))


def rating(i):  # irritation of the source sentence that overlaps this segment most
    g = sent[sent.file == os.path.basename(ix["path"][i])]
    ov = np.minimum(g.end, ix["end"][i]) - np.maximum(g.start, ix["start"][i])
    return g.irritation.iloc[int(np.argmax(ov))]


top, bottom = np.mean([rating(i) for i in order[:10]]), np.mean([rating(i) for i in order[-10:]])
print(f"mean irritation: top 10 = {top:.2f}, bottom 10 = {bottom:.2f}")
assert top > bottom, "angry search does not rank irritated speech first"
print("ok")
