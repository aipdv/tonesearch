"""Join HaessigDB sentence clips into whole-call WAVs for the demo.
Usage: python -I make_demo.py <haessig dir>
Writes <haessig dir>/calls/actor{a}_call{c}.wav and calls/sentences.csv (where each sentence sits in its call).
"""
import sys, os
import numpy as np, pandas as pd, librosa, soundfile as sf

SR, GAP = 16000, np.zeros(8000, np.float32)  # 0.5 s of silence between sentences
root = sys.argv[1]
out = os.path.join(root, "calls")
os.makedirs(out, exist_ok=True)
df = pd.read_csv(os.path.join(root, "all_ratings.csv"))
df["path"] = [os.path.join(root, "audio", f) for f in df.file_name]
df = df[df.path.map(os.path.exists)].drop_duplicates("file_name")  # CSV has rows for clips that do not ship
df["irritation"] = df[["annoyance", "frustration", "aggression"]].mean(axis=1)

rows = []
for (a, c), g in df.sort_values("sentence").groupby(["actor", "call"]):
    name, parts, t = f"actor{a}_call{c}.wav", [], 0.0
    for r in g.itertuples():
        y = librosa.load(r.path, sr=SR, mono=True)[0]
        rows.append((name, t, t + len(y) / SR, r.irritation))
        parts += [y, GAP]
        t += (len(y) + len(GAP)) / SR
    sf.write(os.path.join(out, name), np.concatenate(parts), SR)
pd.DataFrame(rows, columns=["file", "start", "end", "irritation"]).to_csv(os.path.join(out, "sentences.csv"), index=False)
print(f"{df.groupby(['actor', 'call']).ngroups} calls, {len(rows)} sentences -> {out}")
