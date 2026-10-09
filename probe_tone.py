"""Is tone inside the EmbeddingGemma 2 audio vector? Linear probe, leave-one-actor-out.
Usage: python probe_tone.py <haessig dir>  (run eval_tone.py first; it saves audio_emb.npy)
"""
import sys, os
import numpy as np, pandas as pd, librosa
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

ROOT = sys.argv[1]
df = pd.read_csv(os.path.join(ROOT, "all_ratings.csv"))
df["path"] = [os.path.join(ROOT, "audio", f) for f in df.file_name]
df = df[df.path.map(os.path.exists)].reset_index(drop=True)
y = df[["annoyance", "frustration", "aggression"]].mean(axis=1).to_numpy()
hi, lo = y >= np.quantile(y, 0.75), y <= np.quantile(y, 0.25)
A = np.load(os.path.join(ROOT, "audio_emb.npy"))
T = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"vision_config": None, "audio_config": None}).encode(
    [f"title: none | text: {s}" for s in df.transcript])
rms = np.array([librosa.feature.rms(y=librosa.load(p, sr=16000)[0]).mean() for p in df.path])
# ponytail: z-score loudness within each call, a product sees one call at a time
rms_z = pd.Series(rms).groupby(df.call.values).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9)).to_numpy()


def loao(X):
    pred = np.zeros(len(y))
    for a in df.actor.unique():
        tr, te = df.actor != a, df.actor == a
        pred[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr]).predict(X[te])
    return pred


p_text = loao(T)
resid = y - p_text
m = hi | lo
for name, X in [("transcript emb", T), ("audio emb", A), ("loudness (z in call)", rms_z[:, None]),
                ("audio emb + loudness", np.c_[A, rms_z]), ("audio + transcript", np.c_[A, T])]:
    p = loao(X)
    print(f"{name:<22} spearman={spearmanr(p, y).statistic:+.3f}  AUC={roc_auc_score(hi[m], p[m]):.3f}"
          f"  beyond-words={spearmanr(p, resid).statistic:+.3f}")
