"""Same probe as probe_tone.py, but on emotion2vec+ (a speech-emotion model).
Usage: python probe_e2v.py <haessig dir>
"""
import sys, os, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from funasr import AutoModel
from sentence_transformers import SentenceTransformer

ROOT = sys.argv[1]
df = pd.read_csv(os.path.join(ROOT, "all_ratings.csv"))
df["path"] = [os.path.join(ROOT, "audio", f) for f in df.file_name]
df = df[df.path.map(os.path.exists)].reset_index(drop=True)
y = df[["annoyance", "frustration", "aggression"]].mean(axis=1).to_numpy()
hi, lo = y >= np.quantile(y, 0.75), y <= np.quantile(y, 0.25)
m = hi | lo

e2v = AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf", disable_update=True)
t = time.perf_counter()
out = [e2v.generate(p, granularity="utterance", extract_embedding=True)[0] for p in df.path]
print(f"e2v: {time.perf_counter() - t:.1f}s for {len(df)} clips")
E = np.stack([o["feats"] for o in out])
labels = out[0]["labels"]
angry = np.array([o["scores"][[i for i, l in enumerate(labels) if "angry" in l][0]] for o in out])

T = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"vision_config": None, "audio_config": None}).encode(
    [f"title: none | text: {s}" for s in df.transcript])
A = np.load(os.path.join(ROOT, "audio_emb.npy"))


def loao(X):
    pred = np.zeros(len(y))
    for a in df.actor.unique():
        tr, te = df.actor != a, df.actor == a
        pred[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr]).predict(X[te])
    return pred


resid = y - loao(T)
for name, p in [("e2v angry score (0-shot)", angry), ("e2v emb probe", loao(E)),
                ("e2v + EG2 audio", loao(np.c_[E, A]))]:
    print(f"{name:<26} spearman={spearmanr(p, y).statistic:+.3f}  AUC={roc_auc_score(hi[m], p[m]):.3f}"
          f"  beyond-words={spearmanr(p, resid).statistic:+.3f}")

# Timeline check: inside one call, does the angry score follow the rating as the call escalates?
per_call = [spearmanr(g.a, g.y).statistic for _, g in df.assign(a=angry, y=y).groupby(["actor", "call"]) if len(g) >= 5]
print(f"within-call spearman (angry score vs rating): median={np.nanmedian(per_call):+.3f} over {len(per_call)} calls")
