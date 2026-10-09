"""Tone test: does EmbeddingGemma 2 audio separate irritated speech from calm speech?

Data: HaessigDB (CC-BY-4.0), call-center clips rated 1-10 for annoyance,
frustration and aggression. Usage: python eval_tone.py <dir with all_ratings.csv and audio/>
"""
import sys, time, os
import numpy as np, pandas as pd, librosa
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

ROOT = sys.argv[1]
SR = 16000
ANGRY = "task: search result | query: an angry, irritated customer raising their voice"
CALM = "task: search result | query: a calm, polite customer speaking normally"


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def prosody(y):
    # ponytail: two cheap features as the baseline the embedding must beat
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=SR)
    return float(librosa.feature.rms(y=y).mean()), float(np.std(f0))


def report(name, score, rating, hi, lo):
    rho = spearmanr(score, rating).statistic
    mask = hi | lo
    auc = roc_auc_score(hi[mask], score[mask])
    print(f"{name:<28} spearman={rho:+.3f}  AUC(hi vs lo)={auc:.3f}")


def main():
    df = pd.read_csv(os.path.join(ROOT, "all_ratings.csv"))
    df["path"] = [os.path.join(ROOT, "audio", f) for f in df.file_name]
    df = df[df.path.map(os.path.exists)].reset_index(drop=True)  # 1068 rows, 888 clips ship
    df["irritation"] = df[["annoyance", "frustration", "aggression"]].mean(axis=1)
    rating = df.irritation.to_numpy()
    hi, lo = rating >= np.quantile(rating, 0.75), rating <= np.quantile(rating, 0.25)
    print(f"clips={len(df)}  actors={df.actor.nunique()}  calls={df.call.nunique()}")

    model = SentenceTransformer("google/embeddinggemma-2", config_kwargs={"vision_config": None})
    audio = [load(p) for p in df.path]

    t = time.perf_counter()
    A = model.encode([{"array": y, "sampling_rate": SR} for y in audio], batch_size=8, show_progress_bar=True)
    dt = time.perf_counter() - t
    secs = sum(len(y) for y in audio) / SR
    print(f"audio embed: {dt:.1f}s for {secs:.0f}s of audio on {model.device} ({secs / dt:.1f}x real time)")

    q = model.encode([ANGRY, CALM])
    T = model.encode([f"title: none | text: {s}" for s in df.transcript])

    report("audio: angry query", A @ q[0], rating, hi, lo)
    report("audio: angry - calm", A @ q[0] - A @ q[1], rating, hi, lo)
    report("transcript: angry - calm", T @ q[0] - T @ q[1], rating, hi, lo)
    rms, f0sd = np.array([prosody(y) for y in audio]).T
    report("prosody: loudness", rms, rating, hi, lo)
    report("prosody: pitch spread", f0sd, rating, hi, lo)

    # Tone, not words: remove what the transcript explains, then check what audio adds.
    t_score = T @ q[0] - T @ q[1]
    resid = rating - np.polyval(np.polyfit(t_score, rating, 1), t_score)
    print(f"audio vs rating left after words: spearman={spearmanr(s_audio := A @ q[0] - A @ q[1], resid).statistic:+.3f}")
    print(f"loudness vs rating left after words: spearman={spearmanr(rms, resid).statistic:+.3f}")

    # Same transcript: same transcript, different delivery. Does audio rank the more irritated take higher?
    s = A @ q[0] - A @ q[1]
    wins = total = 0
    for _, g in df.assign(s=s).groupby("transcript"):
        if len(g) < 2:
            continue
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                d = g.irritation.iloc[i] - g.irritation.iloc[j]
                if abs(d) >= 1.0:
                    total += 1
                    wins += (g.s.iloc[i] - g.s.iloc[j]) * d > 0
    print(f"same words, different tone: audio picks the angrier take {wins}/{total}"
          + (f" = {wins / total:.0%}" if total else ""))

    np.save(os.path.join(ROOT, "audio_emb.npy"), A)


if __name__ == "__main__":
    main()
