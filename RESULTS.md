# Tone test results (2026-10-09)

Data: HaessigDB (CC-BY-4.0). 888 acted call-center clips, 4 actors, 37 calls, rated 1-10 for irritation.
Hardware: M3 Mac. Probes are leave-one-actor-out (the test voice is never seen in training).
"Beyond words" = Spearman with the rating left over after a transcript-embedding model explains what it can.

| Signal | Spearman | AUC (top vs bottom 25%) | Beyond words |
|---|---|---|---|
| EG2 audio, text query "angry customer" | +0.36 | 0.78 | +0.11 |
| EG2 audio embedding, ridge probe | +0.45 | 0.84 | +0.06 |
| EG2 transcript embedding, ridge probe | +0.45 | 0.84 | n/a |
| Loudness | +0.42 | 0.79 | +0.41 |
| emotion2vec+ base, "angry" score, zero-shot | +0.54 | 0.88 | +0.54 |
| emotion2vec+ embedding, ridge probe | +0.57 | 0.91 | +0.54 |
| emotion2vec+ + EG2 audio, ridge probe | +0.66 | 0.94 | +0.45 |

Within one call, the emotion2vec angry score follows escalation: median Spearman +0.41 over 37 calls.

Speed: EG2 audio 41x real time (MPS). emotion2vec+ base 30x real time.

## What it means

- EG2 audio encodes **what** is said, about as well as a transcript. It does not add tone.
- emotion2vec encodes **how** it is said. Zero-shot, no training.
- Product stack: EG2 for semantic search, emotion2vec for the tone score and the per-call timeline.

## Caveats

- Acted, scripted speech. Real calls are noisier.
- Only 4 actors.
- emotion2vec license is FunASR "model-license". Check commercial terms before a paid product.

Scripts: `eval_tone.py`, `probe_tone.py`, `probe_e2v.py`.
