# tonesearch

Search call recordings by **what** was said and **how** it was said. Runs locally. No transcript needed.

- What: EmbeddingGemma 2 embeds the audio, so a text query finds speech by meaning.
- How: emotion2vec+ scores each segment for tone (angry, sad, happy, ...).
- Timeline: tone over time for one call, with the top 3 spikes.

Validation numbers: [RESULTS.md](RESULTS.md).

## Where

| File | What |
|---|---|
| `tonesearch.py` | Indexer, search, CLI, Gradio UI |
| `make_demo.py` | Builds whole-call demo WAVs from HaessigDB |
| `test_tonesearch.py` | Self-check on the demo index |
| `eval_tone.py`, `probe_tone.py`, `probe_e2v.py` | Research scripts behind RESULTS.md |

## How

```sh
pip install -e .
tonesearch index <audio dir> <index dir>          # any .wav .mp3 .flac .m4a .ogg, resampled to 16 kHz mono
tonesearch search <index dir> "refund request" --tone angry --w 1
tonesearch search <index dir> --tone angry        # tone only
tonesearch ui <index dir>                         # Gradio at http://127.0.0.1:7860
```

Score per segment: `z(cosine to query) + w * z(tone score)`. `w = 0` searches words only.

Demo on HaessigDB (CC-BY-4.0, download from Zenodo record 18745894):

```sh
python -I make_demo.py <haessig dir>
tonesearch index <haessig dir>/calls <haessig dir>/index
python -I test_tonesearch.py <haessig dir>
tonesearch ui <haessig dir>/index
```

License note: emotion2vec+ uses the FunASR model license. Check its commercial terms before a paid product.
