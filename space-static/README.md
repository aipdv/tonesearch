---
title: tonesearch
colorFrom: indigo
colorTo: red
sdk: static
app_file: index.html
license: apache-2.0
short_description: Search call audio by what was said and how it was said
---

# tonesearch demo

Search 8 call recordings by **what** was said (EmbeddingGemma 2, audio in, no transcript) and **how** it was said (emotion2vec+ tone score, prebuilt). Runs fully in the browser.

- Search tab: type a phrase, pick a tone, set the tone weight. Click a row to play it.
- Timeline tab: tone over one call, with the top 3 spikes.
- The query is embedded in your browser with transformers.js (WebGPU if available, else WASM). The audio embeddings and tone scores are precomputed.

Score per segment: `z(cosine to query) + w * z(tone log-odds)`. Empty query = tone only.

Demo audio: 8 calls from "HaessigDB: A Database of Irritable Speech with Intensity Grading" by Niklas Weller, Marc Christopher Grau and Ivo Blohm (2026), doi:10.5281/zenodo.18745894, licensed CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). The calls are acted, not real customers. The sentence clips were joined into whole calls. Converted to mono 48 kbps MP3 for this demo.

Models: `google/embeddinggemma-2` (audio embeddings, computed offline), `onnx-community/embeddinggemma-2-ONNX` q4 (query embeddings, in the browser), `emotion2vec/emotion2vec_plus_base` (tone scores, computed offline; FunASR model license).

Rebuild the data from the repo root: `python -I space-static/build.py` (needs `space/index` and `space/calls`).
