---
title: tonesearch
colorFrom: indigo
colorTo: red
sdk: gradio
app_file: app.py
license: apache-2.0
short_description: Search call audio by what was said and how it was said
---

# tonesearch demo

Search 8 call recordings by **what** was said (EmbeddingGemma 2, audio in, no transcript) and **how** it was said (emotion2vec+ tone score, prebuilt).

- Search tab: type a phrase, pick a tone, set the tone weight.
- Timeline tab: tone over one call, with the top 3 spikes.

Demo audio: 8 calls from "HaessigDB: A Database of Irritable Speech with Intensity Grading" by Niklas Weller, Marc Christopher Grau and Ivo Blohm (2026), doi:10.5281/zenodo.18745894, licensed CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). The calls are acted, not real customers. The sentence clips were joined into whole calls.

Models: `google/embeddinggemma-2` (query and audio embeddings), `emotion2vec/emotion2vec_plus_base` (tone scores, computed offline; FunASR model license).
