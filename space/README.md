---
title: tonesearch
colorFrom: indigo
colorTo: purple
thumbnail: https://raw.githubusercontent.com/aipdv/tonesearch/main/assets/banner.jpg
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
short_description: Search call audio by what was said and how it was said
---

# tonesearch demo

Search 8 call recordings by **what** was said (EmbeddingGemma 2, audio in, no transcript) and **how** it was said (emotion2vec+ tone score, prebuilt).

- Search tab: type a phrase, pick a tone, set the tone weight.
- Timeline tab: tone over one call, with the top 3 spikes.
- Your call tab: upload your own call (up to 5 min). Both models run on a ZeroGPU GPU, then you get its tone timeline and top segments.

Demo audio: 8 calls from "HaessigDB: A Database of Irritable Speech with Intensity Grading" by Niklas Weller, Marc Christopher Grau and Ivo Blohm (2026), doi:10.5281/zenodo.18745894, licensed CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). The calls are acted, not real customers. The sentence clips were joined into whole calls.

Models: `google/embeddinggemma-2` (query and audio embeddings), `emotion2vec/emotion2vec_plus_base` (tone scores; FunASR model license). Demo call scores are prebuilt; uploads run live.
