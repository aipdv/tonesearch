---
title: tonesearch: find the angry minute in your call recordings, on your own machine
published: false
tags: devchallenge, hf26challenge, ai, opensource
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 2: Side Hustle](https://dev.to/challenges/hacktoberfest-week2-2026-10-12)*

## What I Built

tonesearch searches call recordings by **what** was said and **how** it was said.

You type "customer wants a refund" and pick the tone "angry". You get back the 20 audio segments that match both. Click a row and you hear the segment. A second tab shows the tone over one whole call, with the top 3 spikes.

It needs no transcript and runs on a laptop, so no audio leaves the machine.

The people I think would pay are small call teams and small agencies that run voice agents for clients. A voice agent can make hundreds of calls a day. Somebody has to find the calls where the caller got angry, and fix the prompt. Today that person listens to random calls, or reads transcripts that lose the tone.

Big contact centers buy tools like Gong or Cresta for this. Those tools are built for large teams, and a five-person agency is not their customer. Some agencies also cannot upload customer calls to another company's cloud, because their clients forbid it.

So the job is small and specific: "show me the angry moments in this week's calls, without sending the calls anywhere."

## Demo

<!-- TODO-LINK: Hugging Face Space URL. The Space is built but not deployed yet. -->

Search tab. The query is "customer wants a refund", tone "angry", tone weight 1.

<!-- TODO-SCREENSHOT: upload post/img/search.png to DEV and replace this path -->
![Search results for "customer wants a refund" with tone angry](img/search.png)

Timeline tab. The angry score over one call, with the top 3 spikes.

<!-- TODO-SCREENSHOT: upload post/img/timeline.png to DEV and replace this path -->
![Angry score over one call, with the top 3 spikes](img/timeline.png)

The demo data is HaessigDB, a public set of acted call-center speech (CC-BY-4.0, by Niklas Weller, Marc Christopher Grau and Ivo Blohm). These are actors, not real customers.

## Code

<!-- TODO-LINK: GitHub repo. It is private now. Make it public before you post. -->

About 170 lines of Python in one file, plus a self-check:

```sh
pip install -e .
tonesearch index <audio dir> <index dir>
tonesearch search <index dir> "refund request" --tone angry --w 1
tonesearch ui <index dir>
```

## How I Built It

Two open-weight models do the work.

1. **EmbeddingGemma 2** (`google/embeddinggemma-2`) embeds each audio segment. It also embeds the text query, in the same space. So a text query finds speech by meaning, with no speech-to-text step.
2. **emotion2vec+ base** (`emotion2vec/emotion2vec_plus_base`) scores each segment for tone labels such as angry, sad, happy and neutral.

The indexer splits each call on silence into 2 to 10 second segments. It runs both models once and saves one `.npz` file. A search is one matrix multiply:

```
score = z(cosine to query) + w * z(tone score)
```

`w = 0` searches by words only. An empty query searches by tone only.

### Gemma alone did not hear the tone

My first plan was simpler. I wanted EmbeddingGemma 2 to do everything. Embed the audio, then search for "angry customer" and let the audio carry the tone.

I tested that before I built anything. HaessigDB has 888 clips, each rated 1 to 10 for irritation. I measured how well each signal tracks that rating. I also measured what is left after a transcript model explains what it can. I call that column "beyond words". For the trained probes, the test voice was never in the training set.

| Signal | Spearman | Beyond words |
|---|---|---|
| EmbeddingGemma 2 audio, query "angry customer" | +0.36 | +0.11 |
| EmbeddingGemma 2 audio embedding, trained probe | +0.45 | +0.06 |
| EmbeddingGemma 2 transcript embedding, trained probe | +0.45 | n/a |
| Loudness | +0.42 | +0.41 |
| emotion2vec+ "angry" score, no training | +0.54 | +0.54 |
| emotion2vec+ and EmbeddingGemma 2 audio, trained probe | +0.66 | +0.45 |

The audio embedding scored the same as the transcript embedding. So it hears the words, about as well as a transcript does. It adds almost nothing about tone. Plain loudness did better on tone.

emotion2vec+ caught tone with no training at all. Inside one call, its angry score follows the escalation (median Spearman +0.41 over 37 calls).

So each model got one job. Gemma handles what was said. emotion2vec handles how it was said.

### One more fix: ties at the top

emotion2vec returns softmax scores. On the 60-minute demo, 133 of 1176 segments scored exactly 1.0 for "angry". The top of an angry search was a tie.

The fix is small. I now rank on log-odds: the log score minus the mean log score over all labels. That is the model's own logit gap, and it does not saturate.

| Tone-only "angry" search, 1176 demo segments | Spearman | Mean irritation, top 10 |
|---|---|---|
| Softmax score | +0.50 | 7.19 |
| Log-odds | +0.52 | 7.54 |

### Speed

On an M3 Mac, with Gemma on the GPU, indexing runs at 18x real time. On CPU only, it runs at 3.4x real time. Gemma is the slow part on CPU. An int8 ONNX build of the Gemma audio encoder is my next step.

### What it does not do yet

- The test data is acted and scripted. Real calls are noisier.
- HaessigDB has only 4 actors.
- No transcript and no speaker split. You hear the segment, you do not read it.

## Why Does Open Innovation Matter?

Without open weights, I could not offer this to the people above. There are four reasons.

1. Privacy. Call audio has names, account numbers and health details. With open weights, the models run where the audio already is. Nothing is uploaded. For many small agencies, that is the only way they can use a tool like this.
2. Cost. There is no per-minute API fee. An M3 laptop CPU indexes a 5-minute call in about 90 seconds. The cost per call is close to zero, so a small flat monthly price could still leave a margin. I have not set a price yet.
3. Testing. I could pull the embeddings out and run probes on them. That is how I found that Gemma audio hears words, not tone. A closed API gives you an answer, not the vectors and logits behind it.
4. Swappable parts. When one model did not hear tone, I added a second one. When a better tone model comes out, I change one model name. No vendor has to agree.

A license caveat: emotion2vec+ uses the FunASR model license. I must check its commercial terms before I charge anyone.

## My Agent Session

<!-- TODO: optional. Add a DevRelay session link, or remove this section. -->

## My First User

<!-- TODO: nobody has used it yet. Put it in front of one real call team, then write who they were, what they said, and if they would pay. Do not post this section empty. Remove it if there is no user by the deadline. -->

TODO

## Prize Categories

- Best Use of Gemma: EmbeddingGemma 2 is the search engine. It embeds call audio and text queries in one space, so you can search speech by meaning with no transcript.
