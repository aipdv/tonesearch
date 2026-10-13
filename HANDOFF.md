# Handoff: tonesearch (paused 2026-10-09)

## Goal
Ship tonesearch as a Hacktoberfest 2026 DEV Challenge entry: search call audio by what was said (EmbeddingGemma 2) and how it was said (emotion2vec tone score + per-call timeline). Target: Week 2 challenge, Oct 12 to 18 (theme revealed Mon Oct 12). Main prize target: Best Use of Gemma ($200).

## Where things are
- Validation numbers and CPU speed: `RESULTS.md`
- How to run: `README.md`
- Code: `tonesearch.py` (index, search, CLI, Gradio UI), `make_demo.py`, `test_tonesearch.py`
- Validation scripts: `eval_tone.py`, `probe_tone.py`, `probe_e2v.py`
- History: `git log` (each commit message carries its numbers)
- Data (not in git): `~/Coding/tonesearch-data/haessig` (HaessigDB, CC-BY-4.0; `calls/` demo calls, `index/` built index)
- Venv (not in git): `.venv`, Python 3.12
- Challenge rules: https://dev.to/challenges/hf26
- Hacktoberfest dashboard: https://hacktoberfest.com/my/ (DEV account connected, sticker pack earned)

## Decisions already made (do not reopen)
- Users: voice AI teams and call-center QA. Dictation users dropped.
- EG2 audio = words only (no tone signal). emotion2vec = tone. Use both.
- Score: `z(cos) + w * z(tone[label])`.
- Ship as pip package + Gradio UI; HF Space for demo later; CPU deploy target.
- Cut from MVP: entrainment, ASR, webhook/API, diarization, pgvector.

## Next steps (in order)
1. Mon Oct 12: read the Week 2 theme and shape the write-up angle to fit it.
2. ONNX int8 for the EG2 audio encoder (`onnx-community/embeddinggemma-2-ONNX`). Today CPU indexing is 3.4x real time and EG2 is the bottleneck.
3. Fix tone ties: the emotion2vec angry score saturates at 1.0. Try ranking on a logit or on the score margin over "neutral".
4. Try on one real call recording (user's choice of source, ask about privacy first).
5. With user approval only: HF Space demo (2-line `app.py`), then the DEV post. Writing quality is weighted most in judging.

## Open questions for the user
- Publish the HF Space and DEV post? (Nothing public yet.)
- emotion2vec uses the FunASR model license. Check commercial terms before any paid use.

## Gotchas
- `/handoff` into a new folder stalls on the folder-trust prompt if the user is AFK. Use a background Agent instead.
- HaessigDB CSV has 1068 rows but 888 audio files. Group calls by (actor, call).
- Zenodo download can 504. Use `/api/records/18745894/files/haessigDB.zip/content`.
- Commits: author `aipdv <128473068+xprabhudayal@users.noreply.github.com>`, no Co-Authored-By.

## Verify
```
cd ~/Coding/tonesearch
.venv/bin/python -I test_tonesearch.py ~/Coding/tonesearch-data/haessig   # expect: ok
.venv/bin/tonesearch ui ~/Coding/tonesearch-data/haessig/index            # http://127.0.0.1:7860
```

## Suggested skills
- `ponytail` for any code change (keep it minimal).
- `grilling` before new scope (grill a parent agent, not the user, if the user is AFK).
- `ab` to read the Week 2 theme on dev.to while logged in.
- `humanizer` before the DEV post or any public text.
- `huggingface-best` when building the HF Space.
- `no-mistakes` before pushing larger changes.

## Update 2026-10-13
- Live demo (static, in-browser): https://huggingface.co/spaces/pradachan/tonesearch
- Repo is now public: https://github.com/aipdv/tonesearch
- DEV post saved as an UNPUBLISHED draft (account noreplycareerscoutai). User must set AI Disclosure and click Publish before Oct 18.
- After the DEV post is live: publish post/hf-blog.md as a Hugging Face community blog (humanized, no em dashes).
- First-user section dropped by user choice.
