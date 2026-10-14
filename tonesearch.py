"""tonesearch: search call audio by what was said (EmbeddingGemma 2) and how it was said (emotion2vec).

  tonesearch index <audio dir> <index dir>
  tonesearch search <index dir> "query" [--tone angry] [--w 1]
  tonesearch ui <index dir>
"""
import argparse, functools, glob, os, time
import numpy as np, librosa

SR = 16000
MIN_S, MAX_S = 2.0, 10.0
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


@functools.cache
def eg2():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("google/embeddinggemma-2", config_kwargs={"vision_config": None})


@functools.cache
def e2v():
    from funasr import AutoModel
    return AutoModel(model="emotion2vec/emotion2vec_plus_base", hub="hf", disable_update=True, disable_log=True)


def load(path):
    return librosa.load(path, sr=SR, mono=True)[0]


def segments(y):
    """Split on silence, merge pieces shorter than MIN_S, cut pieces longer than MAX_S. Returns [(start, end)] in samples."""
    out = []
    for s, e in librosa.effects.split(y, top_db=30):
        if out and out[-1][1] - out[-1][0] < MIN_S * SR:
            out[-1][1] = e
        else:
            out.append([s, e])
    if len(out) > 1 and out[-1][1] - out[-1][0] < MIN_S * SR:
        last = out.pop()
        out[-1][1] = last[1]
    cut = []
    for s, e in out:
        n = int(np.ceil((e - s) / (MAX_S * SR)))
        edges = np.linspace(s, e, n + 1).astype(int)
        cut += list(zip(edges[:-1], edges[1:]))
    return cut


def index(audio_dir, index_dir):
    files = sorted(f for f in glob.glob(os.path.join(audio_dir, "**", "*"), recursive=True) if f.lower().endswith(AUDIO_EXT))
    clips, meta = [], []
    for f in files:
        y = load(f)
        for s, e in segments(y):
            clips.append(y[s:e])
            meta.append((os.path.relpath(f, index_dir), s / SR, e / SR))  # relative, so the index moves with its audio
    t0 = time.perf_counter()
    emb = eg2().encode([{"array": c, "sampling_rate": SR} for c in clips], batch_size=8,
                       normalize_embeddings=True, show_progress_bar=True)
    res = e2v().generate(clips, granularity="utterance", disable_pbar=True)
    labels = [l.split("/")[-1] for l in res[0]["labels"]]
    os.makedirs(index_dir, exist_ok=True)
    # ponytail: one npz, brute-force matmul at search time; move to a vector DB above ~1M segments
    np.savez(os.path.join(index_dir, "index.npz"), emb=emb.astype(np.float32),
             tone=np.array([r["scores"] for r in res], np.float32), labels=np.array(labels),
             path=np.array([m[0] for m in meta]), start=np.array([m[1] for m in meta]), end=np.array([m[2] for m in meta]))
    secs = sum(map(len, clips)) / SR
    print(f"{len(files)} files, {len(clips)} segments, {secs / 60:.1f} min of speech, "
          f"models ran at {secs / (time.perf_counter() - t0):.0f}x real time -> {index_dir}")


@functools.cache
def open_index(index_dir):
    ix = dict(np.load(os.path.join(index_dir, "index.npz")))
    ix["path"] = np.array([os.path.normpath(os.path.join(os.path.abspath(index_dir), p)) for p in ix["path"]])
    return ix


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def tone_score(ix, tone):
    """Log-odds of `tone` vs the mean label. Raw softmax saturates at 1.0 (133 tied "angry" segments on the demo); the logit gap does not."""
    lp = np.log(ix["tone"].astype(float) + 1e-30)  # float64: float32 shows as 5.289999961853027 in the UI
    return lp[:, list(ix["labels"]).index(tone)] - lp.mean(1)


def embed_query(query):
    return eg2().encode(f"task: search result | query: {query}", normalize_embeddings=True)


def search(index_dir, query="", tone="angry", w=1.0, k=10):
    """Rank segments by z(cosine to query) + w * z(tone score). Empty query = tone only. Returns row indices, scores."""
    ix = open_index(index_dir)
    t = tone_score(ix, tone)
    what = z(ix["emb"] @ embed_query(query)) if query else 0
    score = what + w * z(t)
    top = np.argsort(-score)[:k]
    return top, score[top]


MAX_UPLOAD_S = 300  # ponytail: 5 min keeps one upload inside a free visitor's daily ZeroGPU quota


def ui(index_dir):
    import gradio as gr, pandas as pd, shutil, tempfile
    ix = open_index(index_dir)
    labels = [l for l in ix["labels"] if l not in ("<unk>", "other")]
    calls = sorted(set(ix["path"]))

    def run_search(q, tone, w):
        top, score = search(index_dir, q, tone, w, k=20)
        df = pd.DataFrame({"file": [os.path.basename(ix["path"][i]) for i in top], "start": ix["start"][top].round(1),
                           "end": ix["end"][top].round(1), "score": score.round(2), "row": top})
        return df, play(df, 0)

    def play(df, r):
        if not len(df):
            return None
        i = int(df.row.iloc[r])
        y = load(ix["path"][i])
        return SR, y[int(ix["start"][i] * SR):int(ix["end"][i] * SR)]

    def on_select(df, e: gr.SelectData):
        return play(df, e.index[0])

    def timeline(call, tone):
        m = ix["path"] == call
        t = tone_score(ix, tone)[m]
        df = pd.DataFrame({"time": ix["start"][m], tone: t})
        spikes = pd.DataFrame({"start": ix["start"][m], "end": ix["end"][m], "score": t.round(2)}).nlargest(3, "score")
        return gr.LinePlot(df, x="time", y=tone, y_title=f"{tone} (log-odds)"), spikes.round(1), call

    def run_upload(path, q, tone):
        """Index one uploaded call in a temp dir, then show its tone timeline and top segments."""
        if not path:
            raise gr.Error("Upload a call first.")
        if librosa.get_duration(path=path) > MAX_UPLOAD_S:
            raise gr.Error(f"Calls up to {MAX_UPLOAD_S // 60} minutes, please.")
        d = tempfile.mkdtemp()
        shutil.copy(path, d)
        index(d, os.path.join(d, "index"))
        ux = open_index(os.path.join(d, "index"))
        t = tone_score(ux, tone)
        top, score = search(os.path.join(d, "index"), q, tone, 1.0, k=10)
        plot = gr.LinePlot(pd.DataFrame({"time": ux["start"], tone: t}), x="time", y=tone, y_title=f"{tone} (log-odds)")
        return plot, pd.DataFrame({"start": ux["start"][top].round(1), "end": ux["end"][top].round(1), "score": score.round(2)})

    with gr.Blocks(title="tonesearch") as app:
        gr.Markdown("# tonesearch\nSearch calls by **what** was said and **how** it was said.")
        with gr.Tab("Search"):
            with gr.Row():
                q = gr.Textbox(label="What was said (optional)", placeholder="customer wants a refund")
                tone = gr.Dropdown(labels, value="angry", label="How it was said")
                w = gr.Slider(0, 3, value=1, step=0.1, label="Tone weight (0 = words only)")
            go = gr.Button("Search", variant="primary")
            hits = gr.Dataframe(interactive=False, label="Results (click a row to play)")
            seg = gr.Audio(label="Segment")
            go.click(run_search, [q, tone, w], [hits, seg])
            q.submit(run_search, [q, tone, w], [hits, seg])
            hits.select(on_select, hits, seg)
        with gr.Tab("Timeline"):
            with gr.Row():
                call = gr.Dropdown([(os.path.basename(c), c) for c in calls], value=calls[0], label="Call")
                tl_tone = gr.Dropdown(labels, value="angry", label="Tone")
            plot = gr.LinePlot()
            spikes = gr.Dataframe(label="Top 3 spikes (seconds)", interactive=False)
            player = gr.Audio(type="filepath", label="Full call")
            for ev in (call.change, tl_tone.change, app.load):
                ev(timeline, [call, tl_tone], [plot, spikes, player])
        with gr.Tab("Your call"):
            up = gr.Audio(type="filepath", label=f"Upload a call (up to {MAX_UPLOAD_S // 60} min)")
            with gr.Row():
                up_q = gr.Textbox(label="What was said (optional)", placeholder="customer wants a refund")
                up_tone = gr.Dropdown(labels, value="angry", label="How it was said")
            up_go = gr.Button("Analyze", variant="primary")
            up_plot = gr.LinePlot()
            up_hits = gr.Dataframe(label="Top segments (seconds)", interactive=False)
            up_go.click(run_upload, [up, up_q, up_tone], [up_plot, up_hits])
    app.launch(allowed_paths=sorted({os.path.dirname(c) for c in calls}))


def main():
    p = argparse.ArgumentParser(prog="tonesearch")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("audio_dir"); a.add_argument("index_dir")
    s = sub.add_parser("search"); s.add_argument("index_dir"); s.add_argument("query", nargs="?", default="")
    s.add_argument("--tone", default="angry"); s.add_argument("--w", type=float, default=1.0); s.add_argument("-k", type=int, default=10)
    u = sub.add_parser("ui"); u.add_argument("index_dir")
    args = p.parse_args()
    if args.cmd == "index":
        index(args.audio_dir, args.index_dir)
    elif args.cmd == "search":
        ix = open_index(args.index_dir)
        for i, sc in zip(*search(args.index_dir, args.query, args.tone, args.w, args.k)):
            print(f"{sc:+.2f}  {ix['start'][i]:7.1f}-{ix['end'][i]:<7.1f} {ix['path'][i]}")
    else:
        ui(args.index_dir)


if __name__ == "__main__":
    main()
