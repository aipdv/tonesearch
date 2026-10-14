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


# Look: dark violet, faint grid, film grain, one glowing lens. Matches assets/banner.svg.
HERO = """<div id="hero">
  <div class="mark"><svg width="34" height="34" viewBox="0 0 38 38" fill="none" stroke="#fff" stroke-linecap="round">
    <circle cx="17" cy="17" r="14" stroke-width="2.6"/><line x1="27.5" y1="27.5" x2="36" y2="36" stroke-width="3.6"/>
    <line x1="11" y1="14" x2="11" y2="20" stroke-width="2.6"/><line x1="17" y1="9.5" x2="17" y2="24.5" stroke="#fb7185" stroke-width="2.6"/>
    <line x1="23" y1="13" x2="23" y2="21" stroke-width="2.6"/></svg><span>tonesearch</span></div>
  <p class="tag">Find the angry minute in your call recordings.</p>
  <p class="sub">Search calls by <b>what</b> was said and <b>how</b> it was said. Or upload your own call.</p>
  <div class="lens"><svg viewBox="0 0 200 100" preserveAspectRatio="none"><polyline points="0,78 20,74 35,80 50,72 62,76 75,70 85,62 92,30 100,14 108,30 115,62 125,70 140,74 155,68 170,75 185,70 200,74"/></svg></div>
</div>"""
CSS = """
html { background: #110e22 !important; }
body { background: radial-gradient(900px 520px at 88% -8%, rgba(99,102,241,.38), transparent 60%),
  radial-gradient(700px 420px at 105% 45%, rgba(124,58,237,.18), transparent 60%),
  linear-gradient(135deg, #2b1f57 0%, #1b1638 55%, #110e22 100%) fixed !important; }
body::before { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 40px 40px; }
body::after { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 1000; opacity: .09;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
.gradio-container { background: transparent !important; max-width: 1080px !important; position: relative; z-index: 1; }
#hero { position: relative; padding: 36px 8px 28px; overflow: visible; }
#hero .mark { display: flex; align-items: center; gap: 14px; }
#hero .mark span { font-size: 46px; font-weight: 300; letter-spacing: -1.2px; color: #fff; }
#hero .tag { margin: 44px 0 4px; font-size: 20px; color: #eef2ff; }
#hero .sub { margin: 0; font-size: 15px; font-weight: 300; color: #a5a3c8; }
#hero .lens { position: absolute; right: 40px; top: 22px; width: 170px; height: 170px; border-radius: 50%;
  border: 2px solid #e0e7ff; background: rgba(15,12,36,.55); overflow: hidden;
  box-shadow: 0 0 36px rgba(129,140,248,.55), inset 0 0 28px rgba(99,102,241,.35); }
#hero .lens svg { position: absolute; left: 0; top: 30px; width: 100%; height: 100px; }
#hero .lens polyline { fill: none; stroke: #fb7185; stroke-width: 2.2; vector-effect: non-scaling-stroke;
  filter: drop-shadow(0 0 6px rgba(244,63,94,.8)); }
@media (max-width: 720px) { #hero .lens { display: none; } }
"""
DARK = dict(body_background_fill="transparent", body_text_color="#e0e7ff", body_text_color_subdued="#a5a3c8",
            background_fill_primary="#15122e", background_fill_secondary="#1b1638",
            block_background_fill="rgba(21,18,46,.72)", block_border_color="rgba(165,180,252,.14)",
            block_label_text_color="#a5a3c8", block_title_text_color="#c7d2fe", border_color_primary="rgba(165,180,252,.14)",
            input_background_fill="rgba(15,12,36,.8)", input_border_color="rgba(165,180,252,.22)",
            button_primary_background_fill="linear-gradient(90deg, #6366f1, #8b5cf6)", button_primary_text_color="#ffffff",
            button_secondary_background_fill="rgba(99,102,241,.14)", button_secondary_text_color="#e0e7ff",
            table_even_background_fill="#17142f", table_odd_background_fill="#1b1638", color_accent_soft="rgba(99,102,241,.25)")

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
        gr.HTML(HERO)
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
            app.load(run_search, [q, tone, w], [hits, seg])  # empty query = tone only, no GPU call
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
    theme = gr.themes.Base(primary_hue="indigo", secondary_hue="rose", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter", weights=(300, 400, 600)), "Helvetica Neue", "sans-serif"])
    theme.set(**DARK, **{k + "_dark": v for k, v in DARK.items()})  # same values both modes: always dark
    app.launch(allowed_paths=sorted({os.path.dirname(c) for c in calls}), theme=theme, css=CSS,
               js="() => document.body.classList.add('dark')")


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
