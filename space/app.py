import spaces  # must come before torch: ZeroGPU patches CUDA on import
import glob, os, librosa, tonesearch

# Load both models on CUDA at startup. Outside @spaces.GPU, ZeroGPU emulates CUDA; inside, the real GPU is used.
tonesearch.eg2().to("cuda")
tonesearch.e2v()


def index_seconds(audio_dir, index_dir):
    """GPU time for one upload: model warmup plus about 10x real time, so a visitor's small quota is not over-asked."""
    secs = sum(librosa.get_duration(path=f) for f in glob.glob(os.path.join(audio_dir, "*")) if os.path.isfile(f))
    return int(20 + secs / 10)


tonesearch.embed_query = spaces.GPU(duration=10)(tonesearch.embed_query)
tonesearch.index = spaces.GPU(duration=index_seconds)(tonesearch.index)
tonesearch.ui(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index"))
