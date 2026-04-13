# Semantic Image Search (CLIP + FAISS)

This project is a small, production-style **semantic image search** service: images are embedded with **OpenAI CLIP** (via Hugging Face Transformers), stored in a **FAISS** index, and queried by **natural language** or by **another image**. Retrieval uses **cosine similarity** (implemented as inner product on **L2-normalized** CLIP vectors) or optional **L2** distance on the same vectors. A **Gradio** UI provides two tabs—text → image and image → image—with optional **tag-based filtering** when a `tags.json` file is present next to your images.

The codebase is modular (`backend/` for embedding, indexing, and search; `app.py` for the UI), typed where it matters, and suitable to showcase on a machine learning or AI engineering CV as a complete retrieval pipeline.

## Features

- **Text → image** and **image → image** search with shared CLIP embedding space
- **FAISS** `IndexFlatIP` / `IndexFlatL2` for exact nearest-neighbor search
- **Normalized embeddings** before indexing and querying (cosine = dot product)
- **Batched image encoding** and automatic **GPU** use when CUDA or Apple MPS is available
- **Persisted** index and manifest (paths + optional tags) under `artifacts/`
- **Configurable** image directory, artifacts directory, model id, batch size, and metric
- **Gradio** gallery UI with optional **tag filter** and **metric** selector
- **Dockerfile** for containerized deployment (CPU-oriented; swap base image for CUDA if needed)

## Tech stack

| Layer        | Technology                                      |
|-------------|--------------------------------------------------|
| Embeddings  | OpenAI CLIP (`openai/clip-vit-base-patch32`)     |
| Framework   | PyTorch, Hugging Face Transformers               |
| Vector DB   | Meta FAISS                                       |
| UI          | Gradio                                           |

## Setup

**Requirements:** Python 3.10+ recommended, `pip`, and enough disk for PyTorch and model weights.

```bash
cd /path/to/new
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Sample images (optional)

Put JPEG/PNG/WebP files under `data/images/`, or download a few placeholders:

```bash
python scripts/download_sample_images.py
```

### Build the FAISS index

From the **project root** (so `import backend` resolves):

```bash
python -m backend.embed --images-dir data/images --artifacts-dir artifacts
```

Options include `--batch-size`, `--device` (`cuda` / `mps` / `cpu`), `--model`, and `--metric` (`cosine` or `l2`). The manifest records the metric used; query with the **same** metric for consistent scores.

### Optional tags (`data/images/tags.json`)

Map **relative paths** (from `data/images/`) to string tags for UI filtering:

```json
{
  "sample_1.jpg": ["animal", "outdoor"],
  "sample_2.jpg": ["abstract", "indoor"]
}
```

The UI’s “Required tags” field expects comma-separated substrings; each token must appear in at least one tag (case-insensitive).

### Run the app

```bash
python app.py --host 127.0.0.1 --port 7860
```

Open the printed URL. If the index is missing, the status box explains how to run `backend.embed`.

## Example usage (Python API)

```python
from backend.config import get_settings
from backend.search import SearchEngine

settings = get_settings(artifacts_dir="artifacts")
engine = SearchEngine(settings)
engine.load()

paths, scores = engine.search_by_text("a dog running on grass", k=5)
print(list(zip(paths, scores)))

paths2, scores2 = engine.search_by_image("/path/to/query.jpg", k=5)
print(list(zip(paths2, scores2)))
```

## Project layout

```
backend/
  config.py    # Paths, model id, batch size, defaults
  embed.py     # CLIP image/text encoding, CLI to build index
  index.py     # FAISS build/save/load, manifest I/O, query
  search.py    # search_by_text, search_by_image, tag filtering
app.py         # Gradio UI
data/images/   # Your image corpus (and optional tags.json)
artifacts/     # Generated: index.faiss, manifest.json
requirements.txt
Dockerfile
```

## Docker

Build and run (mount an `artifacts` volume if you already built the index on the host):

```bash
docker build -t clip-search .
docker run --rm -p 7860:7860 -v "$(pwd)/artifacts:/app/artifacts" clip-search
```

If `artifacts/` is empty, build the index inside the container (one-off) or copy prebuilt files into the volume.

## Screenshots

_Add Gradio screenshots here after running the app (text search and image search tabs)._

## License

Model weights are subject to the CLIP / Hugging Face model licenses. Use this project’s code under terms you choose for your portfolio (e.g. MIT) if you add a `LICENSE` file.
