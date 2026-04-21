"""Gradio UI for CLIP + FAISS semantic image search."""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
from pathlib import Path

import gradio as gr

from backend.config import get_settings
from backend.search import SearchEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


def init_engine(artifacts_dir: str | None) -> str:
    global _engine
    settings = get_settings(artifacts_dir=artifacts_dir) if artifacts_dir else get_settings()
    idx = settings.index_path
    man = settings.manifest_path
    if not idx.is_file() or not man.is_file():
        _engine = None
        return (
            f"Index not found. Build it first:\n"
            f"  python -m backend.embed --images-dir {settings.images_dir}\n"
            f"Expected: {idx} and {man}"
        )
    _engine = SearchEngine(settings)
    _engine.load()
    return f"Ready — {_engine.num_indexed} images indexed."


def search_text_ui(
    query: str,
    k: int,
    metric: str,
    tags: str,
) -> tuple[list[str], str]:
    if not query or not str(query).strip():
        return [], "Enter a text query."
    try:
        eng = get_engine()
    except RuntimeError as e:
        return [], str(e)
    req = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    try:
        paths, scores = eng.search_by_text(
            str(query).strip(),
            k=int(k),
            metric=metric,  # type: ignore[arg-type]
            required_tags=req,
        )
    except Exception as e:
        logger.exception("text search failed")
        return [], str(e)
    if not paths:
        return [], "No results (try increasing k or relaxing tag filter)."
    # Gradio Gallery accepts file paths
    caption = "\n".join(f"{Path(p).name}: {s:.4f}" for p, s in zip(paths, scores))
    return paths, caption


def search_image_ui(
    image,
    k: int,
    metric: str,
    tags: str,
) -> tuple[list[str], str]:
    if image is None:
        return [], "Upload a query image."
    path = image if isinstance(image, str) else getattr(image, "name", None) or str(image)
    if not path or not Path(path).is_file():
        return [], "Invalid upload."
    try:
        eng = get_engine()
    except RuntimeError as e:
        return [], str(e)
    req = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    try:
        paths, scores = eng.search_by_image(
            path,
            k=int(k),
            metric=metric,  # type: ignore[arg-type]
            required_tags=req,
        )
    except Exception as e:
        logger.exception("image search failed")
        return [], str(e)
    if not paths:
        return [], "No results (try increasing k or relaxing tag filter)."
    caption = "\n".join(f"{Path(p).name}: {s:.4f}" for p, s in zip(paths, scores))
    return paths, caption


def build_app(status_message: str = "") -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="blue",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    with gr.Blocks(
        title="Semantic Image Search",
        theme=theme,
        css="""
        .gr-button-primary { font-weight: 600; }
        footer { visibility: hidden; }
        """,
    ) as demo:
        gr.Markdown(
            "## Semantic image search\n"
            "CLIP embeddings + FAISS. **Tab 1:** describe an image in words. "
            "**Tab 2:** upload a picture to find similar ones. "
            "Optional **tags** filter (comma-separated) matches `data/images/tags.json` if present."
        )

        status = gr.Textbox(label="Status", value=status_message, interactive=False, lines=3)

        with gr.Row():
            k = gr.Slider(1, 50, value=12, step=1, label="Top-k")
            metric = gr.Dropdown(
                choices=["cosine", "l2"],
                value="cosine",
                label="Similarity metric",
                info="Should match how the index was built (default: cosine / inner product).",
            )
            tag_filter = gr.Textbox(
                label="Required tags (optional)",
                placeholder="e.g. outdoor, night",
                lines=1,
            )

        with gr.Tabs():
            with gr.Tab("Text → image"):
                q = gr.Textbox(
                    label="Query",
                    placeholder="e.g. a red bicycle near the beach",
                    lines=2,
                )
                btn_t = gr.Button("Search", variant="primary")
                gal_t = gr.Gallery(
                    label="Results",
                    columns=4,
                    rows=2,
                    height=420,
                    object_fit="contain",
                )
                cap_t = gr.Textbox(label="Scores", lines=6, interactive=False)
                btn_t.click(search_text_ui, [q, k, metric, tag_filter], [gal_t, cap_t])

            with gr.Tab("Image → image"):
                up = gr.Image(type="filepath", label="Query image")
                btn_i = gr.Button("Find similar", variant="primary")
                gal_i = gr.Gallery(
                    label="Results",
                    columns=4,
                    rows=2,
                    height=420,
                    object_fit="contain",
                )
                cap_i = gr.Textbox(label="Scores", lines=6, interactive=False)
                btn_i.click(search_image_ui, [up, k, metric, tag_filter], [gal_i, cap_i])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for semantic image search.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory containing index.faiss and manifest.json",
    )
    args = parser.parse_args()

    msg = init_engine(args.artifacts_dir)
    logger.info("%s", msg)

    demo = build_app(status_message=msg)
    demo.launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()
