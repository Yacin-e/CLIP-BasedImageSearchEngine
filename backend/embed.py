"""Encode images with CLIP, normalize embeddings, and build the FAISS index."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.config import Settings, get_settings
from backend.index import SimilarityMetric, build_index, save_index, save_manifest

logger = logging.getLogger(__name__)


def pick_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_image_files(root: Path, extensions: frozenset[str]) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"images directory does not exist: {root}")
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in extensions:
            out.append(p)
    return out


def load_clip(
    model_id: str,
    device: torch.device,
) -> tuple[CLIPModel, CLIPProcessor]:
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    return model, processor


def _batched(items: list[Path], batch_size: int) -> Iterator[list[Path]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def encode_images(
    paths: list[Path],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[Path]]:
    """Encode images in batches. Skips unreadable files and logs warnings."""
    if not paths:
        return np.zeros((0, model.config.projection_dim), dtype=np.float32), []

    valid_paths: list[Path] = []
    all_vecs: list[np.ndarray] = []

    dim = model.config.projection_dim

    for batch_paths in _batched(paths, batch_size):
        images: list[Image.Image] = []
        kept: list[Path] = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                kept.append(p)
            except (OSError, ValueError) as e:
                logger.warning("Skipping invalid image %s: %s", p, e)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        all_vecs.append(feats.cpu().numpy().astype(np.float32))
        valid_paths.extend(kept)

    if not all_vecs:
        return np.zeros((0, dim), dtype=np.float32), []

    emb = np.vstack(all_vecs)
    return emb, valid_paths


def encode_texts(
    texts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> np.ndarray:
    if not texts:
        raise ValueError("texts list is empty")
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        t = model.get_text_features(**inputs)
        t = t / t.norm(dim=-1, keepdim=True)
    return t.cpu().numpy().astype(np.float32)


def load_tags_file(images_dir: Path) -> dict[str, list[str]] | None:
    """Optional ``tags.json`` next to images: { \"relative/path.jpg\": [\"tag\", ...] }."""
    tags_path = images_dir / "tags.json"
    if not tags_path.is_file():
        return None
    import json

    raw = json.loads(tags_path.read_text(encoding="utf-8"))
    return {str(k): [str(t) for t in v] for k, v in raw.items()}


def build_index_from_folder(
    settings: Settings,
    metric: SimilarityMetric = "cosine",
) -> tuple[int, Path, Path]:
    """
    Scan ``settings.images_dir``, compute embeddings, write FAISS index + manifest.

    Returns (num_indexed, index_path, manifest_path).
    """
    device = pick_device(settings.device)
    logger.info("Using device: %s", device)

    paths = list_image_files(settings.images_dir, settings.image_extensions)
    if not paths:
        raise RuntimeError(
            f"No images found under {settings.images_dir}. "
            f"Supported extensions: {sorted(settings.image_extensions)}"
        )

    model, processor = load_clip(settings.clip_model_id, device)
    embeddings, kept_paths = encode_images(
        paths,
        model,
        processor,
        device,
        settings.batch_size,
    )

    if embeddings.shape[0] == 0:
        raise RuntimeError("No valid images could be encoded.")

    tag_map = load_tags_file(settings.images_dir)
    tags_per_index: list[list[str]] = []
    images_root = settings.images_dir.resolve()
    for p in kept_paths:
        try:
            rel = str(p.resolve().relative_to(images_root))
        except ValueError:
            rel = p.name
        if tag_map and rel in tag_map:
            tags_per_index.append(tag_map[rel])
        else:
            tags_per_index.append([])

    # For cosine, vectors are already normalized in encode_images; build_index normalizes again for FAISS add — safe idempotent for unit vectors
    index = build_index(embeddings, metric=metric)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, settings.index_path)

    str_paths = [str(p.resolve()) for p in kept_paths]
    save_manifest(str_paths, tags_per_index, settings.manifest_path, extra={"metric": metric, "model": settings.clip_model_id})

    return len(str_paths), settings.index_path, settings.manifest_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build CLIP + FAISS image index from a folder.")
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Folder of images (default: data/images under project root)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Where to write index.faiss and manifest.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu")
    parser.add_argument(
        "--metric",
        type=str,
        choices=("cosine", "l2"),
        default="cosine",
        help="FAISS metric (cosine uses inner product on normalized vectors)",
    )
    args = parser.parse_args()

    settings = get_settings(
        images_dir=args.images_dir,
        artifacts_dir=args.artifacts_dir,
        batch_size=args.batch_size,
        clip_model_id=args.model,
        device=args.device,
    )
    n, ip, mp = build_index_from_folder(settings, metric=args.metric)
    logger.info("Indexed %s images. Index: %s Manifest: %s", n, ip, mp)


if __name__ == "__main__":
    main()
