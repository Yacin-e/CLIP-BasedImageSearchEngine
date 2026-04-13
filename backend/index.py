"""FAISS index creation, persistence, and nearest-neighbor search."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np

logger = logging.getLogger(__name__)

SimilarityMetric = Literal["cosine", "l2"]


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype(np.float32)
    return x


def build_index(
    embeddings: np.ndarray,
    metric: SimilarityMetric = "cosine",
) -> faiss.Index:
    """
    Build a FAISS index from a 2D embedding matrix.

    For ``cosine`` similarity, vectors should be L2-normalized; retrieval uses
    inner product (equivalent to cosine similarity for unit vectors).
    For ``l2``, the index uses squared L2 distance (smaller is more similar).
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array of shape (n, dim)")
    n, dim = embeddings.shape
    if n == 0:
        raise ValueError("cannot build index with zero vectors")

    x = _ensure_float32(np.ascontiguousarray(embeddings))

    if metric == "cosine":
        faiss.normalize_L2(x)
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError(f"unsupported metric: {metric}")

    index.add(x)
    logger.info("Built FAISS index: n=%s dim=%s metric=%s", n, dim, metric)
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s", path)


def load_index(path: str | Path) -> faiss.Index:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    idx = faiss.read_index(str(path))
    logger.info("Loaded FAISS index from %s (ntotal=%s)", path, idx.ntotal)
    return idx


def save_manifest(
    paths: list[str],
    tags: list[list[str]] | None,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist ordered image paths and optional per-image tags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": 1,
        "paths": paths,
        "tags": tags if tags is not None else [[] for _ in paths],
    }
    if extra:
        payload["extra"] = extra
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote manifest with %s entries to %s", len(paths), path)


def load_manifest(path: str | Path) -> tuple[list[str], list[list[str]]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    paths = data["paths"]
    tags = data.get("tags") or [[] for _ in paths]
    if len(tags) != len(paths):
        raise ValueError("manifest paths and tags length mismatch")
    return paths, tags


def query_index(
    index: faiss.Index,
    query_embedding: np.ndarray,
    k: int,
    metric: SimilarityMetric,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (scores, indices) for top-k neighbors.

    For cosine/IP, higher score is more similar.
    For L2, lower distance is more similar (FAISS returns distances).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    q = _ensure_float32(query_embedding.reshape(1, -1).astype(np.float32))
    if metric == "cosine":
        faiss.normalize_L2(q)
    ntotal = index.ntotal
    if ntotal == 0:
        raise ValueError("index is empty")
    k_eff = min(k, ntotal)
    scores, indices = index.search(q, k_eff)
    return scores[0], indices[0]
