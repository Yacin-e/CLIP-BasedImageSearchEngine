"""Text and image query APIs over a persisted FAISS index."""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

from backend.config import Settings, get_settings
from backend.embed import encode_images, encode_texts, load_clip, pick_device
from backend.index import SimilarityMetric, load_index, load_manifest, query_index

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Loads CLIP + FAISS + manifest and exposes ``search_by_text`` / ``search_by_image``.

    Optional **tag filter**: after retrieval, keeps only images whose tag set
    contains all required tags (case-insensitive substring match on tag names).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._device = pick_device(self.settings.device)
        self._model = None
        self._processor = None
        self._index = None
        self._paths: list[str] = []
        self._tags: list[list[str]] = []
        self._indexed_metric: SimilarityMetric = "cosine"

    def load(self) -> None:
        """Load CLIP weights, FAISS index, and manifest from disk."""
        index_path = self.settings.index_path
        manifest_path = self.settings.manifest_path

        self._index = load_index(index_path)
        self._paths, self._tags = load_manifest(manifest_path)

        import json

        extra = {}
        try:
            data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            extra = data.get("extra") or {}
        except (OSError, ValueError):
            pass
        self._indexed_metric = extra.get("metric", "cosine")

        self._model, self._processor = load_clip(self.settings.clip_model_id, self._device)
        logger.info(
            "SearchEngine ready: %s vectors, device=%s, index_metric=%s",
            len(self._paths),
            self._device,
            self._indexed_metric,
        )

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._model is not None and len(self._paths) > 0

    @property
    def num_indexed(self) -> int:
        return len(self._paths)

    def _ensure_query_metric(self, metric: SimilarityMetric | None) -> SimilarityMetric:
        m = metric or self._indexed_metric
        if m != self._indexed_metric:
            logger.warning(
                "Query metric %s differs from index metric %s; results may be inconsistent.",
                m,
                self._indexed_metric,
            )
        return m

    def _filter_by_tags(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
        required_tags: list[str] | None,
        k: int,
    ) -> tuple[list[str], list[float]]:
        if not required_tags:
            out_paths = [self._paths[int(i)] for i in indices if 0 <= int(i) < len(self._paths)]
            out_scores = [float(s) for s in scores]
            return out_paths[:k], out_scores[:k]

        req_lower = [t.strip().lower() for t in required_tags if t.strip()]
        if not req_lower:
            out_paths = [self._paths[int(i)] for i in indices if 0 <= int(i) < len(self._paths)]
            out_scores = [float(s) for s in scores]
            return out_paths[:k], out_scores[:k]

        picked_paths: list[str] = []
        picked_scores: list[float] = []

        for idx, sc in zip(indices, scores):
            ii = int(idx)
            if ii < 0 or ii >= len(self._paths):
                continue
            tags_lower = [t.lower() for t in self._tags[ii]]
            if all(any(r in t for t in tags_lower) for r in req_lower):
                picked_paths.append(self._paths[ii])
                picked_scores.append(float(sc))
                if len(picked_paths) >= k:
                    break

        # If too few after filter, optionally search deeper (simple expansion)
        if len(picked_paths) < k:
            # Re-query with larger k not implemented here; caller can increase k
            logger.debug(
                "Tag filter returned only %s results (requested %s). Try increasing k.",
                len(picked_paths),
                k,
            )

        return picked_paths, picked_scores

    def search_by_text(
        self,
        query: str,
        k: int = 10,
        *,
        metric: SimilarityMetric | None = None,
        required_tags: list[str] | None = None,
        retrieval_factor: int = 8,
    ) -> tuple[list[str], list[float]]:
        """
        Embed ``query`` with CLIP text tower and return top-k similar image paths.

        ``required_tags``: if set, filters results to images whose tags match
        (each required string must appear as substring in some tag, case-insensitive).
        """
        q = (query or "").strip()
        if not q:
            raise ValueError("query must be a non-empty string")

        if self._model is None or self._index is None:
            raise RuntimeError("SearchEngine not loaded. Call load() first.")

        m = self._ensure_query_metric(metric)
        k_eff = min(k * max(retrieval_factor, 1), max(self._index.ntotal, 1))

        text_emb = encode_texts([q], self._model, self._processor, self._device)[0]
        scores, indices = query_index(self._index, text_emb, k_eff, m)

        paths, out_scores = self._filter_by_tags(indices, scores, required_tags, k)
        return paths[:k], out_scores[:k]

    def search_by_image(
        self,
        image_path: str,
        k: int = 10,
        *,
        metric: SimilarityMetric | None = None,
        required_tags: list[str] | None = None,
        retrieval_factor: int = 8,
    ) -> tuple[list[str], list[float]]:
        """Embed a query image and return top-k similar images (excluding exact duplicate path optional)."""
        p = Path(image_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"image not found: {p}")

        if self._model is None or self._index is None:
            raise RuntimeError("SearchEngine not loaded. Call load() first.")

        m = self._ensure_query_metric(metric)
        k_eff = min(k * max(retrieval_factor, 1), max(self._index.ntotal, 1))

        emb, kept = encode_images(
            [p],
            self._model,
            self._processor,
            self._device,
            batch_size=1,
        )
        if emb.shape[0] == 0 or not kept:
            raise ValueError(f"Could not read or encode image: {p}")

        scores, indices = query_index(self._index, emb[0], k_eff, m)

        # Optionally exclude self if present in index
        qp = str(p)
        paths, out_scores = self._filter_by_tags(indices, scores, required_tags, k + 1)
        filtered: list[tuple[str, float]] = []
        for path, s in zip(paths, out_scores):
            if path == qp and len(paths) > 1:
                continue
            filtered.append((path, s))
            if len(filtered) >= k:
                break
        fp, fs = zip(*filtered) if filtered else ([], [])
        return list(fp), list(fs)


def create_engine(settings: Settings | None = None) -> SearchEngine:
    eng = SearchEngine(settings)
    eng.load()
    return eng
