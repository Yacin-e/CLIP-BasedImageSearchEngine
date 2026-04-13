"""Central configuration for paths, model, and runtime parameters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal


@dataclass
class Settings:
    """Application settings with sensible defaults."""

    # Repository root (parent of backend/)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Image folder to index
    images_dir: Path | None = None

    # Where FAISS index and manifest are stored
    artifacts_dir: Path | None = None

    # Hugging Face CLIP checkpoint
    clip_model_id: str = "openai/clip-vit-base-patch32"

    # Encoding
    batch_size: int = 32
    device: str | None = None  # "cuda", "mps", "cpu", or None for auto

    # Supported image extensions (lowercase, with dot)
    image_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
    )

    # Default similarity: cosine = inner product on L2-normalized vectors
    default_metric: Literal["cosine", "l2"] = "cosine"

    def __post_init__(self) -> None:
        if self.images_dir is None:
            self.images_dir = self.project_root / "data" / "images"
        if self.artifacts_dir is None:
            self.artifacts_dir = self.project_root / "artifacts"

    @property
    def index_path(self) -> Path:
        return self.artifacts_dir / "index.faiss"

    @property
    def manifest_path(self) -> Path:
        return self.artifacts_dir / "manifest.json"


def get_settings(
    images_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    clip_model_id: str | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    default_metric: Literal["cosine", "l2"] | None = None,
    **overrides: object,
) -> Settings:
    """Build `Settings`, optionally overriding fields."""
    base = Settings()
    updates: dict[str, object] = {}
    if images_dir is not None:
        updates["images_dir"] = Path(images_dir).expanduser().resolve()
    if artifacts_dir is not None:
        updates["artifacts_dir"] = Path(artifacts_dir).expanduser().resolve()
    if clip_model_id is not None:
        updates["clip_model_id"] = clip_model_id
    if batch_size is not None:
        updates["batch_size"] = batch_size
    if device is not None:
        updates["device"] = device
    if default_metric is not None:
        updates["default_metric"] = default_metric
    for key, value in overrides.items():
        if value is not None and hasattr(base, key):
            updates[key] = value
    return replace(base, **updates) if updates else base
