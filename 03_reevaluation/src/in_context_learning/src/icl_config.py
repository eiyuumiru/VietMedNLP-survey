"""Shared, provider-neutral settings for the ICL experiment bundle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelProfile:
    """Paper-facing model label and request settings."""

    label: str
    model_env: str
    decoding: dict[str, object]


MODEL_PROFILES = {
    "gpt-4o": ModelProfile(
        label="GPT-4o",
        model_env="ICL_MODEL_GPT_4O",
        decoding={"temperature": 0},
    ),
    "gpt-4.1": ModelProfile(
        label="GPT-4.1",
        model_env="ICL_MODEL_GPT_4_1",
        decoding={"temperature": 0},
    ),
    "gpt-5.2": ModelProfile(
        label="GPT-5.2",
        model_env="ICL_MODEL_GPT_5_2",
        decoding={"reasoning_effort": "none"},
    ),
    "gpt-5.4": ModelProfile(
        label="GPT-5.4",
        model_env="ICL_MODEL_GPT_5_4",
        decoding={"reasoning_effort": "none"},
    ),
}
MODELS = tuple(MODEL_PROFILES)
MODEL_LABELS = {key: profile.label for key, profile in MODEL_PROFILES.items()}
BATCH_METHODS = ("direct", "cot")


def load_project_env(root: Path) -> None:
    """Load the method-local .env without overriding process environment."""
    load_dotenv(Path(root) / ".env", override=False)


def resolve_data_root(value: str | Path | None = None) -> Path:
    """Resolve the local ICL data directory from CLI input or environment."""
    load_project_env(Path(__file__).resolve().parents[1])
    configured = value or os.getenv("ICL_DATA_ROOT", "").strip()
    if not configured:
        raise RuntimeError("Pass --data-root or set ICL_DATA_ROOT in .env.")
    root = Path(configured).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"ICL data root is not a directory: {root}")
    return root.resolve()


def dataset_files(root: str | Path, dataset: str) -> dict[str, Path]:
    """Resolve and validate the train/test CSV paths for a task."""
    root = Path(root)
    paths = {
        split: root / split / f"{dataset}.csv"
        for split in ("train", "test")
    }
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {split} CSV for {dataset}: expected {path}"
            )
    return paths


def model_id_for_family(family: str, *, required: bool = True) -> str | None:
    """Resolve a paper-facing profile to the identifier expected by the API."""
    try:
        profile = MODEL_PROFILES[family]
    except KeyError as exc:
        raise ValueError(f"Unknown model family: {family}") from exc
    load_project_env(Path(__file__).resolve().parents[1])
    model_id = os.getenv(profile.model_env, "").strip()
    if not model_id and required:
        raise RuntimeError(f"Set {profile.model_env} in in_context_learning/.env")
    return model_id or None


def single_model_id(*, required: bool = True) -> str | None:
    """Resolve the generic single-run model identifier from local settings."""
    load_project_env(Path(__file__).resolve().parents[1])
    model_id = os.getenv("ICL_MODEL", "").strip()
    if not model_id and required:
        raise RuntimeError("Pass --model, --model-family, or set ICL_MODEL in .env.")
    return model_id or None


def api_settings() -> tuple[str, str, str]:
    """Return normalized provider settings without exposing credentials."""
    load_project_env(Path(__file__).resolve().parents[1])
    return (
        os.getenv("ICL_API_PROVIDER", "").strip().lower()
        or "openai-compatible",
        os.getenv("ICL_API_BASE_URL", "").strip().rstrip("/"),
        os.getenv("ICL_API_VERSION", "").strip(),
    )
