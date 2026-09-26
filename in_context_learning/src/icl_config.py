"""Environment-backed settings shared by the ICL runners and audit."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


MODELS = ("gpt-4o", "gpt-4.1", "gpt-5.2", "gpt-5.4")
MODEL_LABELS = {
    "gpt-4o": "GPT-4o",
    "gpt-4.1": "GPT-4.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.4": "GPT-5.4",
}
MODEL_ENV_VARS = {
    "gpt-4o": "ICL_DEPLOYMENT_GPT_4O",
    "gpt-4.1": "ICL_DEPLOYMENT_GPT_4_1",
    "gpt-5.2": "ICL_DEPLOYMENT_GPT_5_2",
    "gpt-5.4": "ICL_DEPLOYMENT_GPT_5_4",
}


def load_project_env(root: Path) -> None:
    """Load the method-local .env without overriding explicitly set variables."""
    load_dotenv(Path(root) / ".env", override=False)


def dataset_folder_url(root: Path) -> str:
    """Return the authorized Drive folder URL or build one from its folder ID."""
    load_project_env(root)
    value = os.getenv("ICL_DATASET_FOLDER_ID", "").strip()
    if not value:
        raise RuntimeError(
            "Set ICL_DATASET_FOLDER_ID in in_context_learning/.env "
            "or in the process environment."
        )
    if value.startswith(("https://", "http://")):
        return value
    return f"https://drive.google.com/drive/folders/{value}"


def deployment_for_model(model: str, *, required: bool = True) -> str | None:
    """Resolve a paper-facing model family to a private API deployment name."""
    if model not in MODEL_ENV_VARS:
        raise ValueError(f"Unknown model family: {model}")
    load_project_env(Path(__file__).resolve().parents[1])
    env_name = MODEL_ENV_VARS[model]
    deployment = os.getenv(env_name, "").strip()
    if not deployment and required:
        raise RuntimeError(f"Set {env_name} in in_context_learning/.env")
    return deployment or None

