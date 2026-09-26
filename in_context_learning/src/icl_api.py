"""Small adapter for common chat-completion API configurations."""

from __future__ import annotations

import os
from pathlib import Path

from openai import AzureOpenAI, OpenAI

from icl_config import api_settings, load_project_env, single_model_id


ROOT = Path(__file__).resolve().parents[1]


def make_client(model_id: str | None):
    """Create an OpenAI-compatible or Azure OpenAI client from local settings."""
    load_project_env(ROOT)
    api_key = os.getenv("ICL_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set ICL_API_KEY in in_context_learning/.env.")

    provider, base_url, api_version = api_settings()

    if provider == "openai-compatible":
        client = OpenAI(
            api_key=api_key,
            **({"base_url": base_url} if base_url else {}),
            timeout=120,
            max_retries=3,
        )
    elif provider == "azure-openai":
        if not base_url or not api_version:
            raise ValueError(
                "Azure OpenAI requires ICL_API_BASE_URL and ICL_API_VERSION."
            )
        model_id = model_id or single_model_id()
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version,
            azure_deployment=model_id,
            timeout=120,
            max_retries=3,
        )
    else:
        raise ValueError(
            "ICL_API_PROVIDER must be 'openai-compatible' or 'azure-openai'."
        )

    model_id = model_id or single_model_id()
    return client, model_id
