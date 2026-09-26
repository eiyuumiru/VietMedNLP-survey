"""Minimal client for the Bosch LLM FARM Azure OpenAI-compatible endpoint."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def send_prompt(prompt: str) -> str:
    """Send one user prompt and return the model's text response."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    endpoint = _required_env("LLM_FARM_ENDPOINT").rstrip("/")
    api_key = os.getenv("GENAIPLATFORM_FARM_SUBSCRIPTION_KEY") or os.getenv(
        "LLM_FARM_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "Missing environment variable: "
            "GENAIPLATFORM_FARM_SUBSCRIPTION_KEY or LLM_FARM_API_KEY"
        )
    api_version = _required_env("LLM_FARM_CHAT_API_VERSION")
    deployment = _required_env("LLM_FARM_CHAT_DEPLOYMENT")

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        timeout=120,
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Response contains no text content")
    return content


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
    try:
        print(send_prompt(prompt))
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
