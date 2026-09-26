"""Small shared helpers: logging, seeding, JSON/IO, paths."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def get_logger(name: str = "repro") -> logging.Logger:
    """Return a process-wide logger that writes to stdout once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOGGER = get_logger()


def set_seed(seed: int = 42) -> None:
    """Seed python / numpy / torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch may be unavailable in a pure-aggregation context
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(obj: Any):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(rows: list[dict], path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def hf_login_from_env() -> None:
    """Authenticate with the Hugging Face Hub if a token is present in the env.

    Llama-3.1-8B-Instruct is gated, so a token (HF_TOKEN / HUGGING_FACE_HUB_TOKEN) is
    required unless you point --model-name at an ungated mirror.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        LOGGER.info("Logged in to the Hugging Face Hub via environment token.")
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("HF login failed (%s); continuing without explicit login.", exc)
