"""Data loading and prompt formatting for the `nl` (natural-language) prompt style.

Each task CSV has 10 columns; we only use:
  - instruction_prompt_nl : the full natural-language instruction (ends with "Output:")
  - output_prompt_nl      : the target string the model must produce

Training examples are built with the model's chat template and completion-only label
masking (loss is computed on the assistant response only). Long examples are truncated
from the LEFT so the (short) target at the end is always preserved.
"""

from __future__ import annotations

import os

import pandas as pd

from .config import (
    CHAT_DATE_STRING,
    INSTRUCTION_COLUMN,
    SYSTEM_PROMPT,
    TARGET_COLUMN,
)
from .utils import LOGGER


# --------------------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------------------
def split_csv_path(data_root: str, split: str, stem: str) -> str:
    """Path to {data_root}/{split}/{stem}.csv in the local task data layout."""
    return os.path.join(data_root, split, f"{stem}.csv")


def read_split_df(data_root: str, split: str, stem: str) -> pd.DataFrame:
    """Read one split CSV and keep only valid (non-empty) instruction/target rows."""
    path = split_csv_path(data_root, split, stem)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find split file: {path}\n"
            f"Expected layout: <data_root>/{split}/{stem}.csv. "
            f"Point --data-root at the 'Instruct_Datasets' (or 'Instruct_Datasets_EN') folder."
        )
    # keep_default_na=False so the literal string "None" is NOT turned into NaN: for NER
    # tasks "None" is a valid target meaning "no entities" (a useful negative example).
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in (INSTRUCTION_COLUMN, TARGET_COLUMN):
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' missing in {path}. Found columns: {list(df.columns)}"
            )
    before = len(df)
    df = df[[INSTRUCTION_COLUMN, TARGET_COLUMN]].copy()
    df[INSTRUCTION_COLUMN] = df[INSTRUCTION_COLUMN].str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip()
    # Drop only genuinely empty cells (keep "None"/label-like targets).
    df = df[(df[INSTRUCTION_COLUMN] != "") & (df[TARGET_COLUMN] != "")]
    df = df.reset_index(drop=True)
    LOGGER.info("Loaded %s split '%s': %d rows (%d dropped as empty).",
                stem, split, len(df), before - len(df))
    return df


def maybe_subsample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Chat formatting
# --------------------------------------------------------------------------------------
def build_messages(instruction: str, include_response: str | None = None) -> list[dict]:
    """Build a chat `messages` list. If include_response is given, append the assistant turn."""
    messages: list[dict] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": instruction})
    if include_response is not None:
        messages.append({"role": "assistant", "content": include_response})
    return messages


def render_prompt(tokenizer, instruction: str) -> str:
    """Templated prompt text ending right before the assistant turn (for generation)."""
    return tokenizer.apply_chat_template(
        build_messages(instruction),
        tokenize=False,
        add_generation_prompt=True,
        date_string=CHAT_DATE_STRING,
    )


def build_text_dataset(df: pd.DataFrame, tokenizer):
    """Return a datasets.Dataset with a single "text" column for TRL SFTTrainer.

    Each row is the full chat-templated conversation (system + user + assistant). Prompt
    tokens are masked later by Unsloth's `train_on_responses_only`, so loss is on the
    assistant response only. SFTTrainer handles tokenization/truncation to max_seq_length.
    """
    from datasets import Dataset

    texts = [
        tokenizer.apply_chat_template(
            build_messages(row[INSTRUCTION_COLUMN], include_response=row[TARGET_COLUMN]),
            tokenize=False,
            add_generation_prompt=False,
            date_string=CHAT_DATE_STRING,
        )
        for _, row in df.iterrows()
    ]
    ds = Dataset.from_list([{"text": t} for t in texts])
    LOGGER.info("Built text dataset: %d examples.", len(ds))
    return ds


def build_eval_examples(df: pd.DataFrame, tokenizer) -> list[dict]:
    """Return a list of {prompt, gold} dicts; tokenization/batching happens in infer.py."""
    examples = []
    for _, row in df.iterrows():
        examples.append(
            {
                "prompt": render_prompt(tokenizer, row[INSTRUCTION_COLUMN]),
                "gold": row[TARGET_COLUMN],
            }
        )
    return examples
