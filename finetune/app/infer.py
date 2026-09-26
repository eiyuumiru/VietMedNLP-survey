"""Batched, deterministic (greedy) text generation for evaluation."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from .utils import LOGGER


@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 16,
    max_new_tokens: int = 128,
    max_prompt_len: int = 1536,
) -> list[str]:
    """Generate one completion per prompt. Returns the decoded NEW text (response only).

    Prompts are left-padded and left-truncated so the instruction tail + generation
    cue ("Output:") are always kept.
    """
    assert tokenizer.padding_side == "left", "Use left padding for batched generation."
    tokenizer.truncation_side = "left"

    eos_ids = [tokenizer.eos_token_id]
    # Llama-3.x also stops on the end-of-turn token.
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0 and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)

    preds: list[str] = []
    device = next(model.parameters()).device
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate", leave=False):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=False,  # chat template already added them
        ).to(device)

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
        gen = out[:, enc["input_ids"].shape[1]:]  # slice off the (uniform) prompt length
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds.extend(d.strip() for d in decoded)

    LOGGER.info("Generated %d predictions.", len(preds))
    return preds
