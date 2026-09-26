"""Model + tokenizer loading via Unsloth (FastLanguageModel) with LoRA / QLoRA.

Unsloth MUST be imported before transformers/trl/peft so its kernels patch correctly;
we therefore import it at module top (this module is the first to touch the HF stack).

- use_4bit=True  -> QLoRA (4-bit), fits L4.
- use_4bit=False -> LoRA in bf16, recommended on A100.
"""

from __future__ import annotations

# Import Unsloth FIRST (before transformers gets imported anywhere else).
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402,F401

import torch as _torch

# TF32 matmuls: large speedup on Ampere/A100 with negligible quality impact.
try:
    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

from .config import LORA_TARGET_MODULES, DatasetSpec, RunConfig
from .utils import LOGGER


def is_bf16() -> bool:
    return bool(is_bfloat16_supported())


def _apply_chat_template(tokenizer):
    from unsloth.chat_templates import get_chat_template

    return get_chat_template(tokenizer, chat_template="llama-3.1")


def build_model(cfg: RunConfig, max_seq_length: int):
    """Load the base model with Unsloth and attach LoRA adapters. Returns (model, tokenizer)."""
    LOGGER.info(
        "Loading %s via Unsloth (max_seq_len=%d, load_in_4bit=%s) ...",
        cfg.model_name, max_seq_length, cfg.use_4bit,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=max_seq_length,
        dtype=None,                 # auto: bf16 on Ampere+/Ada, else fp16
        load_in_4bit=cfg.use_4bit,  # False -> bf16 LoRA (A100)
    )
    tokenizer = _apply_chat_template(tokenizer)

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,         # 0 is Unsloth-optimized
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's long-context checkpointing
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )
    _log_trainable(model)
    return model, tokenizer


def load_for_inference(cfg: RunConfig, spec: DatasetSpec, adapter_dir: str | None):
    """Load a trained LoRA adapter (or the base model) for generation only."""
    max_seq_length = cfg.max_seq_len or spec.max_seq_len
    LOGGER.info("Loading for inference: %s", adapter_dir or cfg.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir or cfg.model_name,   # Unsloth loads a saved LoRA dir directly
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=cfg.use_4bit,
    )
    tokenizer = _apply_chat_template(tokenizer)
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    return model, tokenizer


def _log_trainable(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("Trainable params: %s / %s (%.4f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1))
