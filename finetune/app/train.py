"""Unsloth LoRA/QLoRA supervised fine-tuning of one Llama model on one dataset.

Pipeline: Unsloth FastLanguageModel -> TRL SFTTrainer (text field) -> train_on_responses_only
(loss on the assistant response only). The dev split, if present, drives eval_loss + best
checkpoint selection. SFTConfig kwargs are filtered to those the installed TRL accepts, so
the code survives TRL field-name churn.
"""

from __future__ import annotations

import inspect
import os

from .config import DatasetSpec, RunConfig
from .data_utils import build_text_dataset, maybe_subsample, read_split_df
from .modeling import build_model, is_bf16
from .utils import LOGGER, ensure_dir

# Llama-3.1 chat markers used to mask everything before the assistant turn.
_INSTRUCTION_PART = "<|start_header_id|>user<|end_header_id|>\n\n"
_RESPONSE_PART = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def train_model(cfg: RunConfig, spec: DatasetSpec):
    """Fine-tune and (optionally) save the LoRA adapter. Returns (model, tokenizer, info)."""
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    max_seq_len = cfg.max_seq_len or spec.max_seq_len

    # Long-context tasks: shrink per-device batch, raise grad-accum to keep eff. batch.
    train_bs = cfg.per_device_train_batch_size
    grad_accum = cfg.gradient_accumulation_steps
    if max_seq_len >= 2048 and train_bs > 2 and not cfg.no_auto_batch:
        grad_accum = grad_accum * (train_bs // 2)
        train_bs = 2
        LOGGER.info("Long context (seq=%d): train_bs=%d, grad_accum=%d.",
                    max_seq_len, train_bs, grad_accum)

    # ---- data ----
    train_df = read_split_df(cfg.data_root, cfg.train_split, spec.stem)
    train_df = maybe_subsample(train_df, cfg.max_train_samples, cfg.seed)

    # ---- model (Unsloth) ----
    model, tokenizer = build_model(cfg, max_seq_len)
    train_ds = build_text_dataset(train_df, tokenizer)

    # ---- optional dev set for eval_loss + best-checkpoint selection ----
    val_ds = None
    if cfg.eval_during_train:
        try:
            val_df = read_split_df(cfg.data_root, cfg.val_split, spec.stem)
            val_df = maybe_subsample(val_df, cfg.max_val_samples, cfg.seed)
            val_ds = build_text_dataset(val_df, tokenizer)
        except FileNotFoundError:
            LOGGER.warning("No '%s' split for %s; training without validation selection.",
                           cfg.val_split, spec.stem)

    use_bf16 = is_bf16()
    run_dir = ensure_dir(os.path.join(cfg.output_root, spec.stem))

    sft_params = inspect.signature(SFTConfig.__init__).parameters
    seq_key = "max_seq_length" if "max_seq_length" in sft_params else (
        "max_length" if "max_length" in sft_params else None)
    eval_key = "eval_strategy" if "eval_strategy" in sft_params else "evaluation_strategy"

    kw = dict(
        output_dir=os.path.join(run_dir, "trainer"),
        dataset_text_field="text",
        dataset_num_proc=4,
        packing=False,                       # required for train_on_responses_only
        group_by_length=True,                # batch similar lengths -> less padding, same quality
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=train_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        optim="adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=cfg.seed,
        report_to="none",
    )
    if seq_key:
        kw[seq_key] = max_seq_len
    if val_ds is not None:
        kw[eval_key] = "epoch"
        kw["save_strategy"] = "epoch"        # must match eval strategy
        kw["save_total_limit"] = 1
        kw["load_best_model_at_end"] = True
        kw["metric_for_best_model"] = "eval_loss"
        kw["greater_is_better"] = False
    else:
        kw["save_strategy"] = "no"
    # Drop any kwarg the installed SFTConfig doesn't accept (TRL version drift).
    kw = {k: v for k, v in kw.items() if k in sft_params}

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(**kw),
    )
    # Mask the prompt: compute loss only on the assistant response.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=_INSTRUCTION_PART,
        response_part=_RESPONSE_PART,
    )

    LOGGER.info("Starting training: %s (epochs=%.1f, max_steps=%d, bs=%d x ga=%d, bf16=%s).",
                spec.stem, cfg.epochs, cfg.max_steps, train_bs, grad_accum, use_bf16)
    train_out = trainer.train()

    train_info = dict(train_out.metrics)
    train_info["num_train_examples"] = len(train_ds)
    if val_ds is not None:
        train_info["num_val_examples"] = len(val_ds)
        try:
            train_info["best_eval_loss"] = (
                float(trainer.state.best_metric)
                if trainer.state.best_metric is not None else None
            )
        except Exception:
            pass

    adapter_dir = None
    if cfg.save_adapter:
        adapter_dir = ensure_dir(os.path.join(run_dir, "adapter"))
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        LOGGER.info("Saved LoRA adapter to %s", adapter_dir)
    train_info["adapter_dir"] = adapter_dir

    if cfg.merge_and_save and not cfg.use_4bit:
        merged_dir = ensure_dir(os.path.join(run_dir, "merged"))
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        LOGGER.info("Saved merged 16-bit model to %s", merged_dir)
        train_info["merged_dir"] = merged_dir
    elif cfg.merge_and_save and cfg.use_4bit:
        LOGGER.warning("merge_and_save skipped: not supported with 4-bit (use --no-4bit).")

    tokenizer.padding_side = "left"  # for the subsequent generation/eval phase
    return model, tokenizer, train_info
