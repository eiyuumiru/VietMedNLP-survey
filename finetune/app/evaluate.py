"""Run inference on an eval split and compute task-appropriate metrics."""

from __future__ import annotations

import gc

import torch

from .config import DatasetSpec, RunConfig
from .data_utils import build_eval_examples
from .infer import generate_predictions
from .metrics import compute_metrics
from .utils import LOGGER


def _prepare_for_generation(model) -> None:
    """Switch the model to Unsloth fast inference (2x) with KV cache enabled.

    Training leaves the model in checkpointing/no-cache mode; this flips it back.
    """
    try:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    except Exception:
        # fallback for non-Unsloth models
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        try:
            model.config.use_cache = True
        except Exception:
            pass
    model.eval()
    # Release the training allocator cache so generation starts from a clean baseline.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_split(model, tokenizer, df, spec: DatasetSpec, cfg: RunConfig):
    """Generate predictions for `df` and score them.

    Returns (metrics: dict, predictions: list[{gold, pred}]).
    """
    _prepare_for_generation(model)
    tokenizer.padding_side = "left"
    examples = build_eval_examples(df, tokenizer)
    prompts = [e["prompt"] for e in examples]
    golds = [e["gold"] for e in examples]

    max_new_tokens = cfg.max_new_tokens or spec.max_new_tokens
    max_seq_len = cfg.max_seq_len or spec.max_seq_len
    max_prompt_len = max(64, max_seq_len - max_new_tokens)

    # Long-context tasks need a smaller eval batch to avoid OOM during generation
    # (skipped on big GPUs via --no-auto-batch).
    eval_bs = cfg.per_device_eval_batch_size
    if max_seq_len >= 2048 and not cfg.no_auto_batch:
        eval_bs = min(eval_bs, 4)

    LOGGER.info(
        "Evaluating %s on '%s' split: %d examples (max_new_tokens=%d, eval_bs=%d).",
        spec.stem, cfg.eval_split, len(prompts), max_new_tokens, eval_bs,
    )
    preds = generate_predictions(
        model, tokenizer, prompts,
        batch_size=eval_bs,
        max_new_tokens=max_new_tokens,
        max_prompt_len=max_prompt_len,
    )

    metrics = compute_metrics(spec.task_type, preds, golds, use_bertscore=cfg.use_bertscore)
    metrics["primary_metric"] = spec.primary_metric
    metrics["primary_score"] = float(metrics.get(spec.primary_metric)) \
        if metrics.get(spec.primary_metric) is not None else None
    metrics["num_eval"] = len(golds)

    predictions = [{"gold": g, "pred": p} for g, p in zip(golds, preds)]
    LOGGER.info(
        "Done. Primary metric %s = %s",
        spec.primary_metric,
        f"{metrics['primary_score']:.4f}" if metrics["primary_score"] is not None else "n/a",
    )
    return metrics, predictions
