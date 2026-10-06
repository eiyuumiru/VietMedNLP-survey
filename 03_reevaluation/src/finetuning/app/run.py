"""CLI entry point: fine-tune + evaluate the model on ONE dataset.

Typical usage (run once per dataset):

    python -m app.run \
        --dataset ViMedNLI_ViMedNLI \
        --data-root /path/to/instruction-data \
        --output-root /path/to/results

Outputs under <output-root>/<dataset>/:
    metrics.json       - all metrics + paper SOTA reference + train info
    predictions.jsonl  - gold/pred pairs on the eval split
    run_config.json    - the exact configuration used
    adapter/           - the trained LoRA adapter
"""

from __future__ import annotations

import argparse
import os
import time

from .config import DEFAULT_MODEL, RunConfig, get_spec, list_datasets
from .utils import LOGGER, hf_login_from_env, save_json, save_jsonl, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune + evaluate Llama on one dataset.")
    p.add_argument("--dataset", help="Dataset file stem, e.g. ViMedNLI_ViMedNLI. "
                                     "Use --list-datasets to see all.")
    p.add_argument("--data-root", help="Folder containing train/ dev/ test/ subdirs "
                                       "(instruction-formatted data; see DATA_FORMAT.md).")
    p.add_argument("--output-root", default="results", help="Where to write outputs.")
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--list-datasets", action="store_true", help="Print dataset stems and exit.")

    # splits / sizes
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--val-split", default="dev")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=1000)
    p.add_argument("--no-eval-during-train", action="store_true",
                   help="Disable dev-based eval_loss + best-checkpoint selection.")

    # quantization / LoRA
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading (bf16 LoRA, as in the reported runs).")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # optimization
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--no-auto-batch", action="store_true",
                   help="Don't shrink batch for seq-2048 tasks (use on big GPUs, e.g. 80GB).")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--logging-steps", type=int, default=20)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-bf16", action="store_true", help="Use fp16 compute instead of bf16.")
    p.add_argument("--use-bertscore", action="store_true")
    p.add_argument("--no-save-adapter", action="store_true")
    p.add_argument("--merge-and-save", action="store_true",
                   help="Merge LoRA into the base model and save (requires --no-4bit).")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load --adapter-dir and only evaluate.")
    p.add_argument("--adapter-dir", default=None, help="Adapter to load when --eval-only.")
    return p


def cfg_from_args(args) -> RunConfig:
    return RunConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        model_name=args.model_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        val_split=args.val_split,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_val_samples=args.max_val_samples,
        eval_during_train=not args.no_eval_during_train,
        use_4bit=not args.no_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.eval_batch_size,
        no_auto_batch=args.no_auto_batch,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        logging_steps=args.logging_steps,
        save_adapter=not args.no_save_adapter,
        seed=args.seed,
        bf16=not args.no_bf16,
        use_bertscore=args.use_bertscore,
        merge_and_save=args.merge_and_save,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.list_datasets:
        print("\n".join(list_datasets()))
        return
    if not args.dataset or not args.data_root:
        raise SystemExit("Both --dataset and --data-root are required (or use --list-datasets).")

    cfg = cfg_from_args(args)
    spec = get_spec(cfg.dataset)
    set_seed(cfg.seed)
    hf_login_from_env()

    run_dir = os.path.join(cfg.output_root, spec.stem)
    os.makedirs(run_dir, exist_ok=True)
    save_json(cfg, os.path.join(run_dir, "run_config.json"))

    t0 = time.time()

    # ---- train (or load) ----
    if args.eval_only:
        if not args.adapter_dir or not os.path.isdir(args.adapter_dir):
            raise SystemExit("--eval-only requires an existing --adapter-dir")
        from .modeling import load_for_inference

        LOGGER.info("Eval-only: loading adapter %s", args.adapter_dir)
        model, tokenizer = load_for_inference(cfg, spec, args.adapter_dir)
        train_info = {"eval_only": True, "adapter_dir": args.adapter_dir}
    else:
        from .train import train_model

        model, tokenizer, train_info = train_model(cfg, spec)

    # ---- evaluate ----
    from .data_utils import maybe_subsample, read_split_df
    from .evaluate import evaluate_split

    eval_df = read_split_df(cfg.data_root, cfg.eval_split, spec.stem)
    eval_df = maybe_subsample(eval_df, cfg.max_eval_samples, cfg.seed)
    metrics, predictions = evaluate_split(model, tokenizer, eval_df, spec, cfg)

    # ---- persist ----
    result = {
        "dataset": spec.stem,
        "display": spec.display,
        "task_type": spec.task_type,
        "model_name": cfg.model_name,
        "prompt_style": "nl",
        "eval_split": cfg.eval_split,
        "metrics": metrics,
        "paper_sota": spec.paper_sota,
        "train_info": train_info,
        "wall_time_sec": round(time.time() - t0, 1),
    }
    save_json(result, os.path.join(run_dir, "metrics.json"))
    save_jsonl(predictions, os.path.join(run_dir, "predictions.jsonl"))

    LOGGER.info("=" * 70)
    LOGGER.info("FINISHED %s | %s = %s | saved to %s",
                spec.stem, metrics.get("primary_metric"),
                f"{metrics.get('primary_score'):.4f}" if metrics.get("primary_score") is not None else "n/a",
                run_dir)
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
