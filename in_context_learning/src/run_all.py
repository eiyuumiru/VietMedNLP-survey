"""Run all four models on all ten datasets: python src/run_all.py."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import gdown
import icl_inference as inference
from icl_config import MODELS, dataset_folder_url, deployment_for_model

from icl_inference import (
    PROTOCOL,
    ROOT,
    checkpoint_rows,
    object_hash,
    read_jsonl,
    recover_checkpoint_tail,
    run_experiment,
    write_json,
)
from icl_tasks import TASKS

PRIMARY_4O_MODEL = "gpt-4o"
METHODS = ("direct", "cot")
SEED = 42
LIMIT = None  # Set to 20 for a smoke run; use a different RESULTS directory.
RESULTS = ROOT / "results" / f"all_models_{PROTOCOL}"
MAX_PROMPT_CHARS = 60000
WORKERS_PER_MODEL = 4
MAX_TOTAL_WORKERS = 16

def manifest_experiment_id(manifest):
    return object_hash(
        {"config": manifest["config"], **manifest["fingerprints"]}
    )


def fallback_examples(fallback_output):
    """Reuse strict demonstrations when an old manifest is available."""
    if fallback_output is None:
        return None
    manifest_path = Path(fallback_output) / "experiment.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest["examples"], manifest["sampling"]


def merge_fallback_checkpoint(args):
    """Copy strict rows into the new checkpoint without touching strict files."""
    fallback_output = getattr(args, "fallback_output", None)
    if fallback_output is None:
        return

    fallback_output = Path(fallback_output)
    source_manifest_path = fallback_output / "experiment.json"
    source_predictions = fallback_output / "predictions.jsonl"
    if not source_manifest_path.exists() or not source_predictions.exists():
        return
    if source_predictions.stat().st_size == 0:
        return

    current_manifest_path = args.output / "experiment.json"
    if not current_manifest_path.exists():
        raise ValueError("Current experiment manifest is missing")
    current_manifest = json.loads(
        current_manifest_path.read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )

    compatible_keys = (
        "protocol",
        "dataset",
        "shots",
        "method",
        "seed",
        "max_prompt_chars",
        "train_id",
        "test_id",
        "task",
        "evaluator_version",
        "max_prompt_tokens",
        "token_encoding",
    )
    source_config = source_manifest["config"]
    current_config = current_manifest["config"]
    if any(
        source_config.get(key) != current_config.get(key)
        for key in compatible_keys
    ):
        raise ValueError("Strict checkpoint is incompatible with this run")
    if source_manifest["fingerprints"] != current_manifest["fingerprints"]:
        raise ValueError(
            "Strict checkpoint uses different few-shot examples or prompt"
        )

    predictions = args.output / "predictions.jsonl"
    recover_checkpoint_tail(predictions)
    current_id = manifest_experiment_id(current_manifest)
    source_id = manifest_experiment_id(source_manifest)
    current_rows = list(read_jsonl(predictions)) if predictions.exists() else []
    repaired_current = False
    for current_row in current_rows:
        content = {
            key: value
            for key, value in current_row.items()
            if key != "record_sha256"
        }
        if current_row.get("record_sha256") == object_hash(content):
            continue
        # Repair the malformed fallback records produced by the first version
        # of this merge code; reject every other checksum mismatch.
        if (
            current_row.get("source_model_family") == source_config["model_family"]
            and current_row.get("source_experiment_id")
        ):
            current_row["record_sha256"] = object_hash(content)
            repaired_current = True
            continue
        raise ValueError("Current checkpoint record checksum mismatch")
    if repaired_current:
        temporary = predictions.with_suffix(".repair.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in current_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(predictions)
    if current_rows:
        checkpoint_rows(predictions, current_id)
    source_rows = list(read_jsonl(source_predictions))
    checkpoint_rows(source_predictions, source_id)

    merged = {row["row"]: row for row in current_rows}
    for source_row in source_rows:
        row = dict(source_row)
        row["source_model_family"] = source_config["model_family"]
        row["source_experiment_id"] = source_row["experiment_id"]
        row["experiment_id"] = current_id
        row.pop("record_sha256", None)
        row["record_sha256"] = object_hash(row)
        merged[row["row"]] = row

    row_ids = sorted(merged)
    if row_ids != list(range(len(row_ids))):
        raise ValueError("Strict checkpoint rows are not a contiguous prefix")
    merged_rows = [merged[row_id] for row_id in row_ids]
    if merged_rows == current_rows and not repaired_current:
        return

    temporary = predictions.with_suffix(".merge.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in merged_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(predictions)
    print(
        f"Merged {len(source_rows)} strict rows into {args.output}; "
        "non-strict will process only the remaining rows.",
        flush=True,
    )


def completed_result(args):
    """Return a verified existing result without reopening its source CSV."""
    metrics_path = args.output / "metrics.json"
    predictions = args.output / "predictions.jsonl"
    if not metrics_path.exists() or not predictions.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not (
        metrics.get("completed_requested_rows")
        and metrics.get("n_scored")
    ):
        return None
    sources = {}
    for row in read_jsonl(predictions):
        source = row.get("source_model_family", args.model_family)
        sources[source] = sources.get(source, 0) + 1
    return {
        "dataset": args.dataset,
        "model": args.model_family,
        "method": args.method,
        "status": "completed",
        "metrics": metrics["metrics"],
        "n_scored": metrics["n_scored"],
        "n_invalid": metrics["n_invalid"],
        "diagnostics": metrics.get("diagnostics", {}),
        "cot_format": metrics.get("cot_format"),
        "n_truncated": metrics.get("n_truncated", 0),
        "row_sources": sources,
        "benchmark_comparability": metrics.get("benchmark_comparability"),
    }


def migrate_incomplete_checkpoint(args):
    """Keep a verified failed-run prefix when only inference code changed."""
    manifest_path = args.output / "experiment.json"
    predictions = args.output / "predictions.jsonl"
    if not manifest_path.exists() or not predictions.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_id = manifest_experiment_id(manifest)
    new_code = inference.source_hashes()
    if manifest["config"].get("code") == new_code:
        return
    checkpoint_rows(predictions, old_id)
    manifest["config"]["code"] = new_code
    new_id = manifest_experiment_id(manifest)
    temporary = predictions.with_suffix(".migrate.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in read_jsonl(predictions):
            row["experiment_id"] = new_id
            row["record_sha256"] = object_hash(
                {key: value for key, value in row.items() if key != "record_sha256"}
            )
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(predictions)
    write_json(manifest_path, manifest)
    print(f"Migrated {count} checkpoint rows in {args.output}.", flush=True)


def run_model(args, files, prepared):
    try:
        existing = completed_result(args)
        if existing is not None:
            return existing
        migrate_incomplete_checkpoint(args)
        # Create the method manifest before seeding the fallback checkpoint.
        run_experiment(
            SimpleNamespace(**{**vars(args), "prepare_only": True}),
            files,
            prepared,
        )
        merge_fallback_checkpoint(args)
        run_experiment(args, files, prepared)
        metrics = json.loads(
            (args.output / "metrics.json").read_text(encoding="utf-8")
        )
        source_counts = {}
        for row in read_jsonl(args.output / "predictions.jsonl"):
            source = row.get("source_model_family", args.model_family)
            source_counts[source] = source_counts.get(source, 0) + 1
        return {
            "dataset": args.dataset,
            "model": args.model_family,
            "method": args.method,
            "status": "completed",
            "metrics": metrics["metrics"],
            "n_scored": metrics["n_scored"],
            "n_invalid": metrics["n_invalid"],
            "diagnostics": metrics.get("diagnostics", {}),
            "cot_format": metrics.get("cot_format"),
            "n_truncated": metrics.get("n_truncated", 0),
            "row_sources": source_counts,
            "benchmark_comparability": metrics.get("benchmark_comparability"),
        }
    except Exception as exc:
        print(
            f"FAILED {args.dataset} | {args.model_family}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "dataset": args.dataset,
            "model": args.model_family,
            "method": args.method,
            "status": "failed",
            "error_type": type(exc).__name__,
        }


def main():
    if MAX_TOTAL_WORKERS < WORKERS_PER_MODEL:
        raise ValueError(
            "MAX_TOTAL_WORKERS must be >= WORKERS_PER_MODEL"
        )
    model_concurrency = min(
        len(MODELS), MAX_TOTAL_WORKERS // WORKERS_PER_MODEL
    )
    deployments = {model: deployment_for_model(model) for model in MODELS}
    strict_4o_deployment = deployment_for_model("gpt-4o", required=False)
    strict_4o_deployment = (
        os.getenv("ICL_DEPLOYMENT_GPT_4O_STRICT", "").strip()
        or (f"{strict_4o_deployment}-strict" if strict_4o_deployment else None)
    )
    files = {
        f.path: f.id
        for f in gdown.download_folder(
            url=dataset_folder_url(ROOT), quiet=True, use_cookies=False, skip_download=True
        )
    }
    runs = []
    for dataset, task in TASKS.items():
        jobs = [
            SimpleNamespace(
                dataset=dataset,
                model_family=model,
                deployment=deployments[model],
                output=RESULTS / dataset / model,
                shots=task[1],
                method="direct",
                seed=SEED,
                limit=LIMIT,
                max_prompt_chars=MAX_PROMPT_CHARS,
                workers=WORKERS_PER_MODEL,
                resume=True,
                prepare_only=False,
                fallback_output=(
                    RESULTS / dataset / strict_4o_deployment
                    if model == PRIMARY_4O_MODEL and strict_4o_deployment
                    else None
                ),
            )
            for model in MODELS
        ]
        print(
            f"\n=== {dataset} | 4 models x "
            f"{WORKERS_PER_MODEL} workers; "
            f"{model_concurrency} models at once ===",
            flush=True,
        )
        completed = []
        for method in METHODS:
            for args in jobs:
                output = args.output if method == "direct" else args.output / method
                result = completed_result(
                    SimpleNamespace(**{**vars(args), "method": method, "output": output})
                )
                if result is not None:
                    completed.append(result)
        if len(completed) == len(MODELS) * len(METHODS):
            runs.extend(completed)
            RESULTS.mkdir(parents=True, exist_ok=True)
            write_json(RESULTS / "summary.json", runs)
            continue
        for method in METHODS:
            for args in jobs:
                migration_args = SimpleNamespace(
                    **{
                        **vars(args),
                        "method": method,
                        "output": (
                            args.output
                            if method == "direct"
                            else args.output / method
                        ),
                    }
                )
                if completed_result(migration_args) is None:
                    migrate_incomplete_checkpoint(migration_args)
        try:
            # Prepare once so every model gets the same shots.
            prepared_from_fallback = fallback_examples(
                jobs[0].fallback_output
            )
            run_experiment(
                SimpleNamespace(**{**vars(jobs[0]), "prepare_only": True}),
                files,
                prepared_from_fallback,
            )
            manifest = json.loads(
                (jobs[0].output / "experiment.json").read_text(
                    encoding="utf-8"
                )
            )
            prepared = manifest["examples"], manifest["sampling"]
        except Exception as exc:
            print(f"Preparation failed for {dataset}: {exc}", flush=True)
            runs.extend(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
                for model in MODELS
                for method in METHODS
            )
            RESULTS.mkdir(parents=True, exist_ok=True)
            write_json(RESULTS / "summary.json", runs)
            continue
        for method in METHODS:
            # Separate method directories within the versioned results root.
            method_jobs = [
                SimpleNamespace(
                    **{
                        **vars(args),
                        "method": method,
                        "output": (
                            args.output
                            if method == "direct"
                            else args.output / method
                        ),
                        "fallback_output": (
                            args.fallback_output
                            if method == "direct"
                            else args.fallback_output / method
                            if args.fallback_output is not None
                            else None
                        ),
                    }
                )
                for args in jobs
            ]
            print(f"=== {dataset} | {method} ===", flush=True)
            with ThreadPoolExecutor(
                max_workers=model_concurrency
            ) as executor:
                futures = [
                    executor.submit(run_model, args, files, prepared)
                    for args in method_jobs
                ]
                for future in as_completed(futures):
                    runs.append(future.result())
                    # Only the coordinator writes the aggregate summary.
                    write_json(RESULTS / "summary.json", runs)
    failed = sum(run["status"] == "failed" for run in runs)
    print(
        f"Finished: {len(runs) - failed}/{len(runs)} completed. "
        f"Summary: {RESULTS / 'summary.json'}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
