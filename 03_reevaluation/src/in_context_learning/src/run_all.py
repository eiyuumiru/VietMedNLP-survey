"""Run the Section 6 (in-context learning branch) model, dataset, and prompting-condition matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from icl_config import (
    BATCH_METHODS,
    MODELS,
    api_settings,
    dataset_files,
    model_id_for_family,
    resolve_data_root,
)
from icl_inference import (
    PROTOCOL,
    ROOT,
    checkpoint_rows,
    file_hash,
    run_experiment,
    source_hashes,
    write_json,
)
from icl_tasks import TASKS


METHODS = BATCH_METHODS
SEED = 42
RESULTS = ROOT / "results" / "icl"
MAX_PROMPT_CHARS = 60000
WORKERS_PER_MODEL = 4
MAX_TOTAL_WORKERS = 16


def completed_result(args):
    """Return a completed result only when its local outputs are verifiable."""
    metrics_path = args.output / "metrics.json"
    predictions = args.output / "predictions.jsonl"
    manifest_path = args.output / "experiment.json"
    if not all(path.exists() for path in (metrics_path, predictions, manifest_path)):
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    data = dataset_files(args.data_root, args.dataset)
    if not metrics.get("completed_requested_rows"):
        return None
    try:
        records = checkpoint_rows(predictions, metrics.get("experiment_id"))
    except (KeyError, TypeError, ValueError):
        return None
    if metrics.get("n_scored") != len(records) or not records:
        return None
    if config.get("protocol") != PROTOCOL or config.get("dataset") != args.dataset:
        return None
    if config.get("model_family") != args.model_family:
        return None
    if config.get("model_fingerprint") != hashlib.sha256(
        args.model_id.encode("utf-8")
    ).hexdigest():
        return None
    if config.get("method") != args.method or config.get("shots") != args.shots:
        return None
    if config.get("seed") != args.seed or config.get("limit") != args.limit:
        return None
    if config.get("max_prompt_chars") != args.max_prompt_chars:
        return None
    api_provider, api_base, api_version = api_settings()
    if config.get("api_provider") != api_provider:
        return None
    if config.get("api_base_fingerprint") != hashlib.sha256(
        api_base.encode("utf-8")
    ).hexdigest():
        return None
    if config.get("api_version") != api_version:
        return None
    if config.get("train_sha256") != file_hash(data["train"]):
        return None
    if config.get("test_sha256") != file_hash(data["test"]):
        return None
    if config.get("code") != source_hashes() or metrics.get("config") != config:
        return None
    if metrics.get("predictions_sha256") != file_hash(predictions):
        return None
    return summarize_result(args, metrics)


def summarize_result(args, metrics):
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
        "benchmark_comparability": metrics.get("benchmark_comparability"),
    }


def run_model(args, prepared):
    try:
        existing = completed_result(args)
        if existing is not None:
            return existing
        run_experiment(args, prepared)
        metrics = json.loads((args.output / "metrics.json").read_text(encoding="utf-8"))
        return summarize_result(args, metrics)
    except Exception as exc:
        print(
            f"FAILED {args.dataset} | {args.model_family} | {args.method}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "dataset": args.dataset,
            "model": args.model_family,
            "method": args.method,
            "status": "failed",
            "error_type": type(exc).__name__,
        }


def experiment_args(dataset, task, data_root, model_family, model_id, method, output, limit):
    return SimpleNamespace(
        dataset=dataset,
        data_root=data_root,
        model_family=model_family,
        model_id=model_id,
        output=output,
        shots=task[1],
        method=method,
        seed=SEED,
        limit=limit,
        max_prompt_chars=MAX_PROMPT_CHARS,
        workers=WORKERS_PER_MODEL,
        resume=True,
        prepare_only=False,
    )


def load_prepared_examples(jobs):
    """Reuse saved demonstrations, or select them once for this dataset."""
    train_hash = file_hash(
        dataset_files(jobs[0].data_root, jobs[0].dataset)["train"]
    )
    for args in jobs:
        manifest_path = args.output / "experiment.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config", {})
            if (
                config.get("protocol") == PROTOCOL
                and config.get("dataset") == args.dataset
                and config.get("train_sha256") == train_hash
                and config.get("shots") == args.shots
                and config.get("seed") == args.seed
                and isinstance(manifest.get("examples"), list)
                and "sampling" in manifest
            ):
                return manifest["examples"], manifest["sampling"]

    first = jobs[0]
    prepare_args = SimpleNamespace(**{**vars(first), "prepare_only": True})
    run_experiment(prepare_args)
    manifest = json.loads(
        (first.output / "experiment.json").read_text(encoding="utf-8")
    )
    return manifest["examples"], manifest["sampling"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Folder containing train/ and test/")
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--limit", type=int, help="Run only the first N test rows")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        args.data_root = resolve_data_root(args.data_root)
    except (NotADirectoryError, RuntimeError) as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    if MAX_TOTAL_WORKERS < WORKERS_PER_MODEL:
        raise ValueError("MAX_TOTAL_WORKERS must be >= WORKERS_PER_MODEL")
    model_concurrency = min(len(MODELS), MAX_TOTAL_WORKERS // WORKERS_PER_MODEL)
    model_ids = {model: model_id_for_family(model) for model in MODELS}
    results_root = args.results_root.expanduser()
    results_root.mkdir(parents=True, exist_ok=True)
    runs = []

    for dataset, task in TASKS.items():
        jobs = {
            (model, method): experiment_args(
                dataset,
                task,
                args.data_root,
                model,
                model_ids[model],
                method,
                results_root / dataset / model / ("" if method == "direct" else method),
                args.limit,
            )
            for model in MODELS
            for method in METHODS
        }
        all_existing = {
            key: completed_result(job) for key, job in jobs.items()
        }
        if all(value is not None for value in all_existing.values()):
            runs.extend(all_existing.values())
            write_json(results_root / "summary.json", runs)
            continue

        print(
            f"\n=== {dataset} | {len(MODELS)} models x {len(METHODS)} methods; "
            f"{WORKERS_PER_MODEL} request workers/model ===",
            flush=True,
        )
        try:
            prepared = load_prepared_examples(
                [jobs[(model, "direct")] for model in MODELS]
            )
        except Exception as exc:
            print(f"Preparation failed for {dataset}: {exc}", flush=True)
            failed = [
                {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                }
                for model in MODELS
                for method in METHODS
            ]
            runs.extend(failed)
            write_json(results_root / "summary.json", runs)
            continue

        dataset_results = {}
        for method in METHODS:
            method_jobs = [jobs[(model, method)] for model in MODELS]
            with ThreadPoolExecutor(max_workers=model_concurrency) as executor:
                futures = {
                    executor.submit(run_model, job, prepared): job
                    for job in method_jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    dataset_results[(job.model_family, method)] = future.result()
                    current = runs + [
                        dataset_results[key]
                        for key in sorted(dataset_results, key=lambda item: (METHODS.index(item[1]), MODELS.index(item[0])))
                    ]
                    write_json(results_root / "summary.json", current)
        runs.extend(
            dataset_results[(model, method)]
            for method in METHODS
            for model in MODELS
        )
        write_json(results_root / "summary.json", runs)

    failed_count = sum(run["status"] == "failed" for run in runs)
    print(
        f"Finished: {len(runs) - failed_count}/{len(runs)} completed. "
        f"Summary: {results_root / 'summary.json'}"
    )
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
