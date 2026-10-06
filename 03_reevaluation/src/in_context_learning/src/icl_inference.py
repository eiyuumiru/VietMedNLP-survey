"""Few-shot inference: stream train -> balance examples -> predict test -> score gold."""

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from importlib.metadata import version

import tiktoken

from icl_tasks import (
    TASKS,
    EVALUATOR_VERSION,
    cot_compliance,
    evaluate,
    groups,
    messages,
    normalize,
    parse_answer,
)
from icl_config import (
    BATCH_METHODS,
    MODEL_PROFILES,
    MODELS,
    api_settings,
    dataset_files,
    model_id_for_family,
    resolve_data_root,
    single_model_id,
)
from icl_api import make_client as _make_client

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "icl-v3"
POOL_PER_GROUP = 32
MAX_EXAMPLE_CHARS = 6000
MAX_PROMPT_TOKENS = 32000
MAX_COMPLETION_TOKENS = 8192
TOKEN_ENCODING = "o200k_base"
def csv_reader(stream, source):
    reader = csv.DictReader(stream)
    if not {"input", "output"}.issubset(reader.fieldnames or []):
        raise ValueError(f"{source}: CSV requires input and output columns")
    return reader


@contextmanager
def local_csv(path):
    """Stream one authorized benchmark CSV from the local data directory."""
    csv.field_size_limit(10_000_000)
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield csv_reader(stream, path)


def checked_row(row, index):
    if any(
        not isinstance(row.get(key), str) or not row[key].strip()
        for key in ("input", "output")
    ):
        raise ValueError(f"Empty input/gold at row {index}")
    return {"row": index, "input": row["input"], "output": row["output"]}


def fingerprint(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def object_hash(value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


def source_hashes():
    """AST fingerprints ignore whitespace-only formatting changes."""
    return {
        name: object_hash(
            ast.dump(
                ast.parse(
                    Path(__file__).with_name(name).read_text(encoding="utf-8")
                )
            )
        )
        for name in (
            "icl_inference.py",
            "icl_tasks.py",
            "icl_config.py",
            "icl_api.py",
        )
    }


def experiment_fingerprints(examples, task, method):
    return {
        "examples": object_hash(examples),
        "prompt": object_hash(messages("", examples, task, method)),
    }


def decoding_options(model_family=None, *, temperature=None, reasoning_effort=None):
    """Build request settings from a profile, with per-run overrides."""
    options = {"max_completion_tokens": MAX_COMPLETION_TOKENS}
    profile = MODEL_PROFILES.get(model_family)
    if profile is not None:
        options.update(profile.decoding)
    else:
        options["temperature"] = 0
    if temperature is not None:
        options.pop("reasoning_effort", None)
        options["temperature"] = temperature
    if reasoning_effort is not None:
        options.pop("temperature", None)
        options["reasoning_effort"] = reasoning_effort
    return options


def prompt_token_estimate(prompt, encoding):
    """Local estimate; allowance for chat framing, not an API-exact count."""
    return 3 + sum(
        6 + len(encoding.encode(message["content"], disallowed_special=()))
        for message in prompt
    )


def select_examples(rows, task, shots, seed):
    """Uniform reservoir per group across the entire train stream, then balance."""
    rng = random.Random(seed)
    pools = defaultdict(list)
    seen, eligible = Counter(), Counter()
    scanned = excluded = 0
    for index, raw in enumerate(rows):
        row = checked_row(raw, index)
        buckets = groups(row, task)
        scanned += 1
        seen.update(buckets)
        if len(row["input"]) + len(row["output"]) > MAX_EXAMPLE_CHARS:
            excluded += 1
            continue
        for bucket in sorted(buckets):
            eligible[bucket] += 1
            pool = pools[bucket]
            if len(pool) < POOL_PER_GROUP:
                pool.append(row)
            else:
                position = rng.randrange(eligible[bucket])
                if position < POOL_PER_GROUP:
                    pool[position] = row

    candidates = {row["row"]: row for pool in pools.values() for row in pool}
    words = {
        i: set(re.findall(r"\w+", normalize(row["input"])))
        for i, row in candidates.items()
    }
    buckets = {i: groups(row, task) for i, row in candidates.items()}
    counts, used_inputs, selected = Counter(), set(), []
    tie_order = list(sorted(pools))
    rng.shuffle(tie_order)
    rank = {bucket: i for i, bucket in enumerate(tie_order)}
    while len(selected) < shots:
        available = [
            i
            for i, row in candidates.items()
            if fingerprint(row["input"]) not in used_inputs
        ]
        active = set().union(*(buckets[i] for i in available))
        if not active:
            raise ValueError(
                f"Only {len(selected)} distinct eligible examples; reduce --shots"
            )
        target = min(active, key=lambda b: (counts[b], rank[b]))
        choices = [i for i in available if target in buckets[i]]

        def priority(i):
            # Lexical distance is a tie-breaker, not a claim of semantic topic balance.
            similarity = max(
                (
                    len(words[i] & words[r["row"]])
                    / max(1, len(words[i] | words[r["row"]]))
                    for r in selected
                ),
                default=0,
            )
            coverage = sum(1 / (1 + counts[b]) for b in sorted(buckets[i]))
            return coverage, -similarity

        rng.shuffle(choices)
        index = max(choices, key=priority)
        row = candidates[index]
        selected.append(row)
        used_inputs.add(fingerprint(row["input"]))
        # NER: update ALL types present, not just target.
        counts.update(buckets[index])

    return selected, {
        "train_rows": scanned,
        "excluded_long_examples": excluded,
        "group_population": dict(seen),
        "selected_groups": dict(counts),
        "uncovered_groups": sorted(set(seen) - set(counts)),
        "pool_per_group": POOL_PER_GROUP,
        "max_example_chars": MAX_EXAMPLE_CHARS,
    }


def make_client(model_id):
    """Thin seam for tests; provider selection lives in the API adapter."""
    return _make_client(model_id)


def read_jsonl(path):
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def recover_checkpoint_tail(path):
    """Repair only an unterminated final line, preserving torn bytes in a backup."""
    if not path.exists():
        return
    with path.open("rb") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                return
            try:
                json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise ValueError(
                        "Corrupt checkpoint line; manual inspection required"
                    )
                backup = path.with_name(
                    path.name
                    + ".torn-"
                    + hashlib.sha256(line).hexdigest()[:16]
                    + ".bak"
                )
                if not backup.exists():
                    with backup.open("xb") as saved:
                        saved.write(line)
                        saved.flush()
                        os.fsync(saved.fileno())
                elif backup.read_bytes() != line:
                    raise ValueError("Checkpoint backup conflict")
                with path.open("r+b") as output:
                    output.truncate(start)
                    output.flush()
                    os.fsync(output.fileno())
                print(
                    f"Recovered incomplete final line; original bytes: {backup}",
                    flush=True,
                )
                return
            if not line.endswith(b"\n"):
                with path.open("ab") as output:
                    output.write(b"\n")
                    output.flush()
                    os.fsync(output.fileno())
                return


def checkpoint_rows(path, experiment_id):
    previous = {}
    for row in read_jsonl(path):
        content = {
            key: value for key, value in row.items() if key != "record_sha256"
        }
        if row.get("record_sha256") != object_hash(content):
            raise ValueError("Checkpoint record checksum mismatch")
        if row.get("row") != len(previous):
            raise ValueError("Checkpoint row IDs are not contiguous")
        if row.get("experiment_id") != experiment_id:
            raise ValueError("Checkpoint belongs to a different experiment")
        if not isinstance(row.get("valid"), bool) or any(
            not isinstance(row.get(key), str)
            for key in ("test_hash", "gold", "prediction", "raw_prediction")
        ):
            raise ValueError("Invalid checkpoint record schema")
        previous[row["row"]] = row["test_hash"]
    return previous


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Atomic metadata/summary writes; predictions are flushed after every success.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=TASKS)
    parser.add_argument("--data-root", type=Path, help="Folder containing train/ and test/")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for this experiment",
    )
    parser.add_argument(
        "--shots", type=int, help="Positive; defaults depend on dataset"
    )
    parser.add_argument(
        "--method",
        choices=BATCH_METHODS,
        default="direct",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", dest="model_id")
    parser.add_argument("--model-family", choices=MODELS)
    decoding = parser.add_mutually_exclusive_group()
    decoding.add_argument("--temperature", type=float)
    decoding.add_argument(
        "--reasoning-effort", choices=("none", "minimal", "low", "medium", "high")
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Smoke-test subset; omit for the full test split",
    )
    parser.add_argument("--max-prompt-chars", type=int, default=60000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Save examples; do not call LLM",
    )
    args = parser.parse_args()
    try:
        args.data_root = resolve_data_root(args.data_root)
        if not args.model_family and not args.model_id:
            args.model_id = single_model_id()
    except (NotADirectoryError, RuntimeError) as exc:
        parser.error(str(exc))
    args.shots = TASKS[args.dataset][1] if args.shots is None else args.shots
    if (
        args.shots < 1
        or (args.limit is not None and args.limit < 1)
        or args.max_prompt_chars < 1
    ):
        parser.error(
            "shots, limit and max-prompt-chars must be positive; zero-shot is disabled"
        )
    return args


def main():
    args = parse_args()
    run_experiment(args)


def ordered_results(function, items, workers):
    """Bound outstanding requests and yield results in input order for checkpoints."""
    if workers < 1:
        raise ValueError("workers must be positive")
    items = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        try:
            for _ in range(workers):
                item = next(items, None)
                if item is None:
                    break
                pending.append(executor.submit(function, item))
            while pending:
                yield pending.popleft().result()
                item = next(items, None)
                if item is not None:
                    pending.append(executor.submit(function, item))
        finally:
            for future in pending:
                future.cancel()


def is_input_content_filter(exc):
    return getattr(exc, "code", None) == "content_filter" or (
        "content_filter" in str(exc)
        and "prompt" in str(exc).lower()
    )


def run_experiment(args, prepared=None):
    model_family = getattr(args, "model_family", None)
    model_id = getattr(args, "model_id", None)
    if model_family and not model_id:
        model_id = model_id_for_family(model_family)
    data_root = resolve_data_root(getattr(args, "data_root", None))
    if getattr(args, "prepare_only", False):
        return _run_experiment(
            args,
            data_root,
            prepared,
            None,
            model_id or model_family or "custom-model",
        )
    client, model_id = make_client(model_id)
    with client:
        return _run_experiment(args, data_root, prepared, client, model_id)


def _run_experiment(args, data_root, prepared, client, model_id):
    task = TASKS[args.dataset]
    paths = dataset_files(data_root, args.dataset)
    model_family = getattr(args, "model_family", None)
    decoding = decoding_options(
        model_family,
        temperature=getattr(args, "temperature", None),
        reasoning_effort=getattr(args, "reasoning_effort", None),
    )
    api_provider, api_base, api_version = api_settings()
    config = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "model_family": model_family,
        "model_fingerprint": hashlib.sha256(
            model_id.encode("utf-8")
        ).hexdigest(),
        "shots": args.shots,
        "method": args.method,
        "seed": args.seed,
        "limit": args.limit,
        "max_prompt_chars": args.max_prompt_chars,
        "train_sha256": file_hash(paths["train"]),
        "test_sha256": file_hash(paths["test"]),
        "task": json.loads(json.dumps(task)),
        "code": source_hashes(),
        "evaluator_version": EVALUATOR_VERSION,
        "decoding": decoding,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "token_encoding": TOKEN_ENCODING,
        "sdk_version": version("openai"),
        "tokenizer_version": version("tiktoken"),
        "api_provider": api_provider,
        "api_base_fingerprint": hashlib.sha256(
            api_base.encode("utf-8")
        ).hexdigest(),
        "api_version": api_version,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "experiment.json"
    predictions = args.output / "predictions.jsonl"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError(
                "Experiment exists: use --resume or a new --output directory"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config:
            raise ValueError(
                "Resume configuration/code differs; use a new output directory"
            )
    else:
        if predictions.exists():
            raise ValueError("Predictions exist without experiment metadata")
        if prepared is None:
            with local_csv(paths["train"]) as rows:
                examples, sampling = select_examples(
                    rows, task, args.shots, args.seed
                )
        else:
            examples, sampling = prepared
        if len(examples) != args.shots:
            raise ValueError(
                "Prepared example count differs from requested shots"
            )
        manifest = {
            "config": config,
            "examples": examples,
            "sampling": sampling,
            "system_prompt": messages("", examples, task, args.method)[0][
                "content"
            ],
            "fingerprints": experiment_fingerprints(
                examples, task, args.method
            ),
        }
        write_json(manifest_path, manifest)
    examples = manifest["examples"]
    current_fingerprints = experiment_fingerprints(examples, task, args.method)
    if manifest.get("fingerprints") != current_fingerprints:
        raise ValueError("Prompt/examples changed; use a new output directory")
    if (
        prepared is not None
        and object_hash(prepared[0]) != current_fingerprints["examples"]
    ):
        raise ValueError(
            "Saved examples differ from the shared batch examples"
        )
    experiment_id = object_hash({"config": config, **current_fingerprints})
    print(json.dumps(manifest["sampling"], ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    recover_checkpoint_tail(predictions)
    previous = checkpoint_rows(predictions, experiment_id)
    metrics_path = args.output / "metrics.json"
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            saved.get("config") == config
            and saved.get("experiment_id") == experiment_id
            and saved.get("completed_requested_rows")
            and saved.get("n_scored") == len(previous) > 0
            and saved.get("predictions_sha256") == file_hash(predictions)
        ):
            print("Already completed; skipping.", flush=True)
            return

    # Resume validates contiguous row IDs and input/gold hashes while streaming test.
    encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    example_inputs = {fingerprint(row["input"]) for row in examples}
    total = 0

    def requests_to_run(rows):
        nonlocal total
        for index, raw in enumerate(rows):
            if args.limit is not None and index >= args.limit:
                break
            row = checked_row(raw, index)
            # Validate gold schema independently from prompt construction.
            groups(row, task)
            total += 1
            test_hash = hashlib.sha256(
                json.dumps([row["input"], row["output"]]).encode()
            ).hexdigest()
            if index in previous:
                if previous[index] != test_hash:
                    raise ValueError(
                        f"Test changed at row {index}; cannot resume"
                    )
                continue
            if fingerprint(row["input"]) in example_inputs:
                raise ValueError(
                    f"Train demonstration overlaps test row {index}; select a new seed"
                )
            prompt = messages(row["input"], examples, task, args.method)
            if (
                sum(len(message["content"]) for message in prompt)
                > args.max_prompt_chars
            ):
                raise ValueError(
                    f"Prompt too long at row {index}; no silent truncation"
                )
            prompt_tokens = prompt_token_estimate(prompt, encoding)
            if prompt_tokens > MAX_PROMPT_TOKENS:
                raise ValueError(
                    f"Prompt token budget exceeded at row {index}: {prompt_tokens}"
                )
            yield index, row, test_hash, prompt, prompt_tokens

    def predict(item):
        index, row, test_hash, prompt, prompt_tokens = item
        response = None
        try:
            response = client.chat.completions.create(
                model=model_id, messages=prompt, **decoding
            )
        except Exception as exc:
            if not is_input_content_filter(exc):
                raise
            finish_reason = "input_content_filter"
            raw_answer = answer = ""
            valid = False
        else:
            finish_reason = response.choices[0].finish_reason
            if finish_reason not in {"stop", "length", "content_filter"}:
                raise RuntimeError(
                    f"Unexpected finish reason at row {index}: {finish_reason}"
                )
            raw_answer = response.choices[0].message.content or ""
            answer, valid = parse_answer(raw_answer, task)
            # A bounded but truncated/filtered output is scored as a failed answer,
            # not retried indefinitely or dropped from the test denominator.
            valid = valid and finish_reason == "stop"
        record = {
            "experiment_id": experiment_id,
            "row": index,
            "test_hash": test_hash,
            "gold": row["output"],
            "prediction": answer,
            "valid": valid,
            "raw_prediction": raw_answer,
            "finish_reason": finish_reason,
            "prompt_tokens_estimate": prompt_tokens,
            "cot_format_compliant": (
                finish_reason == "stop" and cot_compliance(raw_answer)
                if args.method == "cot"
                else None
            ),
            "usage": response.usage.model_dump() if response and response.usage else None,
            "response_model": getattr(response, "model", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
        }
        record["record_sha256"] = object_hash(record)
        return record

    with predictions.open("a", encoding="utf-8") as output, local_csv(
        paths["test"]
    ) as rows:
        for record in ordered_results(
            predict, requests_to_run(rows), getattr(args, "workers", 1)
        ):
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"{model_family or 'custom-model'} | {args.dataset} | {args.method} | "
                f"row={record['row']} valid={record['valid']}",
                flush=True,
            )
    if total < len(previous):
        raise ValueError("Test is shorter than the saved checkpoint")
    if not total:
        raise ValueError("Test CSV has zero rows; refusing to mark completed")
    if args.limit is not None and total != args.limit:
        raise ValueError(f"Expected {args.limit} test rows, received {total}")
    saved_rows = checkpoint_rows(predictions, experiment_id)
    if len(saved_rows) != total:
        raise ValueError("Prediction count differs from processed test rows")
    summary = evaluate(read_jsonl(predictions), task, args.method)
    summary.update(
        {
            "config": config,
            "experiment_id": experiment_id,
            "predictions_sha256": file_hash(predictions),
            "n_test_rows": total,
            "completed_requested_rows": True,
            "scope": "full_test" if args.limit is None else "test_subset",
        }
    )
    write_json(args.output / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
