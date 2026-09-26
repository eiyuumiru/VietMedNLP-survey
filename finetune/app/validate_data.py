"""Exhaustively validate every task CSV before training.

For each of the 12 datasets x {train, dev, test} it checks:
  - the file exists and has EXACTLY the 10 standard columns (no missing/extra),
  - row count,
  - the % of rows whose nl target is empty / "None" (a broken dataset like ViMQ_NER
    shows ~100% here — the entity labels were lost in conversion),
  - the % of rows whose nl instruction is empty,
  - for classification/NLI/MCQA: number of distinct labels + the most common ones,
  - a few sample (instruction, target) pairs so you can eyeball the format.

Big files (e.g. ViSP ~881 MB) are streamed in chunks to limit memory use.

Usage:
    python -m app.validate_data \
        --data-root /path/to/Instruct_Datasets

Writes <data-root>/validation_report.json and prints a summary table. A FAIL means do
not train on that file as-is; a WARN means inspect it (often a high empty/None rate).
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import pandas as pd

from .config import (
    DATASETS,
    EMPTY_TARGET_SENTINELS,
    EXPECTED_COLUMNS,
    INSTRUCTION_COLUMN,
    TARGET_COLUMN,
    get_spec,
)
from .utils import LOGGER, save_json

CLASSIFICATION_TASKS = {"classification", "nli", "mcqa"}
_CHUNK = 20000
_MAX_DISTINCT_LABELS = 300  # cap label counter memory for safety


def _is_empty(value: str) -> bool:
    return str(value).strip().lower() in EMPTY_TARGET_SENTINELS


def validate_file(data_root: str, split: str, stem: str, n_samples: int) -> dict:
    spec = get_spec(stem)
    path = os.path.join(data_root, split, f"{stem}.csv")
    rec: dict = {
        "dataset": stem, "split": split, "task_type": spec.task_type, "path": path,
        "status": "OK", "issues": [],
    }
    if not os.path.exists(path):
        rec["status"] = "MISSING"
        rec["issues"].append("file not found")
        return rec

    # --- column / schema check (header only) ---
    header = pd.read_csv(path, nrows=0).columns.tolist()
    rec["n_columns"] = len(header)
    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    extra = [c for c in header if c not in EXPECTED_COLUMNS]
    if missing:
        rec["status"] = "FAIL"
        rec["issues"].append(f"missing columns: {missing}")
    if extra:
        rec["issues"].append(f"extra columns: {extra}")
    if missing:  # cannot analyse content without the nl columns
        return rec

    # --- streamed content scan ---
    n_rows = 0
    n_empty_target = 0
    n_empty_instr = 0
    label_counter: Counter = Counter()
    label_overflow = False
    samples: list[dict] = []
    is_cls = spec.task_type in CLASSIFICATION_TASKS

    reader = pd.read_csv(
        path, dtype=str, keep_default_na=False,
        usecols=[INSTRUCTION_COLUMN, TARGET_COLUMN], chunksize=_CHUNK,
    )
    for chunk in reader:
        instr = chunk[INSTRUCTION_COLUMN].astype(str)
        tgt = chunk[TARGET_COLUMN].astype(str)
        n_rows += len(chunk)
        n_empty_instr += int(instr.map(_is_empty).sum())
        n_empty_target += int(tgt.map(_is_empty).sum())
        if is_cls and not label_overflow:
            for v in tgt:
                label_counter[v.strip()] += 1
                if len(label_counter) > _MAX_DISTINCT_LABELS:
                    label_overflow = True
                    break
        for i in range(len(chunk)):
            if len(samples) >= n_samples:
                break
            samples.append({
                "instruction": instr.iloc[i][:300],
                "target": tgt.iloc[i][:300],
            })

    rec["n_rows"] = n_rows
    rec["empty_target_rate"] = round(n_empty_target / n_rows, 4) if n_rows else 1.0
    rec["empty_instruction_rate"] = round(n_empty_instr / n_rows, 4) if n_rows else 1.0
    rec["samples"] = samples
    if is_cls:
        rec["num_distinct_labels"] = ">" + str(_MAX_DISTINCT_LABELS) if label_overflow \
            else len(label_counter)
        rec["top_labels"] = label_counter.most_common(8)

    # --- verdict ---
    if n_rows == 0:
        rec["status"] = "FAIL"
        rec["issues"].append("0 usable rows")
    if rec["empty_target_rate"] >= 0.5:
        rec["status"] = "FAIL" if rec["empty_target_rate"] >= 0.95 else "WARN"
        rec["issues"].append(
            f"{rec['empty_target_rate']:.0%} of targets are empty/None "
            f"(labels likely missing for this dataset)"
        )
    if rec["empty_instruction_rate"] > 0.01:
        rec["status"] = "WARN" if rec["status"] == "OK" else rec["status"]
        rec["issues"].append(f"{rec['empty_instruction_rate']:.0%} of instructions empty")
    return rec


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate the instruct datasets before training.")
    p.add_argument("--data-root", required=True,
                   help="Instruct_Datasets[_EN] folder with train/ dev/ test/ subdirs.")
    p.add_argument("--splits", default="train,dev,test")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--out", default=None, help="Report JSON path (default: <data-root>/validation_report.json)")
    args = p.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    records = []
    for stem in DATASETS:
        for split in splits:
            try:
                records.append(validate_file(args.data_root, split, stem, args.samples))
            except Exception as exc:  # pragma: no cover
                records.append({"dataset": stem, "split": split, "status": "ERROR",
                                "issues": [str(exc)]})

    # --- print summary table ---
    LOGGER.info("%-32s %-6s %-7s %9s %9s  %s", "Dataset", "Split", "Status",
                "rows", "empty%", "issues")
    LOGGER.info("-" * 100)
    n_fail = n_warn = 0
    for r in records:
        if r["status"] == "FAIL":
            n_fail += 1
        elif r["status"] == "WARN":
            n_warn += 1
        LOGGER.info(
            "%-32s %-6s %-7s %9s %9s  %s",
            r["dataset"][:32], r.get("split", "")[:6], r["status"],
            str(r.get("n_rows", "-")),
            (f"{r['empty_target_rate']:.0%}" if "empty_target_rate" in r else "-"),
            "; ".join(r.get("issues", [])),
        )

    out = args.out or os.path.join(args.data_root, "validation_report.json")
    try:
        save_json({"records": records, "n_fail": n_fail, "n_warn": n_warn}, out)
        LOGGER.info("Wrote report to %s", out)
    except Exception as exc:  # The data root may be read-only.
        fallback = os.path.join(os.getcwd(), "validation_report.json")
        save_json({"records": records, "n_fail": n_fail, "n_warn": n_warn}, fallback)
        LOGGER.warning("Could not write to %s (%s); wrote %s instead.", out, exc, fallback)

    LOGGER.info("=" * 60)
    LOGGER.info("SUMMARY: %d FAIL, %d WARN, %d total checks.", n_fail, n_warn, len(records))
    if n_fail:
        LOGGER.warning("Do NOT train on FAIL files as-is (e.g. all-empty targets).")


if __name__ == "__main__":
    main()
