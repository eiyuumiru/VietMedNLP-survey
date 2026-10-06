"""Offline audit of the existing Section 6--7 runs. No model or API calls.

Recomputes the aggregate CSVs in ``04_results/`` from the saved fine-tuning
archives and ICL runs, which are not distributed with this repository. Pass
``--check`` to compare against the committed CSVs without overwriting them.
Only aggregate, non-record-level outputs are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "03_reevaluation/src"
MODEL_VERSIONS = {
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5.2": "gpt-5.2-2025-12-11",
    "gpt-5.4": "gpt-5.4-2026-03-05",
}
CONDITIONS = {"direct": "answer_only", "cot": "explanation_request"}  # output labels
FT_SOURCES = {
    "PhoNER_COVID19_NER": "results_1.zip",
    "ViMedNER_NER": "results_1.zip",
    "ViMQ_intent_classification": "results_1.zip",
    "ViMedNLI_ViMedNLI": "results_1.zip",
    "vihealthbert_acrDrAid": "results_1.zip",
    "VMHQA_Multiple_Choice_QA": "results_1.zip",
    "ViNewsQA_Extractive_QA": "results_1.zip",
    "UIT-ViCoV19QA_QA": "results_2.zip",
    "ViMedAQA_Abstract_QA": "results_2.zip",
    "vihealthbert_Summarization": "results_2.zip",
}
FT_TABLE_METRICS = {
    "PhoNER_COVID19_NER": ("entity_micro_f1", "entity_macro_f1"),
    "ViMedNER_NER": ("entity_micro_f1", "entity_macro_f1"),
    "ViMQ_intent_classification": ("micro_f1", "macro_f1"),
    "ViMedNLI_ViMedNLI": ("accuracy",),
    "vihealthbert_acrDrAid": ("macro_f1",),
    "VMHQA_Multiple_Choice_QA": ("accuracy",),
    "ViNewsQA_Extractive_QA": ("squad_em", "squad_f1"),
    "UIT-ViCoV19QA_QA": ("rougeL", "bertscore_f1"),
    "ViMedAQA_Abstract_QA": ("rougeL", "bertscore_f1"),
    "vihealthbert_Summarization": ("rougeL", "bertscore_f1"),
}
DISPLAY = {
    "PhoNER_COVID19_NER": "PhoNER\\_COVID19",
    "ViMedNER_NER": "ViMedNER",
    "ViMQ_intent_classification": "ViMQ",
    "ViMedNLI_ViMedNLI": "ViMedNLI",
    "vihealthbert_acrDrAid": "acrDrAid",
    "VMHQA_Multiple_Choice_QA": "VMHQA",
    "ViNewsQA_Extractive_QA": "UIT-ViNewsQA",
    "UIT-ViCoV19QA_QA": "UIT-ViCoV19QA",
    "ViMedAQA_Abstract_QA": "ViMedAQA",
    "vihealthbert_Summarization": "FAQSum",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str], *, check: bool = False) -> None:
    if check:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        expected = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == fields, f"CSV columns differ: {path.name}"
            assert list(reader) == expected, f"CSV values differ: {path.name}"
        print(f"Matched {path.name}: {len(rows)} rows")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def zip_json(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name))


def check_icl_table(text: str, scores: dict) -> None:
    """Check ordered numeric cells, independently of citations and prose notes."""
    from audit_icl import DISPLAY as ICL_DISPLAY, MODELS, METHODS

    headers = [line for line in text.splitlines() if "multicolumn{2}{c}{GPT-4o}" in line]
    assert len(headers) == 1, "Missing or duplicate ICL model header"
    header = headers[0]
    assert re.findall(r"multicolumn\{2\}\{c\}\{([^}]+)\}", header) == ["GPT-4o", "GPT-4.1", "GPT-5.2", "GPT-5.4"]
    prompt_header = next(line for line in text.splitlines() if r"\textbf{Metric}" in line)
    assert re.findall(r"Answer-only|Explanation-", prompt_header) == ["Answer-only", "Explanation-"] * 4
    for dataset, (name, _, metrics) in ICL_DISPLAY.items():
        rows = [line.split(" & ") for line in text.splitlines() if " & " in line]
        matches = [cells for cells in rows if len(cells) == 12 and cells[1].strip() == name]
        assert len(matches) == 1, dataset
        cells = matches[0]
        metric_cell = cells[3]
        expected_metric = {"macro_f1": "Macro", "micro_f1": "Micro", "accuracy": "Accuracy", "exact_match": "EM", "token_f1": "F1", "rouge_l_f1": "ROUGE-L"}
        for metric in metrics:
            assert expected_metric[metric] in metric_cell, (dataset, metric)
        for cell, (model, method) in zip(cells[4:], [(m, p) for m in MODELS for p in METHODS]):
            assert re.findall(r"\d+\.\d+", cell) == [f"{scores[dataset][method][model][metric]:.2f}" for metric in metrics], (dataset, model, method)


def check_pair(direct: list[dict], cot: list[dict]) -> None:
    assert len({r["row"] for r in direct}) == len(direct), "Duplicate test row"
    assert [(r["row"], r["test_hash"], r["gold"]) for r in direct] == [(r["row"], r["test_hash"], r["gold"]) for r in cot], "Unmatched prompt rows"


def error_rows(dataset: str, model: str, method: str, counts: Counter, task_kind: str) -> list[dict]:
    categories = ["invalid", "truncated"]
    if task_kind in {"intent", "nli", "mcqa", "acronym"}:
        categories += ["incorrect"]
    elif task_kind == "ner":
        categories += ["missed_gold_entities", "spurious_predicted_entities", "boundary_overlap_mismatch"]
    elif task_kind == "extractive":
        categories += ["partial_overlap", "zero_overlap"]
    else:
        categories += ["low_rouge"]
    denominators = {"invalid": "records", "truncated": "records", "missed_gold_entities": "gold_entity_pairs", "spurious_predicted_entities": "predicted_entity_pairs", "boundary_overlap_mismatch": "same_type_mismatch_candidates"}
    rows = []
    for category in categories:
        denominator = counts[denominators[category]] if category in denominators else counts["records"] - counts["invalid"]
        rows.append({"dataset": dataset, "model": model, "condition": CONDITIONS[method], "category": category,
                     "count": counts[category], "denominator": denominator,
                     "rate_percent": round(100 * counts[category] / denominator, 2) if denominator else "NA",
                     "rule": "audit_icl.py:classify_record"})
    return rows


def coverage_rows(validity: list[dict]) -> list[dict]:
    summary = []
    for dataset in dict.fromkeys(row["dataset"] for row in validity):
        rows = [r for r in validity if r["dataset"] == dataset]
        sizes = {int(r["n_predictions"]) for r in rows}
        assert len(rows) == 8 and len(sizes) == 1
        n = sum(int(r["n_predictions"]) for r in rows)
        invalid = sum(int(r["n_invalid"]) for r in rows)
        compliance = [float(r["format_compliance_percent"]) for r in rows if r["condition"] == CONDITIONS["cot"]]
        assert len(compliance) == 4
        summary.append({"dataset": dataset, "n_test_rows": sizes.pop(), "n_configurations": 8,
                        "n_predictions": n, "n_invalid": invalid, "invalid_rate_percent": round(100 * invalid / n, 3),
                        "format_compliance_min_percent": min(compliance), "format_compliance_max_percent": max(compliance)})
    return summary


def score_components(rows: list[dict], kind: str, task=None):
    """Return cached per-row sufficient statistics for a bootstrap metric."""
    if kind == "ft_ner":
        from app.metrics import parse_entity_list

        values = []
        for row in rows:
            gold, pred = set(parse_entity_list(row["gold"])), set(parse_entity_list(row["pred"]))
            values.append((len(gold & pred), len(pred - gold), len(gold - pred)))
        return np.asarray(values, dtype=np.int32)
    if kind == "ft_cls":
        from app.metrics import norm_label

        gold = [norm_label(r["gold"]) for r in rows]
        labels = sorted(set(gold))
        pred = []
        for row in rows:
            value = norm_label(row["pred"])
            if value not in labels:
                hits = [label for label in labels if label and re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", value)]
                value = hits[0] if len(hits) == 1 else "<other>"
            pred.append(value)
        labels += (["<other>"] if "<other>" in pred else [])
        label_index = {label: i for i, label in enumerate(labels)}
        gold_ids = np.asarray([label_index[x] for x in gold], dtype=np.int32)
        pred_ids = np.asarray([label_index[x] for x in pred], dtype=np.int32)
        groups = [np.flatnonzero(gold_ids == i) for i in range(len(labels)) if np.any(gold_ids == i)]
        return gold_ids, pred_ids, len(labels), groups
    if kind == "ft_accuracy":
        from app.metrics import norm_label

        return np.asarray([float(norm_label(r["gold"]) == norm_label(r["pred"])) for r in rows])
    if kind == "ft_squad":
        from app.metrics import _squad_f1, norm_squad

        return np.asarray([[_squad_f1(r["pred"], r["gold"]), float(norm_squad(r["pred"]) == norm_squad(r["gold"]))] for r in rows])
    if kind == "ft_rouge":
        from rouge_score import rouge_scorer
        from app.metrics import _WhitespaceTokenizer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_WhitespaceTokenizer())
        return np.asarray([scorer.score(r["gold"], r["pred"])["rougeL"].fmeasure for r in rows])
    from icl_tasks import entities, normalize, overlap_f1, qa_tokens, rouge_l

    task_kind, _, labels, _ = task
    if task_kind == "ner":
        types = [normalize(label) for label in labels]
        index = {label: i for i, label in enumerate(types)}
        result = np.zeros((len(rows), len(types), 3), dtype=np.int32)
        for i, row in enumerate(rows):
            gold = entities(row["gold"], labels)
            pred = entities(row["prediction"], labels) if row["valid"] else {("__invalid__", "")}
            for label, _ in gold & pred:
                if label in index:
                    result[i, index[label], 0] += 1
            for label, _ in pred - gold:
                if label in index:
                    result[i, index[label], 1] += 1
            for label, _ in gold - pred:
                if label in index:
                    result[i, index[label], 2] += 1
        return result
    if task_kind in {"intent", "nli", "mcqa", "acronym"}:
        gold = [normalize(r["gold"]) for r in rows]
        pred = [normalize(r["prediction"]) if r["valid"] else "__invalid__" for r in rows]
        fixed = [normalize(label) for label in labels] if labels else sorted(set(gold))
        label_index = {label: i for i, label in enumerate(fixed)}
        gold_ids = np.asarray([label_index.get(x, -1) for x in gold], dtype=np.int32)
        pred_ids = np.asarray([label_index.get(x, -1) for x in pred], dtype=np.int32)
        groups = [np.flatnonzero(gold_ids == i) for i in range(len(fixed)) if np.any(gold_ids == i)]
        return gold_ids, pred_ids, len(fixed), groups
    if task_kind == "extractive":
        return np.asarray([float(r["valid"] and qa_tokens(r["gold"]) == qa_tokens(r["prediction"])) for r in rows])
    return np.asarray([rouge_l(normalize(r["gold"]).split(), normalize(r["prediction"]).split()) if r["valid"] else 0.0 for r in rows])


def score_sample(components, kind: str, sample: np.ndarray, metric: str) -> float:
    values = components
    if kind in {"ft_ner", "icl_ner"}:
        counts = values[sample].sum(axis=0)
        if kind == "ft_ner" or metric == "micro_f1":
            tp, fp, fn = counts.sum(axis=0) if kind == "icl_ner" else counts
            return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        denom = 2 * counts[:, 0] + counts[:, 1] + counts[:, 2]
        return float(np.mean(np.divide(2 * counts[:, 0], denom, out=np.zeros(len(denom)), where=denom != 0)))
    if kind in {"ft_cls", "icl_cls"}:
        gold, pred, n_labels = values[:3]
        g, p = gold[sample], pred[sample]
        if metric in {"accuracy", "micro_f1", "exact_match"}:
            return float(np.mean(g == p))
        gold_count = np.bincount(g[g >= 0], minlength=n_labels)
        pred_count = np.bincount(p[p >= 0], minlength=n_labels)
        matched = g[(g == p) & (g >= 0)]
        tp = np.bincount(matched, minlength=n_labels)
        denom = gold_count + pred_count
        return float(np.mean(np.divide(2 * tp, denom, out=np.zeros(n_labels), where=denom != 0)))
    if kind == "ft_squad":
        return float(np.mean(values[sample, 1 if metric == "squad_em" else 0]))
    return float(np.mean(values[sample]))


def ci(components, kind: str, metric: str, rng: np.random.Generator, reps: int):
    n = len(components[0]) if kind in {"ft_cls", "icl_cls"} else len(components)
    full = np.arange(n)
    point = score_sample(components, kind, full, metric)
    draws = np.empty(reps)
    for i in range(reps):
        draws[i] = score_sample(components, kind, sample_indices(components, kind, metric, rng), metric)
    return point, *np.percentile(draws, [2.5, 97.5])


def sample_indices(components, kind: str, metric: str, rng: np.random.Generator) -> np.ndarray:
    n = len(components[0]) if kind in {"ft_cls", "icl_cls"} else len(components)
    # Rare acronym classes disappear in an ordinary row bootstrap, making the
    # percentile interval target a different class inventory. Preserve the
    # observed gold-class counts for macro-F1 in that one high-cardinality case.
    if kind in {"ft_cls", "icl_cls"} and metric == "macro_f1" and len(components[3]) > 20:
        return np.concatenate([rng.choice(group, len(group), replace=True) for group in components[3]])
    return rng.integers(0, n, n)


def ft_kind(dataset: str) -> tuple[str, str]:
    if "NER" in dataset and dataset != "ViMQ_intent_classification":
        return "ft_ner", "entity_micro_f1"
    if dataset in {"ViMQ_intent_classification", "vihealthbert_acrDrAid"}:
        return "ft_cls", "macro_f1"
    if dataset == "ViNewsQA_Extractive_QA":
        return "ft_squad", "squad_f1"
    if dataset in {"ViMedNLI_ViMedNLI", "VMHQA_Multiple_Choice_QA"}:
        return "ft_cls", "accuracy"
    return "ft_rouge", "rougeL"


def icl_kind(task) -> tuple[str, str]:
    kind = task[0]
    if kind == "ner":
        return "icl_ner", "macro_f1"
    if kind in {"intent", "acronym"}:
        return "icl_cls", "macro_f1"
    if kind == "nli":
        return "icl_cls", "accuracy"
    if kind == "mcqa":
        return "icl_cls", "exact_match"
    if kind == "extractive":
        return "icl_scalar", "exact_match"
    return "icl_scalar", "rouge_l_f1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-results", type=Path, required=True, help="Directory containing results_1.zip and results_2.zip")
    parser.add_argument("--finetune-code", type=Path, default=SRC / "finetuning", help="Parent directory of the app Python package")
    parser.add_argument("--icl-root", type=Path, required=True, help="Directory containing the per-dataset ICL run folders")
    parser.add_argument("--icl-code", type=Path, default=SRC / "in_context_learning/src", help="Directory containing icl_tasks.py and audit_icl.py")
    parser.add_argument("--run-dir-map", type=Path, help="Optional JSON {model: saved run folder name} when folders are not named by model")
    parser.add_argument("--table-dir", type=Path, help="Optional manuscript table directory for cross-checking printed cells")
    parser.add_argument("--out-dir", type=Path, default=REPO / "04_results")
    parser.add_argument("--bootstrap-reps", type=int, default=400)
    parser.add_argument("--check", action="store_true", help="Compare regenerated CSVs without overwriting them")
    args = parser.parse_args()
    ft_results, icl_root, out = args.finetune_results.resolve(), args.icl_root.resolve(), args.out_dir.resolve()
    table_dir = args.table_dir.resolve() if args.table_dir else None
    run_dirs = json.loads(args.run_dir_map.read_text(encoding="utf-8")) if args.run_dir_map else {}
    sys.path.insert(0, str(args.finetune_code.resolve()))
    sys.path.insert(0, str(args.icl_code.resolve()))
    from icl_tasks import TASKS, entities, evaluate, cot_compliance
    from audit_icl import classify_record, MODELS, METHODS
    from app.metrics import classification_metrics, ner_metrics, squad_metrics

    rng = np.random.default_rng(20260923)
    fine_manifest, icl_manifest, intervals, paired, errors = [], [], [], [], []
    configuration_errors, validity = [], []
    table61 = (table_dir / "table 6.1.tex").read_text(encoding="utf-8") if table_dir else None
    ft_components = {}
    archives = {}
    for dataset, archive_name in FT_SOURCES.items():
        archive_path = ft_results / archive_name
        archive = archives.setdefault(archive_name, zipfile.ZipFile(archive_path))
        prefix = f"{dataset}/"
        config_name, metrics_name, pred_name = (prefix + name for name in ("run_config.json", "metrics.json", "predictions.jsonl"))
        config, result = zip_json(archive, config_name), zip_json(archive, metrics_name)
        raw_pred = archive.read(pred_name)
        rows = [json.loads(line) for line in io.BytesIO(raw_pred) if line.strip()]
        assert len(rows) == result["metrics"]["num_eval"], dataset
        assert config["use_4bit"] is False and config["model_name"] == result["model_name"], dataset
        if table61:
            table_lines = table61.splitlines()
            position = next(i for i, line in enumerate(table_lines) if DISPLAY[dataset] in line and "\\cite" in line)
            rendered = re.findall(r"\d+\.\d+", table_lines[position].split(" & ")[-1])
            if "bertscore_f1" in FT_TABLE_METRICS[dataset]:
                assert "BERTScore-F1" in table_lines[position + 1]
                rendered += re.findall(r"\d+\.\d+", table_lines[position + 1].split(" & ")[-1])
            assert rendered == [f"{100 * result['metrics'][m]:.2f}" for m in FT_TABLE_METRICS[dataset]], dataset
        preds, golds = [r["pred"] for r in rows], [r["gold"] for r in rows]
        if ft_kind(dataset)[0] == "ft_ner":
            recomputed = ner_metrics(preds, golds)
        elif ft_kind(dataset)[0] == "ft_cls":
            recomputed = classification_metrics(preds, golds)
        elif ft_kind(dataset)[0] == "ft_squad":
            recomputed = squad_metrics(preds, golds)
        else:
            recomputed = {"rougeL": float(score_components(rows, "ft_rouge").mean())}
        for metric in FT_TABLE_METRICS[dataset]:
            if metric != "bertscore_f1":  # Saved BERTScore only; no encoder rerun or download.
                assert f"{100 * recomputed[metric]:.2f}" == f"{100 * result['metrics'][metric]:.2f}", (dataset, metric)
        fine_manifest.append({
            "dataset": dataset, "archive": archive_name, "run_config": config_name,
            "metrics": metrics_name, "predictions": pred_name,
            "run_config_sha256": sha256(archive.read(config_name)),
            "metrics_sha256": sha256(archive.read(metrics_name)),
            "predictions_sha256": sha256(raw_pred), "n_test": len(rows),
            "base_model": config["model_name"], "use_4bit": config["use_4bit"],
            "bf16": config["bf16"], "epochs": config["epochs"], "seed": config["seed"],
            "train_batch": config["per_device_train_batch_size"],
            "gradient_accumulation": config["gradient_accumulation_steps"],
            "learning_rate": config["learning_rate"], "lora_r": config["lora_r"],
            "lora_alpha": config["lora_alpha"], "lora_dropout": config["lora_dropout"],
            "max_val_samples": config["max_val_samples"],
            "table_metrics": json.dumps({m: round(100 * result["metrics"][m], 2) for m in FT_TABLE_METRICS[dataset]}),
        })
        kind, metric = ft_kind(dataset)
        components = score_components(rows, kind)
        ft_components[dataset] = components
        point, low, high = ci(components, kind, metric, rng, args.bootstrap_reps)
        # The exact scorer is still the source of truth for the displayed point estimate.
        assert abs(point - result["metrics"][metric]) < 0.0002, (dataset, point, result["metrics"][metric])
        intervals.append({"pipeline": "fine_tuning", "dataset": dataset, "model": config["model_name"],
                          "method": "LoRA", "metric": metric, "n_test_rows": len(rows),
                          "point_percent": round(100 * point, 2), "ci_low_percent": round(100 * low, 2),
                          "ci_high_percent": round(100 * high, 2), "bootstrap_reps": args.bootstrap_reps})
    for archive in archives.values():
        archive.close()

    configs = {}
    table10_scores: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    total = 0
    by_condition = defaultdict(Counter)
    by_configuration = defaultdict(Counter)
    for dataset, method, model in itertools.product(TASKS, METHODS, MODELS):  # fixed order: the bootstrap RNG is consumed in it
        folder = run_dirs.get(model, model)
        directory = icl_root / dataset / folder
        if method == "cot":
            directory = directory / "cot"
        pred_path, metrics_path, exp_path = (directory / name for name in ("predictions.jsonl", "metrics.json", "experiment.json"))
        raw = pred_path.read_bytes()
        rows = [json.loads(line) for line in io.BytesIO(raw) if line.strip()]
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiment = json.loads(exp_path.read_text(encoding="utf-8"))
        assert len(rows) == result["n_scored"]
        assert sha256(raw) == result["predictions_sha256"]
        assert all(row["experiment_id"] == result["experiment_id"] for row in rows)
        actual = evaluate(rows, TASKS[dataset], method)
        assert actual["metrics"] == result["metrics"], (dataset, model, method)
        assert experiment["config"]["dataset"] == dataset and experiment["config"]["deployment"] == folder and experiment["config"]["method"] == method
        assert len({r["row"] for r in rows}) == len(rows)
        counts = by_configuration[(dataset, model, method)]
        table10_scores[dataset][method][model] = result["metrics"]
        total += len(rows)
        for row in rows:
            counts.update(classify_record(row, TASKS[dataset]))
            if TASKS[dataset][0] == "ner" and row["valid"]:
                gold = entities(row["gold"], TASKS[dataset][2])
                pred = entities(row["prediction"], TASKS[dataset][2])
                missed, spurious = gold - pred, pred - gold
                counts["gold_entity_pairs"] += len(gold)
                counts["predicted_entity_pairs"] += len(pred)
                counts["same_type_mismatch_candidates"] += sum(
                    glabel == plabel for glabel, _ in missed for plabel, _ in spurious
                )
        by_condition[(dataset, method)].update(counts)
        assert counts["invalid"] == result["n_invalid"]
        assert counts["truncated"] == result["n_truncated"]
        compliant = 0
        if method == "cot":
            for row in rows:
                compliance = row.get("finish_reason") == "stop" and cot_compliance(row["raw_prediction"])
                assert compliance == row["cot_format_compliant"]
                compliant += compliance
            assert compliant == result["cot_format"]["n_compliant"]
        validity.append({"dataset": dataset, "model": model, "condition": CONDITIONS[method], "n_predictions": len(rows),
                         "n_invalid": counts["invalid"], "invalid_rate_percent": round(100 * counts["invalid"] / len(rows), 2),
                         "n_truncated": counts["truncated"], "n_format_compliant": compliant if method == "cot" else "NA",
                         "format_compliance_percent": round(100 * compliant / len(rows), 2) if method == "cot" else "NA"})
        configuration_errors.extend(error_rows(dataset, model, method, counts, TASKS[dataset][0]))
        icl_manifest.append({"dataset": dataset, "model": model, "model_version": MODEL_VERSIONS[model], "method": CONDITIONS[method],
                             "n_test": len(rows), "n_invalid": result["n_invalid"],
                             "n_truncated": result["n_truncated"], "predictions_sha256": sha256(raw),
                             "experiment_sha256": sha256(exp_path.read_bytes()),
                             "metrics_sha256": sha256(metrics_path.read_bytes()),
                             "seed": experiment["config"]["seed"],
                             "shots": experiment["config"]["shots"],
                             "decoding": json.dumps(experiment["config"]["decoding"], sort_keys=True),
                             "sdk_version": experiment["config"]["sdk_version"],
                             "prompt_fingerprint": experiment["fingerprints"]["prompt"],
                             "examples_fingerprint": experiment["fingerprints"]["examples"],
                             "experiment_id": result["experiment_id"]})
        kind, metric = icl_kind(TASKS[dataset])
        components = score_components(rows, kind, TASKS[dataset])
        point, low, high = ci(components, kind, metric, rng, args.bootstrap_reps)
        assert abs(100 * point - result["metrics"][metric]) < 0.02, (dataset, model, method, point)
        intervals.append({"pipeline": "ICL", "dataset": dataset, "model": model,
                          "method": CONDITIONS[method], "metric": metric, "n_test_rows": len(rows),
                          "point_percent": round(100 * point, 2), "ci_low_percent": round(100 * low, 2),
                          "ci_high_percent": round(100 * high, 2), "bootstrap_reps": args.bootstrap_reps})
        configs[(dataset, model, method)] = (rows, components, kind, metric)
    assert len(configs) == 80 and total == 171344
    if table_dir:
        check_icl_table((table_dir / "table 10.tex").read_text(encoding="utf-8"), table10_scores)
    for dataset in TASKS:
        dataset_runs = [r for r in icl_manifest if r["dataset"] == dataset]
        assert len({r["examples_fingerprint"] for r in dataset_runs}) == 1, (dataset, "demonstrations differ")
        reference_rows = configs[(dataset, MODELS[0], "direct")][0]
        for model in MODELS:
            check_pair(reference_rows, configs[(dataset, model, "direct")][0])
        for model in sorted({item["model"] for item in icl_manifest}):
            direct = configs[(dataset, model, "direct")]
            cot = configs[(dataset, model, "cot")]
            dr, dc, kind, metric = direct
            cr, cc, _, _ = cot
            check_pair(dr, cr)
            n = len(dr)
            full = np.arange(n)
            difference = 100 * (score_sample(cc, kind, full, metric) - score_sample(dc, kind, full, metric))
            draws = np.empty(args.bootstrap_reps)
            for i in range(args.bootstrap_reps):
                sample = sample_indices(dc, kind, metric, rng)
                draws[i] = 100 * (score_sample(cc, kind, sample, metric) - score_sample(dc, kind, sample, metric))
            low, high = np.percentile(draws, [2.5, 97.5])
            paired.append({"dataset": dataset, "model": model, "metric": metric,
                           "n_paired_test_rows": n, "explanation_minus_answer_only_points": round(difference, 2),
                           "ci_low_points": round(low, 2), "ci_high_points": round(high, 2),
                           "bootstrap_reps": args.bootstrap_reps})
        aggregate = Counter()
        for method in ("direct", "cot"):
            aggregate.update(by_condition[(dataset, method)])
        for method in ("direct", "cot", "pooled"):
            counts = aggregate if method == "pooled" else by_condition[(dataset, method)]
            for category in ("invalid", "incorrect", "missed_gold_entities", "spurious_predicted_entities",
                             "boundary_overlap_mismatch", "partial_overlap", "zero_overlap", "low_rouge"):
                if category in counts:
                    if category == "invalid":
                        denominator = counts["records"]
                    elif category == "missed_gold_entities":
                        denominator = counts["gold_entity_pairs"]
                    elif category == "spurious_predicted_entities":
                        denominator = counts["predicted_entity_pairs"]
                    elif category == "boundary_overlap_mismatch":
                        denominator = counts["same_type_mismatch_candidates"]
                    else:
                        denominator = counts["records"] - counts["invalid"]
                    errors.append({"dataset": dataset, "condition": CONDITIONS.get(method, method), "category": category,
                                   "count": counts[category], "denominator": denominator,
                                   "rate_percent": round(100*counts[category]/denominator, 2) if denominator else "",
                                   "rule": "audit_icl.py:classify_record"})
    write_csv(out / "fine_tuning_manifest.csv", fine_manifest, list(fine_manifest[0]), check=args.check)
    write_csv(out / "icl_run_manifest.csv", icl_manifest, list(icl_manifest[0]), check=args.check)
    write_csv(out / "bootstrap_intervals.csv", intervals, list(intervals[0]), check=args.check)
    write_csv(out / "paired_prompt_differences.csv", paired, list(paired[0]), check=args.check)
    write_csv(out / "error_counts_by_condition.csv", errors, list(errors[0]), check=args.check)
    write_csv(out / "error_counts_by_configuration.csv", configuration_errors, list(configuration_errors[0]), check=args.check)
    write_csv(out / "output_validity_by_configuration.csv", validity, list(validity[0]), check=args.check)
    coverage = coverage_rows(validity)
    write_csv(out / "dataset_output_coverage.csv", coverage, list(coverage[0]), check=args.check)
    if table_dir:
        table7 = (table_dir / "table 7.audit.tex").read_text(encoding="utf-8")
        for row in errors:
            if row["condition"] == "pooled" and row["count"] in {1902, 7719, 3914, 13121, 23940, 10883, 21205, 47369, 734, 6164, 2774}:
                assert f'{row["count"]:,} / {row["denominator"]:,}' in table7, row
    case_specs = [
        ("E1", "PhoNER_COVID19_NER", "gpt-4o", 63, "direct+cot", "prompt_condition_disagreement"),
        ("E2", "ViMedNER_NER", "gpt-4o", 52, "direct+cot", "prompt_condition_disagreement"),
        ("E3", "ViMedNER_NER", "gpt-5.2", 1, "cot", "additional_representative_task_error"),
        ("E4", "UIT-ViCoV19QA_QA", "gpt-4.1", 209, "direct+cot", "prompt_condition_disagreement"),
    ]
    case_trace = []
    for case_id, dataset, model, row_number, methods, rule in case_specs:
        selected = [configs[(dataset, model, method)][0][row_number] for method in methods.split("+")]
        assert all(row["row"] == row_number for row in selected)
        assert len({row["test_hash"] for row in selected}) == 1
        case_trace.append({"case_id": case_id, "dataset": dataset, "model": model,
                           "row": row_number, "test_hash": selected[0]["test_hash"],
                           "methods": "+".join(CONDITIONS[m] for m in methods.split("+")),
                           "selection_rule": rule})
    write_csv(out / "case_trace.csv", case_trace, list(case_trace[0]), check=args.check)
    print(f"Verified {len(fine_manifest)} fine-tuning tasks, {len(icl_manifest)} ICL runs and {total} ICL predictions")


if __name__ == "__main__":
    main()
