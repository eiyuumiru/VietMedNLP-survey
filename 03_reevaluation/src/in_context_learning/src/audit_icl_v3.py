"""Audit the official ICL-v3 runs and generate manuscript-ready artifacts.

The authoritative run universe is ``summary.json`` (80 runs), not every
directory below the results folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from icl_config import BATCH_METHODS as METHODS, MODELS, MODEL_LABELS
from icl_tasks import TASKS, entities, normalize, overlap_f1, qa_tokens, rouge_l


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "all_models_icl-v3"
ANALYSIS = ROOT / "analysis"
DATASETS = tuple(TASKS)

DISPLAY = {
    "ViMQ_intent_classification": ("ViMQ", "Intent\\\\ Macro/Micro F1", ("macro_f1", "micro_f1")),
    "ViMedNLI_ViMedNLI": ("ViMedNLI", "NLI\\\\ Accuracy", ("accuracy",)),
    "VMHQA_Multiple_Choice_QA": ("VMHQA", "Mixed-format QA\\\\ EM/token F1", ("exact_match", "token_f1")),
    "PhoNER_COVID19_NER": ("PhoNER\\_COVID19", "NER\\\\ Macro/Micro F1", ("macro_f1", "micro_f1")),
    "ViMedNER_NER": ("ViMedNER", "NER\\\\ Macro/Micro F1", ("macro_f1", "micro_f1")),
    "vihealthbert_acrDrAid": ("acrDrAid", "Acronym\\\\ Macro F1", ("macro_f1",)),
    "ViNewsQA_Extractive_QA": ("UIT-ViNewsQA", "Extractive QA\\\\ EM/token F1", ("exact_match", "token_f1")),
    "UIT-ViCoV19QA_QA": ("UIT-ViCoV19QA", "Generative QA\\\\ ROUGE-L", ("rouge_l_f1",)),
    "ViMedAQA_Abstract_QA": ("ViMedAQA", "Abstractive QA\\\\ ROUGE-L", ("rouge_l_f1",)),
    "vihealthbert_Summarization": ("FAQSum", "Summarization\\\\ ROUGE-L", ("rouge_l_f1",)),
}

# Published figures already cited in the manuscript's benchmark tables. These
# are descriptive comparison anchors, not paired baselines.
PUBLISHED_BASELINES = {
    "ViMQ_intent_classification": {"micro_f1": 90.65, "macro_f1": 91.65, "citation": "Huy_2021", "note": "Published self-supervised PhoBERT result."},
    "ViMedNLI_ViMedNLI": {"accuracy": 81.65, "citation": "phan-etal-2023-enriching", "note": "Published ViPubmedT5 result."},
    "VMHQA_Multiple_Choice_QA": {"exact_match": 91.50, "citation": "nguyen_enhancing_2025", "note": "Published fine-tuned Qwen-7B accuracy; not a matched ICL baseline."},
    "PhoNER_COVID19_NER": {"micro_f1": 94.76, "macro_f1": 93.18, "citation": "tran-tien-etal-2023-vipubmeddeberta", "note": "Published ViPubmedDeBERTa result."},
    "ViMedNER_NER": {"micro_f1": 72.50, "macro_f1": 64.00, "citation": "Duong_Trinh_Nguyen_Vu_Pham_Tuan_Son_2024", "note": "Published XLM-R result."},
    "vihealthbert_acrDrAid": {"macro_f1": 89.04, "citation": "phan-etal-2023-enriching", "note": "Published ViPubmedT5 result."},
    "ViNewsQA_Extractive_QA": {"exact_match": 76.46, "token_f1": 91.84, "citation": "nguyen2023multi", "note": "MRC result with evidence extraction; input/system conditions differ."},
    "UIT-ViCoV19QA_QA": {"rouge_l_f1": 33.95, "citation": "thai-etal-2022-uit", "note": "Published RNN-2 result."},
    "ViMedAQA_Abstract_QA": {"rouge_l_f1": 59.89, "citation": "tran2024vimedaqa", "note": "Published VinaLLaMA-7B result."},
    "vihealthbert_Summarization": {"rouge_l_f1": 61.30, "citation": "phan-etal-2023-enriching", "note": "Published ViT5 result."},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def experiment_directory(run: dict[str, Any], results_root: Path = RESULTS) -> Path:
    directory = results_root / run["dataset"] / run["model"]
    return directory / "cot" if run["method"] == "cot" else directory


def score_record(record: dict[str, Any], task: tuple[Any, ...]) -> float:
    """Return a per-record task score for direct/CoT case selection."""
    kind, _, labels, _ = task
    if not record["valid"]:
        return 0.0
    gold, prediction = record["gold"], record["prediction"]
    if kind == "ner":
        gold_entities, predicted_entities = entities(gold, labels), entities(prediction, labels)
        return 2 * len(gold_entities & predicted_entities) / (len(gold_entities) + len(predicted_entities)) if gold_entities or predicted_entities else 1.0
    if kind in {"intent", "nli", "mcqa", "acronym"}:
        return float(normalize(gold) == normalize(prediction))
    if kind == "extractive":
        return overlap_f1(qa_tokens(gold), qa_tokens(prediction))
    return rouge_l(normalize(gold).split(), normalize(prediction).split())


def classify_record(record: dict[str, Any], task: tuple[Any, ...]) -> Counter:
    """Classify one prediction using only the evaluator-visible gold/prediction."""
    kind, _, labels, _ = task
    counts: Counter = Counter(records=1, invalid=not record["valid"], truncated=record.get("finish_reason") == "length")
    if not record["valid"]:
        counts["format_or_schema_failure"] += 1
        return counts
    gold, prediction = record["gold"], record["prediction"]
    if kind in {"intent", "nli", "mcqa", "acronym"}:
        counts["correct" if normalize(gold) == normalize(prediction) else "incorrect"] += 1
        counts[f"confusion::{normalize(gold)}->{normalize(prediction)}"] += 1
    elif kind == "ner":
        gold_entities, predicted_entities = entities(gold, labels), entities(prediction, labels)
        exact = gold_entities & predicted_entities
        missed, spurious = gold_entities - predicted_entities, predicted_entities - gold_entities
        counts["exact_entity_pairs"] += len(exact)
        counts["missed_gold_entities"] += len(missed)
        counts["spurious_predicted_entities"] += len(spurious)
        for gold_label, gold_span in missed:
            for predicted_label, predicted_span in spurious:
                if gold_span == predicted_span and gold_label != predicted_label:
                    counts["label_mismatch_same_span"] += 1
                elif gold_label == predicted_label and set(gold_span.split()) & set(predicted_span.split()):
                    counts["boundary_overlap_mismatch"] += 1
    elif kind == "extractive":
        gold_tokens, predicted_tokens = qa_tokens(gold), qa_tokens(prediction)
        f1 = overlap_f1(gold_tokens, predicted_tokens)
        counts["exact_match" if gold_tokens == predicted_tokens else "partial_overlap" if f1 else "zero_overlap"] += 1
        ratio = len(predicted_tokens) / max(1, len(gold_tokens))
        counts["under_extraction" if ratio < 0.5 else "over_extraction" if ratio > 1.5 else "similar_length"] += 1
    else:
        gold_tokens, predicted_tokens = normalize(gold).split(), normalize(prediction).split()
        score = rouge_l(gold_tokens, predicted_tokens)
        counts["high_rouge" if score >= 0.8 else "partial_rouge" if score >= 0.3 else "low_rouge"] += 1
        ratio = len(predicted_tokens) / max(1, len(gold_tokens))
        counts["shorter_than_reference" if ratio < 0.5 else "longer_than_reference" if ratio > 1.5 else "similar_reference_length"] += 1
        counts["added_reference_mismatch"] += bool(set(predicted_tokens) - set(gold_tokens))
        counts["missing_reference_content"] += bool(set(gold_tokens) - set(predicted_tokens))
    return counts


def serializable_case(record: dict[str, Any], source: Path, source_root: Path, dataset: str, model: str, method: str, category: str) -> dict[str, Any]:
    return {
        "category": category,
        "dataset": dataset,
        "model": MODEL_LABELS[model],
        "model_family": model,
        "method": method,
        "row": record["row"],
        "test_hash": record["test_hash"],
        "gold": record["gold"],
        "prediction": record["prediction"],
        "valid": record["valid"],
        "raw_prediction": record["raw_prediction"],
        "source": str(source.relative_to(source_root)),
    }


def choose_case(current: tuple[float, dict[str, Any]] | None, value: float, case: dict[str, Any], prefer_high: bool) -> tuple[float, dict[str, Any]]:
    if current is None or (value > current[0] if prefer_high else value < current[0]):
        return value, case
    return current


def retain_low_score(candidates: list[tuple[float, dict[str, Any]]], value: float, case: dict[str, Any], limit: int = 8) -> None:
    """Keep a small deterministic pool of distinct valid task errors."""
    identity = (case["model_family"], case["method"], case["row"])
    if any((item[1]["model_family"], item[1]["method"], item[1]["row"]) == identity for item in candidates):
        return
    candidates.append((value, case))
    candidates.sort(key=lambda item: (item[0], item[1]["row"], item[1]["model_family"], item[1]["method"]))
    del candidates[limit:]


def build_table(all_row_metrics: dict[str, Any]) -> str:
    lines = [
        "% Generated by in_context_learning/src/audit_icl_v3.py; do not hand-edit.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Few-shot ICL results on ten Vietnamese medical NLP datasets. Scores are on a 0--100 scale and include invalid or truncated outputs as zero. Boldface indicates the highest score for each dataset and metric among the eight configurations.}",
        "\\label{tab:icl_results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{ll*{8}{c}}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{GPT-4o} & \\multicolumn{2}{c}{GPT-4.1} & \\multicolumn{2}{c}{GPT-5.2} & \\multicolumn{2}{c}{GPT-5.4} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
        "\\textbf{Dataset} & \\textbf{Task / metric} & " + " & ".join([r"\makecell{Answer-only\\prompting}", r"\makecell{Explanation-\\and-answer\\prompting}"] * 4) + r" \\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        name, task_metric, metric_names = DISPLAY[dataset]
        scores = {
            (model, method): all_row_metrics[dataset][method][model]
            for model in MODELS for method in METHODS
        }
        maxima = {metric: max(value[metric] for value in scores.values()) for metric in metric_names}
        cells = []
        for model in MODELS:
            for method in METHODS:
                values = []
                for metric in metric_names:
                    number = scores[(model, method)][metric]
                    rendered = f"{number:.2f}"
                    values.append(f"\\textbf{{{rendered}}}" if number == maxima[metric] else rendered)
                cells.append(" / ".join(values))
        lines.append(f"{name} & \\makecell[l]{{{task_metric}}} & " + " & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{minipage}{\\textwidth}",
        "\\vspace{2pt}\\footnotesize\\textit{Note:} Both conditions use the same demonstrations. EM denotes exact match. NER matches unique entity type--text pairs; VMHQA demonstrations mix option letters and free-text answers. Section~\\ref{sec:instruction_tuning} defines local scoring.",
        "\\end{minipage}",
        "\\end{table*}",
        "",
    ])
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS)
    args = parser.parse_args(argv)
    results_root = args.results_root.expanduser()
    analysis_root = args.analysis_root.expanduser()

    summary = read_json(results_root / "summary.json")
    if len(summary) != 80:
        raise ValueError(f"Expected 80 official runs, found {len(summary)}")
    expected = {(dataset, model, method) for dataset in DATASETS for model in MODELS for method in METHODS}
    actual = {(run["dataset"], run["model"], run["method"]) for run in summary}
    if actual != expected:
        raise ValueError("summary.json does not contain exactly the expected 10 x 4 x 2 run universe")

    analysis_root.mkdir(parents=True, exist_ok=True)
    all_row_metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for run in summary:
        all_row_metrics[run["dataset"]][run["method"]][run["model"]] = run["metrics"]
    latex_table = analysis_root / "table 10.tex"
    latex_table.write_text(build_table(all_row_metrics), encoding="utf-8")

    aggregate: dict[str, Counter] = defaultdict(Counter)
    valid_metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    registry, cases = [], []
    direct_records: dict[tuple[str, str, int], tuple[dict[str, Any], Path]] = {}
    dataset_candidates: dict[str, dict[str, Any]] = defaultdict(dict)

    for run in sorted(summary, key=lambda item: (DATASETS.index(item["dataset"]), METHODS.index(item["method"]), MODELS.index(item["model"]))):
        directory = experiment_directory(run, results_root)
        metrics_path, experiment_path, predictions_path = (directory / "metrics.json", directory / "experiment.json", directory / "predictions.jsonl")
        metrics, experiment = read_json(metrics_path), read_json(experiment_path)
        config = experiment["config"]
        if config["dataset"] != run["dataset"] or config.get("model_family") != run["model"] or config["method"] != run["method"]:
            raise ValueError(f"Manifest mismatch for {directory}")
        if metrics["metrics"] != run["metrics"]:
            raise ValueError(f"Summary metric mismatch for {directory}")
        valid_metrics[run["dataset"]][run["method"]][run["model"]] = metrics["valid_only_metrics"]["metrics"]

        task = TASKS[run["dataset"]]
        counts: Counter = Counter()
        prediction_count = 0
        for record in read_jsonl(predictions_path):
            prediction_count += 1
            classified = classify_record(record, task)
            counts.update(classified)
            aggregate[run["dataset"]].update(classified)
            current_case = serializable_case(record, predictions_path, results_root, run["dataset"], run["model"], run["method"], "")
            score = score_record(record, task)
            candidates = dataset_candidates[run["dataset"]]
            if not record["valid"]:
                candidates["format"] = choose_case(candidates.get("format"), -record["row"], current_case | {"category": "format_or_schema_failure"}, True)
            elif score < 1:
                candidates["worst"] = choose_case(candidates.get("worst"), score, current_case | {"category": "representative_task_error"}, False)
                fallbacks = candidates.setdefault("fallbacks", [])
                retain_low_score(fallbacks, score, current_case | {"category": "additional_representative_task_error"})
            key = (run["dataset"], run["model"], record["row"])
            if run["method"] == "direct":
                direct_records[key] = (record, predictions_path)
            else:
                direct = direct_records.get(key)
                if direct:
                    direct_record, direct_path = direct
                    delta = abs(score - score_record(direct_record, task))
                    pair = {
                        "category": "direct_cot_disagreement",
                        "dataset": run["dataset"],
                        "model": MODEL_LABELS[run["model"]],
                        "model_family": run["model"],
                        "row": record["row"],
                        "test_hash": record["test_hash"],
                        "direct": serializable_case(direct_record, direct_path, results_root, run["dataset"], run["model"], "direct", "direct"),
                        "cot": serializable_case(record, predictions_path, results_root, run["dataset"], run["model"], "cot", "cot"),
                        "absolute_per_record_score_delta": round(delta, 6),
                    }
                    candidates["contrast"] = choose_case(candidates.get("contrast"), delta, pair, True)
        if prediction_count != run["n_scored"]:
            raise ValueError(f"Prediction count mismatch for {directory}: {prediction_count} != {run['n_scored']}")
        registry.append({
            "dataset": run["dataset"], "model_family": run["model"], "model": MODEL_LABELS[run["model"]], "method": run["method"],
            "n_scored": run["n_scored"], "n_invalid": run["n_invalid"], "n_truncated": run["n_truncated"],
            "valid_only_metrics": metrics["valid_only_metrics"]["metrics"], "prediction_file": str(predictions_path.relative_to(results_root)),
            "experiment_id": metrics["experiment_id"], "cot_format": run["cot_format"],
        })

    for dataset in DATASETS:
        candidates = dataset_candidates[dataset]
        selected = [candidates.get("format"), candidates.get("contrast"), candidates.get("worst")]
        selected_identities: set[tuple[Any, ...]] = set()
        for candidate in selected:
            if candidate is not None:
                case = candidate[1]
                if case["category"] == "direct_cot_disagreement":
                    identity = (case["model_family"], case["row"])
                else:
                    identity = (case["model_family"], case["method"], case["row"])
                if identity not in selected_identities:
                    cases.append(case)
                    selected_identities.add(identity)
        # Datasets without invalid outputs still receive three reviewed cases;
        # duplicate-free fallbacks are chosen deterministically from low-score cases.
        for _, fallback in candidates.get("fallbacks", []):
            if sum(case["dataset"] == dataset for case in cases) >= 3:
                break
            identity = (fallback["model_family"], fallback["method"], fallback["row"])
            if identity not in selected_identities:
                cases.append(fallback)
                selected_identities.add(identity)

    published_comparison = []
    for dataset in DATASETS:
        for metric, baseline in PUBLISHED_BASELINES[dataset].items():
            if metric in {"citation", "note"}:
                continue
            best = max(
                valid_metrics[dataset][method][model][metric]
                for model in MODELS for method in METHODS
            )
            published_comparison.append({
                "dataset": dataset, "metric": metric, "best_icl_valid_only": best,
                "published_score": baseline, "difference_points": round(best - baseline, 4),
                "citation": PUBLISHED_BASELINES[dataset]["citation"], "comparison_boundary": PUBLISHED_BASELINES[dataset]["note"],
            })

    audit = {
        "protocol": "icl-v3", "official_run_count": len(registry), "datasets": list(DATASETS),
        "models": {model: MODEL_LABELS[model] for model in MODELS}, "methods": list(METHODS),
        "total_predictions": sum(item["n_scored"] for item in registry),
        "run_registry": registry,
        "aggregate_error_counts_by_dataset": {dataset: dict(aggregate[dataset]) for dataset in DATASETS},
        "published_score_comparison": published_comparison,
        "scope_note": "Counts use only the 80 runs in summary.json. Other result folders are not counted.",
        "interpretation_note": "Published scores are descriptive anchors from unmatched protocols; numerical differences are not causal or SOTA claims.",
    }
    valid_metrics_path = analysis_root / "valid_only_metrics.json"
    valid_metrics_path.write_text(json.dumps(valid_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (analysis_root / "icl_v3_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (analysis_root / "icl_v3_case_review.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    with (analysis_root / "icl_v3_run_registry.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["dataset", "model", "model_family", "method", "n_scored", "n_invalid", "n_truncated", "prediction_file", "experiment_id"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in registry:
            writer.writerow({field: item[field] for field in fields})
    manifest = {
        "bundle": "ICL-v3 audit for ACM Sections 6.2 and 7",
        "official_runs": len(registry),
        "total_predictions": audit["total_predictions"],
        "score_source": str(valid_metrics_path.relative_to(analysis_root)),
        "score_source_sha256": hashlib.sha256(valid_metrics_path.read_bytes()).hexdigest(),
        "included_files": [
            "valid_only_metrics.json",
            "icl_v3_audit.json",
            "icl_v3_case_review.json",
            "icl_v3_run_registry.csv",
            "table 10.tex",
        ],
        "scope": "Only the 80 official runs listed in summary.json.",
    }
    (analysis_root / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audited {len(registry)} official runs and {audit['total_predictions']} predictions.")
    print(f"Wrote {latex_table}, {analysis_root} artifacts, and {len(cases)} review cases.")


if __name__ == "__main__":
    main()
