"""Add BERTScore (semantic similarity) to generation-task results — fully offline.

BERTScore only needs (prediction, reference) pairs, which are already saved in each
results/<dataset>/predictions.jsonl during evaluation. So this recomputes nothing from
the trained model: no re-training, no re-generation. It loads its own multilingual
embedding model, scores the saved pairs, and writes "bertscore_f1" into metrics.json.

Run after training, before aggregate:
    python -m app.score_bertscore --results-root /content/results
Then re-run `python -m app.aggregate ...` so the column appears in the summary.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from .config import get_spec
from .utils import LOGGER, load_json, save_json


def _is_generation(stem: str, data: dict) -> bool:
    tt = data.get("task_type")
    if tt:
        return tt == "generation"
    try:
        return get_spec(stem).task_type == "generation"
    except Exception:
        return False


def score_one(metrics_path: str, lang: str, batch_size: int) -> tuple[str, float] | None:
    data = load_json(metrics_path)
    folder = os.path.dirname(metrics_path)
    stem = data.get("dataset") or os.path.basename(folder)
    if not _is_generation(stem, data):
        return None
    pred_path = os.path.join(folder, "predictions.jsonl")
    if not os.path.exists(pred_path):
        LOGGER.warning("%s: no predictions.jsonl, skipping bertscore.", stem)
        return None

    rows = [json.loads(line) for line in open(pred_path, encoding="utf-8")]
    preds = [str(r.get("pred", "")) for r in rows]
    golds = [str(r.get("gold", "")) for r in rows]

    from bert_score import score as bert_score

    _, _, f1 = bert_score(preds, golds, lang=lang, batch_size=batch_size,
                          rescale_with_baseline=False)
    val = float(f1.mean().item())
    data.setdefault("metrics", {})["bertscore_f1"] = val
    save_json(data, metrics_path)
    LOGGER.info("%s: bertscore_f1=%.4f (n=%d)", stem, val, len(rows))
    return stem, val


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Add BERTScore to generation-task results.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--lang", default="vi")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    scored = 0
    for mp in sorted(glob.glob(os.path.join(args.results_root, "*", "metrics.json"))):
        if score_one(mp, args.lang, args.batch_size):
            scored += 1
    LOGGER.info("Scored %d generation dataset(s). Re-run app.aggregate to include it.", scored)
    if scored == 0:
        LOGGER.warning("No generation results found under %s.", args.results_root)


if __name__ == "__main__":
    main()
