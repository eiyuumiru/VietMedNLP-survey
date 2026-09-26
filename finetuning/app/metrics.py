"""Task-appropriate metrics computed from generated text vs. gold strings.

Dispatched by task_type (see config.PRIMARY_METRIC):
  - ner            -> entity-set Micro/Macro F1 (parses list-like outputs)
  - classification -> accuracy + Macro/Micro/Weighted F1 (lenient label matching)
  - nli / mcqa     -> same as classification (primary metric differs)
  - extractive_qa  -> SQuAD-style Exact-Match + token F1
  - generation     -> ROUGE-1/2/L (+ BLEU, optional BERTScore)

All text comparisons are diacritic-preserving (important for Vietnamese): ROUGE uses a
whitespace tokenizer instead of rouge_score's default ASCII tokenizer.
"""

from __future__ import annotations

import ast
import re
import string
from collections import Counter

from .config import EMPTY_TARGET_SENTINELS


# --------------------------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def norm_label(s: str) -> str:
    """Normalize a classification label: lowercase, strip quotes/brackets/punct, collapse ws."""
    s = (s or "").strip().lower()
    s = s.strip("[]()\"' ")
    # if the model answered like "label: entailment" keep the part after the last colon
    if ":" in s and len(s.split(":")[-1].strip()) > 0:
        s = s.split(":")[-1].strip()
    s = s.translate(_PUNCT_TABLE)
    s = _WS.sub(" ", s).strip()
    return s


def norm_squad(s: str) -> str:
    """SQuAD normalization (no English article removal; keep Vietnamese intact)."""
    s = (s or "").lower()
    s = s.translate(_PUNCT_TABLE)
    s = _WS.sub(" ", s).strip()
    return s


# Split NER output into mentions: on newline/semicolon, or on a comma that is
# immediately followed by a new "TYPE:" label (so commas inside a name are kept).
_NER_SPLIT = re.compile(r"\s*[;\n]\s*|,\s*(?=[A-Za-zÀ-ỹ][\wÀ-ỹ ]*:)")


def parse_entity_list(s: str) -> list[str]:
    """Parse an NER output into a list of normalized entity mentions.

    The dataset uses the format ``"Loại thực thể: Tên thực thể"`` (TYPE: name), with
    several mentions separated by newline/semicolon or ``", "`` before a new TYPE; an
    empty answer is the literal string ``None``. Legacy Python list/dict outputs are also
    handled. Returns a lower-cased list; ``None``/empty sentinels yield ``[]``.
    """
    s = (s or "").strip()
    if s.lower() in EMPTY_TARGET_SENTINELS:
        return []
    # Legacy list/dict literal, e.g. "['a','b']" or "{'TYPE': ['a','b']}"
    if s[:1] in "[{":
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, set)):
                return [str(x).strip().lower() for x in v
                        if str(x).strip().lower() not in EMPTY_TARGET_SENTINELS]
            if isinstance(v, dict):
                out: list[str] = []
                for k, val in v.items():
                    items = val if isinstance(val, (list, tuple, set)) else [val]
                    for x in items:
                        xs = str(x).strip().lower()
                        if xs and xs not in EMPTY_TARGET_SENTINELS:
                            out.append(f"{str(k).strip().lower()}: {xs}")
                return out
        except Exception:
            pass
    # "TYPE: name" mentions (the dataset's standard NER format)
    out = []
    for piece in _NER_SPLIT.split(s):
        piece = piece.strip().strip("'\"").strip()
        if piece and piece.lower() not in EMPTY_TARGET_SENTINELS:
            out.append(piece.lower())
    return out


# --------------------------------------------------------------------------------------
# Classification (also NLI / MCQA)
# --------------------------------------------------------------------------------------
def classification_metrics(preds: list[str], golds: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    gold_norm = [norm_label(g) for g in golds]
    label_set = sorted(set(gold_norm))
    label_set_by_len = sorted(label_set, key=len, reverse=True)

    def map_pred(p: str) -> str:
        pn = norm_label(p)
        if pn in label_set:
            return pn
        # Word-boundary containment, only when exactly ONE distinct label matches
        # (avoids crediting a negated/embedded label from a rambling answer).
        hits = [lbl for lbl in label_set_by_len
                if lbl and re.search(r"(?<!\w)" + re.escape(lbl) + r"(?!\w)", pn)]
        if len(set(hits)) == 1:
            return hits[0]
        return "<other>"  # out-of-set -> counted as wrong (NOT excluded from averaging)

    pred_norm = [map_pred(p) for p in preds]
    # NOTE: we intentionally do NOT pass labels=label_set, so an out-of-set prediction
    # ("<other>") is scored as a real error. This keeps micro_f1 == accuracy and prevents
    # inflated precision. (macro_f1 averages over gold labels plus the zero-support
    # "<other>" class when present, which is the intended penalty for invalid outputs.)
    return {
        "accuracy": float(accuracy_score(gold_norm, pred_norm)),
        "macro_f1": float(f1_score(gold_norm, pred_norm, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(gold_norm, pred_norm, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(gold_norm, pred_norm, average="weighted", zero_division=0)),
        "num_labels": len(label_set),
    }


# --------------------------------------------------------------------------------------
# NER (entity-set matching)
# --------------------------------------------------------------------------------------
def ner_metrics(preds: list[str], golds: list[str]) -> dict:
    tp = fp = fn = 0
    per_example_f1 = []
    for p, g in zip(preds, golds):
        pset = {e.lower() for e in parse_entity_list(p)}
        gset = {e.lower() for e in parse_entity_list(g)}
        etp = len(pset & gset)
        efp = len(pset - gset)
        efn = len(gset - pset)
        tp += etp
        fp += efp
        fn += efn
        denom = (2 * etp + efp + efn)
        per_example_f1.append((2 * etp / denom) if denom else 1.0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    macro_f1 = sum(per_example_f1) / len(per_example_f1) if per_example_f1 else 0.0
    return {
        "entity_precision": float(precision),
        "entity_recall": float(recall),
        "entity_micro_f1": float(micro_f1),
        "entity_macro_f1": float(macro_f1),
    }


# --------------------------------------------------------------------------------------
# Extractive QA (SQuAD EM + token F1)
# --------------------------------------------------------------------------------------
def _squad_f1(pred: str, gold: str) -> float:
    p_tokens = norm_squad(pred).split()
    g_tokens = norm_squad(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_metrics(preds: list[str], golds: list[str]) -> dict:
    em = sum(norm_squad(p) == norm_squad(g) for p, g in zip(preds, golds))
    f1 = sum(_squad_f1(p, g) for p, g in zip(preds, golds))
    n = max(len(golds), 1)
    return {"squad_em": float(em / n), "squad_f1": float(f1 / n)}


# --------------------------------------------------------------------------------------
# Generation (ROUGE + BLEU + optional BERTScore)
# --------------------------------------------------------------------------------------
class _WhitespaceTokenizer:
    """Diacritic-preserving tokenizer for rouge_score (lowercase + whitespace split)."""

    def tokenize(self, text: str) -> list[str]:
        return (text or "").lower().split()


def generation_metrics(preds: list[str], golds: list[str], use_bertscore: bool = False) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=_WhitespaceTokenizer()
    )
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, g in zip(preds, golds):
        scores = scorer.score(g or "", p or "")
        for k in agg:
            agg[k] += scores[k].fmeasure
    n = max(len(golds), 1)
    out = {k: float(v / n) for k, v in agg.items()}

    try:
        import sacrebleu

        out["bleu"] = float(sacrebleu.corpus_bleu(preds, [golds]).score)  # 0-100 scale
    except Exception:
        out["bleu"] = None

    if use_bertscore:
        try:
            from bert_score import score as bert_score

            _, _, f1 = bert_score(preds, golds, lang="vi", rescale_with_baseline=False)
            out["bertscore_f1"] = float(f1.mean().item())
        except Exception:
            out["bertscore_f1"] = None
    return out


# --------------------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------------------
def compute_metrics(task_type: str, preds: list[str], golds: list[str],
                    use_bertscore: bool = False) -> dict:
    if task_type == "ner":
        return ner_metrics(preds, golds)
    if task_type in ("classification", "nli", "mcqa"):
        return classification_metrics(preds, golds)
    if task_type == "extractive_qa":
        return squad_metrics(preds, golds)
    if task_type == "generation":
        return generation_metrics(preds, golds, use_bertscore=use_bertscore)
    raise ValueError(f"Unknown task_type: {task_type}")
