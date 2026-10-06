"""Offline checks for the small extensions to the saved-run audit."""
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/in_context_learning/src"))
from icl_tasks import TASKS, cot_compliance, entities
from audit_icl import classify_record, build_table, DATASETS, MODELS, METHODS, DISPLAY
from rebuild import check_pair, check_icl_table, error_rows, coverage_rows, write_csv


class SavedRunAuditTests(unittest.TestCase):
    def test_coverage_uses_all_predictions_and_four_compliance_rates(self):
        rows = [{"dataset": "test", "condition": method, "n_predictions": 10, "n_invalid": 1,
                 "format_compliance_percent": rate if method == "explanation_request" else "NA"}
                for method in ["answer_only", "explanation_request"] for rate in [0, 25, 50, 100]]
        result = coverage_rows(rows)[0]
        self.assertEqual(result["n_predictions"], 80)
        self.assertEqual(result["invalid_rate_percent"], 10)
        self.assertEqual(result["format_compliance_min_percent"], 0)
        self.assertEqual(result["format_compliance_max_percent"], 100)
        with self.assertRaises(AssertionError):
            coverage_rows(rows[:-1])

    def test_strata_preserve_invalid_and_zero_events(self):
        task = TASKS["ViMedNLI_ViMedNLI"]
        rows = [{"gold": "entailment", "prediction": "entailment", "valid": True},
                {"gold": "entailment", "prediction": "", "valid": False, "finish_reason": "length"}]
        strata = [classify_record(row, task) for row in rows]
        pooled = sum(strata, Counter())
        result = {r["category"]: r for r in error_rows("NLI", "model", "direct", pooled, task[0])}
        self.assertEqual(result["invalid"]["denominator"], 2)
        self.assertEqual(result["incorrect"]["denominator"], 1)
        self.assertEqual(result["incorrect"]["count"], 0)
        empty = error_rows("NLI", "model", "direct", Counter(records=1, invalid=1), task[0])
        self.assertEqual(empty[-1]["rate_percent"], "NA")

    def test_unique_entity_pairs(self):
        labels = TASKS["ViMedNER_NER"][2]
        self.assertEqual(len(entities("nguyen_nhan_benh: virus, nguyen_nhan_benh: virus", labels)), 1)

    def test_pair_identity_includes_reference(self):
        row = {"row": 1, "test_hash": "a", "gold": "answer"}
        check_pair([row], [dict(row)])
        for changed in [dict(row, test_hash="b"), dict(row, gold="other")]:
            with self.assertRaises(AssertionError):
                check_pair([row], [changed])
        with self.assertRaises(AssertionError):
            check_pair([row, row], [row, row])

    def test_compliance_is_format_only(self):
        self.assertTrue(cot_compliance(json.dumps({"answer": "x", "explanation": "y"})))
        self.assertFalse(cot_compliance(json.dumps({"answer": "x", "explanation": " "})))
        self.assertFalse(cot_compliance("not JSON"))

    def test_table_allows_citation_but_rejects_wrong_cell(self):
        scores = {d: {p: {m: {k: 12.34 for k in DISPLAY[d][2]} for m in MODELS} for p in METHODS} for d in DATASETS}
        text = build_table(scores).replace(r"\textbf{Task / metric}", r"\textbf{Task} & \textbf{Metric}")
        for name, task, _ in DISPLAY.values():
            text = text.replace(name + " & ", r"\cite{example} & " + name + " & " + task + " & ")
        check_icl_table(text, scores)
        with self.assertRaises(AssertionError):
            check_icl_table(text.replace("12.34", "12.35", 1), scores)
        with self.assertRaises(AssertionError):
            check_icl_table(text.replace("{GPT-4o}", "{GPT-4.1}", 1), scores)
        with self.assertRaises(AssertionError):
            check_icl_table(text.replace(r"\makecell{Answer-only", r"\makecell{Explanation-", 1), scores)

    def test_csv_check_does_not_overwrite_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            rows = [{"count": 2, "rate": "NA"}]
            write_csv(path, rows, ["count", "rate"])
            original = path.read_bytes()
            write_csv(path, rows, ["count", "rate"], check=True)
            with self.assertRaises(AssertionError):
                write_csv(path, [{"count": 3, "rate": "NA"}], ["count", "rate"], check=True)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
