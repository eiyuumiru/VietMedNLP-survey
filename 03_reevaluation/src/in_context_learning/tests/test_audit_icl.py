"""Focused regression tests for ICL audit classifications."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_icl import classify_record, score_record
from icl_tasks import TASKS


class AuditClassificationTests(unittest.TestCase):
    def test_closed_label_confusion(self):
        record = {"gold": "entailment", "prediction": "neutral", "valid": True, "finish_reason": "stop"}
        counts = classify_record(record, TASKS["ViMedNLI_ViMedNLI"])
        self.assertEqual(counts["incorrect"], 1)
        self.assertEqual(counts["confusion::entailment->neutral"], 1)

    def test_ner_boundary_overlap_is_not_exact(self):
        record = {"gold": "trieu_chung_benh: đau đầu", "prediction": "trieu_chung_benh: đầu", "valid": True, "finish_reason": "stop"}
        counts = classify_record(record, TASKS["ViMedNER_NER"])
        self.assertEqual(counts["exact_entity_pairs"], 0)
        self.assertEqual(counts["boundary_overlap_mismatch"], 1)

    def test_invalid_output_has_zero_score(self):
        record = {"gold": "A", "prediction": "", "valid": False, "finish_reason": "stop"}
        self.assertEqual(score_record(record, TASKS["VMHQA_Multiple_Choice_QA"]), 0.0)
        self.assertEqual(classify_record(record, TASKS["VMHQA_Multiple_Choice_QA"])["format_or_schema_failure"], 1)


if __name__ == "__main__":
    unittest.main()
