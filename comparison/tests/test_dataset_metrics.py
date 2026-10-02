import json
import tempfile
import unittest
from pathlib import Path

from comparison.dataset import LABELS, PRECEDENCE_POLICY, derive_gold, generate_cases, load_cases, write_dataset
from comparison.metrics import percentile, summarize
from comparison.models import EvalCase, Prediction


class DatasetTests(unittest.TestCase):
    def test_balance_and_recomputable_gold(self):
        cases = generate_cases()
        self.assertEqual(len(cases), 512)
        self.assertEqual({label: sum(c.gold == label for c in cases) for label in LABELS}, {label: 128 for label in LABELS})
        self.assertTrue(any(c.metadata["has_conflict"] for c in cases))
        for case in cases:
            self.assertIn(PRECEDENCE_POLICY, case.context)
            self.assertEqual(case.gold, derive_gold(case.metadata["rule_inputs"]))

    def test_precedence(self):
        self.assertEqual(derive_gold({"prohibited": True, "escalation_required": True, "human_review_required": True}), "BLOCK")
        self.assertEqual(derive_gold({"prohibited": False, "escalation_required": True, "human_review_required": True}), "ESCALATE")

    def test_loader_rejects_malformed_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"version": 1, "cases": []}))
            with self.assertRaisesRegex(ValueError, "expected 512"):
                load_cases(path)
            write_dataset(path)
            payload = json.loads(path.read_text())
            payload["cases"][0]["gold"] = "BLOCK"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_cases(path)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.case = generate_cases()[0]

    def test_percentiles(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(percentile([1, 2, 3, 4], 95), 3.85)
        self.assertIsNone(percentile([], 50))

    def test_invalid_is_evaluated_and_always_incorrect(self):
        result = summarize([
            (self.case, Prediction.invalid("p", "m", "not_choice", latency_ms=10)),
            (self.case, Prediction.invalid("p", "m", "not_choice", latency_ms=20)),
        ])
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["evaluated"], 2)
        self.assertEqual(result["conditioned_accuracy"], 0.0)
        self.assertEqual(result["effective_accuracy"], 0.0)
        self.assertEqual(result["strict_validity"], 0.0)

    def test_errors_reduce_effective_not_conditioned_accuracy(self):
        result = summarize([
            (self.case, Prediction("p", "m", "ok", self.case.gold, latency_ms=10, exact_valid=True)),
            (self.case, Prediction.error("p", "m", "timeout", 20)),
            (self.case, Prediction.unavailable("p", "m", "missing_key")),
        ])
        self.assertEqual(result["attempted"], 2)
        self.assertEqual(result["evaluated"], 1)
        self.assertEqual(result["conditioned_accuracy"], 1.0)
        self.assertEqual(result["effective_accuracy"], 0.5)
        self.assertEqual(result["errors"], {"missing_key": 1, "timeout": 1})

    def test_success_latency_excludes_invalid_and_error(self):
        result = summarize([
            (self.case, Prediction("p", "m", "ok", self.case.gold, latency_ms=10, exact_valid=True)),
            (self.case, Prediction.invalid("p", "m", "not_choice", latency_ms=100)),
            (self.case, Prediction.error("p", "m", "timeout", 1000)),
        ])
        self.assertEqual(result["success_latency_ms"]["p50"], 10)
        self.assertEqual(result["attempt_latency_ms_by_status"]["invalid"]["p50"], 100)
        self.assertEqual(result["attempt_latency_ms_by_status"]["error"]["p50"], 1000)

    def test_majority_class_baseline(self):
        other = next(case for case in generate_cases() if case.gold == "BLOCK")
        result = summarize([(self.case, Prediction.unavailable("p", "m", "off")), (other, Prediction.unavailable("p", "m", "off")), (other, Prediction.unavailable("p", "m", "off"))])
        self.assertEqual(result["majority_class_baseline"], {"label": "BLOCK", "correct": 2, "n": 3, "accuracy": 2 / 3})

    def test_score_kinds_are_never_aggregated(self):
        result = summarize([
            (self.case, Prediction("p", "m", "ok", self.case.gold, score=.9, score_kind="probability")),
            (self.case, Prediction("p", "m", "ok", self.case.gold, score=2.0, score_kind="logit")),
        ])
        groups = result["score_diagnostics"]
        self.assertEqual(len(groups), 2)
        self.assertEqual({group["score_kind"] for group in groups}, {"probability", "logit"})
        self.assertTrue(all("exploratory" in group["calibration"] for group in groups))


if __name__ == "__main__":
    unittest.main()
