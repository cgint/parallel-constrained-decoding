"""Stdlib tests for comparison model contracts."""

import unittest

from comparison.models import EvalCase, Prediction


class EvalCaseTests(unittest.TestCase):
    def test_round_trip_serialization(self) -> None:
        value = {
            "case_id": "case-1",
            "context": "Context",
            "choices": ["Yes", "No"],
            "gold": "Yes",
            "metadata": {"source": "test"},
        }

        self.assertEqual(EvalCase.from_dict(value).to_dict(), value)


class PredictionConstructorTests(unittest.TestCase):
    def test_ok_constructor_serializes_unchanged(self) -> None:
        prediction = Prediction(
            provider="provider",
            model="model",
            status="ok",
            prediction="Yes",
            score=0.9,
            score_kind="probability",
            latency_ms=12.5,
            exact_valid=True,
            usage={"tokens": 3},
            diagnostics={"raw": "Yes"},
        )

        self.assertEqual(
            prediction.to_dict(),
            {
                "provider": "provider",
                "model": "model",
                "status": "ok",
                "prediction": "Yes",
                "score": 0.9,
                "score_kind": "probability",
                "latency_ms": 12.5,
                "exact_valid": True,
                "error_code": None,
                "usage": {"tokens": 3},
                "diagnostics": {"raw": "Yes"},
            },
        )

    def test_invalid_constructor_records_response_contract_failure(self) -> None:
        diagnostics = {"raw": " yes "}
        prediction = Prediction.invalid(
            "provider", "model", "out_of_choices", 5.0, diagnostics
        )
        diagnostics["raw"] = "mutated"

        self.assertEqual(prediction.status, "invalid")
        self.assertIsNone(prediction.prediction)
        self.assertIsNone(prediction.score)
        self.assertFalse(prediction.exact_valid)
        self.assertEqual(prediction.latency_ms, 5.0)
        self.assertEqual(prediction.error_code, "out_of_choices")
        self.assertEqual(prediction.diagnostics, {"raw": " yes "})
        self.assertEqual(
            prediction.to_dict(),
            {
                "provider": "provider",
                "model": "model",
                "status": "invalid",
                "prediction": None,
                "score": None,
                "score_kind": "none",
                "latency_ms": 5.0,
                "exact_valid": False,
                "error_code": "out_of_choices",
                "usage": {},
                "diagnostics": {"raw": " yes "},
            },
        )

    def test_error_constructor_records_execution_failure(self) -> None:
        prediction = Prediction.error("provider", "model", "timeout", 25.0)

        self.assertEqual(prediction.status, "error")
        self.assertEqual(prediction.error_code, "timeout")
        self.assertEqual(prediction.latency_ms, 25.0)
        self.assertIsNone(prediction.prediction)
        self.assertIsNone(prediction.score)
        self.assertFalse(prediction.exact_valid)
        self.assertEqual(
            prediction.to_dict(),
            {
                "provider": "provider",
                "model": "model",
                "status": "error",
                "prediction": None,
                "score": None,
                "score_kind": "none",
                "latency_ms": 25.0,
                "exact_valid": False,
                "error_code": "timeout",
                "usage": {},
                "diagnostics": {},
            },
        )

    def test_unavailable_constructor_records_no_attempt(self) -> None:
        prediction = Prediction.unavailable("provider", "model", "missing_key")

        self.assertEqual(prediction.status, "unavailable")
        self.assertEqual(prediction.error_code, "missing_key")
        self.assertIsNone(prediction.prediction)
        self.assertIsNone(prediction.score)
        self.assertFalse(prediction.exact_valid)
        self.assertEqual(
            prediction.to_dict(),
            {
                "provider": "provider",
                "model": "model",
                "status": "unavailable",
                "prediction": None,
                "score": None,
                "score_kind": "none",
                "latency_ms": None,
                "exact_valid": False,
                "error_code": "missing_key",
                "usage": {},
                "diagnostics": {},
            },
        )


if __name__ == "__main__":
    unittest.main()
