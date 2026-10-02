from __future__ import annotations

import json
import sys
import types
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from comparison.adapters.jev import JevAdapter
from comparison.adapters.rlcd import RLCDAdapter
from comparison.models import EvalCase

CASE = EvalCase("case-1", "Ticket text", ("returns", "shipping"), "returns")


class _Response:
    def __init__(self, data: dict):
        self.data = data

    def read(self) -> bytes:
        return json.dumps(self.data).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class JevAdapterTests(unittest.TestCase):
    def test_missing_key_is_unavailable_without_request(self):
        # JevAdapter falls back to the ambient TYPESAFE_API_KEY / JEV_API_KEY env vars
        # when api_key is None. Strip them so the no-key path is asserted the same way
        # in every environment and every test order (ambient key => gate passes, mocked
        # transport => invalid_response). See ADA-SUITE-DETERMINISM-20260919.
        env = {"TYPESAFE_API_KEY": "", "JEV_API_KEY": ""}
        with patch.dict("os.environ", env, clear=False):
            with patch("comparison.adapters.jev.urlopen") as call:
                result = JevAdapter(api_key=None).predict(CASE)
        self.assertEqual((result.status, result.error_code), ("unavailable", "missing_api_key"))
        call.assert_not_called()

    def test_posts_documented_choice_payload_and_preserves_score_semantics(self):
        data = {"model": "jev-1.13.0", "answers": {"route": {"type": "choice", "choice": "shipping", "confidence": 0.3, "probabilities": {"returns": 0.25, "shipping": 0.75}}}, "usage": {"input_tokens": 12}}
        with patch("comparison.adapters.jev.urlopen", return_value=_Response(data)) as call:
            result = JevAdapter(api_key="test-key").predict(CASE)
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(json.loads(request.data), {"state": "Ticket text", "model": "jev-1.13.0", "questions": {"route": {"type": "choice", "instructions": "Select the best matching option.", "criteria": {"returns": "returns", "shipping": "shipping"}}}})
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual((result.status, result.prediction, result.score, result.score_kind), ("ok", "shipping", 0.75, "jev_probability"))
        self.assertEqual(result.diagnostics["confidence"], 0.3)

    def test_rejects_malformed_or_case_mismatched_jev_response_as_invalid(self):
        bad = {"model": "jev-1.13.0", "answers": {"route": {"type": "choice", "choice": "Shipping", "confidence": 1.0, "probabilities": {"returns": 0.0, "Shipping": 1.0}}}}
        with patch("comparison.adapters.jev.urlopen", return_value=_Response(bad)):
            result = JevAdapter(api_key="test-key").predict(CASE)
        self.assertEqual((result.status, result.error_code, result.exact_valid), ("invalid", "invalid_response", False))

    def test_jev_transport_failure_is_error(self):
        with patch("comparison.adapters.jev.urlopen", side_effect=HTTPError("url", 401, "unauthorized", {}, None)):
            result = JevAdapter(api_key="test-key").predict(CASE)
        self.assertEqual((result.status, result.error_code), ("error", "http_error"))

    def test_accepts_probabilities_with_1_percent_rounding_gap(self):
        # Jev serializes large choice sets (e.g. 77 BANKING77 labels) with a ~1%
        # rounding artifact: the probability vector sometimes sums to 0.99 even
        # though the choice is valid and in-enum (observed live 2026-09-20, ~50% of
        # draws on that enum). The validator must accept that so the answer is not
        # spuriously dropped as invalid_response.
        probs = {"returns": 0.0, "shipping": 0.99}
        data = {"model": "jev-1.13.0", "answers": {"route": {"type": "choice", "choice": "shipping", "confidence": 0.99, "probabilities": probs}}, "usage": {"input_tokens": 12}}
        with patch("comparison.adapters.jev.urlopen", return_value=_Response(data)):
            result = JevAdapter(api_key="test-key").predict(CASE)
        self.assertEqual((result.status, result.prediction, result.exact_valid), ("ok", "shipping", True))

    def test_still_rejects_probabilities_with_large_missing_mass(self):
        # A genuinely broken vector (sum 0.90, 10% mass missing, beyond the 5% gap
        # Jev's serialization can produce) must still be rejected as invalid_response.
        probs = {"returns": 0.0, "shipping": 0.90}
        data = {"model": "jev-1.13.0", "answers": {"route": {"type": "choice", "choice": "shipping", "confidence": 0.9, "probabilities": probs}}, "usage": {"input_tokens": 12}}
        with patch("comparison.adapters.jev.urlopen", return_value=_Response(data)):
            result = JevAdapter(api_key="test-key").predict(CASE)
        self.assertEqual((result.status, result.error_code, result.exact_valid), ("invalid", "invalid_response", False))

    def test_invalid_case_prevents_jev_request(self):
        malformed = EvalCase("bad", "text", ("returns", "returns"), "returns")
        with patch("comparison.adapters.jev.urlopen") as call:
            result = JevAdapter(api_key="test-key").predict(malformed)
        self.assertEqual((result.status, result.error_code), ("error", "invalid_case"))
        call.assert_not_called()


class RLCDAdapterTests(unittest.TestCase):
    def test_constructs_one_route_and_uses_candidate_softmax(self):
        captured = {}

        class Schema:
            def __init__(self, value):
                captured["schema"] = value

        def generate(context, schema, temperature):
            captured.update(context=context, temperature=temperature)
            return {"mode": "parallel_constrained_calibrated", "elapsed_ms": 2.0, "prefill_ms": 1.0, "suffix_eval_ms": 0.5, "parsed_json": {"route": {"value": "shipping", "prob": 0.8}}}

        fake_engine = types.ModuleType("core.engine")
        fake_engine.run_rlcd_generation = generate
        fake_schema = types.ModuleType("core.schema")
        fake_schema.StructuredSchema = Schema
        with patch.dict(sys.modules, {"core.engine": fake_engine, "core.schema": fake_schema}):
            result = RLCDAdapter().predict(CASE)
        self.assertEqual(captured["schema"], {"route": {"type": "enum", "description": "Select the best matching option.", "choices": ["returns", "shipping"]}})
        self.assertEqual((result.status, result.prediction, result.score, result.score_kind), ("ok", "shipping", 0.8, "candidate_softmax"))
        self.assertEqual(result.diagnostics["engine_timing_ms"], {"elapsed_ms": 2.0, "prefill_ms": 1.0, "suffix_eval_ms": 0.5})

    def test_rejects_case_mismatched_rlcd_output_as_invalid(self):
        fake_engine = types.ModuleType("core.engine")
        fake_engine.run_rlcd_generation = lambda *_args, **_kwargs: {"parsed_json": {"route": {"value": "Shipping", "prob": 1.0}}}
        fake_schema = types.ModuleType("core.schema")
        fake_schema.StructuredSchema = lambda value: value
        with patch.dict(sys.modules, {"core.engine": fake_engine, "core.schema": fake_schema}):
            result = RLCDAdapter().predict(CASE)
        self.assertEqual((result.status, result.error_code, result.exact_valid), ("invalid", "invalid_response", False))

    def test_invalid_case_prevents_rlcd_execution(self):
        malformed = EvalCase("bad", "text", ("returns", "returns"), "returns")
        result = RLCDAdapter().predict(malformed)
        self.assertEqual((result.status, result.error_code), ("error", "invalid_case"))

    def test_rlcd_engine_failure_is_error(self):
        fake_engine = types.ModuleType("core.engine")
        fake_engine.run_rlcd_generation = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend failed"))
        fake_schema = types.ModuleType("core.schema")
        fake_schema.StructuredSchema = lambda value: value
        with patch.dict(sys.modules, {"core.engine": fake_engine, "core.schema": fake_schema}):
            result = RLCDAdapter().predict(CASE)
        self.assertEqual((result.status, result.error_code), ("error", "engine_error"))


if __name__ == "__main__":
    unittest.main()
