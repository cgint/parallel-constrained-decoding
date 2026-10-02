"""Mocked checks for the optional DSPy adapter; no provider calls are made."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from comparison.adapters.dspy_adapter import DEFAULT_MODEL, DSPyAdapter
from comparison.models import EvalCase


class FakeResult:
    def __init__(self, choice: str, usage: dict[str, int] | None = None) -> None:
        self.choice = choice
        self._usage = usage

    def get_lm_usage(self):
        return self._usage


class FakeDSPy:
    class AdapterParseError(Exception):
        pass

    def __init__(self, choice: str = "A", exception: Exception | None = None) -> None:
        self.choice = choice
        self.exception = exception
        self.cache_calls = []
        self.lm_kwargs = None
        self.context_kwargs = None
        self.predict_calls = 0
        self.signature_fields = None

    class InputField:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class OutputField(InputField):
        pass

    class JSONAdapter:
        pass

    def Signature(self, fields, instructions):
        self.signature_fields = fields
        return (fields, instructions)

    def configure_cache(self, **kwargs):
        self.cache_calls.append(kwargs)

    def LM(self, **kwargs):
        self.lm_kwargs = kwargs
        return "lm"

    def Predict(self, signature):
        def predict(**kwargs):
            self.predict_calls += 1
            if self.exception is not None:
                raise self.exception
            return FakeResult(self.choice, {"total_tokens": 7})

        return predict

    def context(self, **kwargs):
        self.context_kwargs = kwargs
        return nullcontext()


CASE = EvalCase("one", "Choose A", ("A", "B", "C", "D"), "A")


class DSPyAdapterTests(unittest.TestCase):
    def test_valid_prediction_uses_json_temperature_zero_and_usage(self):
        fake = FakeDSPy()
        adapter = DSPyAdapter(dspy_module=fake)

        prediction = adapter.predict(CASE)

        self.assertEqual(prediction.status, "ok")
        self.assertEqual(prediction.prediction, "A")
        self.assertTrue(prediction.exact_valid)
        self.assertEqual(prediction.usage, {"total_tokens": 7})
        self.assertEqual(fake.predict_calls, 1)
        self.assertEqual(fake.lm_kwargs, {"model": DEFAULT_MODEL, "temperature": 0, "cache": False})
        self.assertEqual(fake.cache_calls, [{"enable_disk_cache": False, "enable_memory_cache": False}])
        self.assertIsInstance(fake.context_kwargs["adapter"], FakeDSPy.JSONAdapter)
        self.assertTrue(fake.context_kwargs["track_usage"])
        self.assertEqual(fake.signature_fields["choice"][0].__args__, CASE.choices)

    def test_rejects_malformed_case_as_error_without_calling_provider(self):
        fake = FakeDSPy()
        prediction = DSPyAdapter(dspy_module=fake).predict(
            EvalCase("bad", "x", ("A", "B", "C"), "A")
        )
        self.assertEqual(prediction.status, "error")
        self.assertEqual(prediction.error_code, "invalid_case")
        self.assertEqual(fake.predict_calls, 0)

    def test_rejects_provider_value_outside_case_choices_as_invalid(self):
        fake = FakeDSPy(choice="a")  # Exact membership: no case-folding or trimming.
        prediction = DSPyAdapter(dspy_module=fake).predict(CASE)
        self.assertEqual(prediction.status, "invalid")
        self.assertEqual(prediction.error_code, "invalid_provider_choice")
        self.assertEqual(fake.predict_calls, 1)

    def test_structured_parse_failure_is_invalid(self):
        fake = FakeDSPy(exception=FakeDSPy.AdapterParseError("unparseable JSON"))
        prediction = DSPyAdapter(dspy_module=fake).predict(CASE)
        self.assertEqual(prediction.status, "invalid")
        self.assertEqual(prediction.error_code, "dspy_adapter_parse_error")
        self.assertEqual(fake.predict_calls, 1)

    def test_installed_dspy_parse_error_is_recognized_without_provider_call(self):
        from dspy.utils.exceptions import AdapterParseError

        fake = FakeDSPy()
        adapter = DSPyAdapter(dspy_module=fake)

        parse_error = AdapterParseError("JSONAdapter", SimpleNamespace(output_fields={}), "{}")
        self.assertTrue(adapter._is_adapter_parse_error(parse_error))
        self.assertEqual(fake.predict_calls, 0)

    def test_generic_provider_exception_remains_error(self):
        fake = FakeDSPy(exception=RuntimeError("network failure"))
        prediction = DSPyAdapter(dspy_module=fake).predict(CASE)
        self.assertEqual(prediction.status, "error")
        self.assertEqual(prediction.error_code, "dspy_provider_error")
        self.assertEqual(fake.predict_calls, 1)


if __name__ == "__main__":
    unittest.main()
