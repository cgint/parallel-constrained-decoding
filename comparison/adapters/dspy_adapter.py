"""Lazy DSPy adapter for four-way finite-choice evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from importlib.util import find_spec
from time import perf_counter
from typing import Any, Literal

from comparison.models import EvalCase, Prediction

DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"


class DSPyAdapter:
    """Run one DSPy structured prediction for each valid evaluation case.

    DSPy is imported only when a prediction is requested, so the comparison
    harness remains importable without the optional dependency installed.
    """

    provider = "dspy"

    def __init__(self, model: str = DEFAULT_MODEL, dspy_module: Any | None = None) -> None:
        self.model = model
        self._dspy = dspy_module
        self._predictors: dict[tuple[str, ...], Any] = {}

    def availability(self) -> bool:
        return self._dspy is not None or find_spec("dspy") is not None

    def _load_dspy(self) -> Any:
        if self._dspy is None:
            import dspy  # type: ignore[import-not-found]

            self._dspy = dspy
        return self._dspy

    def _ensure_predictor(self, choices: tuple[str, ...]) -> Any:
        if choices in self._predictors:
            return self._predictors[choices]

        dspy = self._load_dspy()
        choice_type = Literal[choices]
        signature = dspy.Signature(
            {
                "context": (str, dspy.InputField(desc="Text to classify.")),
                "choice": (
                    choice_type,
                    dspy.OutputField(desc="Select exactly one permitted choice."),
                ),
            },
            "Choose the single best answer from the permitted choices.",
        )
        # Both per-LM cache=False and global cache configuration are intentional:
        # evaluation calls must not be satisfied by a prior cached completion.
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        lm = dspy.LM(model=self.model, temperature=0, cache=False)
        predictor = (dspy.Predict(signature), lm, dspy.JSONAdapter())
        self._predictors[choices] = predictor
        return predictor

    def _is_adapter_parse_error(self, exc: Exception) -> bool:
        """Recognize structured-output parse failures from either DSPy source."""
        exception_types: list[type[Exception]] = []

        # An injected module is the DSPy implementation used for this prediction.
        injected_error = getattr(self._dspy, "AdapterParseError", None)
        if isinstance(injected_error, type) and issubclass(injected_error, Exception):
            exception_types.append(injected_error)

        # Keep this optional: importing this adapter must not require DSPy.
        try:
            from dspy.utils.exceptions import AdapterParseError
        except ImportError:
            pass
        else:
            exception_types.append(AdapterParseError)

        return isinstance(exc, tuple(exception_types))

    @staticmethod
    def _usage(result: Any) -> dict[str, Any]:
        getter = getattr(result, "get_lm_usage", None)
        usage = getter() if callable(getter) else getattr(result, "usage", None)
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        try:
            return dict(usage)
        except (TypeError, ValueError):
            return {"raw": str(usage)}

    def predict(self, case: EvalCase) -> Prediction:
        if (
            len(case.choices) != 4
            or not all(isinstance(choice, str) and choice for choice in case.choices)
            or len(set(case.choices)) != 4
        ):
            return Prediction.error(self.provider, self.model, "invalid_case")
        if not self.availability():
            return Prediction.unavailable(self.provider, self.model, "dspy_not_installed")

        started = perf_counter()
        try:
            predictor, lm, adapter = self._ensure_predictor(case.choices)
            dspy = self._load_dspy()
            context = dspy.context(lm=lm, adapter=adapter, track_usage=True)
            with context if context is not None else nullcontext():
                result = predictor(context=case.context)
            latency_ms = (perf_counter() - started) * 1000
            selected = getattr(result, "choice", None)
            if not isinstance(selected, str) or selected not in case.choices:
                return Prediction.invalid(self.provider, self.model, "invalid_provider_choice", latency_ms)
            return Prediction(
                provider=self.provider,
                model=self.model,
                status="ok",
                prediction=selected,
                latency_ms=latency_ms,
                exact_valid=True,
                usage=self._usage(result),
            )
        except Exception as exc:
            if self._is_adapter_parse_error(exc):
                return Prediction.invalid(
                    self.provider, self.model, "dspy_adapter_parse_error", (perf_counter() - started) * 1000
                )
            return Prediction.error(
                self.provider, self.model, "dspy_provider_error", (perf_counter() - started) * 1000
            )
