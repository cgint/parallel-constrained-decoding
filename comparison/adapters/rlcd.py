"""Adapter for this repository's parallel constrained decoding engine."""

from __future__ import annotations

import math
import time
from typing import Any

from comparison.models import EvalCase, Prediction


class RLCDAdapter:
    """Run one finite-choice question through a single RLCD enum field."""

    provider = "rlcd"

    def __init__(self, *, model: str = "Qwen2.5-1.5B-Instruct-4bit", temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = temperature

    def availability(self) -> bool:
        """RLCD is built in; defer backend import and its failure to ``predict``."""
        return True

    def predict(self, case: EvalCase) -> Prediction:
        if not self._valid_choices(case.choices):
            return Prediction.error(self.provider, self.model, "invalid_case")

        started = time.perf_counter()
        try:
            # Keep these imports here: importing this adapter must not select/load a backend.
            from core.engine import run_rlcd_generation
            from core.schema import StructuredSchema
        except (ImportError, ModuleNotFoundError):
            return Prediction.unavailable(self.provider, self.model, "engine_unavailable")

        try:
            schema = StructuredSchema(
                {"route": {"type": "enum", "description": "Select the best matching option.", "choices": list(case.choices)}}
            )
            result = run_rlcd_generation(case.context, schema, temperature=self.temperature)
        except Exception:
            return Prediction.error(self.provider, self.model, "engine_error", (time.perf_counter() - started) * 1000)

        try:
            choice, probability = self._validated_route(result, case.choices)
        except ValueError:
            return Prediction.invalid(self.provider, self.model, "invalid_response", (time.perf_counter() - started) * 1000)
        latency_ms = (time.perf_counter() - started) * 1000

        return Prediction(
            provider=self.provider,
            model=self.model,
            status="ok",
            prediction=choice,
            score=probability,
            score_kind="candidate_softmax",
            latency_ms=latency_ms,
            exact_valid=True,
            diagnostics={
                "engine_timing_ms": {
                    key: result[key]
                    for key in ("elapsed_ms", "prefill_ms", "suffix_eval_ms")
                    if key in result
                },
                "engine_mode": result.get("mode"),
            },
        )

    @staticmethod
    def _valid_choices(choices: Any) -> bool:
        return (
            isinstance(choices, tuple)
            and bool(choices)
            and all(isinstance(choice, str) and choice for choice in choices)
            and len(set(choices)) == len(choices)
        )

    @staticmethod
    def _validated_route(result: Any, choices: tuple[str, ...]) -> tuple[str, float]:
        if not isinstance(result, dict):
            raise ValueError("result must be an object")
        parsed = result.get("parsed_json")
        if not isinstance(parsed, dict) or set(parsed) != {"route"}:
            raise ValueError("missing or extra route fields")
        route = parsed["route"]
        if not isinstance(route, dict):
            raise ValueError("route must be an object")
        choice = route.get("value")
        probability = route.get("prob")
        if choice not in choices or isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("invalid route")
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid route probability")
        return choice, probability
