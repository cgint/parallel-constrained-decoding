"""Direct, optional stdlib-HTTP adapter for TypeSafe Jev (Choice and Noul primitives)."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from comparison.models import EvalCase, Prediction

# Noul->boolean threshold, per https://docs.typesafe.ai/primitives/noul: 0.5 for
# symmetric yes/no costs; edges documented: p >= 0.5 -> True.
NOUL_THRESHOLD = 0.5


class JevAdapter:
    """Call TypeSafe System One only when an API key is explicitly available."""

    provider = "jev"
    endpoint = "https://api.typesafe.ai/v1/systemone"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "jev-1.13.0",
        timeout_seconds: float = 30.0,
        instructions: str = "Select the best matching option.",
    ) -> None:
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY")
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.instructions = instructions

    def availability(self) -> bool:
        return bool(self.api_key)

    def predict(self, case: Any) -> Prediction:
        # Lane selection: EvalCase (context, choices, ...) -> documented Choice
        # path, byte-for-byte as before; NoulQuestionCase (questions map) -> Noul
        # path. The branch test below is on attribute shape, so neither payload
        # or validation path is altered for the other lane.
        questions = getattr(case, "questions", None)
        if isinstance(questions, dict):
            return self._predict_noul(case)
        return self._predict_choice(case)

    def _predict_choice(self, case: EvalCase) -> Prediction:
        if not self._valid_choices(case.choices):
            return Prediction.error(self.provider, self.model, "invalid_case")
        if not self.availability():
            return Prediction.unavailable(self.provider, self.model, "missing_api_key")

        payload = {
            "state": case.context,
            "model": self.model,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": {choice: choice for choice in case.choices},
                }
            },
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except HTTPError:
            return Prediction.error(self.provider, self.model, "http_error", (time.perf_counter() - started) * 1000)
        except (URLError, TimeoutError, OSError):
            return Prediction.error(self.provider, self.model, "network_error", (time.perf_counter() - started) * 1000)

        try:
            response_data = json.loads(body.decode("utf-8"))
            choice, probability, confidence, usage, response_model = self._validated_answer(response_data, case.choices)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            return Prediction.invalid(self.provider, self.model, "invalid_response", (time.perf_counter() - started) * 1000)

        return Prediction(
            provider=self.provider,
            model=response_model,
            status="ok",
            prediction=choice,
            score=probability,
            score_kind="jev_probability",
            latency_ms=(time.perf_counter() - started) * 1000,
            exact_valid=True,
            usage=usage,
            diagnostics={"confidence": confidence},
        )

    def _predict_noul(self, case: Any) -> Prediction:
        if not self._valid_noul_case(case):
            return Prediction.error(self.provider, self.model, "invalid_case")
        if not self.availability():
            return Prediction.unavailable(self.provider, self.model, "missing_api_key")

        questions = {key: {"type": "noul", "instructions": spec["instructions"]}
                     for key, spec in case.questions.items()}
        payload = {"state": case.context, "model": self.model, "questions": questions}
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except HTTPError:
            return Prediction.error(self.provider, self.model, "http_error", (time.perf_counter() - started) * 1000)
        except (URLError, TimeoutError, OSError):
            return Prediction.error(self.provider, self.model, "network_error", (time.perf_counter() - started) * 1000)

        try:
            response_data = json.loads(body.decode("utf-8"))
            probabilities, usage, response_model = self._validated_noul_answers(response_data, set(case.questions))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            return Prediction.invalid(self.provider, self.model, "invalid_response", (time.perf_counter() - started) * 1000)

        # Documented edge: p >= 0.5 -> True (NOUL_THRESHOLD, docs default for
        # symmetric yes/no costs). Raw P(yes) floats are exposed as score and
        # per-field noul_scores so they remain auditable.
        prediction = {key: value >= NOUL_THRESHOLD for key, value in probabilities.items()}
        return Prediction(
            provider=self.provider,
            model=response_model,
            status="ok",
            prediction=prediction,
            score=min(probabilities.values()) if probabilities else None,
            score_kind="jev_noul_probability",
            latency_ms=(time.perf_counter() - started) * 1000,
            exact_valid=True,
            usage=usage,
            diagnostics={"noul_scores": probabilities, "noul_threshold": NOUL_THRESHOLD},
        )

    @staticmethod
    def _valid_noul_case(case: Any) -> bool:
        questions = getattr(case, "questions", None)
        context = getattr(case, "context", None)
        return (
            isinstance(questions, dict)
            and bool(questions)
            and isinstance(context, str) and context
            and all(
                isinstance(key, str) and key
                and isinstance(spec, dict) and set(spec) == {"instructions"}
                and isinstance(spec["instructions"], str) and spec["instructions"]
                for key, spec in questions.items()
            )
        )

    @staticmethod
    def _validated_noul_answers(data: Any, expected_keys: set[str]) -> tuple[dict[str, float], dict[str, Any], str]:
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ValueError("response must contain answers")
        answers = data["answers"]
        if set(answers) != expected_keys:
            raise ValueError("answer keys must match question ids exactly")
        probabilities: dict[str, float] = {}
        for key in expected_keys:
            answer = answers[key]
            # A Noul answer has no 'confidence' field (docs: unlike Choice/Score);
            # its absence is correct, not an error. We do require the declared
            # type marker and a single finite probability in [0, 1].
            if not isinstance(answer, dict) or answer.get("type") != "noul":
                raise ValueError("missing noul answer")
            value = answer.get("noul")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("invalid noul probability")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("invalid noul probability")
            probabilities[key] = value
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            raise ValueError("invalid usage")
        response_model = data.get("model", "")
        if not isinstance(response_model, str) or not response_model:
            raise ValueError("invalid model")
        return probabilities, usage, response_model

    @staticmethod
    def _valid_choices(choices: Any) -> bool:
        return (
            isinstance(choices, tuple)
            and bool(choices)
            and all(isinstance(choice, str) and choice for choice in choices)
            and len(set(choices)) == len(choices)
        )

    @staticmethod
    def _validated_answer(data: Any, choices: tuple[str, ...]) -> tuple[str, float, float, dict[str, Any], str]:
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ValueError("response must contain answers")
        answer = data["answers"].get("route")
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("missing choice answer")
        choice = answer.get("choice")
        probabilities = answer.get("probabilities")
        confidence = answer.get("confidence")
        if choice not in choices or not isinstance(probabilities, dict) or set(probabilities) != set(choices):
            raise ValueError("invalid choices")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("invalid confidence")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("invalid confidence")
        total = 0.0
        for value in probabilities.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("invalid probability")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("invalid probability")
            total += value
        # Jev serializes large choice sets (e.g. 77 BANKING77 labels) with a
        # ~1% rounding artifact: probability vectors sometimes sum to 0.99
        # (observed live 2026-09-20, ~50% of draws on that enum). A tight
        # abs_tol=1e-6 (float-epsilon level) rejects those valid answers as
        # 'probabilities must sum to one' even though the choice itself is
        # in-enum. Tolerate the documented serialization gap while still
        # rejecting genuinely broken vectors (mass far from 1.0).
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=0.05):
            raise ValueError("probabilities must sum to one")
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            raise ValueError("invalid usage")
        response_model = data.get("model", "")
        if not isinstance(response_model, str) or not response_model:
            raise ValueError("invalid model")
        return choice, float(probabilities[choice]), confidence, usage, response_model
