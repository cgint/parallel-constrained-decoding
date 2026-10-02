"""Provider-neutral contracts for the comparison harness.

Exact validity is deliberately adapter-owned: an ``ok`` prediction must be an
exact member of ``EvalCase.choices``.  Do not trim whitespace or case-fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["ok", "invalid", "error", "unavailable"]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    context: str
    choices: tuple[str, ...]
    gold: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalCase":
        return cls(
            case_id=str(value["case_id"]),
            context=str(value["context"]),
            choices=tuple(str(choice) for choice in value["choices"]),
            gold=str(value["gold"]),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["choices"] = list(self.choices)
        return value


@dataclass
class Prediction:
    provider: str
    model: str
    status: Status
    prediction: str | None = None
    score: float | None = None
    score_kind: str = "none"
    latency_ms: float | None = None
    exact_valid: bool = False
    error_code: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unavailable(cls, provider: str, model: str, reason: str) -> "Prediction":
        return cls(
            provider=provider,
            model=model,
            status="unavailable",
            error_code=reason,
        )

    @classmethod
    def invalid(
        cls,
        provider: str,
        model: str,
        code: str,
        latency_ms: float | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> "Prediction":
        """Record a response that is not an exact permitted choice.

        Adapters determine this via exact membership in ``EvalCase.choices``;
        they must not trim or case-fold provider output.
        """
        return cls(
            provider=provider,
            model=model,
            status="invalid",
            latency_ms=latency_ms,
            error_code=code,
            diagnostics=dict(diagnostics or {}),
        )

    @classmethod
    def error(cls, provider: str, model: str, code: str, latency_ms: float | None = None) -> "Prediction":
        return cls(
            provider=provider,
            model=model,
            status="error",
            latency_ms=latency_ms,
            error_code=code,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
