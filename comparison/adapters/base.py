"""Shared adapter protocol for provider-neutral finite-choice evaluation."""

from __future__ import annotations

from typing import Protocol

from comparison.models import EvalCase, Prediction


class ProviderAdapter(Protocol):
    """A provider that can classify one finite-choice evaluation case."""

    def availability(self) -> bool: ...

    def predict(self, case: EvalCase) -> Prediction: ...
