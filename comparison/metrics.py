"""Provider-neutral summaries for paired evaluation cases and predictions."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable, Sequence

from .models import EvalCase, Prediction

_ATTEMPTED = frozenset(("ok", "invalid", "error"))
_EVALUATED = frozenset(("ok", "invalid"))


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear-interpolated percentile; returns None where no measurements exist."""
    if not values:
        return None
    if not 0 <= quantile <= 100:
        raise ValueError("quantile must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95), "p99": percentile(values, 99), "sum_ms": sum(values)}


def summarize(pairs: Iterable[tuple[EvalCase, Prediction]]) -> dict[str, object]:
    """Summarize explicit case/prediction pairs.

    Invalid results are evaluated but always incorrect. Errors are attempted but not
    evaluated. Only successful (`ok`) calls contribute to success-latency percentiles.
    Score diagnostics remain segregated by provider, model, and score kind.
    """
    rows = list(pairs)
    statuses: dict[str, int] = defaultdict(int)
    error_codes: dict[str, int] = defaultdict(int)
    latency_by_status: dict[str, list[float]] = defaultdict(list)
    success_latencies: list[float] = []
    score_groups: dict[tuple[str, str, str], list[tuple[float, bool]]] = defaultdict(list)
    attempted = evaluated = correct = exact_valid = 0

    for case, prediction in rows:
        status = prediction.status
        statuses[status] += 1
        if status != "ok":
            error_codes[prediction.error_code or status] += 1
        if status in _ATTEMPTED:
            attempted += 1
            if prediction.latency_ms is not None:
                latency_by_status[status].append(prediction.latency_ms)
        if status in _EVALUATED:
            evaluated += 1
            exact_valid += int(prediction.exact_valid)
        if status == "ok":
            success_latencies.extend(() if prediction.latency_ms is None else (prediction.latency_ms,))
            is_correct = prediction.prediction == case.gold
            correct += int(is_correct)
            if prediction.score is not None and prediction.score_kind != "none":
                score_groups[(prediction.provider, prediction.model, prediction.score_kind)].append((prediction.score, is_correct))

    diagnostics = []
    for (provider, model, score_kind), observations in sorted(score_groups.items()):
        scores = [score for score, _ in observations]
        correct_flags = [is_correct for _, is_correct in observations]
        diagnostics.append({
            "provider": provider, "model": model, "score_kind": score_kind,
            "n": len(scores), "mean_score": sum(scores) / len(scores),
            "accuracy_at_scored": sum(correct_flags) / len(correct_flags),
            "calibration": "exploratory; score semantics are provider-defined and not comparable across score_kind/provider",
        })

    gold_counts = Counter(case.gold for case, _ in rows)
    majority_label = min(gold_counts, key=lambda label: (-gold_counts[label], label)) if gold_counts else None
    majority_correct = gold_counts.get(majority_label, 0) if majority_label else 0
    return {
        "n": len(rows), "status_counts": dict(sorted(statuses.items())),
        "attempted": attempted,
        "evaluated": evaluated,
        "correct": correct,
        "conditioned_accuracy": correct / evaluated if evaluated else None,
        "effective_accuracy": correct / attempted if attempted else None,
        "strict_validity": exact_valid / evaluated if evaluated else None,
        "errors": dict(sorted(error_codes.items())),
        "success_latency_ms": _latency_summary(success_latencies),
        "attempt_latency_ms_by_status": {status: _latency_summary(latency_by_status[status]) for status in sorted(latency_by_status)},
        "throughput_inputs": {"successful_requests": statuses["ok"], "attempted_requests": attempted, "success_latency_sum_ms": sum(success_latencies), "wall_clock_required": True},
        "majority_class_baseline": {"label": majority_label, "correct": majority_correct, "n": len(rows), "accuracy": majority_correct / len(rows) if rows else None},
        "score_diagnostics": diagnostics,
    }
