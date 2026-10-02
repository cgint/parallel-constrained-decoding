"""Deterministic mechanism-baseline dataset for routing-label comparisons."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .models import EvalCase

LABELS = ("ALLOW", "REVIEW", "BLOCK", "ESCALATE")
DATASET_PATH = Path(__file__).with_name("datasets") / "routing_v1.json"
PRECEDENCE_POLICY = ("Policy: prohibited => BLOCK; else escalation_required => ESCALATE; "
                     "else human_review_required => REVIEW; else ALLOW.")


def derive_gold(rule_inputs: dict[str, Any]) -> str:
    """Apply the documented precedence: BLOCK > ESCALATE > REVIEW > ALLOW."""
    if not isinstance(rule_inputs, dict):
        raise ValueError("rule_inputs must be an object")
    for key in ("prohibited", "escalation_required", "human_review_required"):
        if not isinstance(rule_inputs.get(key), bool):
            raise ValueError(f"rule_inputs.{key} must be a boolean")
    if rule_inputs["prohibited"]:
        return "BLOCK"
    if rule_inputs["escalation_required"]:
        return "ESCALATE"
    if rule_inputs["human_review_required"]:
        return "REVIEW"
    return "ALLOW"


def _inputs_for(label: str, index: int) -> dict[str, bool]:
    # Cycle through lower-priority signals to deliberately include precedence conflicts.
    low_a, low_b = bool(index & 1), bool(index & 2)
    if label == "BLOCK":
        return {"prohibited": True, "escalation_required": low_a, "human_review_required": low_b}
    if label == "ESCALATE":
        return {"prohibited": False, "escalation_required": True, "human_review_required": low_a}
    if label == "REVIEW":
        return {"prohibited": False, "escalation_required": False, "human_review_required": True}
    if label == "ALLOW":
        return {"prohibited": False, "escalation_required": False, "human_review_required": False}
    raise ValueError(f"unknown label: {label}")


def generate_cases() -> list[EvalCase]:
    """Generate exactly 512 balanced, reproducible routing cases."""
    cases: list[EvalCase] = []
    subjects = ("payment", "account", "shipment", "invoice", "access request", "customer record", "contract", "refund")
    for label_index, target in enumerate(LABELS):
        for index in range(128):
            inputs = _inputs_for(target, index)
            subject = subjects[(index + label_index) % len(subjects)]
            metadata = {
                "dataset": "routing_v1",
                "rule_inputs": inputs,
                "precedence": ["BLOCK", "ESCALATE", "REVIEW", "ALLOW"],
                "has_conflict": sum(inputs.values()) > 1,
            }
            cases.append(EvalCase(
                case_id=f"routing-v1-{label_index * 128 + index + 1:03d}",
                context=(f"Route {subject} request {index + 1}. "
                         f"Prohibited={inputs['prohibited']}; "
                         f"escalation_required={inputs['escalation_required']}; "
                         f"human_review_required={inputs['human_review_required']}. "
                         f"{PRECEDENCE_POLICY}"),
                choices=LABELS,
                gold=derive_gold(inputs),
                metadata=metadata,
            ))
    validate_cases(cases)
    return cases


def validate_cases(cases: Iterable[EvalCase]) -> None:
    cases = list(cases)
    if len(cases) != 512:
        raise ValueError(f"expected 512 cases, got {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    for case in cases:
        if case.choices != LABELS:
            raise ValueError(f"{case.case_id}: choices must be {LABELS}")
        if PRECEDENCE_POLICY not in case.context:
            raise ValueError(f"{case.case_id}: context must contain the complete precedence policy")
        inputs = case.metadata.get("rule_inputs")
        expected = derive_gold(inputs)
        if case.gold != expected:
            raise ValueError(f"{case.case_id}: gold {case.gold!r} disagrees with rule inputs ({expected})")
    counts = Counter(case.gold for case in cases)
    if counts != Counter({label: 128 for label in LABELS}):
        raise ValueError(f"dataset is not balanced: {dict(counts)}")


def write_dataset(path: Path = DATASET_PATH) -> None:
    cases = generate_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "cases": [case.to_dict() for case in cases]}, indent=2) + "\n")


def load_cases(path: Path = DATASET_PATH) -> list[EvalCase]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load dataset {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError("dataset must be an object with version 1 and a cases list")
    try:
        cases = [EvalCase.from_dict(value) for value in payload["cases"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed case: {exc}") from exc
    validate_cases(cases)
    return cases


if __name__ == "__main__":
    write_dataset()
