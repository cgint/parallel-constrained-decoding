"""Sequential, reproducible runner for the finite-choice comparison dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .adapters.dspy_adapter import DEFAULT_MODEL, DSPyAdapter
from .adapters.jev import JevAdapter
from .adapters.rlcd import RLCDAdapter
from .dataset import DATASET_PATH, LABELS, load_cases
from .metrics import summarize
from .models import EvalCase, Prediction

PROVIDERS = ("rlcd", "dspy", "jev")
AdapterFactory = Callable[[], object]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _parse_providers(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names or len(set(names)) != len(names) or any(name not in PROVIDERS for name in names):
        raise argparse.ArgumentTypeError(f"--providers must be a nonempty comma list of: {', '.join(PROVIDERS)}")
    return names


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", type=_parse_providers, required=True, help="Comma list: rlcd,dspy,jev")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--limit", type=_positive)
    parser.add_argument("--warmup", type=lambda value: _nonnegative(value), default=0)
    parser.add_argument("--repetitions", type=_positive, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dspy-model", default=DEFAULT_MODEL)
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--allow-paid", action="store_true")
    return parser


def _factories(args: argparse.Namespace) -> dict[str, AdapterFactory]:
    return {
        "rlcd": lambda: RLCDAdapter(),
        "dspy": lambda: DSPyAdapter(model=args.dspy_model),
        "jev": lambda: JevAdapter(model=args.jev_model),
    }


def _select_cases(cases: list[EvalCase], limit: int | None) -> list[EvalCase]:
    """Round-robin labels for bounded runs; preserve the complete dataset unchanged."""
    if limit is None or limit >= len(cases):
        return cases
    by_label = {label: [case for case in cases if case.gold == label] for label in LABELS}
    selected: list[EvalCase] = []
    index = 0
    while len(selected) < limit:
        for label in LABELS:
            bucket = by_label[label]
            if index < len(bucket):
                selected.append(bucket[index])
                if len(selected) == limit:
                    return selected
        index += 1
    return selected


def _provider_status(name: str, adapter: object, allow_paid: bool) -> tuple[bool, str | None]:
    if name == "jev" and not allow_paid:
        return False, "paid_provider_not_allowed"
    try:
        available = adapter.availability()  # type: ignore[attr-defined]
    except Exception:
        return False, "availability_check_failed"
    if available:
        return True, None
    return False, "missing_api_key" if name == "jev" else "adapter_unavailable"


def run(args: argparse.Namespace, *, adapter_factories: dict[str, AdapterFactory] | None = None) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Execute configured adapters.  Adapter failures are recorded as data, not raised."""
    cases = _select_cases(load_cases(args.dataset), args.limit)
    if not cases:
        raise ValueError("selected dataset contains no cases")
    dataset_bytes = args.dataset.read_bytes()
    utc_started = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    factories = adapter_factories or _factories(args)
    adapters: dict[str, object] = {}
    provider_configs: dict[str, dict[str, object]] = {}
    for name in args.providers:
        adapter = factories[name]()
        adapters[name] = adapter
        available, reason = _provider_status(name, adapter, args.allow_paid)
        provider_configs[name] = {
            "model": getattr(adapter, "model", None), "available": available, "reason": reason,
        }

    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for name in args.providers:
        adapter = adapters[name]
        status = provider_configs[name]
        if not status["available"]:
            continue
        for case in cases[:args.warmup]:
            try:
                adapter.predict(case)  # type: ignore[attr-defined]
            except Exception:
                pass
        provider_started = time.perf_counter()
        pairs: list[tuple[EvalCase, Prediction]] = []
        for repetition in range(args.repetitions):
            for case in cases:
                try:
                    prediction = adapter.predict(case)  # type: ignore[attr-defined]
                except Exception:
                    prediction = Prediction.error(name, str(status["model"] or "unknown"), "adapter_exception")
                pairs.append((case, prediction))
                rows.append({"case_id": case.case_id, "gold": case.gold, "repetition": repetition, **prediction.to_dict()})
        wall_seconds = time.perf_counter() - provider_started
        # A response model may differ from the configured model, so retain isolation by model.
        model_groups: dict[str, list[tuple[EvalCase, Prediction]]] = {}
        for pair in pairs:
            model_groups.setdefault(pair[1].model, []).append(pair)
        summaries[name] = {}
        for model, model_pairs in model_groups.items():
            summary = summarize(model_pairs)
            summary["observed_wall_clock"] = {
                "seconds": wall_seconds,
                "requests": len(model_pairs),
                "requests_per_second": len(model_pairs) / wall_seconds if wall_seconds else None,
            }
            summaries[name][model] = summary

    manifest: dict[str, object] = {
        "dataset": {"path": str(args.dataset), "sha256": hashlib.sha256(dataset_bytes).hexdigest()},
        "utc_started": utc_started,
        "platform": platform.platform(),
        "python": sys.version,
        "providers": provider_configs,
        "settings": {
            "providers": list(args.providers), "limit": args.limit, "warmup": args.warmup,
            "repetitions": args.repetitions, "allow_paid": args.allow_paid,
            "dspy_model": args.dspy_model, "jev_model": args.jev_model, "concurrency": 1,
            "selection_strategy": "round_robin_gold_labels_dataset_order_when_limited; full_dataset_order_when_unlimited_or_limit_at_least_dataset_size", 
        },
        "git": _git_state(),
        "overall_wall_seconds": time.perf_counter() - started,
    }
    return manifest, rows, {"providers": summaries}


def main(argv: Sequence[str] | None = None, *, adapter_factories: dict[str, AdapterFactory] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest, rows, summary = run(args, adapter_factories=adapter_factories)
        args.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.out / "manifest.json", manifest)
        _atomic_jsonl(args.out / "results.jsonl", rows)
        _atomic_json(args.out / "summary.json", summary)
    except (OSError, ValueError, KeyError) as exc:
        print(f"comparison runner failed: {exc}", file=sys.stderr)
        return 2
    print("provider\tmodel\trows\taccuracy\trequests/s")
    for provider, models in summary["providers"].items():
        for model, result in models.items():
            rate = result["observed_wall_clock"]["requests_per_second"]
            print(f"{provider}\t{model}\t{result['n']}\t{result['conditioned_accuracy']}\t{rate}")
    for provider, config in manifest["providers"].items():
        if not config["available"]:
            print(f"{provider}\t{config['model']}\tunavailable\t{config['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
