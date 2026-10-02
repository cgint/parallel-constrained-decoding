"""No-network tests for the sequential comparison runner."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
import unittest
from pathlib import Path

from comparison.dataset import write_dataset
from comparison.models import Prediction
from comparison.run import main


class FakeAdapter:
    def __init__(self, provider: str, *, available: bool = True, error: bool = False, model: str = "fake-model"):
        self.provider, self.model, self.available, self.error = provider, model, available, error
        self.calls = 0

    def availability(self):
        return self.available

    def predict(self, case):
        self.calls += 1
        if self.error:
            return Prediction.error(self.provider, self.model, "fake_failure")
        return Prediction(self.provider, self.model, "ok", case.gold, latency_ms=1, exact_valid=True)


class RunnerTests(unittest.TestCase):
    def invoke(self, directory: Path, providers="rlcd", *, factories, extra=()):
        dataset = directory / "dataset.json"
        write_dataset(dataset)
        output = directory / "out"
        code = main(["--providers", providers, "--dataset", str(dataset), "--out", str(output), *extra], adapter_factories=factories)
        return code, output

    def test_limit_repetitions_warmup_and_artifact_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = FakeAdapter("rlcd")
            code, output = self.invoke(Path(temp), factories={"rlcd": lambda: adapter}, extra=("--limit", "2", "--repetitions", "3", "--warmup", "1"))
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(adapter.calls, 7)  # one warmup, excluded from measured rows
            self.assertEqual(set(rows[0]), {"case_id", "gold", "repetition", "provider", "model", "status", "prediction", "score", "score_kind", "latency_ms", "exact_valid", "error_code", "usage", "diagnostics"})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("sha256", manifest["dataset"])
            self.assertEqual(manifest["settings"]["concurrency"], 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["providers"]["rlcd"]["fake-model"]["n"], 6)
            self.assertIn("requests_per_second", summary["providers"]["rlcd"]["fake-model"]["observed_wall_clock"])

    def test_limited_selection_is_round_robin_and_near_balanced(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            adapter = FakeAdapter("rlcd")
            code, output = self.invoke(directory, factories={"rlcd": lambda: adapter}, extra=("--limit", "4"))
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["gold"] for row in rows], ["ALLOW", "REVIEW", "BLOCK", "ESCALATE"])

            adapter = FakeAdapter("rlcd")
            code, output = self.invoke(directory, factories={"rlcd": lambda: adapter}, extra=("--limit", "6"))
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            counts = Counter(row["gold"] for row in rows)
            self.assertEqual(counts, {"ALLOW": 2, "REVIEW": 2, "BLOCK": 1, "ESCALATE": 1})
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIn("round_robin_gold_labels", manifest["settings"]["selection_strategy"])

    def test_unavailable_provider_has_no_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = FakeAdapter("dspy", available=False)
            code, output = self.invoke(Path(temp), providers="dspy", factories={"dspy": lambda: adapter})
            self.assertEqual(code, 0)
            self.assertEqual((output / "results.jsonl").read_text(), "")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["providers"]["dspy"]["available"])
            self.assertEqual(adapter.calls, 0)

    def test_jev_paid_gate_does_not_call_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = FakeAdapter("jev")
            code, output = self.invoke(Path(temp), providers="jev", factories={"jev": lambda: adapter})
            self.assertEqual(code, 0)
            self.assertEqual(adapter.calls, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["providers"]["jev"]["reason"], "paid_provider_not_allowed")

    def test_provider_error_is_retained_and_secret_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = FakeAdapter("rlcd", error=True, model="not-a-secret")
            code, output = self.invoke(Path(temp), factories={"rlcd": lambda: adapter}, extra=("--limit", "1"))
            self.assertEqual(code, 0)
            row = json.loads((output / "results.jsonl").read_text())
            self.assertEqual((row["status"], row["error_code"]), ("error", "fake_failure"))
            combined = "".join(path.read_text() for path in output.iterdir())
            self.assertNotIn("TYPESAFE_API_KEY", combined)
            self.assertNotIn("test-key", combined)


if __name__ == "__main__":
    unittest.main()
