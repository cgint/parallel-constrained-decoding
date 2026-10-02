import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from core.engine_mlx import _enum_candidate_sequences, _score_enum_candidates_teacher_forced
from core.engine_torch import _enum_candidate_sequences_torch
from core.schema import FieldDefinition, StructuredSchema


class CharacterTokenizer:
    """A deterministic tokenizer whose complete-string encodings preserve boundaries."""
    pad_token_id = 0

    def __init__(self):
        self._ids = {chr(i): i + 1 for i in range(128)}

    def encode(self, text, add_special_tokens=False):
        return [self._ids[c] for c in text]


class FixedLogitModel:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, inputs, **_kwargs):
        array = np.asarray(inputs)
        self.calls.append(tuple(array.shape))
        # Token score is independent of row, allowing a direct manual recomputation.
        logits = np.tile(np.arange(self.vocab_size, dtype=np.float32),
                         (array.shape[0], array.shape[1], 1)) / 17.0
        import mlx.core as mx
        return mx.array(logits)


class EnumTeacherForcingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.base = "system\n{\n"
        self.suffix = '  "route": "'

    def test_scores_complete_continuations_and_groups_equal_lengths(self):
        model = FixedLogitModel()
        choices = ["aa", "zz", "alpha"]
        raw, probabilities, forward_passes = _score_enum_candidates_teacher_forced(
            model, self.tokenizer, self.base, self.suffix, choices, 1.0
        )
        prefix, candidates = _enum_candidate_sequences(self.tokenizer, self.base, self.suffix, choices)
        expected = []
        logits = np.arange(256, dtype=float) / 17.0
        normalizer = math.log(np.exp(logits - logits.max()).sum()) + logits.max()
        for _full, continuation in candidates:
            expected.append(sum(logits[token] - normalizer for token in continuation))
        np.testing.assert_allclose(raw, expected, rtol=1e-6)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=7)
        # "aa" and "zz" have equal complete sequence lengths and share one batch.
        self.assertIn((2, len(prefix) + len(candidates[0][1])), model.calls)
        self.assertEqual(forward_passes, 2)
        self.assertEqual(choices[int(np.argmax(probabilities))], "zz")

    def test_two_single_token_choices_are_joint_scored_without_defaulting(self):
        model = FixedLogitModel()
        raw, probabilities, _forward_passes = _score_enum_candidates_teacher_forced(
            model, self.tokenizer, self.base, self.suffix, ["a", "z"], 1.0
        )
        self.assertEqual(len(raw), 2)
        self.assertGreater(probabilities[1], probabilities[0])
        self.assertEqual(int(np.argmax(probabilities)), 1)  # not the first configured choice

    def test_prefix_candidate_is_disambiguated_by_closing_quote(self):
        _prefix, candidates = _enum_candidate_sequences(
            self.tokenizer, self.base, self.suffix, ["cash", "cash withdrawal"]
        )
        short_continuation = candidates[0][1]
        self.assertEqual(short_continuation[-1], self.tokenizer.encode('"')[0])
        self.assertNotEqual(candidates[0][1], candidates[1][1][:len(short_continuation)])

    def test_json_serialization_and_banking_shaped_lengths(self):
        choices = ["ab", "abc", "abcd", "abcde", "abcdef", "abcdefg", "abcdefgh", "abcdefghi"]
        _prefix, candidates = _enum_candidate_sequences(self.tokenizer, self.base, self.suffix, choices)
        self.assertEqual([len(c) for _, c in candidates], list(range(3, 11)))  # text plus closing quote
        escaped = 'a"\\é'
        _prefix, escaped_candidate = _enum_candidate_sequences(
            self.tokenizer, self.base, self.suffix, [escaped]
        )
        expected_text = json.dumps(escaped, ensure_ascii=True)[1:]
        self.assertEqual(escaped_candidate[0][1], self.tokenizer.encode(expected_text))

    def test_inconsistent_complete_prefix_tokenization_is_rejected(self):
        class BoundaryChangingTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=False):
                encoded = super().encode(text, add_special_tokens=add_special_tokens)
                return [999] if text.endswith('a"') else encoded

        with self.assertRaisesRegex(ValueError, "prefix boundary"):
            _enum_candidate_sequences(BoundaryChangingTokenizer(), self.base, self.suffix, ["a"])

    def test_invalid_duplicate_and_empty_choices_rejected(self):
        for choices in ([], [""], ["  "], ["same", "same"], ["ok", 1]):
            with self.assertRaises(ValueError):
                FieldDefinition("route", "enum", "", choices)

    def test_boolean_metadata_is_still_batched(self):
        schema = StructuredSchema({"approved": {"type": "boolean", "description": "x"}})
        metadata = schema.compile_parallel_metadata(self.tokenizer)
        self.assertEqual(metadata["cands_per_field"][0], [
            self.tokenizer.encode("true")[0], self.tokenizer.encode("false")[0]
        ])
        self.assertEqual(metadata["has_collisions"], [False])

    def test_singleton_enum_has_probability_one(self):
        raw, probabilities, forward_passes = _score_enum_candidates_teacher_forced(
            FixedLogitModel(), self.tokenizer, self.base, self.suffix, ["only"], 1.0
        )
        self.assertEqual(len(raw), 1)
        self.assertEqual(probabilities, [1.0])
        self.assertEqual(forward_passes, 1)

    def test_torch_candidate_sequence_helper_matches_complete_json_semantics(self):
        prefix, candidates = _enum_candidate_sequences_torch(
            self.tokenizer, self.base, self.suffix, ["a", "z"]
        )
        self.assertEqual(prefix, self.tokenizer.encode(self.base + self.suffix))
        self.assertEqual(candidates[0][1][-1], self.tokenizer.encode('"')[0])
        self.assertNotEqual(candidates[0][1], candidates[1][1])


class RuntimeContractTests(unittest.TestCase):
    def _run(self, schema_dict):
        from core import engine_mlx

        tokenizer = CharacterTokenizer()
        model = FixedLogitModel()
        with patch.object(engine_mlx, "get_engine", return_value=(model, tokenizer)), \
             patch.object(engine_mlx, "make_prompt_cache", return_value=[]):
            return engine_mlx.run_parallel_generation("ticket", StructuredSchema(schema_dict))

    def test_boolean_only_result_is_calibrated(self):
        result = self._run({"approved": {"type": "boolean", "description": "x"}})
        self.assertEqual(result["mode"], "parallel_constrained_calibrated")
        self.assertTrue(result["has_calibrated_probabilities"])
        self.assertEqual(result["sequential_forward_passes"], 1)
        self.assertEqual(result["field_telemetry"]["approved"]["score_semantics"], "boolean_next_token_softmax")

    def test_enum_result_is_candidate_scored_and_not_first_choice(self):
        result = self._run({"route": {"type": "enum", "description": "x", "choices": ["a", "z"]}})
        self.assertEqual(result["mode"], "parallel_constrained_candidate_scored")
        self.assertFalse(result["has_calibrated_probabilities"])
        self.assertEqual(result["parsed_json"]["route"]["value"], "z")
        self.assertEqual(result["sequential_forward_passes"], 2)
        telemetry = result["field_telemetry"]["route"]
        self.assertEqual(telemetry["score_semantics"], "candidate_joint_softmax")
        self.assertIn("joint_log_probability", telemetry["top_choices"][0])

    def test_singleton_enum_runtime_contract_is_uncalibrated(self):
        result = self._run({"route": {"type": "enum", "description": "x", "choices": ["only"]}})
        self.assertEqual(result["mode"], "parallel_constrained_candidate_scored")
        self.assertFalse(result["has_calibrated_probabilities"])
        self.assertEqual(result["parsed_json"]["route"]["prob"], 1.0)


if __name__ == "__main__": 
    unittest.main()
