"""Tests for the CLINC150 dataset adapter (label integrity + normalization).

Runs fully in the comparison venv: label parsing is exercised directly against a
fixture card, and the parquet path is exercised by injecting a synthetic reader,
so no pyarrow/pandas is required (the venv has neither).
"""
import unittest
from comparison.multi_field.dataset import (
    normalize_clinc150,
    parse_clinc150_labels,
    parse_clinc150_parquet,
)
from comparison.multi_field.models import StructuredCase

# Canonical 151-way CLINC-OOS label set: 150 in-domain intents + OOS.
# 'oos' sits at index 42, matching the HF card's class_label mapping.
FIXTURE_LABELS = [f'label_{i}' for i in range(42)] + ['oos'] + \
                 [f'label_{i}' for i in range(43, 151)]


def _card_bytes(labels):
    """Build a minimal HF-card README containing one class_label block."""
    lines = [f"          '{i}': {l}" for i, l in enumerate(labels)]
    block = (
        "dataset_info:\n"
        "- config_name: imbalanced\n"
        "  features:\n"
        "  - name: intent\n"
        "    dtype:\n"
        "      class_label:\n"
        "        names:\n"
        + "\n".join(lines) + "\n"
    )
    return block.encode()


class Clinc150LabelTests(unittest.TestCase):
    def test_151_unique_labels(self):
        labels = parse_clinc150_labels(_card_bytes(FIXTURE_LABELS))
        self.assertEqual(len(labels), 151)
        self.assertEqual(len(set(labels)), 151)
        self.assertIn('oos', labels)
        self.assertEqual(labels, tuple(FIXTURE_LABELS))
        self.assertEqual(labels[42], 'oos')

    def test_rejects_missing_oos(self):
        bad = [f'label_{i}' for i in range(151)]  # no 'oos'
        with self.assertRaises(ValueError):
            parse_clinc150_labels(_card_bytes(bad))

    def test_rejects_too_few_labels(self):
        bad = [f'label_{i}' for i in range(150)]
        with self.assertRaises(ValueError):
            parse_clinc150_labels(_card_bytes(bad))

    def test_rejects_duplicate_labels(self):
        bad = [f'label_{i}' for i in range(149)] + ['dup', 'dup']
        with self.assertRaises(ValueError):
            parse_clinc150_labels(_card_bytes(bad))


class Clinc150ParquetTests(unittest.TestCase):
    """Parse-level tests using a synthetic reader (no pyarrow needed)."""

    def test_parse_roundtrip(self):
        labels = tuple(FIXTURE_LABELS)
        # Synthetic rows: source_id, text, intent_index (0, 42=oos, 150)
        synthetic = [('0', 'text_0', 0), ('1', 'text_1', 42), ('2', 'text_2', 150), ('3', 'text_3', 0)]
        rows = parse_clinc150_parquet(b'', labels, reader=lambda _data: synthetic)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], ('0', 'text_0', 'label_0'))
        self.assertEqual(rows[1], ('1', 'text_1', 'oos'))  # index 42 -> 'oos'
        self.assertEqual(rows[2], ('2', 'text_2', 'label_150'))

    def test_rejects_out_of_range_index(self):
        labels = tuple(FIXTURE_LABELS)
        synthetic = [('0', 'text_0', 151)]  # index 151 is out of range
        with self.assertRaises(ValueError):
            parse_clinc150_parquet(b'', labels, reader=lambda _data: synthetic)


class Clinc150NormalizeTests(unittest.TestCase):
    def test_normalize_structure(self):
        labels = tuple(FIXTURE_LABELS)
        synthetic = [('0', 'text_0', 0), ('1', 'text_1', 42), ('2', 'text_2', 150)]
        cases = normalize_clinc150(b'', _card_bytes(labels), reader=lambda _data: synthetic)
        self.assertEqual(len(cases), 3)
        self.assertIsInstance(cases[0], StructuredCase)
        # schema: single enum field with 151 choices
        self.assertEqual(list(cases[0].schema), ['intent'])
        self.assertEqual(len(cases[0].schema['intent']['choices']), 151)
        self.assertEqual(cases[0].gold, {'intent': 'label_0'})
        self.assertEqual(cases[1].gold, {'intent': 'oos'})
        # metadata carries source_id and intent_index
        self.assertEqual(cases[1].metadata['source_id'], '1')
        self.assertEqual(cases[1].metadata['intent_index'], 42)


if __name__ == '__main__':
    unittest.main()
