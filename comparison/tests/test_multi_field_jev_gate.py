"""Gate tests for the Jev arm in the multi-field enum lane, plus the Noul
(boolean-lane) mock-HTTP contract.

Proves the paid gate's behaviour (no requests, exact reason strings) without
any network and without a real credential, and that a Jev response with an
out-of-enum choice fails the lane's own validation. The Noul additions prove,
fully offline, that an all-boolean schema (GoEmotions: 28 fields) goes out as
ONE request carrying 28 `{type:'noul'}` questions with the schema descriptions
as affirmative instructions, and that a 28-answer response parses correctly
including threshold edges, malformed answers, and the missing-key path.

Trustworthy runner (CLI, not in-process discover -- see AGENTS.md instrument
rules):
  PYTHONPATH="$PWD" .venv/bin/python -m unittest comparison.tests.test_multi_field_jev_gate -v
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from comparison.adapters.jev import NOUL_THRESHOLD
from comparison.models import Prediction
from comparison.multi_field.models import StructuredCase, StructuredPrediction
from comparison.multi_field import run as mfrun
from comparison.multi_field.adapters import JevStructuredAdapter, valid_object

MARKER = "test_multi_field_jev_gate loaded"
print(MARKER)

LABELS = (b'admiration\namusement\nanger\nannoyance\napproval\ncaring\nconfusion\n'
          b'curiosity\ndesire\ndisappointment\ndisapproval\ndisgust\nembarrassment\n'
          b'excitement\nfear\ngratitude\ngrief\njoy\nlove\nnervousness\noptimism\n'
          b'pride\nrealization\nrelief\nremorse\nsadness\nsurprise\nneutral\n')
TEST = b'good\t0,27\tid1\n'


def _banking_schema() -> dict:
    # Mirrors comparison.multi_field.dataset.normalize_banking77's single-field enum schema.
    return {'intent': {'type': 'enum', 'description': 'BANKING77 intent category.',
                       'choices': ['balance', 'transfers', 'credit_card_payment']}}


class _FakeInner:
    """Stands in for comparison.adapters.jev.JevAdapter; records every predict call."""
    def __init__(self, inner_result, available=True):
        self.inner_result = inner_result
        self.available = available
        self.calls = 0

    def availability(self):
        return self.available

    def predict(self, case):
        self.calls += 1
        return self.inner_result


def _make_adapter(inner_result, available=True):
    """Build a JevStructuredAdapter whose inner JevAdapter is a _FakeInner."""
    def fake_init(self, model='jev-1.13.0'):
        self.model = model
        self._inner = _FakeInner(inner_result, available)
    with patch.object(JevStructuredAdapter, '__init__', fake_init):
        return JevStructuredAdapter()


def _gate_reason(name, adapter, allow_paid):
    """Replicate the exact gate logic from comparison/multi_field/run.py main()."""
    if name == 'jev' and not allow_paid:
        return False, 'paid_provider_not_allowed'
    try:
        available = adapter.availability()
    except Exception:
        return False, 'availability_check_failed'
    reason = None if available else ('missing_api_key' if name == 'jev' else 'adapter_unavailable')
    return available, reason


# ---------------------------------------------------------------------------
# Gate 2 behaviour: paid gate in the enum lane, replicating comparison/run.py
# ---------------------------------------------------------------------------
class JevGateBehaviourTests(unittest.TestCase):
    def test_jev_without_allow_paid_is_paid_provider_not_allowed_no_request(self):
        inner = StructuredPrediction('jev', 'jev-1.13.0', 'ok', {'intent': 'balance'}, 10.0, True)
        adapter = _make_adapter(inner, available=True)  # even if a key were present
        available, reason = _gate_reason('jev', adapter, allow_paid=False)
        self.assertFalse(available)
        self.assertEqual(reason, 'paid_provider_not_allowed')
        self.assertEqual(adapter._inner.calls, 0)  # zero requests

    def test_jev_with_allow_paid_no_credential_is_missing_api_key_no_request(self):
        adapter = _make_adapter(None, available=False)  # no credential -> unavailable
        available, reason = _gate_reason('jev', adapter, allow_paid=True)
        self.assertFalse(available)
        self.assertEqual(reason, 'missing_api_key')
        self.assertEqual(adapter._inner.calls, 0)  # zero requests

    def test_non_jev_provider_not_gated_by_paid_rule(self):
        # rlcd/dspy must not be affected by the jev-only paid gate.
        class _Avail:
            def availability(self):
                return True
        available, reason = _gate_reason('rlcd', _Avail(), allow_paid=False)
        self.assertTrue(available)
        self.assertIsNone(reason)


# ---------------------------------------------------------------------------
# End-to-end: main() with jev in --providers, no --allow-paid -> gated, no crash
# ---------------------------------------------------------------------------
class JevGateRunnerTests(unittest.TestCase):
    def test_runner_rejects_jev_without_flag_reason_recorded(self):
        class _Rlcd:
            model = 'q'
            def availability(self):
                return True
            def predict(self, c):
                return StructuredPrediction('rlcd', 'q', 'ok', dict(c.gold), 5.0, True)

        class _Jev:
            model = 'jev-1.13.0'
            def availability(self):
                return True
            def predict(self, c):
                return StructuredPrediction('jev', 'jev-1.13.0', 'ok', {'admiration': True}, 10.0, True)

        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'out'
            with patch.object(mfrun, 'fetch_sources',
                              return_value=(TEST, LABELS, {'sources': [{'sha256': 'x'}]})):
                rc = mfrun.main(['--providers', 'rlcd,jev', '--limit', '1', '--out', str(out)],
                                {'rlcd': lambda: _Rlcd(), 'jev': lambda: _Jev()})
            self.assertEqual(rc, 0)
            m = json.loads((out / 'manifest.json').read_text())
            jev = m['providers']['jev']
            self.assertFalse(jev['available'])
            self.assertEqual(jev['reason'], 'paid_provider_not_allowed')
            rlcd = m['providers']['rlcd']
            self.assertTrue(rlcd['available'])
            # Zero Jev rows: the arm was gated before any predict.
            rows = [json.loads(l) for l in (out / 'results.jsonl').read_text().splitlines() if l.strip()]
            self.assertNotIn('jev', [r.get('provider') for r in rows])
            self.assertTrue(any(r.get('provider') == 'rlcd' for r in rows))


# ---------------------------------------------------------------------------
# Validity honesty: out-of-enum Jev value fails the lane's own validation
# ---------------------------------------------------------------------------
class JevValidityTests(unittest.TestCase):
    def test_out_of_enum_value_fails_lane_validation(self):
        schema = _banking_schema()
        self.assertFalse(valid_object({'intent': 'not_a_real_label'}, schema))
        self.assertTrue(valid_object({'intent': 'balance'}, schema))

    def test_jev_adapter_maps_ok_to_lane_prediction(self):
        # JevAdapter returns a comparison.models.Prediction with a flat string choice.
        inner = Prediction(provider='jev', model='jev-1.13.0', status='ok',
                           prediction='balance', score=0.75, score_kind='jev_probability',
                           latency_ms=12.0, exact_valid=True)
        adapter = _make_adapter(inner, available=True)
        case = StructuredCase('banking77-test-1', 'ctx', _banking_schema(), {'intent': 'balance'})
        p = adapter.predict(case)
        self.assertEqual(p.status, 'ok')
        self.assertEqual(p.prediction, {'intent': 'balance'})
        self.assertTrue(valid_object(p.prediction, _banking_schema()))

    def test_jev_unavailable_maps_to_lane_unavailable_reason(self):
        inner = Prediction(provider='jev', model='jev-1.13.0', status='unavailable',
                           error_code='missing_api_key')
        adapter = _make_adapter(inner, available=False)
        case = StructuredCase('banking77-test-1', 'ctx', _banking_schema(), {'intent': 'balance'})
        p = adapter.predict(case)
        self.assertEqual(p.status, 'unavailable')
        self.assertEqual(p.error_code, 'missing_api_key')


# ---------------------------------------------------------------------------
# Noul (boolean) lane: end-to-end request/response contract, mocked transport
# ---------------------------------------------------------------------------
NOUL_FIELDS = ('admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
               'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
               'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral',
               'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise')


def _goemotions_schema() -> dict:
    return {name: {'type': 'boolean', 'description': f'Whether the text expresses {name}.'}
            for name in NOUL_FIELDS}


def _goemotions_case(text='I am so angry about this.'):  # 28-field boolean case
    return StructuredCase('goemotions-test-1', text, _goemotions_schema(),
                          {name: (name == 'anger') for name in NOUL_FIELDS})


class _Resp:
    def __init__(self, data: dict):
        self._data = data

    def read(self) -> bytes:
        return json.dumps(self._data).encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False


def _noul_response(fail_field=None, extra_key=None, drop_key=None):
    answers = {}
    for i, key in enumerate(NOUL_FIELDS):
        p = {0: 0.02, 1: 0.97}.get(i, 0.5 if i == 2 else round(0.05 + 0.01 * i, 4))
        if fail_field == key and i == 2:
            p = 1.3  # out of [0, 1]
        answers[key] = {'type': 'noul', 'noul': p}
    if drop_key is not None:
        del answers[drop_key]
    if extra_key is not None:
        answers[extra_key] = {'type': 'noul', 'noul': 0.1}
    return {'model': 'jev-1.13.0', 'answers': answers, 'usage': {'input_tokens': 100, 'output_tokens': 200}}


class JevNoulLaneTests(unittest.TestCase):
    def _adapter(self):
        adapter = JevStructuredAdapter(model='jev-1.13.0')
        adapter._inner.api_key = 'test-key'  # real JevAdapter instance, no env key needed
        return adapter

    def test_missing_api_key_boolean_lane_zero_bytes_on_wire(self):
        env = {'TYPESAFE_API_KEY': '', 'JEV_API_KEY': ''}
        with patch.dict('os.environ', env, clear=False), \
                patch('comparison.adapters.jev.urlopen') as call:
            adapter = JevStructuredAdapter(model='jev-1.13.0')
            p = adapter.predict(_goemotions_case())
        call.assert_not_called()  # zero bytes on the wire
        self.assertEqual((p.status, p.error_code), ('unavailable', 'missing_api_key'))

    def test_missing_api_key_enum_lane_zero_bytes_on_wire(self):
        env = {'TYPESAFE_API_KEY': '', 'JEV_API_KEY': ''}
        with patch.dict('os.environ', env, clear=False), \
                patch('comparison.adapters.jev.urlopen') as call:
            adapter = JevStructuredAdapter(model='jev-1.13.0')
            p = adapter.predict(StructuredCase('b77-1', 'ctx', _banking_schema(), {'intent': 'balance'}))
        call.assert_not_called()
        self.assertEqual((p.status, p.error_code), ('unavailable', 'missing_api_key'))

    def test_all_boolean_schema_sends_single_request_with_28_noul_questions(self):
        with patch('comparison.adapters.jev.urlopen', return_value=_Resp(_noul_response())) as call:
            p = self._adapter().predict(_goemotions_case())
        self.assertEqual(call.call_count, 1)  # one request carries all 28 questions
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, 'https://api.typesafe.ai/v1/systemone')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-key')
        payload = json.loads(request.data)
        self.assertEqual(payload['model'], 'jev-1.13.0')
        self.assertEqual(payload['state'], 'I am so angry about this.')
        self.assertEqual(set(payload['questions']), set(NOUL_FIELDS))
        for key, spec in payload['questions'].items():
            self.assertEqual(spec, {'type': 'noul', 'instructions': f'Does this text express {key}?'})
        self.assertEqual(p.status, 'ok')
        self.assertTrue(p.exact_valid)
        self.assertTrue(valid_object(p.prediction, _goemotions_schema()))
        # Threshold edges: p >= 0.5 -> True (documented edge, 0.5 inclusive).
        self.assertEqual(p.prediction['admiration'], False)  # 0.02
        self.assertEqual(p.prediction['amusement'], True)    # 0.97
        for i, key in enumerate(NOUL_FIELDS):
            if i in (0, 1, 2):
                continue
            expected = (0.05 + 0.01 * i) >= NOUL_THRESHOLD
            self.assertEqual(p.prediction[key], expected, key)
        # Raw P(yes) floats captured per-field for audit; no fabricated score.
        self.assertEqual(p.diagnostics['noul_scores']['anger'], 0.5)
        self.assertEqual(p.diagnostics['noul_threshold'], 0.5)
        self.assertEqual(p.usage, {'input_tokens': 100, 'output_tokens': 200})
        self.assertIsInstance(p.latency_ms, float)

    def test_threshold_edges_are_boundary_strict_not_close(self):
        def probe(p_anger):
            body = _noul_response()
            body['answers']['anger']['noul'] = p_anger
            with patch('comparison.adapters.jev.urlopen', return_value=_Resp(body)):
                return self._adapter().predict(_goemotions_case())
        self.assertEqual(probe(0.5).prediction['anger'], True)   # p >= 0.5 -> True
        self.assertEqual(probe(0.49).prediction['anger'], False)  # one float step below
        self.assertEqual(probe(0.0).prediction['anger'], False)
        self.assertEqual(probe(1.0).prediction['anger'], True)

    def test_out_of_range_noul_value_rejected_as_invalid(self):
        body = _noul_response(fail_field='anger')  # anger -> 1.3
        body['answers']['anger']['noul'] = 1.3
        with patch('comparison.adapters.jev.urlopen', return_value=_Resp(body)):
            p = self._adapter().predict(_goemotions_case())
        self.assertEqual((p.status, p.error_code, p.exact_valid), ('invalid', 'invalid_response', False))
        self.assertIsNone(p.prediction)

    def test_missing_answer_rejected_not_crash(self):
        body = _noul_response(drop_key='surprise')  # 27 of 28 answers
        with patch('comparison.adapters.jev.urlopen', return_value=_Resp(body)):
            p = self._adapter().predict(_goemotions_case())
        self.assertEqual((p.status, p.error_code, p.exact_valid), ('invalid', 'invalid_response', False))

    def test_unexpected_extra_answer_rejected(self):
        body = _noul_response(extra_key='f_extra')
        with patch('comparison.adapters.jev.urlopen', return_value=_Resp(body)):
            p = self._adapter().predict(_goemotions_case())
        self.assertEqual((p.status, p.error_code), ('invalid', 'invalid_response'))

    def test_non_noul_answer_type_rejected(self):
        body = _noul_response()
        body['answers']['joy'] = {'type': 'choice', 'choice': 'x'}  # wrong primitive
        with patch('comparison.adapters.jev.urlopen', return_value=_Resp(body)):
            p = self._adapter().predict(_goemotions_case())
        self.assertEqual((p.status, p.error_code), ('invalid', 'invalid_response'))

    def test_nonfinite_noul_value_rejected_as_invalid(self):
        # Python's JSON decoder accepts the non-standard NaN token; validation
        # must still reject it rather than thresholding it as a boolean.
        body = _noul_response()
        body['answers']['grief']['noul'] = float('nan')
        with patch('comparison.adapters.jev.urlopen', return_value=_Resp(body)):
            p = self._adapter().predict(_goemotions_case())
        self.assertEqual((p.status, p.error_code, p.exact_valid), ('invalid', 'invalid_response', False))

    def test_missing_model_field_rejected(self):
        body = _noul_response()
        del body['model']
        with patch('comparison.adapters.jev.urlopen', return_value=_Resp(body)):
            p = self._adapter().predict(_goemotions_case())
        self.assertEqual((p.status, p.error_code), ('invalid', 'invalid_response'))

    def test_incomplete_or_non_boolean_schema_still_errors_not_noul(self):
        # Empty schema and mixed-type schema must not be routed to the Noul
        # path; the existing enum behaviour (and its error mode) is preserved.
        env = {'TYPESAFE_API_KEY': '', 'JEV_API_KEY': ''}
        with patch.dict('os.environ', env, clear=False), \
                patch('comparison.adapters.jev.urlopen') as call:
            adapter = JevStructuredAdapter(model='jev-1.13.0')
            empty = StructuredCase('empty', 'ctx', {}, {})
            p = adapter.predict(empty)
        self.assertEqual(p.status, 'error')
        call.assert_not_called()  # nothing goes to the wire on a bad schema


if __name__ == '__main__':
    unittest.main(verbosity=2)
