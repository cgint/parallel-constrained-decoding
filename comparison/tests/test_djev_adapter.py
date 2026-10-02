"""Offline tests for DjevStructuredAdapter.

These run with NO live djev endpoint: the transport is monkeypatched to a
mock that returns a canned /v1/systemone response. They lock the two things
that could silently break the grid cell if we only trusted a live run:

  1. The 77/150-way `choice` -> prediction mapping (the adapter must consume
     the server's returned `choice` verbatim and validate it against the
     case schema's enum).
  2. The top-20 logprob floor: djev's softmax is over top_logprobs=20, so for
     a 77-way set most labels fall outside and the server floors them at
     (min-5.0). The adapter must trust the server's `probabilities` dict
     (floor already applied) and NOT re-derive it.

Also covered: the boolean (goe) lane, the invalid-case guard (multi-enum),
and the out-of-enum rejection path.
"""
import unittest
from unittest.mock import patch

from comparison.multi_field.djev_adapter import DjevStructuredAdapter
from comparison.multi_field.models import StructuredCase, StructuredPrediction


def _enum_case(choices, gold, context='Payment failed'):
    return StructuredCase(
        case_id='c1', context=context,
        schema={'intent': {'type': 'enum', 'description': 'intent.', 'choices': list(choices)}},
        gold={'intent': gold})


def _bool_case(n=3):
    schema = {f'f{i}': {'type': 'boolean', 'description': f'f{i}.'} for i in range(n)}
    return StructuredCase(case_id='b1', context='x', schema=schema,
                          gold={f'f{i}': bool(i % 2) for i in range(n)})


class DjevAdapterTest(unittest.TestCase):

    def _adapter(self, canned_response):
        ad = DjevStructuredAdapter(endpoint='http://mock:8080')
        captured = {}

        def fake_post(payload):
            captured['payload'] = payload
            return canned_response
        ad._post_systemone = fake_post
        return ad, captured

    # -- enum lane: 77-way mapping ---------------------------------------

    def test_enum_77way_consumes_server_choice_verbatim(self):
        choices = [f'cls_{i:02d}' for i in range(77)]
        canned = {
            'model': 'djev-dgemma',
            'answers': {'intent': {
                'choice': 'cls_07',
                # top-20 floored: only 20 labels present, rest at a small floor
                'probabilities': {f'cls_{i:02d}': 0.001 for i in range(77)},
                'confidence': 0.31}},
            'diagnostics': {'timing': {'total_ms': 116.3}},
        }
        ad, captured = self._adapter(canned)
        p = ad.predict(_enum_case(choices, gold='cls_07'))
        self.assertEqual(p.status, 'ok')
        self.assertEqual(p.prediction, {'intent': 'cls_07'})
        self.assertTrue(p.exact_valid)
        # probability dict must be the server's, not re-derived
        self.assertEqual(p.diagnostics['probabilities']['intent'][choices[0]], 0.001)
        self.assertEqual(p.diagnostics['server_total_ms'], 116.3)
        # payload must carry all 77 criteria as [[name, None], ...]
        self.assertEqual(len(captured['payload']['questions'][0]['choices']), 77)
        self.assertEqual(captured['payload']['questions'][0]['type'], 'choice')
        self.assertEqual(captured['payload']['state'], 'Payment failed')

    def test_enum_out_of_enum_is_invalid_not_ok(self):
        choices = [f'cls_{i}' for i in range(77)]
        canned = {'model': 'djev-dgemma',
                  'answers': {'intent': {'choice': 'NOT_A_CHOICE',
                                          'probabilities': {'NOT_A_CHOICE': 1.0},
                                          'confidence': 1.0}},
                  'diagnostics': {'timing': {'total_ms': 50.0}}}
        ad, _ = self._adapter(canned)
        p = ad.predict(_enum_case(choices, gold='cls_0'))
        self.assertEqual(p.status, 'invalid')
        self.assertEqual(p.error_code, 'out_of_enum')
        self.assertIsNone(p.prediction)

    # -- boolean (goe) lane ----------------------------------------------

    def test_boolean_lane_maps_noul_to_bool(self):
        canned = {'model': 'djev-dgemma',
                  'answers': {
                      'f0': {'noul': 0.9, 'probability': 0.9},
                      'f1': {'noul': 0.1, 'probability': 0.1},
                      'f2': {'noul': 0.7, 'probability': 0.7}},
                  'diagnostics': {'timing': {'total_ms': 88.0}}}
        ad, captured = self._adapter(canned)
        p = ad.predict(_bool_case(3))
        self.assertEqual(p.status, 'ok')
        # noul is a probability; the boolean prediction must be the argmax-vs-0.5
        # interpretation the server's `noul` field already gives as P(yes).
        # The adapter casts bool(noul) -- so we assert the deterministic cast:
        self.assertEqual(p.prediction, {'f0': True, 'f1': False, 'f2': True})
        # all three questions were sent as noul, WITH the yes/no choices+labels
        # the server's noul branch needs to read probs[0] as P(yes).
        q0 = captured['payload']['questions'][0]
        self.assertTrue(all(q['type'] == 'noul' for q in captured['payload']['questions']))
        self.assertEqual(q0['choices'], [['yes', None], ['no', None]])
        self.assertEqual(q0['labels'], ['yes', 'no'])

    # -- guards -----------------------------------------------------------

    def test_multi_enum_rejected_locally(self):
        case = StructuredCase(
            case_id='m', context='x',
            schema={'a': {'type': 'enum', 'choices': ['x', 'y']},
                    'b': {'type': 'enum', 'choices': ['p', 'q']}},
            gold={'a': 'x', 'b': 'p'})
        ad, _ = self._adapter({})
        p = ad.predict(case)
        self.assertEqual(p.status, 'error')
        self.assertEqual(p.error_code, 'invalid_case')
        # nothing was sent to the endpoint
        self.assertEqual(p.latency_ms, None)

    def test_missing_answer_key_is_malformed(self):
        choices = [f'cls_{i}' for i in range(5)]
        canned = {'model': 'djev-dgemma', 'answers': {},
                  'diagnostics': {'timing': {'total_ms': 10.0}}}
        ad, _ = self._adapter(canned)
        p = ad.predict(_enum_case(choices, gold='cls_0'))
        self.assertEqual(p.status, 'invalid')
        self.assertEqual(p.error_code, 'malformed_response')


if __name__ == '__main__':
    unittest.main()
