"""Djev adapter: DiffusionGemma-Jev served via taeold/djev-run.

Djev is a DIFFERENT substrate from every other grid arm:
  - RLCD   = KV-cache sub-vocabulary logit slicing over a 1.5B model.
  - DSPy   = standard autoregressive LLM (local qwen or hosted Gemini).
  - Jev    = TypeSafe hosted API.
  - Djev   = NVIDIA DiffusionGemma-26B-A4B-NVFP4, parallel *diffusion*
             generation over a 128-token seed canvas, served through vLLM's
             nightly diffusion mode with an ASGI middleware exposing a
             stateless POST /v1/systemone endpoint.

The adapter speaks the pre-built `ghcr.io/taeold/djev-run:latest`
`/v1/systemone` contract (NOT the raw vLLM path, which would require manual
tokenize + vllm_xargs canvas building per request). That contract:

  request:  {"state": <str>, "steps": 1,
             "questions": [{"id","type":"choice","choices":[[name,desc],...],
                            "labels":["a","b",...]}]}
  response: {"model","answers":{<id>:{"choice","probabilities",
            "confidence"}},"diagnostics":{"timing":{"total_ms"}}}

The server maps criteria -> labels a,b,c... and returns the argmax choice plus
a normalized probability dict. Its softmax is over top_logprobs (top-20), so
for very large choice sets (BANKING77=77, CLINC150=150) some labels fall
outside the top-20 and the server applies a floor of (min-5.0); the returned
`probabilities` dict already reflects that floor. We consume `choice` and
`probabilities` directly; we do NOT re-derive them.

Schema-mapping strategy (one djev request per record):
  - all-boolean schema  -> one `noul` question per field (yes/no), matching the
                           Jev boolean lane so goemotions lines up with the
                           other grid arms.
  - single-enum schema  -> one `choice` question, choices=criteria list.
  - multi-enum / mixed  -> not sent; return an invalid-case error (we would
                           never fire a malformed request at a live endpoint).

Latency: we measure client-side wall-clock here (same semantics as the other
grid arms, so it includes the LAN hop to the GPU host), AND we record the
server's own `diagnostics.timing.total_ms` as a diagnostic. The grid headline
latency is the client-side number; the two are comparable but not identical
(server total_ms excludes the network round-trip).
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from .models import StructuredCase, StructuredPrediction


def _all_boolean(schema: dict[str, dict[str, Any]]) -> bool:
    return bool(schema) and all(spec['type'] == 'boolean' for spec in schema.values())


class DjevStructuredAdapter:
    provider = 'djev'

    def __init__(self, endpoint: str, model: str = 'djev-dgemma',
                 timeout_s: float = 30.0, steps: int = 1):
        """endpoint: full base URL of the djev-run systemone service, e.g.
        http://twins:8080  (no trailing path; /v1/systemone is appended)."""
        self.model = model
        self.endpoint = endpoint.rstrip('/')
        self.timeout_s = timeout_s
        self.steps = steps

    # -- transport ---------------------------------------------------------

    def _post_systemone(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            f'{self.endpoint}/v1/systemone',
            data=body,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode('utf-8'))

    # -- contract checks ---------------------------------------------------

    @staticmethod
    def _valid_choice_prediction(value: Any, choices: list[str]) -> bool:
        return isinstance(value, str) and value in choices

    @staticmethod
    def _valid_boolean_prediction(value: Any) -> bool:
        return type(value) is bool

    # -- public interface (mirrors the other adapters) --------------------

    def availability(self) -> bool:
        """Cheap reachability probe: GET the service root (the middleware serves
        a demo HTML page at '/'). No generation is triggered, so this is free
        and safe to run at provider-registration time."""
        try:
            request = urllib.request.Request(f'{self.endpoint}/', method='GET')
            with urllib.request.urlopen(request, timeout=min(5.0, self.timeout_s)) as response:
                return response.status == 200
        except Exception:
            return False

    def predict(self, case: StructuredCase) -> StructuredPrediction:
        """Predict one StructuredCase via /v1/systemone.

        Only two schema shapes are ever sent to the endpoint:
          * a single enum field -> one `choice` question (choices=criteria).
          * all-boolean fields  -> one `noul` question per field (yes/no).
        Anything else is rejected locally with an invalid_case error so we never
        fire a malformed request at a live (possibly paid) endpoint.
        """
        schema = case.schema
        fields = list(schema)

        if _all_boolean(schema):
            lane = 'boolean'
        elif len(schema) == 1 and schema[fields[0]]['type'] == 'enum':
            lane = 'enum'
        else:
            return StructuredPrediction.error(self.provider, self.model, 'invalid_case')

        # Build the systemone payload.
        if lane == 'boolean':
            # One noul question per field. Field ids remain stable for matching.
            # The server's noul branch expects labels ["yes","no"] and reads
            # probs[0] as P(yes). We must provide them explicitly in the list
            # form; the server's dict-form auto-generates them, but we use the
            # list form for uniformity with the enum lane.
            questions = []
            for key in fields:
                questions.append({
                    'id': key,
                    'type': 'noul',
                    'instructions': f"Does this text express {key.replace('_', ' ')}?",
                    'choices': [['yes', None], ['no', None]],
                    'labels': ['yes', 'no'],
                })
        else:
            key = fields[0]
            choices = list(schema[key]['choices'])
            questions = [{
                'id': key,
                'type': 'choice',
                'instructions': schema[key].get('description', ''),
                'choices': [[c, None] for c in choices],
                # The server re-derives labels a,b,c... itself from `choices`;
                # we pass them for parity but the contract is `choices`.
                'labels': [chr(97 + i) for i in range(len(choices))],
            }]

        payload = {'state': case.context, 'steps': self.steps, 'questions': questions}

        started = time.perf_counter()
        try:
            response = self._post_systemone(payload)
        except Exception:
            latency = (time.perf_counter() - started) * 1000
            return StructuredPrediction.error(self.provider, self.model, 'provider_error', latency)
        latency = (time.perf_counter() - started) * 1000

        answers = response.get('answers') or {}
        timing = ((response.get('diagnostics') or {}).get('timing') or {})
        server_total_ms = timing.get('total_ms')

        try:
            if lane == 'boolean':
                # The server's `noul` field is P(yes) (a float), NOT a boolean.
                # bool(0.1) is True in Python, so we MUST threshold at 0.5 --
                # the documented default for symmetric yes/no cost, identical to
                # the Jev-goe Noul lane. (A raw bool(noul) cast silently
                # misclassified every P(yes)<1.0 as True.)
                prediction = {key: (answers[key]['noul'] >= 0.5) for key in fields}
                exact = all(self._valid_boolean_prediction(prediction[k]) for k in fields)
            else:
                key = fields[0]
                choice = answers[key]['choice']
                prediction = {key: choice}
                exact = self._valid_choice_prediction(choice, list(schema[key]['choices']))
        except (KeyError, TypeError):
            return StructuredPrediction.invalid(self.provider, self.model, 'malformed_response', latency)

        if not exact:
            return StructuredPrediction.invalid(self.provider, self.model, 'out_of_enum', latency)

        # Preserve the server's normalized probabilities as a diagnostic so the
        # cell can be re-scored (e.g. at a different threshold) without a new
        # live run -- mirroring how Jev's goe Noul probabilities are persisted.
        diagnostics = {
            'server_total_ms': server_total_ms,
            'probabilities': {k: answers[k].get('probabilities') for k in fields},
            'model': response.get('model'),
        }
        return StructuredPrediction(
            self.provider, self.model, 'ok', prediction, latency,
            exact_valid=True, diagnostics=diagnostics)
