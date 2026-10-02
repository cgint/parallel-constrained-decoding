"""One-call-per-record structured adapters for Boolean and enum lanes."""
from __future__ import annotations

from contextlib import nullcontext
from importlib.util import find_spec
import os
from time import perf_counter
from typing import Any, Literal

from .models import StructuredCase, StructuredPrediction


DEFAULT_DSPY_MODEL = 'gemini/gemini-3.5-flash-lite'


def valid_object(value: Any, schema: dict[str, dict[str, Any]]) -> bool:
    if not isinstance(value, dict) or set(value) != set(schema):
        return False
    for key, spec in schema.items():
        if spec['type'] == 'boolean' and type(value[key]) is not bool:
            return False
        if spec['type'] == 'enum' and (type(value[key]) is not str or value[key] not in spec['choices']):
            return False
    return True


def split_model_endpoint(model: str) -> tuple[str, str | None]:
    """Parse '<model>[@host:port]' into (model, base_url|None).

    A plain name (no '@') keeps legacy behavior (default providers via env keys);
    'a@host' or 'a@host:port' becomes an OpenAI-compatible endpoint at
    http://host:port/v1 (https when a scheme is given explicitly). Used to point
    the dspy arm at local servers (e.g. a LiteLLM proxy) without hardcoding any
    address."""
    if '@' in model:
        name, endpoint = model.rsplit('@', 1)
        endpoint = endpoint.strip()
        if not endpoint:
            raise ValueError('empty endpoint after @ in model string')
        base = endpoint if endpoint.startswith(('http://', 'https://')) else f'http://{endpoint}'
        if not base.endswith('/v1'):
            base = f'{base}/v1'
        return name, base
    return model, None


class RLCDStructuredAdapter:
    provider = 'rlcd'

    def __init__(self, model='Qwen2.5-1.5B-Instruct-4bit'):
        self.model = model

    def availability(self):
        return True

    def predict(self, case: StructuredCase):
        started = perf_counter()
        try:
            from core.engine import run_rlcd_generation
            from core.schema import StructuredSchema
            result = run_rlcd_generation(case.context, StructuredSchema(case.schema), temperature=1.0)
        except (ImportError, ModuleNotFoundError):
            return StructuredPrediction.unavailable(self.provider, self.model, 'engine_unavailable')
        except Exception:
            return StructuredPrediction.error(self.provider, self.model, 'engine_error', (perf_counter() - started) * 1000)
        parsed = result.get('parsed_json') if isinstance(result, dict) else None
        try:
            prediction = {key: value['value'] for key, value in parsed.items()}
        except (AttributeError, TypeError):
            prediction = None
        latency = (perf_counter() - started) * 1000
        if not valid_object(prediction, case.schema):
            return StructuredPrediction.invalid(self.provider, self.model, 'invalid_response', latency)
        keys = ('elapsed_ms', 'prefill_ms', 'suffix_eval_ms', 'sequential_forward_passes', 'has_calibrated_probabilities')
        return StructuredPrediction(
            self.provider, self.model, 'ok', prediction, latency, True,
            diagnostics={'engine_timing_ms': {k: result[k] for k in keys[:3] if k in result}, 'engine_mode': result.get('mode'),
                         **{k: result.get(k) for k in keys[3:]}})


class DSPyStructuredAdapter:
    provider = 'dspy'

    def __init__(self, model=DEFAULT_DSPY_MODEL, dspy_module=None, no_thinking=False):
        self.model = model
        self.no_thinking = no_thinking
        self._dspy = dspy_module
        self._predictors = {}

    def availability(self):
        return self._dspy is not None or (find_spec('dspy') is not None and bool(
            os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY') or split_model_endpoint(self.model)[1]
        ))

    def _load(self):
        if self._dspy is None:
            import dspy
            self._dspy = dspy
        return self._dspy

    def _lm_params(self):
        """Return (model, extra_kwargs) for dspy.LM.

        A plain name (no '@') keeps legacy behavior: dspy resolves the default
        provider from the name prefix + ambient keys (e.g. gemini/gemini-3.5-flash-lite).
        An explicit endpoint ('name@host:port') is routed through LiteLLM's OpenAI
        provider against that OpenAI-compatible base_url; the model is prefixed
        with 'openai/' so LiteLLM does not try to infer a provider for the bare name.
        """
        model, base = split_model_endpoint(self.model)
        if base is None:
            return model, {}
        prefixed = model if model.startswith('openai/') else f'openai/{model}'
        extra = {'api_base': base, 'api_key': os.environ.get('OPENAI_API_KEY') or 'unused'}
        # LiteLLM on this proxy rejects top-level 'thinking' and ignores plain
        # 'enable_thinking'; chat_template_kwargs.enable_thinking is the switch
        # that actually reaches Qwen3.x chat templates (verified 2026-09-20).
        if self.no_thinking:
            extra['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
        return prefixed, extra

    def predict(self, case: StructuredCase):
        if not self.availability():
            return StructuredPrediction.unavailable(self.provider, self.model, 'dspy_not_installed')
        started = perf_counter()
        keys = tuple(case.schema)
        try:
            dspy = self._load()
            if keys not in self._predictors:
                fields = {'context': (str, dspy.InputField(desc='Text to classify.'))}
                for key, spec in case.schema.items():
                    field_type = bool if spec['type'] == 'boolean' else Literal[tuple(spec['choices'])]
                    fields[key] = (field_type, dspy.OutputField(desc=spec['description']))
                sig = dspy.Signature(fields, 'Return every field as a native JSON value matching its declared type.')
                dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
                lm_model, lm_extra = self._lm_params()
                self._predictors[keys] = (dspy.Predict(sig), dspy.LM(model=lm_model, temperature=0, cache=False, **lm_extra), dspy.JSONAdapter())
            predictor, lm, adapter = self._predictors[keys]
            with (dspy.context(lm=lm, adapter=adapter, track_usage=True) or nullcontext()):
                result = predictor(context=case.context)
            prediction = {key: getattr(result, key, None) for key in keys}
            latency = (perf_counter() - started) * 1000
            if not valid_object(prediction, case.schema):
                return StructuredPrediction.invalid(self.provider, self.model, 'invalid_provider_object', latency)
            getter = getattr(result, 'get_lm_usage', None)
            usage = getter() if callable(getter) else {}
            return StructuredPrediction(self.provider, self.model, 'ok', prediction, latency, True, usage=dict(usage or {}))
        except Exception as exc:
            latency = (perf_counter() - started) * 1000
            parse = getattr(self._dspy, 'AdapterParseError', ())
            if isinstance(parse, type) and isinstance(exc, parse):
                return StructuredPrediction.invalid(self.provider, self.model, 'dspy_adapter_parse_error', latency)
            return StructuredPrediction.error(self.provider, self.model, 'dspy_provider_error', latency)


class _JevEvalCase:
    """Minimal EvalCase-shaped object for the shared JevAdapter (context + choices)."""

    def __init__(self, context, choices):
        self.context = context
        self.choices = tuple(choices)


class _JevNoulCase:
    """Minimal shape for the shared JevAdapter's Noul path.

    `questions` maps a question id (the schema field name) to its
    affirmative yes/no instructions. The shared adapter branches on the
    presence of a `questions` dict, so enum/case objects never take the
    Noul path (and vice versa)."""

    def __init__(self, context, questions):
        self.context = context
        self.questions = questions


def _all_boolean(schema: dict[str, dict[str, Any]]) -> bool:
    return bool(schema) and all(spec['type'] == 'boolean' for spec in schema.values())


class JevStructuredAdapter:
    """Enum-lane adapter for TypeSafe Jev, built on the shared JevAdapter.

    Reuses the shared adapter's endpoint, request shape, response validation, and
    missing-key path (JevAdapter.predict returns status 'unavailable' with
    error_code 'missing_api_key' when no credential is present -- no request is
    sent). The paid gate itself lives in comparison.multi_field.run (the
    --allow-paid flag); this adapter is only ever constructed for prediction
    after that gate has passed.
    """

    provider = 'jev'

    def __init__(self, model='jev-1.13.0'):
        self.model = model
        from comparison.adapters.jev import JevAdapter
        self._inner = JevAdapter(model=model)

    def availability(self):
        return self._inner.availability()

    def predict(self, case: StructuredCase):
        boolean_lane = _all_boolean(case.schema)
        if boolean_lane:
            # One Jev request carries all boolean fields as affirmative yes/no
            # Noul questions. Field ids remain stable for response matching.
            questions = {
                key: {'instructions': f"Does this text express {key.replace('_', ' ')}?"}
                for key in case.schema
            }
            inner = self._inner.predict(_JevNoulCase(case.context, questions))
        else:
            # The pre-existing Jev Choice lane is exactly one enum field. Do
            # not send malformed/mixed schemas to a paid endpoint.
            if len(case.schema) != 1:
                return StructuredPrediction.error(self.provider, self.model, 'invalid_case')
            key, spec = next(iter(case.schema.items()))
            if spec.get('type') != 'enum' or not isinstance(spec.get('choices'), list):
                return StructuredPrediction.error(self.provider, self.model, 'invalid_case')
            choices = tuple(spec['choices'])
            inner = self._inner.predict(_JevEvalCase(case.context, choices))
        if inner.status == 'ok':
            prediction = inner.prediction if boolean_lane else {key: inner.prediction}
            return StructuredPrediction(self.provider, inner.model, 'ok', prediction,
                                        inner.latency_ms, True, usage=dict(inner.usage or {}),
                                        diagnostics=inner.diagnostics or {})
        if inner.status == 'unavailable':
            return StructuredPrediction.unavailable(self.provider, self.model, inner.error_code or 'missing_api_key')
        if inner.status == 'invalid':
            return StructuredPrediction.invalid(self.provider, inner.model, inner.error_code or 'invalid_response', inner.latency_ms)
        return StructuredPrediction.error(self.provider, self.model, inner.error_code or 'adapter_error', inner.latency_ms)
