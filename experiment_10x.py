"""Smoke test: support_triage preset with 10x input token size, naive vs parallel."""
import json
import time

from core.schema import StructuredSchema
from core import engine

model, tokenizer = engine.get_engine()

preset = json.load(open("presets/support_triage.json"))
schema = StructuredSchema(preset["schema"])
base_ctx = preset["context"]

base_tok = len(tokenizer.encode(base_ctx))
target_tok = base_tok * 10
print(f"base context: {base_tok} tokens -> target {target_tok} tokens (10x)")

FILLER = (
    "The operations team reviewed the logs and recorded the observations in the "
    "incident database, then handed the summary to the on-call rotation for follow-up. "
)
ctx = base_ctx
while len(tokenizer.encode(ctx)) < target_tok:
    ctx += FILLER
ctx_tok = len(tokenizer.encode(ctx))
print(f"padded context: {ctx_tok} tokens")

t0 = time.perf_counter()
naive = engine.run_naive_generation(ctx, schema)
t_naive = time.perf_counter() - t0

t0 = time.perf_counter()
par = engine.run_parallel_generation(ctx, schema)
t_par = time.perf_counter() - t0

print(
    f"\nnaive:    {t_naive*1000:8.0f} ms | generated {naive.get('total_tokens_generated','?')} tokens | "
    f"valid_json={naive.get('is_valid_json')} schema_match={naive.get('schema_match')}"
)
print(
    f"parallel: {t_par*1000:8.0f} ms | prefill {par.get('prefill_ms')} ms, suffix {par.get('suffix_eval_ms')} ms | "
    f"valid_json={par.get('is_valid_json')} schema_match={par.get('schema_match')}"
)
print(f"\nspeedup at 10x input: {t_naive/t_par:.2f}x")
