"""Second-run test: same 10x context, run each engine twice.
Question: is the prefill (KV cache) reused across calls, or recomputed?
"""
import json
import time

from core.schema import StructuredSchema
from core import engine

model, tokenizer = engine.get_engine()

preset = json.load(open("presets/support_triage.json"))
schema = StructuredSchema(preset["schema"])
base_ctx = preset["context"]

FILLER = (
    "The operations team reviewed the logs and recorded the observations in the "
    "incident database, then handed the summary to the on-call rotation for follow-up. "
)
ctx = base_ctx
target = len(tokenizer.encode(base_ctx)) * 10
while len(tokenizer.encode(ctx)) < target:
    ctx += FILLER
print(f"context: {len(tokenizer.encode(ctx))} tokens\n")

for i in (1, 2):
    t0 = time.perf_counter()
    par = engine.run_parallel_generation(ctx, schema)
    wall = (time.perf_counter() - t0) * 1000
    print(
        f"parallel run {i}: wall {wall:7.0f} ms | prefill {par['prefill_ms']:7.2f} ms | "
        f"suffix {par['suffix_eval_ms']:6.2f} ms | reported total {par['elapsed_ms']} ms"
    )

for i in (1, 2):
    t0 = time.perf_counter()
    naive = engine.run_naive_generation(ctx, schema)
    wall = (time.perf_counter() - t0) * 1000
    print(f"naive    run {i}: wall {wall:7.0f} ms | valid_json={naive.get('is_valid_json')}")
