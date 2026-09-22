"""Scaling experiment: naive vs parallel constrained latency across input sizes.

Pads the support_triage context with neutral filler to ~500/2k/6k/12k tokens,
times both engines, and prints the speedup curve.
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


def pad_to_tokens(ctx: str, target: int) -> str:
    while len(tokenizer.encode(ctx)) < target:
        ctx += FILLER
    return ctx


rows = []
for target in [500, 2000, 6000, 12000]:
    ctx = pad_to_tokens(base_ctx, target)
    n_ctx = len(tokenizer.encode(ctx))

    t0 = time.perf_counter()
    naive = engine.run_naive_generation(ctx, schema)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    par = engine.run_parallel_generation(ctx, schema)
    t_par = time.perf_counter() - t0

    rows.append(
        (
            n_ctx,
            round(t_naive * 1000),
            naive.get("total_tokens_generated", "?"),
            round(t_par * 1000),
            par.get("prefill_ms", "?"),
            round(t_naive / t_par, 2),
        )
    )
    print(
        f"ctx={n_ctx:>6d} tok | naive {t_naive*1000:7.0f} ms "
        f"({naive.get('total_tokens_generated','?')} gen tok) | "
        f"parallel {t_par*1000:7.0f} ms (prefill {par.get('prefill_ms','?')} ms) | "
        f"speedup {t_naive/t_par:.2f}x",
        flush=True,
    )

print("\nTable: ctx_tok | naive_ms | gen_tok | par_ms | prefill_ms | speedup")
for r in rows:
    print(r)
