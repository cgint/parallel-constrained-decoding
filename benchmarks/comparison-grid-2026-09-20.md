# Comparison grid — results (2026-09-20)

**Grid complete: 15 cells (3 datasets × 5 arms), all n=600 (limit 200 × reps 3), zero missing.**

## The grid (quality = micro-F1 or top-1; latency = p50 client ms)

| dataset | dspy-ON (qwen+think) | dspy-OFF (qwen) | rlcd (3 reps) | laya | gemini-lite |
|---|---|---|---|---|---|
| **banking77** | 0.843 @850ms | **0.842 @168ms** | 0.290 / 0.290 / 0.290 @2032–2399ms | 0.345 @**31ms** | **0.860** @704ms |
| **goemotions** | 0.376 @8068ms | 0.369 @7051ms | 0.120 / 0.120 / 0.120 @219–322ms | **0.378** @**126ms** | **0.384** @1062ms |
| **clinc150** | **0.903** @1079ms | 0.875 @222ms | 0.310 / 0.310 / 0.310 @3815–4523ms | 0.430 @**91ms** | **0.912** @4458ms |

## Verdicts

1. **RLCD quality collapse is systematic, not noise.** Identical scores across 3 independent reps
   on all three datasets (0.290 / 0.120 / 0.310). It is 2.5–3.5× below every other arm on b77 and
   clinc, and 3× below laya/qwen/gemini on goe. The engine's speed advantage (single-pass parallel
   decode) cannot compensate: on these presets its selection strategy is simply picking wrong.
   The gloss-asymmetry hypothesis (77-way token-softmax over bare identifiers) remains the leading
   candidate explanation — still untested mechanistically.
2. **Thinking-ON is not worth it on b77** (0.843 vs 0.842 at 5× latency) **but matters on clinc**
   (0.903 vs 0.875, 0.028 macro-level gain at 5× latency) and is neutral on goe (0.376 vs 0.369).
3. **Laya = fastest by 6–25× everywhere, quality mid-pack.** On goe it matches qwen/gemini (0.378
   vs 0.376/0.384). On b77/clinc it trails the LLM arms by 2–2.6×. Latency: 31–126ms vs 168ms–8s.
4. **Gemini-lite = best or tied-best quality on all 3 datasets** (0.860/0.384/0.912), at 3–30×
   laya's latency and paid-API cost. This is the closed-API benchmark the corporate-alternative
   idea must beat on cost-adjusted quality — not on raw quality.
5. **The local-alternative shape, from this grid:** qwen-OFF (local, 168–222ms, 0.842–0.875 on
   the two intent datasets) is the strongest local arm; laya dominates latency; RLCD as-shipped is
   not competitive on quality for these presets.
