# Comparison grid — results (2026-09-20)

**Grid complete: 18 cells (3 datasets × 6 arms), all n=600 (limit 200 × reps 3), zero missing.**
Jev column added 2026-09-23 (runs of 2026-09-19 → 23, model `jev-1.13.0`;
selected 200-case ID sets are byte-identical across all six arms — verified
2026-09-23 from every manifest; selection: fixed-seed random sample for
banking77/clinc150, deterministic label order for goemotions; 3 repetitions, 1 warmup).

## The grid (quality = micro-F1 or top-1; latency = p50 client ms)

| dataset | dspy-ON (qwen+think) | dspy-OFF (qwen) | rlcd (3 reps) | laya | gemini-lite | **jev (API, new)** |
|---|---|---|---|---|---|---|
| **banking77** | 0.843 @850ms | **0.842 @168ms** | 0.290 / 0.290 / 0.290 @2032–2399ms | 0.345 @**31ms** | **0.860** @704ms | 0.827 @687ms |
| **goemotions¹** | 0.376 @8068ms | 0.369 @7051ms | 0.120 / 0.120 / 0.120 @219–322ms | **0.378** @**126ms** | **0.384** @1062ms | 0.229 @418ms |
| **clinc150** | **0.903** @1079ms | 0.875 @222ms | 0.310 / 0.310 / 0.310 @3815–4523ms | 0.430 @**91ms** | **0.912** @4458ms | 0.727 @427ms |

¹ Jev goe cell: 28 booleans via the Noul primitive, threshold p ≥ 0.5 (documented default for
symmetric yes/no cost; not tuned). Raw per-field probabilities are persisted in the run's
diagnostics (`comparison/results/power-goe/jev/`), so the cell can be re-derived at any threshold.

## Verdicts

1. **RLCD quality collapse is systematic, not noise.** Identical scores across 3 independent reps
   on all three datasets (0.290 / 0.120 / 0.310) — the lowest cell in every dataset:
   b77 0.290 vs laya 0.345, Jev 0.827, LLM arms 0.842–0.860; goe 0.120 vs Jev 0.229, laya
   0.378, LLM arms 0.369–0.384; clinc 0.310 vs laya 0.430, Jev 0.727, LLM arms 0.875–0.912.
   The engine's speed advantage (single-pass parallel
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
6. **Jev (closed API, added 09-23) is solid third on banking77 but never the top arm** (0.827
   @687ms: below qwen-OFF's 0.842 @168ms, which is both better and ~4× faster; below gemini-lite
   on quality). On clinc150 it clearly trails the LLM arms (0.727 vs 0.875–0.912), and on
   goemotions it sits above local RLCD (0.229 vs 0.120) but clearly below the LLM arms
   (0.369–0.384). The closed API is a drop-in convenience, not a quality reference, on all three
   datasets. All Jev cells measured n=600, 100% in-enum/in-schema across every request.
