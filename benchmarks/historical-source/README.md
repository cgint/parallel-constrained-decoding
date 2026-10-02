# Historical source snapshot (2026-09-20 benchmark grid)

The retained original benchmark harness and byte-identical copies of saved
orchestration associated with the published 2026-09-20 comparison grid
([`benchmarks/comparison-grid-2026-09-20.md`](../comparison-grid-2026-09-20.md)).
The shared decision/measurement code is the retained original harness in
`comparison/`; this package adds the verbatim archived orchestration
associated with that grid. Published AS-IS for transparency:
**no code was improved, no behavior changed, no new numbers computed, no
driver executed during packaging.**

## What is published, and what it drove

| Published column | Source-run dirs (per dataset) | Actual runner / driver |
|---|---|---|
| RLCD (3 repetitions) | `power-rlcd-banking77-rep1/2/3`, `power-rlcd-goemotions-rep1/2/3`, `camp-rlcd-clinc150` + `power-rlcd-clinc150-rep2/3` | `comparison.multi_field.run --providers rlcd --limit 200 --warmup 1 --repetitions 3` (shared CLI, see Reproduction notes), invoked by `laya/camp_runner2.py` (`run_rlcd()`) |
| DSPy qwen, thinking ON | `camp-dspy-on-banking77`, `camp-dspy-on-goemotions`, `camp-dspy-on-clinc150` | same shared CLI, `--providers dspy --dspy-model qwen3.8-27b-nvfp4-dflash2-direct@<local-host>:40115` (no `--dspy-no-thinking`), via `drivers/nova_serial.sh` / `drivers/remote_serial.sh` |
| DSPy qwen, thinking OFF | `power-b77-qwen-off`, `camp-dspy-off-goemotions`, `camp-dspy-off-clinc150` | same, + `--dspy-no-thinking`, via the serial drivers and `drivers/banking77_run_arms.sh` |
| DSPy Gemini (hosted) | `camp-gemini-banking77`, `camp-gemini-goemotions`, `camp-gemini-clinc150` | same, `--dspy-model gemini/gemini-3.5-flash-lite` |
| Laya | `camp-laya-banking77`, `camp-laya-goemotions`, `camp-laya-clinc150` | `laya/camp_runner2.py` directly (imports and reuses `comparison.multi_field.{dataset,metrics,models}`) |
| Jev | `power-b77-jev`, `power-goe/jev`, `clinc150-20260920-power/jev` | shared CLI, `--providers jev --allow-paid` |

Full per-run mapping, settings, and stored summary metrics:
[`provenance.json`](provenance.json). Run names map 1:1 to the `ARMS` table in
the original (private, not published) tool
`agent/tools/check_cohort_identity.py`.

Standard protocol of every published arm: 200 selected cases x 3
repetitions = 600 attempted rows, 1 warmup case discarded, concurrency 1.
These are not 600 independent cases. DSPy thinking-ON recorded 1 execution
error on BANKING77, 1 on GoEmotions, and 3 on CLINC150. The original quality
metrics exclude execution errors from their denominators (599, 599, and
597 evaluated rows respectively); this publication does not change that
method. Status counts are retained in `provenance.json`.

## Repository code (shared modules)

The RLCD / DSPy / Gemini / Jev columns were produced by the shared harness in
`comparison/`: `multi_field/{run,dataset,adapters,metrics,models}.py` (with
`djev_adapter.py` as part of the import closure of the retained source — the
Djev adapter did **not** participate in producing the 6-arm grid; no Djev grid
was ever published), the original scalar modules `comparison/{run,dataset,
metrics,models,__init__}.py`, and `comparison/adapters/` (imported by the
published tests and by `multi_field` Jev plumbing). The `routing_v1` fixture
lives in `comparison/datasets/`.

**Snapshot honesty:** the published shared code is the *retained original
harness*, **not** a per-run immutable snapshot. No artifact pins the exact
on-disk revision of `comparison/*` that ran on 2026-09-20. What IS pinned:
(a) the archived driver/runner files below, verbatim; (b) the stored
`summary.json` metrics per run; (c) a 2026-10-02 read-only cross-check that
recomputed quality from the stored per-row results and confirmed it matches
the stored quality in all 24 runs, with identical 200-case cohorts per
dataset. Do not read this as "every byte was frozen at measurement time" —
that guarantee is not made and cannot be made.

## Archived drivers and runners

- `laya/camp_runner2.py` — the **actual** runner for all three Laya arms and
  the clinc150/rlcd camp cells. It accepts the cell name
  (`camp-laya-banking77`, ...); the manifest's `command` string form
  (`camp_runner2.py laya banking77`) is a stale label and **not accepted** by
  the parser. It reuses `comparison.multi_field.{dataset,metrics,models}` for
  case selection and scoring.
- `drivers/banking77_run_arms.sh`, `drivers/goemotions_run_all_arms.sh` —
  per-dataset arm drivers (qwen ON/OFF, Gemini, Jev) for the `power-*` cells.
- `drivers/clinc150_run_arms.py` and `drivers/clinc150_run_phase2.py` — two
  variants of the clinc150 driver. **Which one drove the final published
  clinc150 dspy cells is not recorded in any retained artifact.** Both point
  at the same shared CLI; the difference is env-var key plumbing and the
  default output dir. Retained for provenance completeness, not as proof of
  the final invocation.
- `drivers/nova_serial.sh`, `drivers/remote_serial.sh` — serial lock-guarded
  drivers that invoked the shared CLI for the `camp-dspy-*` / `power-*`
  dspy/Jev cells.

**These archived scripts contain host-specific historical configuration: the
original absolute repository path (`/Users/cgint/...`) and private LAN
endpoint host `pluto` / `192.168.1.124:40115`. These are NOT credentials and
are retained verbatim for fidelity.** They are historical, host-specific
orchestration — **not portable commands and not recommended for execution.**
Do not run them on another machine as-is.

**Early 1-repetition variants.** `nova_serial.sh`, `remote_serial.sh`, and
the `*_run_arms*` scripts above are retained exactly as they existed; some
cell invocations in them use `--repetitions 1`. This is **NOT** proof that
those variants drove the final published 3-repetition runs — the published
grid's authoritative settings are the per-run manifests (reproduced in
`provenance.json`: `repetitions: 3` for all 24 runs).

## Laya dependency

- Upstream: <https://github.com/mizorewww/laya-mlx>, **Apache-2.0** (see the
  upstream LICENSE/NOTICE for the dependency; the package is not vendored
  here).
- Exact commit used by the local tracked tree:
  `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337` (local tracked tree had no
  modifications).
- Model: Hugging Face `aac6fef/laya-mlx` — this is a **model ID, not a git
  revision**; weights are fetched at runtime and are **not** redistributed
  with this package.
- Only `camp_runner2.py` (the actual runner) is archived from the local
  Laya scratch directory. The superseded `camp_runner.py` (earlier,
  1-repetition, lock-based variant) was **not** archived: the final Laya
  arm manifests name `camp_runner2.py` verbatim in their `command` field.

## Reproduction notes (historical, not a promise)

The shared CLI run shape for the non-Laya arms was:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m comparison.multi_field.run \
  --providers <rlcd|dspy|jev> --dataset <banking77|goemotions|clinc150> \
  --limit 200 --warmup 1 --repetitions 3 --out comparison/results/<cell>
# dspy arms: + --dspy-model <model-or-endpoint>; OFF arms: + --dspy-no-thinking
# jev arms:  + --allow-paid
```

The historical dspy model string was
`qwen3.8-27b-nvfp4-dflash2-direct@<local-host>:40115` (a local LiteLLM
proxy endpoint on the original machine; `<local-host>` placeholder in docs —
the archived drivers retain the original host). The Gemini arm used the
hosted `gemini/gemini-3.5-flash-lite`.

**Cost warning:** `--allow-paid` gates the Jev arm only. The default DSPy
arm can target a hosted model (e.g. Gemini) and **may incur cost**;
reproducing dspy/Gemini/Jev arms requires API credentials and spending.

Known limitations (honest, unchanged from the original work):

- Datasets were fetched from dynamic `main`/`master` upstream sources
  (banking77, GoEmotions, CLINC150); the selection was seeded
  (seed 20260919 where applicable) but upstream source drift is possible.
- Model/service versions were **unpinned** at run time (only
  `requirements-comparison.txt` pins `dspy>=3.2.1,<3.3` for the DSPy
  baseline).
- The clinc150 path fell back to the **system `python3`** for parquet
  reads; that interpreter needed `pyarrow` + `pandas` installed.
- Enum-token diagnostics in the engine (bounded-choice candidate scoring in
  `core/`) are unrelated to Djev.
- The original `comparison/README.md` describes a small **synthetic smoke**
  run, not this real 200-case x 3-repetition grid.
- The `djev` import in `comparison/multi_field/run.py` was added **after**
  the grid was produced; the dependency is retained unchanged (no Djev grid
  was ever published).

No live rerun is claimed anywhere in this package; the archived drivers were
**not executed** during packaging.
