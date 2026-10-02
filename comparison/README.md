# Finite-choice comparison harness

Compare the local **RLCD** decision engine, a **DSPy** structured-output baseline, and optional paid **Jev** Choice calls on the same four-label routing cases. DSPy is the workflow/baseline layer; RLCD and Jev are decision backends.

Workstream state: [`comparison/STATUS.md`](STATUS.md). Dated evaluations: [`comparison/evaluations/`](evaluations/).

## Run

From the repository root, use the existing MLX environment and install the optional DSPy baseline when needed:

```bash
.venv/bin/python -m pip install -r requirements-comparison.txt
```

Configure the credentials required by your chosen DSPy model, then run the eight-case local comparison:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m comparison.run \
  --providers rlcd,dspy --limit 8 --out comparison/results/rlcd-dspy-8
```

Jev is deliberately opt-in because it can incur API charges; set `TYPESAFE_API_KEY` and pass `--allow-paid`:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m comparison.run \
  --providers rlcd,dspy,jev --limit 8 --allow-paid \
  --out comparison/results/rlcd-dspy-jev-8
```

Use `--help` for model, warm-up, and repetition options. Limited runs are round-robin across gold labels; execution is sequential.

## Read the artifacts

Each output directory contains `manifest.json` (configuration, availability, platform and git state), `results.jsonl` (one row per attempted prediction), and `summary.json` (accuracy, latency, throughput, and score diagnostics). `comparison/results/` is local-only and ignored by Git.

- `ok`: exact permitted label returned.
- `invalid`: provider responded, but not with an exact permitted label.
- `error`: adapter/provider execution failed.
- `unavailable`: no attempt was made (for example, missing key, missing adapter, or Jev without `--allow-paid`).

**Do not compare scores across providers.** RLCD records candidate-slice softmax; Jev records the selected Choice probability and separately retains its distribution-shape confidence; DSPy has no common score here. Score diagnostics remain separated by provider/model/kind.

## Scope of a result

`routing_v1` is a deterministic, balanced **synthetic** routing-policy dataset, not evidence of production decision quality or calibration. Before any selection claim, run the same harness on a representative labeled production dataset and assess accuracy, calibration, p95 latency, concurrent throughput, privacy, and cost.

## Djev arm (seventh, in prep)

Djev (NVIDIA DiffusionGemma-26B-A4B, served via `taeold/djev-run`) is a parallel *diffusion* LLM — a distinct substrate from every other arm. It consumes the pre-built image's stateless `POST /v1/systemone` endpoint (no canvas logic is re-implemented in this repo; the adapter stays a clean consumer). See `comparison/STATUS.md` (section 2026-09-24) for the contract, the `chr(97+i)` post-`z` label caveat, and the argmax-only headline-metric rationale.

**Single ready-check (fire the moment the `/v1/systemone` endpoint is up):**

```bash
PYTHONPATH="$PWD" .venv/bin/python comparison/tools/djev_ready_check.py \
    --endpoint http://sparkz:8080
```

Sequence with baked-in exit codes: `0` = reachability + tokenization-alignment probe + 3-case dry-run all passed (77/150-way cells trustworthy; full sweep is a spend decision) · `1` = endpoint not up · `2` = align probe failed (do NOT publish 77/150-way; goemotions-only) · `3` = dry-run produced no `ok` rows. The full n=600 sweep is separate and spend-gated (see STATUS).
