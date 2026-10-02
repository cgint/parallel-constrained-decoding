# Jev from a user's perspective: a practical comparison

## Statement

**As a user, I wanted to understand what Jev offers me compared with the classification tools I can already use. In these runs, ordinary, unoptimized DSPy setups returned valid structured answers and scored higher than Jev on all three datasets. Looking at those outcomes, I do not see a particular “wow” for my use case.**

This is a humble comparison of the options as I used them, not an attempt to prove something about Jev. It says nothing universal about which system is better, whether marketing claims are justified, or why the results differ internally.

## The relevant comparison

The user-facing question is what the system delivers: given text and a declared set of outputs, does it return a valid answer, and how often is that answer correct?

The comparison includes both a local Qwen model and hosted Gemini through DSPy. Consequently, explaining Jev's value by contrasting a hosted API with the burden of running a local model does not address the hosted-Gemini alternative already tested.

This assessment concerns classification quality and observed structured-output success. It does not decide a total-cost, latency, throughput, or calibration comparison. Those properties require their own evidence and a user requirement that makes them relevant.

## No task-specific DSPy training or optimization

The retained implementation in `comparison/multi_field/adapters.py` constructs:

- A typed DSPy signature from the supplied schema: Boolean fields or enum choices represented as `Literal` types.
- Plain `dspy.Predict`.
- `dspy.JSONAdapter`.
- A language-model configuration with temperature zero and caching disabled.

There is no DSPy optimizer, optimizer compilation, few-shot demonstration set, or fine-tuning step in this execution path. The predictor receives the text as `context`; the signature supplies field descriptions and permitted values. It does not receive the case's gold answer.

In `comparison/multi_field/run.py`, gold answers are used for scoring and result recording. The BANKING77 training CSV is fetched and inspected to validate the canonical category set, not to train the predictor or populate demonstrations.

Therefore, the higher DSPy scores cannot be attributed to task-specific training or DSPy optimization performed by the user. These are unoptimized, zero-shot structured-output baselines, not bare models without instructions or schemas.

The historical model identifiers are:

- Local Qwen: `qwen3.8-27b-nvfp4-dflash2-direct` through a local endpoint, with thinking ON and OFF configurations.
- Hosted Gemini: `gemini/gemini-3.5-flash-lite`.
- Jev: requested model `jev-1.13.0`.

The local Qwen arm should not be described as Qwen2.5-27B.

## Recorded results

The preserved run metrics in `benchmarks/historical-source/provenance.json` report:

| Task | Metric | DSPy · Gemini | DSPy · local Qwen, thinking OFF | Jev |
|---|---|---:|---:|---:|
| BANKING77 | Top-1 accuracy | 0.860000 | 0.841667 | 0.826667 |
| CLINC150 + OOS | Top-1 accuracy | 0.911667 | 0.875000 | 0.726667 |
| GoEmotions | Multi-label micro-F1 | 0.384150 | 0.369054 | 0.228839 |

Gemini exceeds Jev by 3.33 percentage points of top-1 accuracy on BANKING77 and 18.50 percentage points on CLINC150 + OOS. The GoEmotions difference is approximately 0.15531 micro-F1; it is not an ordinary accuracy difference.

Each of the nine cells above records:

- 600 attempted predictions.
- 600 `ok` statuses.
- Strict schema adherence of 1.0.

Thus, both tested DSPy alternatives achieved higher classification scores than Jev on all three tasks, with no observed execution or schema failures in these cells. Architectural differences in output enforcement do not establish a measured schema-success advantage for Jev here. Observed 100% success is not a universal guarantee for either system.

## What the evidence supports

For the evaluated tasks, the unoptimized DSPy alternatives already provide the user-facing functionality under discussion: classification into declared outputs, valid structured responses, and higher measured classification quality.

The benchmark does not show that adopting Jev unlocks a classification capability unavailable through these alternatives. It also does not show that their stronger results required collecting training examples, running an optimization cycle, or maintaining a local model: the Gemini arm is hosted and unoptimized.

Other differences—such as calibrated probabilities, total operating cost, or operational reliability—might matter to another use case. This comparison does not settle them. For my question, different internals alone do not explain what I would gain as a user.

## What the evidence does not support

These results do not establish:

- A hard accuracy ceiling for Jev.
- A causal explanation for the quality differences.
- Jev's proprietary architecture or an alleged absence of semantic reasoning.
- A necessary relationship between higher classification quality and a particular response time.
- An empirically validated calibration advantage for Jev.
- Universal superiority of DSPy, Gemini, or Qwen across other tasks and deployments.
- Guaranteed improvements from future DSPy optimization.

The grid's gloss-asymmetry hypothesis concerns the local RLCD engine and remains mechanistically untested. It must not be transferred to Jev as an established explanation.

## Interpretation and provenance boundaries

### Sample size

Each published cell comprises 200 selected test cases, repeated three times, with one warmup case discarded and concurrency one. These are **600 attempts, not 600 independent test examples**.

The archive reports identical selected 200-case cohorts across arms for each dataset. BANKING77 and CLINC use fixed-seed random sampling; GoEmotions uses deterministic selection beginning with an unseen positive example for each canonical label, followed by remaining dataset order. GoEmotions should therefore not be represented as a random sample of its full test distribution.

### Task definitions

BANKING77 is a 77-choice intent task.

The task labeled CLINC150 in the grid actually includes 150 in-domain intents plus the `oos` label: 151 permitted choices, drawn from the CLINC-OOS imbalanced test source.

GoEmotions is a multi-label task represented by 28 Boolean fields, not a single 28-way choice. Jev uses its Noul primitive and converts each returned probability to a Boolean at `p >= 0.5`. This threshold was not tuned. Its score describes that default operating point, not the best score obtainable at other thresholds.

### System-level comparison

The arms receive the same selected texts and target labels, but use different interface contracts and prompt construction. DSPy uses typed signatures and a JSON adapter. Jev's Choice path supplies candidate identifiers as criteria with the instruction “Select the best matching option”; its Noul path asks whether the text expresses each emotion.

This is a comparison of usable configured systems, not an experiment that isolates base-model capability or architecture. That distinction limits causal claims without invalidating the user-level comparison of these tested configurations.

### Failure accounting

The published quality metrics exclude execution errors from their denominators. The Qwen thinking-ON arm recorded one execution error on BANKING77, one on GoEmotions, and three on CLINC150. The Gemini, Qwen thinking-OFF, and Jev cells presented above have no recorded execution errors, so this denominator issue does not explain their reported ordering.

### Historical evidence, not a new rerun

This assessment is based on reading the retained implementation, published grid, archive documentation, and stored provenance metrics. No live benchmark was run and no individual prediction scores were independently recomputed for this assessment.

The inspected checkout does not contain the raw per-row result files referenced by the historical runs. The archive documents an earlier cross-check that recomputed quality from stored rows and verified cohort identity, but that is a documented historical check, not a newly reproduced verification here.

The shared harness is retained original code, not an immutable per-run source snapshot. Model/service versions were unpinned at run time, and dataset sources were fetched from dynamic upstream branches. These limitations constrain reproducibility and generalization.

## Bottom line

**I am not trying to defend or dismiss Jev. I am simply comparing what I get from the available options. Here, the unoptimized DSPy alternatives gave me higher classification scores and equally successful structured outputs in the recorded runs. That leaves me without a clear reason to be excited about Jev for this use case.**

It may be useful for other needs. My question is narrower: what would I gain by choosing it over what already works for me?
