#!/usr/bin/env python
"""CLINC150 powered comparison — Phase 2 driver.

200 cases x 3 reps x 5 arms. Each arm is a single `comparison.multi_field.run`
invocation (one --providers) writing to its own out dir.

Arms (name -> providers, dspy-model, dspy-no-thinking, needs-paid):
  rlcd            : rlcd, local MLX (free)
  dspy-hosted     : dspy, gemini/gemini-3.5-flash-lite (GEMINI_API_KEY)
  jev             : jev, paid TypeSafe (JEV_API_KEY, --allow-paid)
  dspy-local-off  : dspy, local Qwen via pluto LiteLLM proxy, thinking OFF
  dspy-local-on   : dspy, local Qwen via pluto LiteLLM proxy, thinking ON

The local Qwen endpoint is 192.168.1.124:40115 (pluto LiteLLM proxy, verified
reachable 2026-09-20). The adapter auto-prefixes bare model names with
'openai/' (comparison/multi_field/adapters.py:113), so we pass the bare name.

JEV key: read from .env (JEV_API_KEY) by this driver and passed only to the
jev subprocess env — never echoed or written. If absent, the jev arm reports
'unavailable/missing_api_key' (no request sent).

Usage (repo root):
  PYTHONPATH="$PWD" .venv/bin/python \
    agent/raw/2026-09-20-power-clinc150/run_phase2.py \
    --out comparison/results/clinc150-20260920-power \
    --limit 200 --warmup 1 --repetitions 3
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # .../rlcd-engine
LOCAL_QWEN = 'qwen3.8-27b-nvfp4-dflash2-direct@192.168.1.124:40115'
HOSTED_DSPY = 'gemini/gemini-3.5-flash-lite'

ARM_CMDS = {
    'rlcd':           ['--providers', 'rlcd'],
    'dspy-hosted':    ['--providers', 'dspy', '--dspy-model', HOSTED_DSPY],
    'jev':            ['--providers', 'jev', '--allow-paid'],
    'dspy-local-off': ['--providers', 'dspy', '--dspy-model', LOCAL_QWEN, '--dspy-no-thinking'],
    'dspy-local-on':  ['--providers', 'dspy', '--dspy-model', LOCAL_QWEN],
}


def jev_key_from_env_file() -> str | None:
    """Return the JEV_API_KEY value from REPO/.env (name only logged by caller)."""
    env_file = REPO / '.env'
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith('JEV_API_KEY='):
            v = line.split('=', 1)[1].strip().strip('"').strip("'")
            return v or None
    return None


def run_arm(arm: str, out_root: Path, limit: int, warmup: int, reps: int,
            jev_key: str | None) -> int:
    out = out_root / arm
    cmd = [sys.executable, '-m', 'comparison.multi_field.run'] + ARM_CMDS[arm] + [
        '--dataset', 'clinc150',
        '--limit', str(limit),
        '--warmup', str(warmup),
        '--repetitions', str(reps),
        '--out', str(out),
    ]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO)
    if arm == 'jev' and jev_key:
        env['JEV_API_KEY'] = jev_key
    t0 = time.time()
    print(f'[{arm}] START {" ".join(cmd[2:])}', flush=True)
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    dt = time.time() - t0
    print(f'[{arm}] DONE rc={proc.returncode} wall={dt:.1f}s', flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='comparison/results/clinc150-20260920-power')
    ap.add_argument('--arms', default=','.join(ARM_CMDS))
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--repetitions', type=int, default=3)
    a = ap.parse_args()

    out_root = REPO / a.out
    out_root.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(',') if x]
    for arm in arms:
        if arm not in ARM_CMDS:
            raise SystemExit(f'unknown arm: {arm}')

    jev_key = jev_key_from_env_file()
    if jev_key is None and 'jev' in arms:
        print('WARNING: JEV_API_KEY not found in .env; jev arm will be unavailable', flush=True)

    failures = []
    for arm in arms:
        rc = run_arm(arm, out_root, a.limit, a.warmup, a.repetitions, jev_key)
        if rc != 0:
            failures.append(arm)

    # Cross-arm case-id check (byte-identical selected_case_ids)
    ids: dict[str, list] = {}
    for arm in arms:
        m = out_root / arm / 'manifest.json'
        if m.exists():
            ids[arm] = json.loads(m.read_text())['selected_case_ids']
    print('\n=== CROSS-ARM case-id check ===')
    if len(ids) < 2:
        print('fewer than 2 arms produced a manifest; cannot compare')
    else:
        ref = next(iter(ids))
        for arm, lst in ids.items():
            same = lst == ids[ref]
            print(f'{arm:16s} n={len(lst)} identical_to_{ref}={same}')

    # Per-arm summary table
    print('\n=== PER-ARM SUMMARY (min-max across reps) ===')
    for arm in arms:
        sp = out_root / arm / 'summary.json'
        if not sp.exists():
            print(f'{arm:16s} MISSING summary.json')
            continue
        data = json.loads(sp.read_text())
        for model, s in data.get('providers', {}).items():
            print(f'{arm:16s} {model:45s} acc={s.get("top1_accuracy")} '
                  f'valid={s.get("strict_schema_adherence")} '
                  f'p50={s.get("client_latency_ms", {}).get("p50")} '
                  f'p95={s.get("client_latency_ms", {}).get("p95")} '
                  f'status={s.get("status_counts")}')
    print('\nFAILURES:', failures if failures else 'NONE')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
