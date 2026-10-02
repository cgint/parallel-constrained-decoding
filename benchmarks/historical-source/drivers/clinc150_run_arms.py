#!/usr/bin/env python
"""Phase-2 run driver: 200 cases x 3 reps x 5 arms on CLINC150.

Arms (single --providers invocation each, one out dir per arm):
  1. rlcd             -> --providers rlcd
  2. dspy-hosted      -> --providers dspy --dspy-model gemini/gemini-3.5-flash-lite
  3. jev              -> --providers jev --allow-paid
  4. dspy-local-off   -> --providers dspy --dspy-model qwen3.8-27b-nvfp4-dflash2-direct@192.168.1.124:40115 --dspy-no-thinking
  5. dspy-local-on    -> same as 4 without --dspy-no-thinking

Env: export JEV_API_KEY before invoking (JEV arm is a no-op/unavailable otherwise).
Usage:
  PYTHONPATH="$PWD" .venv/bin/python agent/raw/2026-09-20-power-clinc150/run_clinc150_arms.py \
      --out comparison/results/clinc150-20260920-3x200
      [--arms rlcd,dspy-hosted,jev,dspy-local-off,dspy-local-on]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARM_CMDS = {
    "rlcd": [
        "--providers", "rlcd",
    ],
    "dspy-hosted": [
        "--providers", "dspy",
        "--dspy-model", "gemini/gemini-3.5-flash-lite",
    ],
    "jev": [
        "--providers", "jev",
        "--allow-paid",
    ],
    "dspy-local-off": [
        "--providers", "dspy",
        "--dspy-model", "qwen3.8-27b-nvfp4-dflash2-direct@192.168.1.124:40115",
        "--dspy-no-thinking",
    ],
    "dspy-local-on": [
        "--providers", "dspy",
        "--dspy-model", "qwen3.8-27b-nvfp4-dflash2-direct@192.168.1.124:40115",
    ],
}


def run_one(arm: str, out_root: Path, limit: int, warmup: int, reps: int) -> int:
    out = out_root / arm
    cmd = [sys.executable, "-m", "comparison.multi_field.run"] + ARM_CMDS[arm] + [
        "--dataset", "clinc150",
        "--limit", str(limit),
        "--warmup", str(warmup),
        "--repetitions", str(reps),
        "--out", str(out),
    ]
    t0 = time.time()
    print(f"[{arm}] START {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO)
    dt = time.time() - t0
    print(f"[{arm}] DONE rc={proc.returncode} wall={dt:.1f}s", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="comparison/results/clinc150-20260920-3x200")
    ap.add_argument("--arms", default=",".join(ARM_CMDS))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repetitions", type=int, default=3)
    a = ap.parse_args()

    out_root = REPO / a.out
    out_root.mkdir(parents=True, exist_ok=True)
    arms = [x for x in a.arms.split(",") if x]
    failures = []
    for arm in arms:
        if arm not in ARM_CMDS:
            raise SystemExit(f"unknown arm: {arm}")
        rc = run_one(arm, out_root, a.limit, a.warmup, a.repetitions)
        if rc != 0:
            failures.append(arm)
    # case-id cross-arm check (byte-identical selected_case_ids)
    ids = {}
    for arm in arms:
        m = out_root / arm / "manifest.json"
        if m.exists():
            ids[arm] = json.loads(m.read_text())["selected_case_ids"]
    if ids:
        ref_arm = next(iter(ids))
        ok = all(ids[arm] == ids[ref_arm] for arm in ids)
        print(f"CROSS-ARM case-id identical: {ok}")
        if not ok:
            for arm in ids:
                if ids[arm] != ids[ref_arm]:
                    diff = sum(1 for x, y in zip(ids[arm], ids[ref_arm]) if x != y)
                    print(f"  {arm}: {diff} positions differ from {ref_arm}")
    print("FAILURES:", failures if failures else "NONE")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
