#!/bin/bash
# W-GOEMO power runs: 4 functional arms (rlcd, dspy-local-off, dspy-local-on, dspy-gemini).
# Jev arm is structurally incompatible (enum-lane adapter vs Boolean schema) — see probe_jev_boolean.py.
set -u
cd /Users/cgint/dev-external/rlcd-engine
export JEV_API_KEY="$(sed -n 's/^JEV_API_KEY=//p' .env)"
STAMP=20260920-power
BASE=comparison/results/power-goe
run_arm() {
  local name="$1"; shift
  local dir="$BASE/$name"
  echo "===== ARM $name -> $dir ====="
  PYTHONPATH="$PWD" .venv/bin/python -m comparison.multi_field.run \
    "$@" --dataset goemotions --limit 200 --warmup 1 --repetitions 3 \
    --out "$dir" 2>&1 | tail -5
  echo "ARM $name exit=$?"
}

# Arm 1: RLCD local MLX engine (free)
run_arm rlcd --providers rlcd

# Arm 2: local Qwen via DSPy adapter, thinking OFF
run_arm dspy-loc-off --providers dspy --dspy-model qwen3.8-27b-nvfp4-dflash2-direct@pluto:40115 --dspy-no-thinking

# Arm 3: local Qwen via DSPy adapter, thinking ON
run_arm dspy-loc-on --providers dspy --dspy-model qwen3.8-27b-nvfp4-dflash2-direct@pluto:40115

# Arm 4: hosted Gemini via DSPy adapter
run_arm dspy-gemini --providers dspy --dspy-model gemini/gemini-3.5-flash-lite

echo "ALL ARMS DONE"
