#!/usr/bin/env bash
# Run BANKING77 power-run arms. Each arm gets its own results dir.
# Usage:
#   bash run_arms.sh qwen-off
#   bash run_arms.sh qwen-on
#   bash run_arms.sh gemini
#   bash run_arms.sh jev
#   bash run_arms.sh rlcd
#   bash run_arms.sh all        (qwen-off, qwen-on, gemini, jev, rlcd in that order)
set -u
cd /Users/cgint/dev-external/rlcd-engine
RAW=agent/raw/2026-09-20-power-banking77
mkdir -p "$RAW"
DS=(--dataset banking77 --limit 200 --warmup 1 --repetitions 3)
LOG() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RAW/arms.log"; }

arm() {
  local name="$1"; shift
  LOG "START arm=$name"
  PYTHONPATH="$PWD" .venv/bin/python -m comparison.multi_field.run \
    "$@" \
    --out "comparison/results/power-b77-$name" --allow-paid \
    > "$RAW/$name.log" 2>&1
  local rc=$?
  LOG "END arm=$name rc=$rc"
  return $rc
}

which_arm="${1:-all}"
case "$which_arm" in
  qwen-off) arm qwen-off --providers dspy "${DS[@]}" --dspy-model qwen3.8-27b-nvfp4-dflash2-direct@pluto:40115 --dspy-no-thinking;;
  qwen-on)  arm qwen-on  --providers dspy "${DS[@]}" --dspy-model qwen3.8-27b-nvfp4-dflash2-direct@pluto:40115;;
  gemini)   arm gemini   --providers dspy "${DS[@]}" --dspy-model gemini/gemini-3.5-flash-lite;;
  jev)      export JEV_API_KEY="$(grep '^JEV_API_KEY=' .env | cut -d= -f2)"; arm jev --providers jev "${DS[@]}";;
  rlcd)     arm rlcd     --providers rlcd "${DS[@]}";;
  all)
    arm qwen-off --providers dspy "${DS[@]}" --dspy-model qwen3.8-27b-nvfp4-dflash2-direct@pluto:40115 --dspy-no-thinking
    arm qwen-on  --providers dspy "${DS[@]}" --dspy-model qwen3.8-27b-nvfp4-dflash2-direct@pluto:40115
    arm gemini   --providers dspy "${DS[@]}" --dspy-model gemini/gemini-3.5-flash-lite
    export JEV_API_KEY="$(grep '^JEV_API_KEY=' .env | cut -d= -f2)"
    arm jev      --providers jev "${DS[@]}"
    arm rlcd     --providers rlcd "${DS[@]}"
    ;;
  *) echo "unknown arm: $which_arm"; exit 2;;
esac
LOG "DONE $which_arm"
