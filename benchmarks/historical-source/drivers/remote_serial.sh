#!/usr/bin/env bash
set -u
cd /Users/cgint/dev-external/rlcd-engine
GATEDIR="agent/scratch/ragna/rlcd-gate"; LOCKDIR="$GATEDIR/lock2"; EV="$GATEDIR/remote-evidence.jsonl"
mkdir -p agent/raw/2026-09-20-power-remote
log(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$EV"; }
acq(){ local n=0; while ! mkdir "$LOCKDIR" 2>/dev/null; do n=$((n+1)); [ "$n" -eq 1 ] && log "LOCK-WAIT $1 holder=$(cat "$LOCKDIR/owner.txt" 2>/dev/null)"; [ "$n" -ge 240 ] && { log "LOCK-TIMEOUT $1"; return 1; }; sleep 30; done; echo "remote $(date -u +%FT%TZ)" > "$LOCKDIR/owner.txt"; log "LOCK-ACQUIRED $1 waits=$n"; return 0; }
rel(){ rm -f "$LOCKDIR/owner.txt" 2>/dev/null; rmdir "$LOCKDIR" 2>/dev/null; log "LOCK-RELEASED $1"; }
excl(){ p=$(ps -ax -o command= | grep -E 'multi_field.run|eir_power_rlcd' | grep -v grep | grep -v remote_serial || true); if [ -n "$p" ]; then log "EXCLUSIVITY-FAIL $1"; return 1; fi; log "EXCLUSIVITY-PASS $1 marker=PS-CLEAN"; return 0; }
run(){ # $1 label  $2 provider-spec args...  
  local label="$1"; shift
  log "RUN-BEGIN $label"; acq "$label" || { log "SKIP $label lock"; return 1; }
  excl "$label" || { rel "$label"; log "SKIP $label busy"; return 1; }
  PYTHONPATH="$PWD" .venv/bin/python -m comparison.multi_field.run "$@" --limit 200 --warmup 1 --repetitions 1 \
    --out "comparison/results/power-$label" --allow-paid > "agent/raw/2026-09-20-power-remote/$label.log" 2>&1
  local rc=$?; log "RUN-END $label rc=$rc"
  [ -f "comparison/results/power-$label/summary.json" ] && log "SUMMARY-OK $label" || log "NO-SUMMARY $label"
  rel "$label"; }

QW="qwen3.8-27b-nvfp4-dflash2-direct@192.168.1.124:40115"
# goemotions: local qwen ON/OFF + hosted + jev
for d in goemotions banking77 clinc150; do
  run "qwen-on-$d"    --providers dspy  --dataset "$d" --dspy-model "$QW"
  run "qwen-off-$d"   --providers dspy  --dataset "$d" --dspy-model "$QW" --dspy-no-thinking
  run "hosted-$d"     --providers dspy  --dataset "$d" --dspy-model "gemini/gemini-3.5-flash-lite"
  run "jev-$d"        --providers jev   --dataset "$d"
done
log "REMOTE-LOOP-COMPLETE"
echo "REMOTE-SERIAL-DONE"
