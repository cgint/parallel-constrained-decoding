#!/usr/bin/env bash
set -u
cd /Users/cgint/dev-external/rlcd-engine

GATEDIR="agent/scratch/ragna/rlcd-gate"
LOCKDIR="$GATEDIR/lock4"
GRID="$GATEDIR/grid.jsonl"
LOG="agent/scratch/ragna/nova_serial.log"
mkdir -p agent/raw/2026-09-20-nova
: > "$LOG"

log(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG" "$GRID"; }

excl(){
  p=$(ps -ax -o command= | grep -E 'multi_field.run|engine_mlx|laya_mlx|remote_serial|eir_power_rlcd' | grep -v grep | grep -v nova_serial || true)
  if [ -n "$p" ]; then
    log "EXCLUSIVITY-FAIL $1"
    printf '%s\n' "$p" >> "$LOG"
    return 1
  fi
  log "EXCLUSIVITY-PASS $1 marker=PS-CLEAN"
  return 0
}

acq(){
  local n=0
  while ! mkdir "$LOCKDIR" 2>/dev/null; do
    n=$((n+1))
    [ "$n" -eq 1 ] && log "LOCK-WAIT $1 holder=$(cat "$LOCKDIR/owner" 2>/dev/null || cat "$LOCKDIR/owner.txt" 2>/dev/null)"
    [ "$n" -ge 480 ] && { log "LOCK-TIMEOUT $1"; return 1; }
    sleep 30
  done
  echo "nova $1 $(date -u +%FT%TZ)" > "$LOCKDIR/owner"
  log "LOCK-ACQUIRED $1 waits=$n"
}

rel(){
  rm -f "$LOCKDIR/owner" 2>/dev/null
  rmdir "$LOCKDIR" 2>/dev/null
  log "LOCK-RELEASED $1"
}

QW="qwen3.8-27b-nvfp4-dflash2-direct@pluto:40115"
CELLS=0; OKC=0

cell(){ # $1=name  $2=dataset  $3=off?
  local name="$1" d="$2" off="$3"
  local out="comparison/results/camp-$name"
  CELLS=$((CELLS+1))

  if [ -f "$out/summary.json" ]; then
    log "NO-OP $name (summary.json already exists)"
    OKC=$((OKC+1))
    return 0
  fi

  local cmd="PYTHONPATH=\$PWD .venv/bin/python -m comparison.multi_field.run --providers dspy --dataset $d --dspy-model $QW --limit 200 --warmup 1 --repetitions 1 --allow-paid --out $out"
  if [ "$off" = "1" ]; then
    cmd="$cmd --dspy-no-thinking"
  fi

  log "CELL-BEGIN $name"
  acq "$name" || { log "CELL-SKIP $name lock-timeout"; return 1; }
  excl "$name" || { rel "$name"; log "CELL-SKIP $name not-exclusive"; return 1; }

  local start ts
  start=$(date -u +%FT%TZ)
  log "CELL-RUN $name marker=PS-CLEAN start=$start command=$cmd"

  if [ "$off" = "1" ]; then
    PYTHONPATH="$PWD" .venv/bin/python -m comparison.multi_field.run \
      --providers dspy --dataset "$d" --dspy-model "$QW" \
      --limit 200 --warmup 1 --repetitions 1 \
      --dspy-no-thinking \
      --out "$out" --allow-paid \
      > "agent/raw/2026-09-20-nova/$name.log" 2>&1
  else
    PYTHONPATH="$PWD" .venv/bin/python -m comparison.multi_field.run \
      --providers dspy --dataset "$d" --dspy-model "$QW" \
      --limit 200 --warmup 1 --repetitions 1 \
      --out "$out" --allow-paid \
      > "agent/raw/2026-09-20-nova/$name.log" 2>&1
  fi
  local rc=$?
  rel "$name"
  ts=$(date -u +%FT%TZ)
  if [ -f "$out/summary.json" ]; then
    log "CELL-OK $name rc=$rc end=$ts"
    OKC=$((OKC+1))
  else
    log "CELL-NOFILE $name rc=$rc end=$ts"
    tail -5 "agent/raw/2026-09-20-nova/$name.log" >> "$GRID" 2>/dev/null || true
  fi
}

wait_exclusive(){
  # Wait for the in-flight remote_serial.sh (w1:p17) to finish.
  log "WAIT-EXCLUSIVE waiting for remote_serial.sh to exit"
  local waited=0
  while [ $waited -lt 1440 ]; do
    p=$(ps -ax -o command= | grep -E 'multi_field.run|engine_mlx|laya_mlx|remote_serial|eir_power_rlcd' | grep -v grep | grep -v nova_serial || true)
    [ -z "$p" ] && break
    sleep 30
    waited=$((waited+30))
    [ $((waited % 300)) -eq 0 ] && log "WAIT-EXCLUSIVE still busy after ${waited}s"
  done
  [ $waited -ge 1440 ] && { log "WAIT-EXCLUSIVE-TIMEOUT 4h"; return 1; }
  log "WAIT-EXCLUSIVE-CLEARED after ${waited}s"
  return 0
}

wait_exclusive || { log "ABORT: timeout waiting for exclusivity"; exit 1; }

cell dspy-on-goemotions  goemotions 0
cell dspy-off-goemotions goemotions 1
cell dspy-on-banking77   banking77  0
cell dspy-off-banking77  banking77  1
cell dspy-on-clinc150    clinc150   0
cell dspy-off-clinc150   clinc150   1

log "NOVA-CELLS-DONE ok=$OKC/$CELLS"
echo "NOVA-SERIAL-DONE"
