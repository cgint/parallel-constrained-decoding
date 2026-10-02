"""Serial camp cells for sol (lock-free relaunch, 2026-09-20).

Cells: camp-laya-goemotions, camp-laya-clinc150 (laya-mlx, aac6fef/laya-mlx)
       camp-rlcd-clinc150 (our engine via comparison.multi_field.run).
Standard protocol: --limit 200, warmup 1, repetitions 3  (summarize() n=600).
Laya extras: per-case client_latency_ms p50/p95/p99 + observed_wall_clock in summary.
No locks, no cross-lane checks: PS-CLEAN marker = no other LOCAL MLX/model process
(remote dspy arms never touch the local GPU and do NOT block this lane).
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/cgint/dev-external/rlcd-engine")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

GRID = REPO / "agent/scratch/ragna/rlcd-gate/grid.jsonl"
RAW = REPO / "agent/raw/2026-09-20-camp-sol"
RAW.mkdir(parents=True, exist_ok=True)

LIMIT = 200
WARMUP = 1
REPS = 3


def now():
    return datetime.now(timezone.utc).strftime("%FT%TZ")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(RAW / "runner.log", "a") as f:
        f.write(line + "\n")


def ps_clean():
    """True if no other LOCAL MLX/model process is running.

    Only real model processes block the lane (a second concurrent local model would
    double-allocate GPU memory / contend for the Metal device, and MLX re-creates its
    KV cache per call so stale state is not the concern — concurrent *load* is).
    Excluded by construction:
      - this process tree (pid == me or parent),
      - any remote dspy arm (multi_field.run --providers dspy): disjoint resource per
        brief, must NOT block this lane,
      - shell / one-liner shells whose text mentions a needle name but is not itself a
        model process (detected via comm / script-name match).
    """
    me = os.getpid()
    try:
        ps = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=", "-o", "ppid=", "-o", "command="],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        return None, f"PS-UNAVAILABLE:{e.__class__.__name__}"

    # Direct blockers: a second local model process in the same lane.
    # "multi_field.run --providers rlcd" = our engine; "engine_mlx" = any direct use.
    rlcd_run_needles = ("multi_field.run", "engine_mlx", "run_rlcd_generation")
    # laya-mlx model processes: a python process whose argv names the laya_mlx package,
    # NOT a shell quoting the path (shells show up as /bin/bash -c ... camp_runner2.py ...).
    laya_needles = ("/laya_mlx/", "laya-mlx/camp_runner", "laya_mlx/__main__", "laya-mlx/smoke")

    hits = []
    for line in ps.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid, ppid, cmd = parts
        if pid == str(me) or ppid == str(me):
            continue
        # skip shells: their command starts with /bin/sh, /bin/bash, zsh, sh, etc.
        cmd_first = cmd.split()[0] if cmd.split() else ""
        is_shell = any(cmd_first.endswith(s) for s in ("/sh", "/bash", "/zsh", "/dash", "sh", "bash", "zsh", "dash"))
        if is_shell:
            continue
        # remote dspy arm: never blocks local lane
        if "multi_field.run" in cmd and "--providers dspy" in cmd:
            continue
        if any(n in cmd for n in rlcd_run_needles):
            hits.append(f"{pid}:{cmd[:160]}")
        elif any(n in cmd for n in laya_needles):
            hits.append(f"{pid}:{cmd[:160]}")
    if hits:
        return False, f"PS-BUSY {hits[0]}"
    return True, "PS-CLEAN"


def grid(cell, lane, cmd, extra=""):
    line = f"{now()} CELL {cell} lane={lane} cmd={cmd} {extra}".strip()
    with open(GRID, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def grid_done(cell, n, rc):
    line = f"{now()} CELL-DONE {cell} n={n} rc={rc}"
    with open(GRID, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    pos = (len(s) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (pos - lo)


def run_laya(cell_name, dataset):
    from comparison.multi_field.dataset import (
        fetch_sources, fetch_clinc150_sources,
        fetch_banking77_sources, normalize, normalize_clinc150, normalize_banking77,
        select_cases, select_banking77_cases,
    )
    from comparison.multi_field.metrics import summarize
    from comparison.multi_field.models import StructuredPrediction

    import laya_mlx as laya
    t0 = time.perf_counter()
    agent = laya.load("aac6fef/laya-mlx")
    load_secs = time.perf_counter() - t0
    if dataset == "clinc150":
        try:
            agent.cfg["head_max_len"] = 512
            agent.cfg["max_len"] = 1024
        except Exception:
            pass
    log(f"MARKER agent-loaded {cell_name} load_seconds={load_secs:.1f}")

    outdir = REPO / "comparison/results" / cell_name
    outdir.mkdir(parents=True, exist_ok=True)

    if dataset == "goemotions":
        test, labels, source = fetch_sources(outdir / "source")
        cases = select_cases(normalize(test, labels), LIMIT)
        log(f"MARKER goemotions-cases n={len(cases)} labels={len(labels)}")
    elif dataset == "clinc150":
        test_parquet, card, source = fetch_clinc150_sources(outdir / "source")
        cases = select_banking77_cases(normalize_clinc150(test_parquet, card), LIMIT)
        log(f"MARKER clinc150-cases n={len(cases)} choices={len(cases[0].schema['intent']['choices'])}")
    else:
        script, train, test, source = fetch_banking77_sources(outdir / "source")
        cases = select_banking77_cases(normalize_banking77(test, script), LIMIT)
        log(f"MARKER banking77-cases n={len(cases)} choices={len(cases[0].schema['intent']['choices'])}")

    def make_questions(case):
        if dataset == "goemotions":
            return {label: {"type": "noul", "instructions": f"Does the text express the emotion '{label}'?"}
                    for label in case.schema}
        return {"intent": {"type": "choice", "instructions": "What is the user's intent?",
                           "criteria": case.schema["intent"]["choices"]}}

    # warmup (standard protocol): WARMUP cases, discarded
    for c in cases[:WARMUP]:
        agent.predict(c.context, make_questions(c))
    log(f"MARKER warmup-done {cell_name} warmup={WARMUP}")

    rows, pairs, latencies = [], [], []
    provider_started = time.perf_counter()
    for rep in range(REPS):
        for i, case in enumerate(cases):
            t0 = time.perf_counter()
            try:
                res = agent.predict(case.context, make_questions(case))
                lat = (time.perf_counter() - t0) * 1000
            except Exception as e:
                pred = StructuredPrediction.error("laya", "laya-mlx", "predict_exception")
                rows.append({"case_id": case.case_id, "gold": case.gold, "repetition": rep, "error": str(e)})
                pairs.append((case, pred))
                print(f"  {dataset} r{rep} {i+1} ERROR: {e}", flush=True)
                continue
            if dataset == "goemotions":
                prediction = {}
                for label in case.schema:
                    ans = res["answers"].get(label)
                    prediction[label] = bool(ans and ans.get("noul", -1) >= 0.5)
                ok = set(prediction) == set(case.schema)
            else:  # clinc150 or banking77: single enum field 'intent'
                ans = res["answers"].get("intent")
                got = ans.get("choice") if ans else None
                if got is None:
                    pred = StructuredPrediction.invalid("laya", "laya-mlx", "no_choice_answer", lat)
                    rows.append({"case_id": case.case_id, "gold": case.gold, "repetition": rep,
                                 "error": "no choice answer", "latency_ms": round(lat, 1)})
                    pairs.append((case, pred))
                    continue
                ok = got in case.schema["intent"]["choices"]
                prediction = {"intent": got}
            pred = StructuredPrediction(
                "laya", "laya-mlx", "ok" if ok else "invalid",
                prediction, lat, ok, None if ok else ("invalid_response" if dataset == "goemotions" else "choice_outside_enum"))
            rows.append({"case_id": case.case_id, "gold": case.gold, "repetition": rep,
                         "prediction": prediction, "latency_ms": round(lat, 1), "schema_valid": ok})
            pairs.append((case, pred))
            latencies.append(lat)
        if (rep + 1) % REPS == 0:
            log(f"MARKER rep-{rep+1} done {cell_name} last_lat={lat:.0f}ms")

    wall = time.perf_counter() - provider_started
    summary = summarize(pairs)
    summary["client_latency_ms"] = {"n": len(latencies), "p50": pct(latencies, 50),
                                    "p95": pct(latencies, 95), "p99": pct(latencies, 99)}
    summary["observed_wall_clock"] = {"seconds": wall, "requests": len(pairs),
                                      "requests_per_second": (len(pairs) / wall) if wall else None}
    (outdir / "results.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    manifest = {
        "utc_started": now(), "cell": cell_name, "provider": "laya",
        "model": "laya-mlx (aac6fef/laya-mlx)", "dataset": dataset,
        "limit": LIMIT, "warmup": WARMUP, "repetitions": REPS, "concurrency": 1,
        "allow_paid": False, "selected_case_ids": [c.case_id for c in cases],
        "command": "PYTHONPATH=. .venv/bin/python agent/scratch/ragna/laya-mlx/camp_runner2.py laya " + dataset,
        "git": {"commit": None},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (outdir / "summary.json").write_text(json.dumps({"providers": {"laya": summary}}, indent=2, sort_keys=True) + "\n")
    log(f"MARKER {cell_name}-done summary={outdir/'summary.json'}")
    keys = (("n", "micro_f1", "macro_f1_supported", "hamming_loss", "all_field_exact_record_match", "strict_schema_adherence")
            if dataset == "goemotions" else ("n", "top1_accuracy", "macro_f1_present_gold_classes", "strict_schema_adherence"))
    log(f"MARKER {cell_name} quality={json.dumps({k: summary.get(k) for k in keys if k in summary})}")
    print(json.dumps({k: summary.get(k) for k in keys if k in summary}, indent=2))
    print(json.dumps(summary["client_latency_ms"], indent=2))
    print(json.dumps(summary["observed_wall_clock"], indent=2))
    return len(pairs)


def run_rlcd(cell_name, dataset="clinc150", out_cell=None):
    out_cell = out_cell or cell_name
    cmd = ("PYTHONPATH=. .venv/bin/python -m comparison.multi_field.run --providers rlcd --dataset "
           f"{dataset} --limit {LIMIT} --warmup {WARMUP} --repetitions {REPS} "
           f"--out comparison/results/{out_cell}")
    log(f"RUN-BEGIN {cell_name}")
    rc = subprocess.call(cmd, shell=True, cwd=str(REPO),
                         stdout=open(RAW / f"{cell_name}.log", "w"), stderr=subprocess.STDOUT)
    log(f"RUN-END {cell_name} rc={rc}")
    outdir = REPO / "comparison/results" / out_cell
    if (outdir / "summary.json").exists():
        log(f"SUMMARY-OK {cell_name}")
        n = len((outdir / "results.jsonl").read_text().splitlines())
        return n
    log(f"NO-SUMMARY {cell_name}")
    return 0


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else ""
    table = {
        "camp-laya-goemotions": ("laya", "goemotions"),
        "camp-laya-clinc150": ("laya", "clinc150"),
        "camp-laya-banking77": ("laya", "banking77"),
        "camp-rlcd-clinc150": ("rlcd", "clinc150"),
        "power-rlcd-banking77-rep3": ("rlcd", "banking77"),
        "power-rlcd-clinc150-rep2": ("rlcd", "clinc150"),
        "power-rlcd-clinc150-rep3": ("rlcd", "clinc150"),
    }
    if cell not in table:
        print(f"unknown cell: {cell}", file=sys.stderr)
        sys.exit(2)
    provider, dataset = table[cell]
    ok, marker = ps_clean()
    if ok is None:
        log(f"PROBE-FAIL {cell} {marker}")
        print("probe failed: " + marker)
        sys.exit(3)
    if not ok:
        log(f"SKIP {cell} {marker}")
        print("lane busy, not started: " + marker)
        sys.exit(4)
    if provider == "laya":
        cmd = ("PYTHONPATH=. .venv/bin/python agent/scratch/ragna/laya-mlx/camp_runner2.py laya " + dataset)
    elif cell == "camp-rlcd-clinc150":
        cmd = (f"PYTHONPATH=. .venv/bin/python -m comparison.multi_field.run --providers rlcd --dataset "
               f"{dataset} --limit {LIMIT} --warmup {WARMUP} --repetitions {REPS} --out comparison/results/{cell}")
    else:
        cmd = (f"PYTHONPATH=. .venv/bin/python -m comparison.multi_field.run --providers rlcd --dataset "
               f"{dataset} --limit {LIMIT} --warmup {WARMUP} --repetitions {REPS} --out comparison/results/{cell}")
    grid(cell, "mlx", cmd, f"marker={marker}")
    try:
        n = run_laya(cell, dataset) if provider == "laya" else run_rlcd(cell, dataset, out_cell=cell)
    except Exception as e:
        log(f"CELL-FAIL {cell} {e.__class__.__name__}: {e}")
        import traceback
        traceback.print_exc()
        grid_done(cell, 0, 1)
        sys.exit(1)
    grid_done(cell, n, 0)
    log(f"BATCH-END {cell}")


if __name__ == "__main__":
    main()
