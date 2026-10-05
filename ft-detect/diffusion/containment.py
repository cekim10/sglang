"""Phase 3c: containment timeline for an injection run with FT_ABORT_PROBE=1.

Joins inject.jsonl (t_inject), steps_rank*.jsonl (progress, abort_probe phases, step/stage
exceptions, req_end), requests.jsonl (client-visible errors) and probe.jsonl (pid states,
engine strings). All times in seconds after t_inject, per surviving rank:

  t_last_progress      last quantum completed before the hang
  t_deadline_miss      probe fired (= last progress + FT_ABORT_DEADLINE_S)
  t_abort_returned     ncclCommAbort path returned in the probe thread   -> T_abort_call
  t_main_exception     the blocked collective raised in the main thread  -> T_abort_effect
  t_req_end            scheduler saw the request fail                      (containment of the in-flight request)
  t_loop_idle          rank back at its receive loop (ready for the next request or for restart)
  t_first_client_error first HTTP client that got an error after t_inject
  rank_alive_after     did the rank survive the abort (probe pid state)
  post_abort_errors    requests after the abort: how fast they fail (fail-fast vs hang again)

    python containment.py runs/diff_B_* [--md out.md] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read_jsonl  # noqa: E402

NS = 1e9


def _s(t, t0):
    return None if t is None or t0 is None else round((t - t0) / NS, 3)


def analyze_run(run: Path) -> list[dict]:
    injs = [r for r in read_jsonl(run / "inject.jsonl") if r.get("ev") == "inject"]
    if not injs:
        return []
    inj = injs[0]
    t_inj, target = inj["t_ns"], inj.get("rank")
    env = {}
    try:
        env = json.loads((run / "launch_env.json").read_text())
    except Exception:
        pass
    reqs = [r for r in read_jsonl(run / "requests.jsonl") if "rid" in r and "ev" not in r]
    stops = [r["t_ns"] for r in read_jsonl(run / "inject.jsonl") if r.get("ev") in ("sigcont", "sigterm", "sigkill", "stopped", "resume")]
    t_stop = min(stops) if stops else None
    probe = [r for r in read_jsonl(run / "probe.jsonl") if t_stop is None or r["t_ns"] < t_stop]
    rows = []
    for p in sorted(run.glob("logs/steps_rank*.jsonl")):
        rank = int(p.stem.replace("steps_rank", ""))
        if rank == target:
            continue
        ev = [r for r in read_jsonl(p) if t_stop is None or r["t_ns"] < t_stop]
        prog = [r["t_ns"] for r in ev if r.get("ev") in ("ds", "stage") and r["t_ns"] < t_inj + int(2 * NS)]
        ap = {r["phase"]: r for r in ev if r.get("ev") == "abort_probe"}
        exc = next((r for r in ev if r.get("ev") == "step_exception" and r["t_ns"] >= t_inj), None)
        stage_err = next((r for r in ev if r.get("ev") == "stage" and r.get("error") and r["t_ns"] >= t_inj), None)
        req_end = next((r for r in ev if r.get("ev") == "req_end" and r["t_ns"] >= t_inj), None)
        miss, ret, idle = ap.get("deadline_miss"), ap.get("abort_returned"), ap.get("loop_idle_again")
        attempts = (ret or {}).get("result", {}).get("attempts", [])
        alive = next((r for r in probe if r.get("ev") == "pid" and r.get("who") == f"rank{rank}"
                      and r.get("state") in ("dead", "zombie") and r["t_ns"] >= t_inj), None)
        first_err = min((r["last_chunk_ts"] or r["send_ts"] for r in reqs if r.get("error") and (r["last_chunk_ts"] or r["send_ts"]) >= t_inj), default=None)
        # requests admitted after the abort: time to failure each
        post = []
        if ret:
            for r in reqs:
                if r["send_ts"] >= ret["t_ns"] and r.get("error"):
                    post.append(((r["last_chunk_ts"] or r["send_ts"]) - r["send_ts"]) / NS)
        rows.append({
            "run": run.name, "model": env.get("model"), "rank": rank, "deadline_s": (miss or {}).get("deadline_s"),
            "abort_mode": None if not attempts else attempts[0].get("group"),
            "abort_how": None if not attempts else attempts[0].get("how"),
            "abort_ok": None if not attempts else all(a.get("ok") for a in attempts),
            "abort_error": "; ".join(a.get("error", "") for a in attempts if not a.get("ok")) or None,
            "t_last_progress": _s(max(prog), t_inj) if prog else None,
            "t_deadline_miss": _s((miss or {}).get("t_ns"), t_inj),
            "t_abort_returned": _s((ret or {}).get("t_ns"), t_inj),
            "T_abort_call_ms": None if not (miss and ret) else round((ret["t_ns"] - miss["t_ns"]) / 1e6, 1),
            "t_main_exception": _s((exc or stage_err or {}).get("t_ns"), t_inj),
            "T_abort_effect_ms": None if not (ret and (exc or stage_err)) else round(((exc or stage_err)["t_ns"] - ret["t_ns"]) / 1e6, 1),
            "main_exception": ((exc or stage_err or {}).get("error") or "")[:120] or None,
            "t_req_end": _s((req_end or {}).get("t_ns"), t_inj),
            "t_loop_idle": _s((idle or {}).get("t_ns"), t_inj),
            "cuda_mem_before_MiB": round(((miss or {}).get("cuda_mem_alloc") or 0) / 2**20),
            "cuda_mem_after_MiB": round(((idle or ret or {}).get("cuda_mem_alloc") or 0) / 2**20),
            "t_first_client_error": _s(first_err, t_inj),
            "rank_dead_at": _s((alive or {}).get("t_ns"), t_inj),
            "observed_until": _s(t_stop, t_inj),
            "abort_still_blocked_at_stop": bool(miss and not ret and t_stop is not None),
            "post_abort_errors": len(post),
            "post_abort_time_to_error_p50_s": round(sorted(post)[len(post) // 2], 2) if post else None,
        })
    return rows


def print_events(run: Path, window_s: float = 200.0) -> None:
    """Raw per-rank event timeline around t_inject, plus matching log lines and pid states."""
    import re
    import subprocess

    injs = [r for r in read_jsonl(run / "inject.jsonl") if r.get("ev") == "inject"]
    if not injs:
        print(f"{run}: no injection"); return
    t = injs[0]["t_ns"]
    print(f"=== {run.name} (t_inject = 0; window -5..+{window_s:.0f} s) ===")
    for p in sorted(run.glob("logs/steps_rank*.jsonl")):
        for r in read_jsonl(p):
            if r.get("ev") == "coexist_baseline":
                print(f"  {p.stem[-5:]} baseline (no hang): {json.dumps(r.get('bench') or r.get('error'))[:300]}")
            if r.get("ev") == "abort_probe" and r.get("phase") == "abort_returned":
                for at in (r.get("result") or {}).get("attempts", []):
                    if at.get("bench_during_hang"):
                        print(f"  {p.stem[-5:]} during hang:        {json.dumps(at['bench_during_hang'])[:300]}")
            if r.get("ev") in ("req_start", "req_end", "step_exception", "abort_probe", "inventory_error") \
                    and t - 5 * NS < r["t_ns"] < t + window_s * NS:
                print(f"  {p.stem[-5:]} {(r['t_ns'] - t) / NS:+9.3f}s {r['ev']:15s} {r.get('phase', '') or ''} {(r.get('error') or '')[:110]}")
    rx = re.compile(r"Error executing|DistBackendError|Watchdog caught|terminate|NONBLOCKING|nonblocking|Timeout\(ms\)|FT_ABORT_PROBE|Abort|abort")
    for p in sorted(run.glob("logs/rank*.log")):
        hits = [l.rstrip()[:200] for l in open(p, errors="replace") if rx.search(l) and not l.startswith("frame #")]
        for l in hits[-8:]:
            print(f"  {p.name}: {l}")
    for r in read_jsonl(run / "probe.jsonl"):
        if r.get("ev") == "pid" and r.get("state") != "alive" and r["t_ns"] >= t - 5 * NS:
            print(f"  probe {(r['t_ns'] - t) / NS:+9.3f}s {r['who']} -> {r['state']}")
    reqs = [r for r in read_jsonl(run / "requests.jsonl") if "rid" in r and "ev" not in r and r.get("error")]
    for r in sorted(reqs, key=lambda r: r["send_ts"])[:3]:
        print(f"  first client errors: sent {(r['send_ts'] - t) / NS:+8.3f}s ended {((r['last_chunk_ts'] or r['send_ts']) - t) / NS:+8.3f}s {str(r['error'])[:100]}")


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_no containment data (run with FT_ABORT_PROBE=1 and an injection)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--events", action="store_true", help="print the raw event timeline per run instead of the table")
    a = ap.parse_args()
    if a.events:
        for r in a.runs:
            if Path(r).is_dir():
                print_events(Path(r))
        return 0
    rows = []
    for r in a.runs:
        if Path(r).is_dir():
            rows.extend(analyze_run(Path(r)))
    df = pd.DataFrame(rows)
    text = ("Containment timeline per surviving rank (s after t_inject).\n\n" + md_table(df) +
            "\n\n`T_abort_call` = probe thread's abort call duration; `T_abort_effect` = until the blocked collective raised in the main thread; "
            "`t_req_end` = scheduler saw the in-flight request fail; `t_loop_idle` = rank back at its receive loop; "
            "`post_abort_time_to_error` = how quickly later requests fail (fail-fast) instead of hanging again.")
    print(text)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(text + "\n")
    if a.csv and not df.empty:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
