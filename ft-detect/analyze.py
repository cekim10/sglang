"""Join requests.jsonl + probe.jsonl + inject.jsonl (+ steps_rank*.jsonl) for one or more runs.

Injection run (inject.jsonl has an 'inject' record) -> columns:
  t_inject, t_first_symptom (client-visible output stall), t_stall_engine (last
  scheduler step on a surviving rank), t_detect_engine (+ which detector),
  t_detect_health (/health or /health_generate first non-200), t_detect_gen1
  (1-token /generate first non-200 and first 3-in-a-row), t_server_dead,
  n_inflight_hung, n_new_errored, TTFT p50/p99 before vs during.
Floor run (no injection) -> spurious-kill columns: any engine detector event,
  any pid death, health non-200 count, request error count.

All times in seconds relative to t_inject (or load start for floor runs).

    python analyze.py run                       # one run -> run/summary.md, row appended to results/summary.csv
    python analyze.py runs/B_* --csv results/caseB.csv --md results/caseB.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl  # noqa: E402

ENGINE_PATTERNS = {
    "watchdog_fire", "scheduler_exception", "subprocess_crashed", "sigquit", "kill_tree",
    "nccl_timeout", "nccl_error", "torch_dist_error", "abort", "cuda_error", "scheduler_terminated",
    # sglang.multimodal_gen (diffusion)
    "diff_recv_error", "diff_exec_error", "diff_max_errors", "diff_worker_dead", "diff_worker_shutdown", "diff_ipc_a2a_timeout",
    "ft_abort_miss", "ft_abort_done",
}
NS = 1e9


def _s(t_ns, t0_ns):
    return None if t_ns is None or t0_ns is None else round((t_ns - t0_ns) / NS, 3)


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[k], 1)


def _load_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def stall_from_steps(run: Path, t_inject: int, target_rank: int | None, gap_s: float):
    """Last run_batch timestamp on a surviving rank before a >= gap_s gap (or before its log ends)."""
    best = None
    for p in sorted(run.glob("logs/steps_rank*.jsonl")):
        rank = int(p.stem.replace("steps_rank", ""))
        if target_rank is not None and rank == target_rank:
            continue
        steps = [r for r in read_jsonl(p) if r.get("ev") == "rb"]
        if not steps:
            continue
        stall = steps[-1]["t_ns"]
        for a, b in zip(steps, steps[1:]):
            if a["t_ns"] >= t_inject - int(2 * NS) and b["t0_ns"] - a["t_ns"] >= gap_s * NS:
                stall = a["t_ns"]
                break
        cand = (stall, rank, steps[-1].get("fc"))
        if best is None or rank == 0:
            best = cand
    return best


STOP_EVENTS = {"sigcont", "sigterm", "sigkill", "stopped", "resume"}


def t_stop_of(run: Path):
    """First teardown action by stop.py / inject.py --resume; nothing after it is a detection."""
    ts = [r["t_ns"] for r in read_jsonl(run / "inject.jsonl") if r.get("ev") in STOP_EVENTS]
    return min(ts) if ts else None


def analyze_injection(run: Path, inj: dict, reqs: list, probe: list, gap_s: float):
    t_inj = inj["t_ns"]
    target = inj.get("rank")
    target_name = f"rank{target}"
    t_stop = t_stop_of(run)
    probe = [r for r in probe if t_stop is None or r["t_ns"] < t_stop]
    end_probe = max((r["t_ns"] for r in probe), default=t_inj)

    # engine-side detection
    eng = []
    for r in probe:
        if r["t_ns"] < t_inj:
            continue
        if r.get("ev") == "log" and r.get("pattern") in ENGINE_PATTERNS:
            eng.append((r["t_ns"], f"log:{r['pattern']}@{r['file']}"))
        elif r.get("ev") == "pid" and r.get("state") in ("dead", "zombie") and r.get("who") != target_name:
            eng.append((r["t_ns"], f"pid:{r['who']}:{r['state']}"))
    eng.sort()
    t_eng, eng_who = (eng[0] if eng else (None, None))
    t_server_dead = next((r["t_ns"] for r in probe if r.get("ev") == "pid" and r.get("who") == "http_server"
                          and r.get("state") in ("dead", "zombie") and r["t_ns"] >= t_inj), None)
    t_target_seen = next((r["t_ns"] for r in probe if r.get("ev") == "pid" and r.get("who") == target_name
                          and r["t_ns"] >= t_inj), None)

    def first_probe(kinds, sustained=None):
        # 'probe' = state change; 'probe_bad' = a repeated bad result (the probe was already
        # failing, e.g. a generation probe stuck in the queue before injection)
        for r in probe:
            if r["t_ns"] < t_inj or r.get("kind") not in kinds:
                continue
            if sustained is None and r.get("ev") in ("probe", "probe_bad") and r.get("state") != "200":
                return r["t_ns"], r["state"]
            if sustained is not None and r.get("consecutive_bad", 0) >= sustained and r.get("state") != "200":
                return r["t_ns"], r["state"]
        return None, None

    pre_lo = t_inj - 300 * NS
    pre_health_bad = [r for r in probe if r.get("ev") == "probe" and r.get("kind") in ("health", "health_generate")
                      and r.get("state") != "200" and pre_lo <= r["t_ns"] < t_inj]
    pre_gen1_bad = [r for r in probe if r.get("ev") == "probe" and r.get("kind") == "gen1"
                    and r.get("state") != "200" and pre_lo <= r["t_ns"] < t_inj]
    def health_with_client_timeout(x_s: float):
        """When a health checker with an x-second client timeout would first have flagged the replica."""
        best = None
        for r in probe:
            if r.get("ev") not in ("probe", "probe_sustained", "heartbeat") or r.get("kind") not in ("health", "health_generate"):
                continue
            if r["t_ns"] < t_inj:
                continue
            cand = None
            if r.get("state") != "200":
                cand = r["t_ns"]
            if r.get("latency_ms") is not None and r["latency_ms"] > x_s * 1e3 and r.get("t_sent_ns"):
                cand = min(cand or 1 << 62, r["t_sent_ns"] + int(x_s * NS))
            if cand is not None and cand >= t_inj and (best is None or cand < best):
                best = cand
        return best

    t_health_ct5 = health_with_client_timeout(5.0)
    t_health, health_state = first_probe({"health", "health_generate"})
    t_gen1, gen1_state = first_probe({"gen1"})
    t_gen1_3, _ = first_probe({"gen1"}, sustained=3)

    t_end = t_eng or t_server_dead or end_probe
    rq = [r for r in reqs if "rid" in r and "ev" not in r]

    def failed(r):
        return bool(r.get("error")) or r.get("status") != 200 or not r.get("done", True)

    inflight = [r for r in rq if r["send_ts"] < t_inj and failed(r)]
    # Client-visible stall: last *token* any request received before the engine-side
    # detection (teardown can flush error chunks, which must not count as output).
    cap = t_eng or t_server_dead or end_probe
    tok_ts = [r["last_token_ts"] for r in rq if r.get("last_token_ts") and r["last_token_ts"] < cap]
    t_stall_client = max(tok_ts) if tok_ts else None
    steps = stall_from_steps(run, t_inj, target, gap_s)
    new_all = [r for r in rq if t_inj <= r["send_ts"] <= t_end]
    new_err = [r for r in new_all if failed(r)]

    def ttft(rs):
        return [(r["first_token_ts"] - r["send_ts"]) / 1e6 for r in rs if r.get("first_token_ts")]

    before = ttft([r for r in rq if t_inj - 60 * NS <= r["send_ts"] < t_inj])
    during = ttft(new_all)

    return {
        "mode": "inject",
        "case": inj.get("case"),
        "target_rank": target,
        "t_inject_ns": t_inj,
        "t_first_symptom": _s(t_stall_client, t_inj),
        "t_stall_engine": _s(steps[0], t_inj) if steps else None,
        "stall_engine_rank": steps[1] if steps else None,
        "t_target_state_seen": _s(t_target_seen, t_inj),
        "t_detect_engine": _s(t_eng, t_inj),
        "engine_detector": eng_who,
        "t_detect_health": _s(t_health, t_inj),
        "health_state": health_state,
        "t_detect_health_if_5s_client_timeout": _s(t_health_ct5, t_inj),
        "t_detect_gen1": _s(t_gen1, t_inj),
        "t_detect_gen1_sustained3": _s(t_gen1_3, t_inj),
        "t_server_dead": _s(t_server_dead, t_inj),
        "n_health_false_pos_5min_before": len(pre_health_bad),
        "n_gen1_false_pos_5min_before": len(pre_gen1_bad),
        "window_end": _s(t_end, t_inj),
        "n_inflight_hung": len(inflight),
        "n_new_sent": len(new_all),
        "n_new_errored": len(new_err),
        "ttft_p50_before_ms": _pct(before, 0.5),
        "ttft_p99_before_ms": _pct(before, 0.99),
        "n_before": len(before),
        "ttft_p50_during_ms": _pct(during, 0.5),
        "ttft_p99_during_ms": _pct(during, 0.99),
        "n_during_with_first_token": len(during),
        "engine_events_all": "; ".join(f"{_s(t, t_inj)}s {w}" for t, w in eng[:8]),
    }


def analyze_floor(run: Path, reqs: list, probe: list):
    t_stop = t_stop_of(run)
    probe = [r for r in probe if t_stop is None or r["t_ns"] < t_stop]
    rq = [r for r in reqs if "rid" in r and "ev" not in r]
    starts = [r for r in reqs if r.get("ev") == "load_start"]
    t0 = starts[0]["t_ns"] if starts else (min((r["send_ts"] for r in rq), default=None))
    t_last = max((r.get("last_chunk_ts") or r["send_ts"] for r in rq), default=t0)
    eng = sorted((r["t_ns"], f"log:{r['pattern']}@{r['file']}") for r in probe
                 if r.get("ev") == "log" and r.get("pattern") in ENGINE_PATTERNS)
    deaths = sorted((r["t_ns"], f"pid:{r['who']}:{r['state']}") for r in probe
                    if r.get("ev") == "pid" and r.get("state") in ("dead", "zombie"))
    health_bad = [r for r in probe if r.get("ev") == "probe" and r.get("kind") in ("health", "health_generate") and r.get("state") != "200"]
    gen1_bad = [r for r in probe if r.get("ev") == "probe" and r.get("kind") == "gen1" and r.get("state") != "200"]
    max_bad = max((r.get("consecutive_bad", 0) for r in probe if r.get("kind") == "gen1"), default=0)
    errs = [r for r in rq if r.get("error") or r.get("status") != 200 or not r.get("done", True)]
    tt = [(r["first_token_ts"] - r["send_ts"]) / 1e6 for r in rq if r.get("first_token_ts")]
    spurious = bool(eng or deaths)
    return {
        "mode": "floor",
        "duration_s": _s(t_last, t0),
        "n_requests": len(rq),
        "n_errors": len(errs),
        "error_kinds": "; ".join(sorted({str(r["error"])[:40] for r in errs})[:5]),
        "ttft_p50_ms": _pct(tt, 0.5),
        "ttft_p99_ms": _pct(tt, 0.99),
        "spurious_kill": spurious,
        "first_engine_event": f"{_s(eng[0][0], t0)}s {eng[0][1]}" if eng else None,
        "first_pid_death": f"{_s(deaths[0][0], t0)}s {deaths[0][1]}" if deaths else None,
        "health_non200_changes": len(health_bad),
        "gen1_non200_changes": len(gen1_bad),
        "gen1_max_consecutive_bad": max_bad,
        "engine_events_all": "; ".join(f"{_s(t, t0)}s {w}" for t, w in eng[:8]),
    }


def analyze_run(run: Path, gap_s: float) -> dict:
    env = _load_json(run / "launch_env.json", {}) or {}
    reqs = read_jsonl(run / "requests.jsonl")
    probe = read_jsonl(run / "probe.jsonl")
    injs = [r for r in read_jsonl(run / "inject.jsonl") if r.get("ev") == "inject"]
    loads = [r for r in reqs if r.get("ev") == "load_start"]
    row = {
        "run": run.name,
        "model": env.get("model"),
        "tp": env.get("tp"),
        "watchdog_timeout": env.get("watchdog_timeout"),
        "dist_timeout": env.get("dist_timeout"),
        "rpc_timeout": env.get("rpc_timeout"),
        "stack": env.get("stack", "sglang"),
        "profile": loads[0].get("profile") if loads else None,
        "rate": loads[0].get("rate") if loads else None,
    }
    if injs:
        row.update(analyze_injection(run, injs[0], reqs, probe, gap_s))
    else:
        row.update(analyze_floor(run, reqs, probe))
    # marker files written by the drivers
    row["launch_failed"] = (run / "launch_failed").exists()
    row["not_steady"] = (run / "not_steady").exists()
    row["no_engine_detect"] = (run / "no_engine_detect").exists()
    row["no_self_teardown"] = (run / "no_self_teardown").exists()
    if row["launch_failed"]:
        row["spurious_kill"] = True
    return row


def to_markdown(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c not in ("engine_events_all", "t_inject_ns")]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def aggregate_md(df: pd.DataFrame) -> str:
    out = []
    if (df["mode"] == "inject").any():
        d = df[df["mode"] == "inject"]
        g = d.groupby(["case", "watchdog_timeout"], dropna=False)
        agg = g.agg(runs=("run", "count"),
                    t_first_symptom_med=("t_first_symptom", "median"),
                    t_detect_engine_med=("t_detect_engine", "median"),
                    t_detect_engine_min=("t_detect_engine", "min"),
                    t_detect_engine_max=("t_detect_engine", "max"),
                    t_detect_health_med=("t_detect_health", "median"),
                    t_detect_gen1_med=("t_detect_gen1", "median"),
                    n_inflight_hung_med=("n_inflight_hung", "median"),
                    n_new_errored_med=("n_new_errored", "median")).reset_index()
        out.append("### T_detect per case (seconds after t_inject, median over runs)\n\n" + to_markdown(agg.round(2)))
    if (df["mode"] == "floor").any():
        d = df[df["mode"] == "floor"].copy()
        for c in ("watchdog_timeout", "dist_timeout", "rate"):
            d[c] = d[c].astype(str)   # CSV round-trips mix int/str; group on a stable key
        g = d.groupby(["profile", "rate", "watchdog_timeout", "dist_timeout"], dropna=False)
        agg = g.agg(runs=("run", "count"), spurious=("spurious_kill", "sum"),
                    errors=("n_errors", "sum"), ttft_p99_ms=("ttft_p99_ms", "max")).reset_index()
        out.append("### Timeout floor sweep (spurious kills per setting)\n\n" + to_markdown(agg))
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", default=None, help="append rows here (default results/summary.csv)")
    ap.add_argument("--md", default=None, help="write combined markdown here (default results/summary.md)")
    ap.add_argument("--gap", type=float, default=5.0, help="step gap (s) that defines an engine stall")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    csv = Path(a.csv) if a.csv else here / "results" / "summary.csv"
    md = Path(a.md) if a.md else here / "results" / "summary.md"
    csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in a.runs:
        run = Path(r).resolve()
        if not run.is_dir():
            print(f"skip {run}: not a directory", file=sys.stderr)
            continue
        row = analyze_run(run, a.gap)
        rows.append(row)
        (run / "summary.json").write_text(json.dumps(row, indent=2))
        (run / "summary.md").write_text(to_markdown(pd.DataFrame([row])))
        print(json.dumps(row, indent=1))
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    if csv.exists():
        old = pd.read_csv(csv)
        df_all = pd.concat([old[~old["run"].isin(df["run"])], df], ignore_index=True)
    else:
        df_all = df
    df_all.to_csv(csv, index=False)
    md.write_text("## Per run\n\n" + to_markdown(df_all) + "\n\n" + aggregate_md(df_all) + "\n")
    print(f"\nwrote {csv} and {md}")
    print(aggregate_md(df_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
