"""Step 4: assemble REPORT.md from results/*.csv, the latest recon.json and results/notes.md.

    python report.py                    # -> ft-detect/REPORT.md
    python report.py --min-detect 60 --min-floor 10

Verdict rule (from the task): GO if median T_detect at defaults >= --min-detect s
AND the smallest zero-false-positive timeout in both workloads >= --min-floor s.
T_detect is reported per detector; the verdict line is computed for the engine
watchdog AND for "any component" (min over engine / health / gen1) so both readings
of the criterion are visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def md_table(df: pd.DataFrame, cols=None) -> str:
    if df is None or df.empty:
        return "_no data_"
    cols = cols or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def latest_recon():
    cands = sorted(glob.glob(str(HERE / "runs" / "*" / "recon.json")) + glob.glob(str(HERE / "run" / "recon.json")),
                   key=os.path.getmtime)
    return json.loads(Path(cands[-1]).read_text()) if cands else {}


def load_csv(name):
    p = HERE / "results" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def floor_value(df: pd.DataFrame):
    """Smallest T with zero spurious kills AND zero launch failures, requiring every larger T to be clean too."""
    if df.empty or "watchdog_timeout" not in df:
        return None, {}
    d = df.copy()
    d["T"] = pd.to_numeric(d["watchdog_timeout"].where(d["watchdog_timeout"] != "default", d.get("dist_timeout")), errors="coerce")
    d = d.dropna(subset=["T"])
    per = d.groupby("T").agg(spurious=("spurious_kill", "max"), failed=("launch_failed", "max") if "launch_failed" in d else ("spurious_kill", "min")).sort_index(ascending=False)
    floor = None
    for T, row in per.iterrows():
        if bool(row["spurious"]) or bool(row.get("failed", False)):
            break
        floor = T
    return floor, per.reset_index().to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-detect", type=float, default=60.0)
    ap.add_argument("--min-floor", type=float, default=10.0)
    ap.add_argument("--out", default=str(HERE / "REPORT.md"))
    a = ap.parse_args()

    rc = latest_recon()
    caseA, caseB = load_csv("caseA.csv"), load_csv("caseB.csv")
    fm, fh = load_csv("floor_mixed.csv"), load_csv("floor_harsh.csv")
    inj = pd.concat([caseA, caseB], ignore_index=True) if not (caseA.empty and caseB.empty) else pd.DataFrame()

    L = []
    L.append("# Failure-detection latency kill test — Phase 1 (SGLang, TP=4)\n")
    L.append("## 1. Stack\n")
    L.append(f"- SGLang version: `{rc.get('sglang_version')}` at `{rc.get('sglang_path')}`; git commit: `{rc.get('sglang_git_commit')}`")
    L.append(f"- torch `{rc.get('torch_version')}`, NCCL `{rc.get('nccl_version')}`, python `{rc.get('python')}`")
    L.append(f"- torch default process-group timeout: `{rc.get('torch_default_pg_timeout_s')}` s; NCCL default: `{rc.get('torch_default_pg_nccl_timeout_s')}` s")
    L.append(f"- relevant env at recon time: `{json.dumps({k: v for k, v in (rc.get('env') or {}).items() if v is not None})}`\n")

    L.append("## 2. Timeout defaults found in the installed code\n")
    rows = [{"finding": k, "value": v.get("value"), "file:line": f"{v.get('file')}:{v.get('line')}" if v.get("line") else "NOT FOUND", "note": v.get("note", "")}
            for k, v in (rc.get("findings") or {}).items()]
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append("Interpretation (from the Step 0 code read):\n"
             "- The scheduler watchdog counts `forward_ct`, polls every `timeout/2`, so a hang is noticed 1.0x-1.5x the timeout after the last step, then sleeps 5 s and SIGQUITs the parent, which kills the whole tree. Detection == replica death; there is no 'unhealthy' state.\n"
             "- A dead rank (non-zero exit code) is noticed by `SubprocessWatchdog` polling every 1 s in the HTTP-server process; a SIGSTOPped rank still counts as alive.\n"
             "- `/health` and `/health_generate` share one handler: 200 as soon as *any* engine output arrives, 503 after `SGLANG_HEALTH_CHECK_TIMEOUT` (20 s) of silence.\n"
             "- `--dist-timeout` defaults to None (torch default). TP all-reduce under load goes through custom all-reduce / pynccl, which torch's NCCL watchdog does not cover; the gloo CPU group for the per-step request broadcast has a hard-coded 2 h timeout.\n")

    L.append("## 3. T_detect per case and per detector (s after t_inject)\n")
    if inj.empty:
        L.append("_no injection runs yet_\n")
    else:
        cols = ["run", "case", "target_rank", "watchdog_timeout", "t_first_symptom", "t_stall_engine", "t_detect_engine", "engine_detector",
                "t_detect_health", "t_detect_gen1", "t_detect_gen1_sustained3", "t_server_dead", "n_inflight_hung", "n_new_errored",
                "ttft_p50_before_ms", "ttft_p99_before_ms", "ttft_p50_during_ms", "ttft_p99_during_ms", "not_steady"]
        L.append(md_table(inj, cols) + "\n")
        g = inj.groupby(["case", "watchdog_timeout"], dropna=False).agg(
            runs=("run", "count"),
            engine_med=("t_detect_engine", "median"), engine_min=("t_detect_engine", "min"), engine_max=("t_detect_engine", "max"),
            health_med=("t_detect_health", "median"), gen1_med=("t_detect_gen1", "median"),
            hung_med=("n_inflight_hung", "median"), new_err_med=("n_new_errored", "median")).reset_index().round(2)
        L.append("Median over runs:\n\n" + md_table(g) + "\n")

    L.append("## 4. Timeout floor (no injection)\n")
    floor_m, per_m = floor_value(fm)
    floor_h, per_h = floor_value(fh)
    for name, df, fl, per in (("mixed", fm, floor_m, per_m), ("harsh (30% 32k prefill + 3x bursts)", fh, floor_h, per_h)):
        L.append(f"### {name}\n")
        L.append(md_table(df, ["run", "watchdog_timeout", "dist_timeout", "duration_s", "n_requests", "n_errors", "ttft_p50_ms", "ttft_p99_ms",
                               "spurious_kill", "first_engine_event", "first_pid_death", "health_non200_changes", "gen1_max_consecutive_bad", "launch_failed"]))
        L.append(f"\nSmallest clean setting: **{fl}** s\n")
    floor_both = None if floor_m is None or floor_h is None else max(floor_m, floor_h)
    L.append(f"Timeout floor (both workloads): **{floor_both}** s\n")

    L.append("## 5. Verdict\n")
    if inj.empty or floor_both is None:
        L.append("_incomplete: need both injection runs and both floor sweeps_\n")
    else:
        dflt = inj[inj["watchdog_timeout"].astype(str) == "default"]
        if dflt.empty:
            L.append("_no injection runs at stock defaults (WATCHDOG_TIMEOUT unset); the verdict needs those_")
        for case in sorted(dflt["case"].dropna().unique()):
            d = dflt[dflt["case"] == case]
            eng = d["t_detect_engine"].median()
            anyc = d[["t_detect_engine", "t_detect_health", "t_detect_gen1"]].min(axis=1).median()
            go_eng = (eng >= a.min_detect) and (floor_both >= a.min_floor)
            go_any = (anyc >= a.min_detect) and (floor_both >= a.min_floor)
            L.append(f"- Case {case}: engine-watchdog T_detect median = {eng:.1f} s -> **{'GO' if go_eng else 'KILL'}**; "
                     f"any-component T_detect median = {anyc:.1f} s -> **{'GO' if go_any else 'KILL'}** "
                     f"(criteria: T_detect >= {a.min_detect} s and floor {floor_both} >= {a.min_floor} s)")
        L.append("")

    L.append("## 6. Workarounds and observations\n")
    notes = HERE / "results" / "notes.md"
    L.append(notes.read_text() if notes.exists() else "_add free-text observations to results/notes.md_")
    L.append("\nHarness-level workarounds baked in (each one is a finding about the stock stack):\n"
             "- Per-rank logs and PIDs are not available from a stock launch (children inherit the parent's stdout); `launch_wrapped.py` redirects fds and records PIDs inside each spawned rank.\n"
             "- Scheduler step boundaries are not exported; `run_batch`/`process_batch_result` are wrapped from the launcher to timestamp them.\n"
             "- `/health` blocks up to 20 s server-side, so the probe keeps one request in flight per endpoint instead of polling synchronously at 100 ms.\n"
             "- A SIGSTOPped rank must be SIGCONTed before it can be killed; `stop.py` does this before tearing the tree down.\n"
             "- After the watchdog fires, the parent kills the tree; the load generator observes hung requests as connection resets, not as timeouts, so in-flight hang counts come from `send_ts < t_inject and error != None`.\n")

    Path(a.out).write_text("\n".join(L))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
