"""Phase 2: assemble REPORT-diffusion.md from diffusion/results/*.csv, the latest recon.json
and diffusion/results/notes.md. Same criteria as Phase 1 (GO if T_detect >= --min-detect s
AND floor >= --min-floor s), with two floors reported: the swept RPC/dist timeouts and the
*hypothetical* per-step watchdog floor derived from measured step times (step_stats.py).

    python report_diff.py
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
sys.path.insert(0, str(HERE.parent))
from report import md_table  # noqa: E402


def latest_recon():
    c = sorted(glob.glob(str(HERE / "runs" / "*" / "recon.json")) + glob.glob(str(HERE / "run" / "recon.json")), key=os.path.getmtime)
    return json.loads(Path(c[-1]).read_text()) if c else {}


def load_csv(name):
    p = HERE / "results" / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def swept_floor(df: pd.DataFrame, knob: str):
    """Smallest clean value of `knob` ('rpc_timeout' or 'dist_timeout'), requiring every larger value clean too."""
    if df.empty or knob not in df:
        return None
    d = df.copy()
    d["T"] = pd.to_numeric(d[knob], errors="coerce")
    d = d.dropna(subset=["T"])
    if d.empty:
        return None
    per = d.groupby("T").agg(bad=("spurious_kill", "max"), failed=("launch_failed", "max")).sort_index(ascending=False)
    floor = None
    for T, row in per.iterrows():
        if bool(row["bad"]) or bool(row["failed"]):
            break
        floor = T
    return floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-detect", type=float, default=60.0)
    ap.add_argument("--min-floor", type=float, default=10.0)
    ap.add_argument("--out", default=str(HERE / "REPORT-diffusion.md"))
    a = ap.parse_args()

    rc = latest_recon()
    caseA, caseB = load_csv("caseA.csv"), load_csv("caseB.csv")
    fm, fh = load_csv("floor_mixed.csv"), load_csv("floor_harsh.csv")
    ssm, ssh = load_csv("step_stats_mixed.csv"), load_csv("step_stats_harsh.csv")
    inj = pd.concat([d for d in (caseA, caseB) if not d.empty], ignore_index=True) if not (caseA.empty and caseB.empty) else pd.DataFrame()
    ngpu = None
    for d in (inj, fm, fh):
        if not d.empty and "tp" in d:
            ngpu = int(d["tp"].dropna().iloc[0]); break

    L = [f"# Failure-detection latency kill test — Phase 2 (SGLang Diffusion, {ngpu or '?'} GPUs, sequence parallel)\n"]
    L.append("## 1. Stack\n")
    L.append(f"- SGLang `{rc.get('sglang_version')}` (sglang.multimodal_gen) at `{rc.get('sglang_path')}`, commit `{rc.get('sglang_git_commit')}`; "
             f"torch `{rc.get('torch_version')}`, torch NCCL PG default `{rc.get('torch_default_pg_nccl_timeout_s')}` s")
    L.append(f"- env at recon: `{json.dumps({k: v for k, v in (rc.get('env') or {}).items() if v is not None})}`\n")

    L.append("## 2. Detectors and timeouts present in the installed diffusion runtime\n")
    rows = [{"finding": k, "value": (v.get("value") or "")[:60].replace("\n", " "),
             "file:line": f"{v.get('file')}:{v.get('line')}" if v.get("line") else "NOT FOUND", "note": v.get("note", "")}
            for k, v in (rc.get("findings") or {}).items()]
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append("Reading (from the code; the runs below test it): there is no per-step watchdog and no rank-death poll "
             "after startup; `/health` is a readiness flag that stays 200 once warm; the HTTP->scheduler RPC waits "
             "forever by default; rank 0 blocks without timeout on a hung peer's result pipe; the only sub-minute "
             "timeout anywhere is the CUDA-IPC all-to-all used by 2-rank Ulysses (10 s), and the NCCL fallback path "
             "is covered only by `--dist-timeout` (3600 s).\n")

    L.append("## 3. T_detect per case and per detector (s after t_inject)\n")
    if inj.empty:
        L.append("_no injection runs yet_\n")
    else:
        cols = ["run", "case", "target_rank", "dist_timeout", "t_first_symptom", "t_stall_engine", "t_detect_engine", "engine_detector",
                "t_detect_health", "t_detect_gen1", "t_detect_gen1_sustained3", "t_server_dead", "n_inflight_hung", "n_new_errored",
                "ttft_p50_before_ms", "ttft_p99_before_ms", "no_engine_detect", "no_self_teardown", "not_steady"]
        L.append(md_table(inj, cols) + "\n")
        g = inj.groupby(["case"], dropna=False).agg(
            runs=("run", "count"), engine_med=("t_detect_engine", "median"), engine_min=("t_detect_engine", "min"), engine_max=("t_detect_engine", "max"),
            no_detect_runs=("no_engine_detect", "sum"), health_med=("t_detect_health", "median"), gen1_med=("t_detect_gen1", "median"),
            teardown_med=("t_server_dead", "median"), no_self_teardown_runs=("no_self_teardown", "sum"),
            hung_med=("n_inflight_hung", "median"), new_err_med=("n_new_errored", "median")).reset_index().round(2)
        L.append("Median over runs (`no_detect_runs` = runs where nothing fired within MAX_DETECT_WAIT; `teardown` = HTTP server gone on its own):\n\n" + md_table(g) + "\n")
        L.append("Note on `TTFT`: images are not streamed, so these are end-to-end latencies; `gen1` is a 1-step 256x256 request with a 60 s client timeout.\n")

    L.append("## 4. Timeout floors (no injection)\n")
    for name, df, ss in (("mixed", fm, ssm), ("harsh (40% 1536x1536 @ 50 steps + 3x bursts)", fh, ssh)):
        L.append(f"### {name}\n")
        L.append(md_table(df, ["run", "rate", "rpc_timeout", "dist_timeout", "duration_s", "n_requests", "n_errors", "error_kinds", "ttft_p50_ms", "ttft_p99_ms",
                               "spurious_kill", "first_engine_event", "health_non200_changes", "gen1_max_consecutive_bad", "launch_failed"]))
        fr, fd = swept_floor(df, "rpc_timeout"), swept_floor(df, "dist_timeout")
        L.append(f"\nSmallest clean `--scheduler-rpc-timeout`: **{fr}** s; smallest clean `--dist-timeout`: **{fd}** s "
                 "(an RPC timeout 'false positive' here means a healthy request was cut off: it is a request-latency timeout, not a step timeout).\n")
        L.append("Hypothetical per-step watchdog floor from measured step durations:\n")
        if ss.empty:
            L.append("_no step_stats CSV; run step_stats.py over the floor runs_\n")
        else:
            L.append(md_table(ss) + "\n")
            L.append(f"Global floor across shapes: **{ss['per_shape_floor_s'].max():.2f} s**; smallest per-shape floor: "
                     f"**{ss['per_shape_floor_s'].min():.2f} s** (ratio {ss['slack_vs_global_x'].max():.0f}x).\n")

    L.append("## 5. Verdict\n")
    if inj.empty:
        L.append("_incomplete_\n")
    else:
        hyp = max([d["per_shape_floor_s"].max() for d in (ssm, ssh) if not d.empty], default=None)
        rows = []
        for case in sorted(inj["case"].dropna().unique()):
            d = inj[inj["case"] == case]
            eng = d["t_detect_engine"].median()
            nodet = int(d["no_engine_detect"].sum()) if "no_engine_detect" in d else 0
            anyc = d[["t_detect_engine", "t_detect_health", "t_detect_gen1"]].min(axis=1).median()
            rows.append({"case": case, "runs": len(d), "runs with no engine detection": nodet,
                         "T_detect engine (median s, detected runs)": None if pd.isna(eng) else round(eng, 1),
                         "T_detect any component (median s)": None if pd.isna(anyc) else round(anyc, 1),
                         "T_evict (median s)": None if d["t_server_dead"].isna().all() else round(d["t_server_dead"].median(), 1)})
        L.append(md_table(pd.DataFrame(rows)) + "\n")
        L.append(f"Criteria: T_detect >= {a.min_detect:.0f} s AND floor >= {a.min_floor:.0f} s. "
                 f"Hypothetical step-watchdog global floor from step times: **{hyp if hyp is None else round(hyp, 2)} s**. "
                 "If the engine never detected within the wait (no_detect runs), T_detect is bounded below by MAX_DETECT_WAIT and the "
                 "first condition holds trivially; the verdict then rests on whether any timeout that *could* be configured has a floor "
                 f"above {a.min_floor:.0f} s. Write the GO/KILL line in results/notes.md after reading sections 3-4; this generator does not guess it.\n")

    L.append("## 6. Workarounds and observations\n")
    notes = HERE / "results" / "notes.md"
    L.append(notes.read_text() if notes.exists() else "_add observations to diffusion/results/notes.md_")
    Path(a.out).write_text("\n".join(L))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
