"""Per-shape execution-quantum durations -> the step-watchdog floor the diffusion stack *would* need.

Input: run/logs/steps_rank*.jsonl written by launch_wrapped_diff.py
  {"ev":"ds", i, n, w, h, nf, rid, t0_ns, t_ns}          one denoising step
  {"ev":"stage", name, cls, w, h, nf, rid, t0_ns, t_ns}  one pipeline stage (encode / denoise / decode ...)

Per (model, shape = WxH[xFf]):
  steps, requests, step p50 / p99 / p99.9 / max (ms), max inter-step gap, and the same max
  split by *cold category* of the request the step belongs to:
    first_after_launch : the first request this rank served after start-up
    first_of_shape     : the first request at this shape (compile / autotune / allocator warm-up)
    after_idle         : first request after >= --idle-s of no steps on this rank
    warm               : everything else
  -> D_w  = alpha * Q(T_step | w, warm)       per-workload deadline (Q = --quantile, default p99.9)
  -> D_global = max_w D_w                     one static timeout for every served shape
  -> slack = D_global / D_w                   what a static watchdog costs each workload
Also a per-shape stage table (max duration of each non-denoising stage), because a VAE decode
is a single execution quantum with no step boundary inside it.

    python step_stats.py <run_dir>... [--md out.md] [--csv out.csv] [--quantile 0.999] [--safety 2]
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


def _model_of(run: Path) -> str:
    try:
        return json.loads((run / "launch_env.json").read_text()).get("model", run.name)
    except Exception:
        return run.name


def _shape(r):
    nf = r.get("nf") or 1
    return f"{r.get('w')}x{r.get('h')}" + (f"x{int(nf)}f" if nf and int(nf) > 1 else "")


def load_events(run_dirs):
    steps, stages = [], []
    for rd in run_dirs:
        rd = Path(rd)
        model = _model_of(rd)
        for p in sorted(rd.glob("logs/steps_rank*.jsonl")):
            rank = int(p.stem.replace("steps_rank", ""))
            for r in read_jsonl(p):
                ev = r.get("ev")
                if ev == "ds":
                    steps.append({"run": rd.name, "model": model, "rank": rank, "rid": r.get("rid"), "shape": _shape(r),
                                  "w": r.get("w"), "h": r.get("h"), "nf": r.get("nf") or 1, "n": r.get("n"), "i": r.get("i"),
                                  "t0_ns": r["t0_ns"], "t_ns": r["t_ns"], "dur_ms": (r["t_ns"] - r["t0_ns"]) / 1e6})
                elif ev == "stage":
                    stages.append({"run": rd.name, "model": model, "rank": rank, "rid": r.get("rid"), "shape": _shape(r),
                                   "stage": r.get("name") or r.get("cls"), "t0_ns": r["t0_ns"], "t_ns": r["t_ns"],
                                   "dur_ms": (r["t_ns"] - r["t0_ns"]) / 1e6})
    return pd.DataFrame(steps), pd.DataFrame(stages)


def classify_cold(df: pd.DataFrame, idle_s: float) -> pd.DataFrame:
    """Tag every step with the cold category of its request (per run+rank)."""
    df = df.sort_values(["run", "rank", "t0_ns"]).reset_index(drop=True)
    df["cold"] = "warm"
    for (run, rank), g in df.groupby(["run", "rank"], sort=False):
        first_rid_overall = g.iloc[0]["rid"]
        first_rid_of_shape = g.groupby("shape")["rid"].first().to_dict()
        # idle gap before each request = time since the previous step of a *different* request
        req_start = g.groupby("rid")["t0_ns"].min()
        prev_end = {}
        order = sorted(req_start.items(), key=lambda kv: kv[1])
        last_end_ns = None
        idle_rids = set()
        for rid, start in order:
            if last_end_ns is not None and (start - last_end_ns) / 1e9 >= idle_s:
                idle_rids.add(rid)
            last_end_ns = max(last_end_ns or 0, g[g["rid"] == rid]["t_ns"].max())
        idx = g.index
        cat = pd.Series("warm", index=idx)
        cat[g["rid"].isin(idle_rids)] = "after_idle"
        cat[g.apply(lambda r: first_rid_of_shape.get(r["shape"]) == r["rid"], axis=1)] = "first_of_shape"
        cat[g["rid"] == first_rid_overall] = "first_after_launch"
        df.loc[idx, "cold"] = cat
    # gap to previous step of the same request (what a watchdog sees between step completions)
    df["gap_ms"] = df.groupby(["run", "rank", "rid"])["t_ns"].diff() / 1e6
    return df


def summarize(df: pd.DataFrame, q: float, safety: float):
    if df.empty:
        return pd.DataFrame(), None
    df["pixels_frames"] = df["w"].fillna(0) * df["h"].fillna(0) * df["nf"].fillna(1)
    warm = df[df["cold"] == "warm"]
    g = df.groupby(["model", "shape"])
    out = g.agg(pixels_frames=("pixels_frames", "first"), steps=("dur_ms", "count"), requests=("rid", "nunique"),
                step_p50_ms=("dur_ms", "median"), step_p99_ms=("dur_ms", lambda s: s.quantile(0.99)),
                step_p999_ms=("dur_ms", lambda s: s.quantile(0.999)), step_max_ms=("dur_ms", "max"),
                gap_max_ms=("gap_ms", "max")).reset_index()
    gw = warm.groupby(["model", "shape"]).agg(warm_steps=("dur_ms", "count"),
                                              warm_q_ms=("dur_ms", lambda s: s.quantile(q)),
                                              warm_max_ms=("dur_ms", "max")).reset_index()
    out = out.merge(gw, on=["model", "shape"], how="left")
    for cat in ("first_after_launch", "first_of_shape", "after_idle"):
        m = df[df["cold"] == cat].groupby(["model", "shape"])["dur_ms"].max().rename(f"max_{cat}_ms").reset_index()
        out = out.merge(m, on=["model", "shape"], how="left")
    out["D_w_s"] = (out["warm_q_ms"].fillna(out["step_max_ms"]) * safety / 1e3).round(3)
    out["D_w_all_s"] = (out[["step_max_ms", "gap_max_ms"]].max(axis=1) * safety / 1e3).round(3)
    d_global = float(out["D_w_s"].max())
    out["slack_x"] = (d_global / out["D_w_s"]).round(1)
    out = out.sort_values("pixels_frames")
    for c in [c for c in out.columns if c.endswith("_ms")]:
        out[c] = out[c].round(1)
    return out, d_global


def stage_table(st: pd.DataFrame) -> pd.DataFrame:
    if st.empty:
        return pd.DataFrame()
    g = st.groupby(["model", "shape", "stage"]).agg(n=("dur_ms", "count"), p50_ms=("dur_ms", "median"),
                                                     max_ms=("dur_ms", "max")).reset_index()
    for c in ("p50_ms", "max_ms"):
        g[c] = g[c].round(1)
    return g


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_no data_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--quantile", type=float, default=0.999)
    ap.add_argument("--safety", type=float, default=2.0)
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--md", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    runs = [r for r in a.runs if Path(r).is_dir()]
    steps, stages = load_events(runs)
    if steps.empty:
        print("_no denoising-step records (check steps_rank*.jsonl 'hook' records)_")
        return 1
    steps = classify_cold(steps, a.idle_s)
    out, d_global = summarize(steps, a.quantile, a.safety)
    stg = stage_table(stages[~stages["stage"].astype(str).str.lower().str.contains("denois")]) if not stages.empty else pd.DataFrame()
    text = [f"Execution-quantum durations over {len(runs)} run(s). D_w = {a.safety:g} x Q{a.quantile:g}(warm step); D_global = max_w D_w.\n",
            md_table(out),
            f"\n**D_global = {d_global:.2f} s; smallest D_w = {out['D_w_s'].min():.2f} s; max slack = {out['slack_x'].max():.0f}x.** "
            f"Cold categories: `max_first_after_launch_ms`, `max_first_of_shape_ms`, `max_after_idle_ms` are the slowest step of such requests "
            f"(a static watchdog must tolerate them; a progress-aware one can know they are coming).",
            "\nNon-denoising stages (single execution quanta without step boundaries):\n", md_table(stg)]
    text = "\n".join(text)
    print(text)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(text + "\n")
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.csv, index=False)
        if not stg.empty:
            stg.to_csv(str(a.csv).replace(".csv", "_stages.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
