"""Per-shape denoising-step durations -> the step-watchdog floor the diffusion stack *would* need.

The LLM result (Phase 1) was that a per-step watchdog can sit at 2 s with zero false
positives because chunked prefill bounds step time independently of the request.
For diffusion the step time is set by the request's shape (pixels x frames), so one
fixed timeout either fires on big shapes or is slack on small ones. This script
measures that from run/logs/steps_rank*.jsonl (written by launch_wrapped_diff.py):

  per (w, h, frames): n_steps, step p50 / p99 / max (ms), and the max gap between
  consecutive steps of the same request (what a watchdog actually sees)
  -> global floor  = max over shapes of max-step (plus a safety factor)
  -> per-shape floor = that shape's max-step; the ratio between the two is the
     cost of a single global timeout.

    python step_stats.py runs/diff_floor_mixed_*            # prints markdown
    python step_stats.py <run_dir> --md out.md --csv out.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read_jsonl  # noqa: E402


def load_steps(run_dirs):
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        for p in sorted(rd.glob("logs/steps_rank*.jsonl")):
            rank = int(p.stem.replace("steps_rank", ""))
            prev_t = {}
            for r in read_jsonl(p):
                if r.get("ev") != "ds":
                    continue
                key = (r.get("rid"), rank)
                dur_ms = (r["t_ns"] - r["t0_ns"]) / 1e6
                gap_ms = (r["t_ns"] - prev_t[key]) / 1e6 if key in prev_t else None
                prev_t[key] = r["t_ns"]
                rows.append({"run": rd.name, "rank": rank, "rid": r.get("rid"), "w": r.get("w"), "h": r.get("h"),
                             "nf": r.get("nf") or 1, "n": r.get("n"), "i": r.get("i"), "dur_ms": dur_ms, "gap_ms": gap_ms})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, safety: float):
    if df.empty:
        return pd.DataFrame(), None
    df = df.copy()
    df["shape"] = df.apply(lambda r: f"{r['w']}x{r['h']}" + (f"x{int(r['nf'])}f" if r["nf"] and int(r["nf"]) > 1 else ""), axis=1)
    df["pixels_frames"] = df["w"].fillna(0) * df["h"].fillna(0) * df["nf"].fillna(1)
    # order within each (run, rank, shape) so "first request of this shape" is well defined
    df = df.sort_values(["run", "rank", "shape"]).reset_index(drop=True)
    df["_ord"] = df.groupby(["run", "rank", "shape"]).cumcount()
    first_rid = df[df["_ord"] == 0].set_index(["run", "rank", "shape"])["rid"]
    df["is_first_req_of_shape"] = df.apply(lambda r: first_rid.get((r["run"], r["rank"], r["shape"])) == r["rid"], axis=1)
    p50 = df.groupby("shape")["dur_ms"].transform("median")
    df["outlier"] = df["dur_ms"] > 5 * p50

    g = df.groupby("shape")
    out = g.agg(pixels_frames=("pixels_frames", "first"), steps=("dur_ms", "count"), requests=("rid", "nunique"),
                step_p50_ms=("dur_ms", "median"), step_p99_ms=("dur_ms", lambda s: s.quantile(0.99)),
                step_max_ms=("dur_ms", "max"), gap_max_ms=("gap_ms", "max"),
                outliers_gt5x_p50=("outlier", "sum")).reset_index().sort_values("pixels_frames")
    # was the slowest step the first request ever seen at this shape (cold-shape cost) ?
    idx = df.groupby("shape")["dur_ms"].idxmax()
    cold = df.loc[idx].set_index("shape")["is_first_req_of_shape"]
    out["max_is_cold_shape"] = out["shape"].map(cold).astype(bool)
    warm = df[~df["is_first_req_of_shape"]].groupby("shape")["dur_ms"].max()
    out["step_max_warm_ms"] = out["shape"].map(warm)
    out["per_shape_floor_s"] = (out[["step_max_ms", "gap_max_ms"]].max(axis=1) * safety / 1e3).round(2)
    out["per_shape_floor_warm_s"] = (out["step_max_warm_ms"].fillna(out["step_max_ms"]) * safety / 1e3).round(2)
    global_floor = float(out["per_shape_floor_s"].max())
    out["slack_vs_global_x"] = (global_floor / out["per_shape_floor_s"]).round(1)
    for c in ("step_p50_ms", "step_p99_ms", "step_max_ms", "gap_max_ms", "step_max_warm_ms"):
        out[c] = out[c].round(1)
    return out, global_floor


def md(out: pd.DataFrame, global_floor, safety, n_runs) -> str:
    if out.empty:
        return "_no denoising-step records (is the DenoisingStage hook installed? see steps_rank*.jsonl 'hook' records)_"
    cols = list(out.columns)
    lines = [f"Denoising-step durations by shape over {n_runs} run(s); floor = max observed step (or inter-step gap) x {safety:g} safety.\n",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in out.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    gw = float(out["per_shape_floor_warm_s"].max())
    lines.append(f"\n**Global step-watchdog floor (one timeout for every served shape): {global_floor:.2f} s "
                 f"(all steps) / {gw:.2f} s (excluding each shape's first request, i.e. cold-shape warm-up).** "
                 f"Smallest per-shape floor: {out['per_shape_floor_s'].min():.2f} s "
                 f"(a single global timeout is {out['slack_vs_global_x'].max():.0f}x slack on that shape). "
                 f"`max_is_cold_shape` says whether a shape's slowest step was its very first request.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--safety", type=float, default=2.0, help="multiply the observed max by this (default 2x)")
    ap.add_argument("--md", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    runs = [r for r in a.runs if Path(r).is_dir()]
    df = load_steps(runs)
    out, gf = summarize(df, a.safety)
    text = md(out, gf, a.safety, len(runs))
    print(text)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(text + "\n")
    if a.csv and not out.empty:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
