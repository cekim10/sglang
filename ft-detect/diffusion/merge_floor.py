"""Merge step_stats CSVs from several characterization runs (models x shapes) into the
static-vs-progress-aware watchdog comparison:

  D_w       = per-workload deadline (alpha x Q(T_step | w, warm))      from each CSV
  D_global  = max_w D_w over EVERY workload in the merged set            (one static timeout)
  slack     = D_global / D_w                                             (cost of the static timeout)
  D_cold    = max over cold categories, i.e. what a static timeout must ALSO tolerate

plus a text chart: safe timeout per workload on a log axis.

    python merge_floor.py results/char_*.csv [--md results/floor_merged.md]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def chart(df: pd.DataFrame, d_global: float, width: int = 48) -> str:
    """Log-scale bar per workload: safe per-workload deadline vs the global static one."""
    lo = max(min(df["D_w_s"].min(), 0.01), 0.01)
    hi = max(d_global, df["D_cold_s"].max()) * 1.2
    lo_l, hi_l = math.log10(lo), math.log10(hi)

    def pos(v):
        v = max(v, lo)
        return int(round((math.log10(v) - lo_l) / (hi_l - lo_l) * (width - 1)))

    g = pos(d_global)
    lines = [f"safe step deadline per workload (log axis {lo:g} s .. {hi:.3g} s); '#' = D_w, 'c' = worst cold step, '|' = D_global={d_global:.2f} s", ""]
    for _, r in df.iterrows():
        row = [" "] * width
        pw, pc = pos(r["D_w_s"]), pos(r["D_cold_s"]) if not pd.isna(r["D_cold_s"]) else None
        for i in range(pw + 1):
            row[i] = "#"
        if pc is not None and pc < width:
            row[pc] = "c"
        if g < width:
            row[g] = "|"
        label = f"{r['model'].split('/')[-1][:18]:18s} {r['shape']:>16s}"
        lines.append(f"{label} {''.join(row)} D_w={r['D_w_s']:.2f}s slack={r['slack_x']:.0f}x")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    frames = [pd.read_csv(p) for p in a.csvs if Path(p).exists()]
    if not frames:
        print("no CSVs")
        return 1
    df = pd.concat(frames, ignore_index=True)
    df = df[["model", "shape", "pixels_frames", "steps", "requests", "step_p50_ms", "step_p999_ms", "step_max_ms", "warm_q_ms",
             "max_first_after_launch_ms", "max_first_of_shape_ms", "max_after_idle_ms", "D_w_s"]].copy()
    df["D_cold_s"] = (df[["max_first_after_launch_ms", "max_first_of_shape_ms", "max_after_idle_ms"]].max(axis=1) * 2 / 1e3).round(2)
    d_global = float(df["D_w_s"].max())
    d_global_cold = float(max(d_global, df["D_cold_s"].max()))
    df["slack_x"] = (d_global / df["D_w_s"]).round(1)
    df["slack_incl_cold_x"] = (d_global_cold / df["D_w_s"]).round(1)
    df = df.sort_values("pixels_frames")
    out = [f"Merged over {len(frames)} characterization CSV(s), {df['model'].nunique()} model(s), {len(df)} workload(s).\n",
           md_table(df),
           f"\n**D_global (warm) = {d_global:.2f} s; D_global including cold steps = {d_global_cold:.2f} s; "
           f"smallest D_w = {df['D_w_s'].min():.3f} s; max slack = {df['slack_x'].max():.0f}x (incl. cold: {df['slack_incl_cold_x'].max():.0f}x).**\n",
           "```\n" + chart(df, d_global) + "\n```"]
    text = "\n".join(out)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
