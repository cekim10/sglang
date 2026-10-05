"""Oracle cost model for recovery after a rank failure in SP=2 diffusion serving.

Question: after a failure at denoising step k of N, which recovery action finishes the request
soonest, how big is the gap to the runner-up (G), and how much would a single static policy
lose against the oracle (regret)? Decides whether "failure-elastic reconfiguration" is a real
optimization problem or whether one fixed policy is within noise of optimal.

Actions (completion time measured from the failure instant):
  degrade     : D + fence + ready(SP1) + restore + (N-k) q1 + decode1
                continue on the surviving GPU in a fresh SP=1 process
  restore_sp2 : D + fence + ready(SP2) + restore + (N-k) q2 + decode2
                bring both ranks back and continue at the original degree
  restart     : D + fence + ready(SP2) + N q2 + decode2
                throw the trajectory away and regenerate at SP=2
with
  D       = progress deadline = alpha x the quantum running at failure (step or decode)
  fence   = terminate the poisoned process(es) (measured 0.1-8 s)
  ready() = 0 if a warm standby process exists for that degree, else the measured cold restart
  restore = checkpoint load + inject (measured 2-18 ms)
Inputs come from results/recovery_inputs.json (recovery_inputs.py) with documented defaults.

    python recovery_cost_model.py [--inputs results/recovery_inputs.json] [--md results/recovery_oracle.md]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

DEFAULTS = {
    "alpha": 2.0,                 # deadline = alpha x expected quantum (Phase 2b: p99.9 within 1-3% of p50)
    "fence_s": [0.1, 8.0],        # process exit -> gone (3c exit mode); both ends are swept
    "restore_s": 0.02,            # 3b-1: 2-18 ms
    "cold_ready_s_default": 37.0, # Wan 5B launch -> ready (24-43 s measured); per-model override from startup_decomp
    "standby": ["none", "sp1", "sp1+sp2"],   # warm standby availability: none / one spare on the surviving GPU / spares for both ranks
    "N_list": [9, 20, 30, 50],    # denoising steps per request (turbo 9; typical 20-50)
    "k_fracs": [0.1, 0.25, 0.5, 0.75, 0.9],  # failure point as fraction of N
}


def load_inputs(path: Path | None):
    inp = {"workloads": {}, "models": {}}
    if path and path.exists():
        inp = json.loads(path.read_text())
    return inp


def workload_rows(inp):
    rows = []
    for key, w in inp["workloads"].items():
        q2 = w.get("q_sp2_ms")
        if not q2:
            continue
        model = w["model"]
        mi = inp["models"].get(model, {})
        ratio = mi.get("q_sp1_over_sp2", 1.5)
        q1 = w.get("q_sp1_ms") or q2 * ratio
        d2 = w.get("decode_sp2_ms") or 0.0
        d1 = w.get("decode_sp1_ms") or d2 * mi.get("decode_sp1_over_sp2", 1.0)
        ready = (mi.get("restart") or {}).get("ready_s") or DEFAULTS["cold_ready_s_default"]
        rows.append({"model": model.split("/")[-1], "shape": w["shape"], "pixels_frames": w.get("pixels_frames", 0),
                     "q1_s": q1 / 1e3, "q2_s": q2 / 1e3, "q1_extrapolated": "q_sp1_ms" not in w,
                     "decode1_s": d1 / 1e3, "decode2_s": d2 / 1e3, "encode_s": (w.get("encode_ms") or 0) / 1e3,
                     "cold_ready_s": ready, "state_MiB": w.get("dynamic_state_MiB") or w.get("state_MiB")})
    return sorted(rows, key=lambda r: (r["model"], r["pixels_frames"]))


def completion_times(w, N, k, standby, fence, alpha, restore):
    rem = N - k
    D = alpha * w["q2_s"]                               # failure during a denoising step at SP=2
    ready1 = 0.0 if standby in ("sp1", "sp1+sp2") else w["cold_ready_s"]
    ready2 = 0.0 if standby == "sp1+sp2" else w["cold_ready_s"]
    return {
        "degrade": D + fence + ready1 + restore + rem * w["q1_s"] + w["decode1_s"],
        "restore_sp2": D + fence + ready2 + restore + rem * w["q2_s"] + w["decode2_s"],
        "restart": D + fence + ready2 + w["encode_s"] + N * w["q2_s"] + w["decode2_s"],
    }


def run(inp, md_path: Path | None):
    rows = workload_rows(inp)
    if not rows:
        print("no workloads with q_sp2 in inputs"); return 1
    cells = []
    for w, N, kf, standby, fence in itertools.product(rows, DEFAULTS["N_list"], DEFAULTS["k_fracs"], DEFAULTS["standby"], DEFAULTS["fence_s"]):
        k = max(1, int(round(kf * N)))
        t = completion_times(w, N, k, standby, fence, DEFAULTS["alpha"], DEFAULTS["restore_s"])
        best = min(t, key=t.get)
        second = min((a for a in t if a != best), key=t.get)
        # compound static policies a runtime could hard-code without any cost model
        pol_warmest = "restore_sp2" if standby == "sp1+sp2" else "degrade"          # use the warmest configuration available
        gap = w["q1_s"] - w["q2_s"]
        ready2 = 0.0 if standby == "sp1+sp2" else w["cold_ready_s"]
        ready1 = 0.0 if standby in ("sp1", "sp1+sp2") else w["cold_ready_s"]
        pol_elastic = "restore_sp2" if gap > 0 and (N - k) * gap > (ready2 - ready1) + (w["decode2_s"] - w["decode1_s"]) else "degrade"  # the crossover rule
        cells.append({
            "pol_warmest": pol_warmest, "regret_warmest": round(t[pol_warmest] / t[best] - 1, 3),
            "pol_elastic": pol_elastic, "regret_elastic": round(t[pol_elastic] / t[best] - 1, 3),"model": w["model"], "shape": w["shape"], "N": N, "k": k, "remaining": N - k, "standby": standby, "fence_s": fence,
                      "q1_s": round(w["q1_s"], 3), "q2_s": round(w["q2_s"], 3), "q1_extrapolated": w["q1_extrapolated"],
                      **{f"T_{a}": round(v, 2) for a, v in t.items()},
                      "best": best, "second": second, "G": round(t[second] / t[best], 3),
                      "regret_degrade": round(t["degrade"] / t[best] - 1, 3),
                      "regret_restore_sp2": round(t["restore_sp2"] / t[best] - 1, 3),
                      "regret_restart": round(t["restart"] / t[best] - 1, 3)})
    df = pd.DataFrame(cells)
    L = ["# Recovery oracle (cost model from measured inputs)\n",
         f"alpha={DEFAULTS['alpha']}, fence in {DEFAULTS['fence_s']} s, restore={DEFAULTS['restore_s']} s, N in {DEFAULTS['N_list']}, "
         f"failure at k/N in {DEFAULTS['k_fracs']}, standby in {DEFAULTS['standby']}. {len(df)} cells.\n",
         "## Inputs per workload\n", pd.DataFrame(rows).round(3).to_markdown(index=False), ""]
    # headline metrics
    L.append("## Headline\n")
    frac_g2 = (df["G"] >= 2).mean()
    L.append(f"- cells where the best action beats the runner-up by >= 2x: **{100 * frac_g2:.1f}%** (median G {df['G'].median():.2f})")
    L.append(f"- best action distribution: {df['best'].value_counts().to_dict()}")
    for pol in ("degrade", "restore_sp2", "restart", "warmest", "elastic"):
        r = df[f"regret_{pol}"]
        label = {"warmest": "two-rule: restore_sp2 if both spares are warm, else degrade",
                 "elastic": "crossover rule: restore_sp2 iff remaining x (q1-q2) > extra ready time"}.get(pol, f"always {pol}")
        L.append(f"- static policy **{label}**: median regret {100 * r.median():.0f}%, p95 {100 * r.quantile(0.95):.0f}%, max {100 * r.max():.0f}%, "
                 f"cells with regret >= 100%: {100 * (r >= 1.0).mean():.0f}%, cells with regret >= 10%: {100 * (r >= 0.10).mean():.0f}%")
    # where does the two-rule policy lose, and by how much?
    lose = df[df["regret_warmest"] >= 0.10]
    if not lose.empty:
        L.append(f"- cells where the two-rule policy loses >= 10%: {len(lose)} of {len(df)}; standby={lose['standby'].value_counts().to_dict()}, "
                 f"remaining steps median {lose['remaining'].median():.0f}, models {lose['model'].value_counts().to_dict()}")
    L.append("\n## GO/KILL reading (criteria fixed in advance)\n")
    rw = df["regret_warmest"]
    L.append(f"- KILL the planner if one simple static policy has p95 regret <= 10% and no large worst case: two-rule policy p95 = {100 * rw.quantile(0.95):.0f}%, max = {100 * rw.max():.0f}%.")
    L.append(f"- GO needs >= 2x penalties for every strong static policy in a realistic region: fraction of cells with G >= 2 is {100 * frac_g2:.0f}%, "
             f"but those cells are {df[df['G'] >= 2]['standby'].value_counts().to_dict()} by standby and are all resolved by the two-rule policy "
             f"(its max regret there: {100 * df[df['G'] >= 2]['regret_warmest'].max():.0f}%).")
    # regime maps: for each workload/standby, best action by (N, k_frac) with fence=0.1
    L.append("\n## Regime maps (best action; fence = 0.1 s; cell = N x remaining)\n")
    show = {(r["model"], r["shape"]) for r in rows if not r["q1_extrapolated"]} | {(rows[-1]["model"], rows[-1]["shape"])}
    big = max((r for r in rows if r["model"].startswith("Wan")), key=lambda r: r["pixels_frames"], default=None)
    if big:
        show.add((big["model"], big["shape"]))
    for (model, shape), g in df[df["fence_s"] == DEFAULTS["fence_s"][0]].groupby(["model", "shape"], sort=False):
        if (model, shape) not in show:
            continue
        L.append(f"### {model} {shape} (q1={g['q1_s'].iloc[0]} s, q2={g['q2_s'].iloc[0]} s{', q1 extrapolated' if g['q1_extrapolated'].iloc[0] else ''})\n")
        for standby, gg in g.groupby("standby", sort=False):
            piv = gg.pivot_table(index="N", columns="remaining", values="best", aggfunc="first")
            gpiv = gg.pivot_table(index="N", columns="remaining", values="G", aggfunc="first")
            L.append(f"standby = {standby}:\n")
            L.append("| N \\\\ remaining | " + " | ".join(str(c) for c in piv.columns) + " |")
            L.append("|---|" + "---|" * len(piv.columns))
            for N in piv.index:
                L.append(f"| {N} | " + " | ".join(f"{piv.loc[N, c]} (G={gpiv.loc[N, c]:.1f})" if pd.notna(piv.loc[N, c]) else "" for c in piv.columns) + " |")
            L.append("")
    # crossover: remaining steps where restore_sp2 overtakes degrade (standby none vs sp1)
    L.append("## Crossover: remaining steps beyond which waiting for SP=2 beats degrading now\n")
    for w in rows:
        gap = w["q1_s"] - w["q2_s"]
        if gap <= 0:
            L.append(f"- {w['model']} {w['shape']}: SP1 not slower than SP2 in the inputs; degrade always wins"); continue
        # standby sp1 (degrade has no restart), restore_sp2 pays a cold restart
        x = (w["cold_ready_s"] + (w["decode2_s"] - w["decode1_s"])) / gap
        L.append(f"- {w['model']} {w['shape']}: remaining > **{x:.0f}** steps (standby=sp1 vs cold SP2 restart {w['cold_ready_s']:.0f} s; per-step gap {gap:.2f} s)")
    text = "\n".join(L)
    print(text)
    if md_path:
        md_path.write_text(text + "\n")
        df.to_csv(str(md_path).replace(".md", ".csv"), index=False)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default=str(HERE / "results" / "recovery_inputs.json"))
    ap.add_argument("--md", default=str(HERE / "results" / "recovery_oracle.md"))
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--cold-ready", type=float, default=None, help="override cold restart seconds for every model")
    a = ap.parse_args()
    if a.alpha:
        DEFAULTS["alpha"] = a.alpha
    inp = load_inputs(Path(a.inputs))
    if a.cold_ready:
        for m in inp.get("models", {}).values():
            m.setdefault("restart", {})["ready_s"] = a.cold_ready
        DEFAULTS["cold_ready_s_default"] = a.cold_ready
    return run(inp, Path(a.md) if a.md else None)


if __name__ == "__main__":
    sys.exit(main())
