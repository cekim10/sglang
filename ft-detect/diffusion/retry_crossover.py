"""Ideal warm retry vs in-place continuation: crossover audit from measured inputs only.

Failure while step k of N runs (progress p = k / N; steps 0..k-1 are complete). Times are from the
failure to the finished result; both strategies pay the same detection delay T_d first.

  B1 ideal warm retry : T_d + E + N*q2 + D2 + O          healthy SP=2 replica, already warm, no queue
  ours (in place)     : T_d + S + C + (N-k)*q1 + D1 + O  same process continues at SP=1

  E encode, q2/q1 step time at SP=2/SP=1, D2/D1 VAE decode at SP=2/SP=1, O fixed request overhead
  (HTTP, video encode, polling), S fence+abort+shrink, C one-off cost of the first SP=1 use after
  the switch. S, C, O, D1/D2 and q1/q2 are calibrated on the Test 3 runs (832x480x81, 9 steps, k=4)
  so the model reproduces the measured failing-request latency exactly.

Crossover (T_d cancels): ours wins iff S + C + (N-k)*q1 + D1 < E + N*q2 + D2. Ignoring the fixed
terms this needs p > 1 - q2/q1: a one-GPU continuation cannot beat a two-GPU retry before that.

    python retry_crossover.py [--md results/retry_crossover.md] [--csv results/retry_crossover.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
# Recommended request settings: Wan2.2-TI2V-5B SGLang default sampling (1280x704x121, 50 steps);
# Z-Image-Turbo cookbook (8 steps).
STEPS = {"Wan-AI/Wan2.2-TI2V-5B-Diffusers": 50, "Tongyi-MAI/Z-Image-Turbo": 8}
PROGRESS = (0.2, 0.4, 0.5, 0.6, 0.8, 0.95)


def _f(x):
    return float(x) if x not in (None, "") else None


def calibrate() -> dict:
    rows = list(csv.DictReader(open(RES / "test3_contain.csv")))
    rows = [r for r in rows if r.get("encoder_parallel") == "replicate" and r.get("fail_video_fhw") and r.get("T_detect_s")]
    med = lambda k: sorted(_f(r[k]) for r in rows)[len(rows) // 2]
    N, k = int(_f(rows[0]["steps"])), int(_f(rows[0]["fail_step"]))
    q2, q1 = med("sp2_step_ms_median") / 1e3, med("sp1_step_ms_median") / 1e3
    ref, fail, after = med("ref_latency_s"), med("fail_latency_s"), med("after_latency_s")
    Td, S = med("T_detect_s"), med("T_switch_s")
    inp = json.load(open(RES / "recovery_inputs.json"))["workloads"]["Wan-AI/Wan2.2-TI2V-5B-Diffusers|832x480x81f"]
    E, D2 = inp["encode_ms"] / 1e3, inp["decode_sp2_ms"] / 1e3
    O = ref - N * q2 - E - D2                        # fixed overhead of a normal request
    D1 = after - N * q1 - E - O                      # decode at SP=1, from the next request (all SP=1)
    pre = E + k * q2                                 # work done before the failure
    C = fail - pre - Td - S - (N - k) * q1 - D1 - O  # first-SP=1-use residual inside the failing request
    return {"runs": len(rows), "N": N, "k": k, "q2": q2, "q1": q1, "r_step": q1 / q2, "E": E, "D2": D2, "D1": D1,
            "r_decode": D1 / D2, "O": O, "S": S, "C": C, "Td": Td, "ref": ref, "fail": fail, "after": after}


def workloads() -> list:
    inp = json.load(open(RES / "recovery_inputs.json"))["workloads"]
    out = []
    for key, w in inp.items():
        model, shape = key.split("|")
        if model not in STEPS or not w.get("q_sp2_ms") or w.get("decode_sp2_ms") is None:
            continue
        out.append({"model": model, "shape": shape, "q2": w["q_sp2_ms"] / 1e3, "D2": w["decode_sp2_ms"] / 1e3,
                    "E": (w.get("encode_ms") or 0) / 1e3, "q1_measured": (w.get("q_sp1_ms") or 0) / 1e3 or None,
                    "source": "measured"})
    # SGLang's default Wan request is 121 frames; only 17/49/81 were measured at 704p. Step time is
    # extrapolated with the 49->81 scaling exponent over latent frames, decode linearly.
    a = next((w for w in out if w["shape"] == "1280x704x49f"), None); b = next((w for w in out if w["shape"] == "1280x704x81f"), None)
    if a and b:
        lat = lambda f: (f - 1) // 4 + 1
        expo = math.log(b["q2"] / a["q2"]) / math.log(lat(81) / lat(49))
        out.append({"model": b["model"], "shape": "1280x704x121f", "q2": b["q2"] * (lat(121) / lat(81)) ** expo,
                    "D2": b["D2"] * lat(121) / lat(81), "E": b["E"], "q1_measured": None,
                    "source": f"extrapolated (step ~ latent_frames^{expo:.2f}, decode linear)"})
    return out


def times(w, cal, N, k, r_step, r_decode, C):
    q2, D2, E, O, S, Td = w["q2"], w["D2"], w["E"], cal["O"], cal["S"], cal["Td"]
    q1 = w["q1_measured"] if (w["q1_measured"] and r_step is None) else q2 * (r_step or cal["r_step"])
    D1 = D2 * r_decode
    retry = Td + E + N * q2 + D2 + O
    ours = Td + S + C + (N - k) * q1 + D1 + O
    return retry, ours, q1, D1


def p_star(w, cal, N, r_step, r_decode, C):
    lo = None
    for k in range(N):
        retry, ours, _, _ = times(w, cal, N, k, r_step, r_decode, C)
        if ours < retry:
            lo = k / N; break
    return lo


SCEN = {
    "measured": dict(r_step=None, r_decode=None, C=None),           # Test 3 ratios and residual
    "pessimistic": dict(r_step=2.0, r_decode=2.0, C=None),          # perfect SP=2 scaling: SP=1 exactly 2x slower
    "prewarmed": dict(r_step=None, r_decode=None, C=0.0),           # first SP=1 use made free (warm SP=1 shapes)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(RES / "retry_crossover.md"))
    ap.add_argument("--csv", default=str(RES / "retry_crossover.csv"))
    a = ap.parse_args()
    cal = calibrate()
    ws = workloads()
    rows = []
    for w in ws:
        N = STEPS[w["model"]]
        for sname, sc in SCEN.items():
            r_decode = sc["r_decode"] or cal["r_decode"]
            C = cal["C"] if sc["C"] is None else sc["C"]
            row = {"model": w["model"].split("/")[1], "shape": w["shape"], "steps": N, "scenario": sname, "source": w["source"]}
            r0, _, q1, D1 = times(w, cal, N, 0, sc["r_step"], r_decode, C)
            row.update({"q2_s": w["q2"], "q1_s": q1, "D2_s": w["D2"], "D1_s": D1, "full_request_s": r0 - cal["Td"],
                        "retry_after_failure_s": r0})
            ps = p_star(w, cal, N, sc["r_step"], r_decode, C)
            row["p_star"] = ps
            row["p_bound"] = 1 - w["q2"] / q1   # no fixed costs: one GPU cannot beat a two-GPU retry earlier
            tr, to = zip(*[times(w, cal, N, k, sc["r_step"], r_decode, C)[:2] for k in range(N)])
            row["mean_retry_over_ours"] = sum(tr) / sum(to)                     # failure uniform over steps
            row["mean_retry_over_best"] = sum(tr) / sum(min(x, y) for x, y in zip(tr, to))   # pick the faster per k
            for p in PROGRESS:
                k = min(N - 1, int(round(p * N)))
                retry, ours, _, _ = times(w, cal, N, k, sc["r_step"], r_decode, C)
                row[f"x{int(p * 100)}"] = retry / ours
            rows.append(row)
    # verdict on the user's pre-agreed gates, measured scenario, video workloads at recommended steps
    vid = [r for r in rows if r["scenario"] == "measured" and "f" in r["shape"].split("x")[-1]]
    good = [r for r in vid if r["p_star"] is not None and r["p_star"] <= 0.5 and r["x80"] >= 1.5]
    late = [r for r in vid if r["p_star"] is not None and r["p_star"] > 0.5]
    never = [r for r in vid if r["p_star"] is None]
    md = ["# Ideal warm retry vs in-place SP=2 -> SP=1 continuation (crossover audit)", "",
          "Cells: T_retry / T_ours, both measured from the failure (detection included; >1 means ours finishes first). "
          "p* = earliest progress k/N at which ours wins. Steps: Wan2.2-TI2V-5B 50 (SGLang default sampling), "
          "Z-Image-Turbo 8 (cookbook). Test 3 itself ran 9 steps at 832x480x81, which nobody deploys.", "",
          "## Calibration on Test 3 (832x480x81, 9 steps, failure at step 4)", "",
          "| term | value |", "|---|---|"]
    for key, label in (("runs", "runs"), ("q2", "step at SP=2 (s)"), ("q1", "step at SP=1 (s)"), ("r_step", "SP=1 / SP=2 step"),
                       ("D2", "decode at SP=2 (s)"), ("D1", "decode at SP=1 (s, from the next request)"), ("r_decode", "SP=1 / SP=2 decode"),
                       ("E", "text encode (s)"), ("O", "fixed request overhead (s)"), ("Td", "detection (s)"), ("S", "switch (s)"),
                       ("C", "first-SP=1-use residual in the failing request (s)")):
        v = cal[key]; md.append(f"| {label} | {v:.2f} |" if isinstance(v, float) else f"| {label} | {v} |")
    md += ["", f"Check: Test 3 measured failing request {cal['fail']:.1f} s; B1 retry would have taken "
           f"{cal['Td'] + cal['ref']:.1f} s from the failure plus the {cal['E'] + cal['k'] * cal['q2']:.1f} s already spent, "
           f"so at this setting retry wins.", ""]
    for sname, desc in (("measured", "SP=1 slowdown and first-use residual as measured in Test 3"),
                        ("pessimistic", "SP=1 exactly 2x slower than SP=2 for steps and decode (perfect SP scaling)"),
                        ("prewarmed", "as measured, but the first-SP=1-use residual removed (SP=1 shapes warmed in advance)")):
        md += [f"## Scenario: {sname} ({desc})", "",
               "| workload | steps | full request (s) | p* | bound 1-q2/q1 | " + " | ".join(f"{int(p * 100)}%" for p in PROGRESS)
               + " | uniform: retry/ours | uniform: retry/best-of-both | source |",
               "|---|---|---|---|---|" + "---|" * len(PROGRESS) + "---|---|---|"]
        for r in [r for r in rows if r["scenario"] == sname]:
            ps = "never" if r["p_star"] is None else f"{r['p_star']:.2f}"
            md.append(f"| {r['model']} {r['shape']} | {r['steps']} | {r['full_request_s']:.1f} | {ps} | {r['p_bound']:.2f} | " +
                      " | ".join(f"{r[f'x{int(p * 100)}']:.2f}" for p in PROGRESS) +
                      f" | {r['mean_retry_over_ours']:.2f} | {r['mean_retry_over_best']:.2f} | {r['source']} |")
        md.append("")
    md += ["## Gate (pre-agreed): measured scenario, video workloads at recommended steps", "",
           f"- p* <= 0.5 and >= 1.5x at 80% progress: {', '.join(r['shape'] for r in good) or 'none'}",
           f"- crossover only after 50%: {', '.join(f'{r['shape']} (p*={r['p_star']:.2f})' for r in late) or 'none'}",
           f"- ours never wins: {', '.join(r['shape'] for r in never) or 'none'}", ""]
    Path(a.md).write_text("\n".join(md))
    with open(a.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
