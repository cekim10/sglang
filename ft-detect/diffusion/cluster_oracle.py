"""Cluster-level oracle: what does one GPU failure cost a diffusion serving cluster under each recovery policy?

Discrete-event simulation of R replicas x N GPUs (Ulysses SP=N), FCFS dispatch to any free replica,
one request per replica at a time, Poisson arrivals at utilisation rho of the failure-free capacity.
One GPU fails at t_f on a random replica. Every policy gets the same detection delay T_d (5 s, the
Test 3 deadline); stock SGLang's 600 s non-detection is shown only as context.

  P1 retry+restart     in-flight request re-queued at the front from scratch; replica down for T_restart
  P2 migrate+restart   in-flight trajectory (k steps done, conditioning kept) re-queued at the front and
                       resumed on whichever replica takes it (+0.1 s transfer); replica down for T_restart.
                       The strongest baseline: progress kept, healthy replicas absorb the request.
  P3 ours              same process continues at SP=N-1 (switch S + first-use C), no downtime
  P3i ours-idle        as P3 for hard failures; for transient faults re-expands only when idle

Failure kinds. hard: the GPU stays out for T_repair (default 4 h, beyond the window), so P1/P2 restart
the survivors at SP=N-1. transient: the GPU is usable after the processes are killed, so P1/P2
restart at SP=N, P3 restarts to SP=N after the in-flight request, P3i at the first idle moment.
After every restart the first request pays the measured first-of-shape warm-up W_first.

Inputs. Step, decode and encode times per workload at SP=2 (characterisation), the SP=1/SP=2 ratios,
switch S, first-use C, overhead O and detection from the Test 3 calibration (retry_crossover.py).
SP=n step time from an Amdahl fit to the measured SP=1/SP=2 ratio (or perfect scaling, --scaling).
SP=N-1 must divide Wan's 24 heads (SP=3 does). Restart time is swept: 40 s is the measured 5B value.

Outputs per (N, rho, T_restart, kind, policy), mean over seeds with common random numbers against a
no-failure run: extra SLO violations per failure, extra request-seconds, healthy-GPU-seconds lost.
Fleet view: failure-attributable violation fraction = violations per event x events per request,
for per-GPU failure rates swept around the order of magnitude Meta reports for Llama 3 pre-training
(~419 unexpected interruptions in 54 days on 16K H100, ~4.7e-4 per GPU-day; approximate).

    python cluster_oracle.py [--seeds 40] [--md results/cluster_oracle.md] [--csv results/cluster_oracle.csv]
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retry_crossover import STEPS, calibrate, workloads  # noqa: E402

RES = HERE / "results"
MIXES = {
    "video": ["832x480x81f", "1280x704x49f", "1280x704x81f"],
    "long-video": ["1280x704x81f", "1280x704x121f"],
}
RHOS = (0.3, 0.5, 0.7, 0.9)
RESTARTS = (40.0, 120.0, 300.0, 900.0)
KINDS = ("hard", "transient")
POLICIES = ("P1", "P2", "P3", "P3i")
GPU_FAIL_PER_DAY = (4.7e-4, 4.7e-3, 4.7e-2)
SLO_MULT = 2.0
WARMUP_S, WINDOW_S = 1800.0, 3600.0


class Model:
    """Service times as a function of SP degree, from measured SP=2 values and the Test 3 ratios."""

    def __init__(self, cal, mix, scaling):
        ws = {w["shape"]: w for w in workloads() if w["model"].startswith("Wan")}
        self.ws = [ws[s] for s in MIXES[mix]]
        self.src = {s: ws[s]["source"] for s in MIXES[mix]}
        self.steps = STEPS["Wan-AI/Wan2.2-TI2V-5B-Diffusers"]
        self.E_O = cal["O"]
        # serial fraction f with t(n) = t1 * (f + (1 - f) / n), fitted so t(2) / t(1) matches Test 3
        self.f_step = 0.0 if scaling == "perfect" else 2 / cal["r_step"] - 1
        self.f_dec = 0.0 if scaling == "perfect" else 2 / cal["r_decode"] - 1
        self.r_step, self.r_dec = cal["r_step"], cal["r_decode"]
        self.S, self.C, self.Td = cal["S"], cal["C"], cal["Td"]
        self.W_first = 14.1   # measured: first request of the shape 29.2 s vs warm 15.1 s (Test 3 runs)

    def q(self, w, n):
        q1 = w["q2"] * self.r_step
        return q1 * (self.f_step + (1 - self.f_step) / n)

    def D(self, w, n):
        d1 = w["D2"] * self.r_dec
        return d1 * (self.f_dec + (1 - self.f_dec) / n)

    def service(self, w, n, steps_done=0, encoded=False):
        return (0 if encoded else w["E"]) + (self.steps - steps_done) * self.q(w, n) + self.D(w, n) + self.E_O


class Sim:
    def __init__(self, m, R, N, rho, seed, fail=None):
        self.m, self.R, self.N = m, R, N
        rng = random.Random(seed)
        mean_ref = sum(m.service(w, N) for w in m.ws) / len(m.ws)
        lam = rho * R / mean_ref
        self.reqs, t = [], 0.0
        t_end = WARMUP_S + WINDOW_S
        while True:
            t += rng.expovariate(lam)
            if t > t_end:
                break
            w = m.ws[rng.randrange(len(m.ws))]
            self.reqs.append({"t": t, "w": w, "ref": m.service(w, N), "done": 0, "enc": False, "fin": None})
        self.t_f = WARMUP_S + rng.random() * 60.0
        self.fail_rep = rng.randrange(R)
        self.fail = fail      # None or (policy, kind, T_restart, T_d)
        self.lost_gpu_s = 0.0

    def run(self):
        m, N = self.m, self.N
        reps = [{"n": N, "up": True, "busy": None, "start": 0.0, "extra": 0.0, "tok": 0, "first": False,
                 "expand": False} for _ in range(self.R)]
        q = deque()
        ev = []
        seq = 0

        def push(t, kind, *a):
            nonlocal seq
            seq += 1
            heapq.heappush(ev, (t, seq, kind, a))

        def start(i, r, t, extra=0.0):
            rep = reps[i]
            rep["busy"], rep["start"], rep["tok"] = r, t, rep["tok"] + 1
            ex = extra + (m.W_first if rep["first"] else 0.0)
            if rep["first"]:
                self.lost_gpu_s += rep["n"] * m.W_first
            rep["first"] = False
            rep["extra"] = ex
            push(t + ex + m.service(r["w"], rep["n"], r["done"], r["enc"]), "done", i, rep["tok"])

        def dispatch(t):
            for i, rep in enumerate(reps):
                if not q:
                    return
                if rep["up"] and rep["busy"] is None:
                    if rep["expand"] and self.fail[0] == "P3":
                        go_down(i, t, back_n=N)
                        continue
                    r = q.popleft()
                    start(i, r, t, extra=r.pop("resume_extra", 0.0))

        def go_down(i, t, back_n):
            rep = reps[i]
            rep["up"], rep["expand"] = False, False
            self.lost_gpu_s += rep["n"] * self.fail[2]   # the healthy GPUs of this replica sit out the restart
            push(t + self.fail[2], "up", i, back_n)

        for r in self.reqs:
            push(r["t"], "arr", r)
        if self.fail:
            push(self.t_f, "fail", self.fail_rep)
        while ev:
            t, _, kind, a = heapq.heappop(ev)
            if kind == "arr":
                q.append(a[0]); dispatch(t)
            elif kind == "done":
                i, tok = a
                rep = reps[i]
                if tok != rep["tok"] or rep["busy"] is None:
                    continue
                rep["busy"]["fin"] = t
                rep["busy"] = None
                if rep["expand"] and (self.fail[0] == "P3" or not q):
                    go_down(i, t, back_n=N)
                dispatch(t)
            elif kind == "up":
                i, back_n = a
                reps[i].update({"up": True, "n": back_n, "first": True})
                dispatch(t)
            elif kind == "fail":
                self._fail(reps, a[0], t, q, push, start)
                dispatch(t)
            elif kind == "detect":
                self._detect(reps, a[0], a[1], t, q, push, start, go_down)
                dispatch(t)
        return self.reqs

    def _fail(self, reps, i, t, q, push, start):
        rep = reps[i]
        rep["tok"] += 1          # the in-flight completion never happens
        rep["up"] = False        # stalled until detection: takes no new work
        r = rep["busy"]
        k = 0
        if r is not None:
            m, w = self.m, r["w"]
            elapsed = t - rep["start"] - rep["extra"] - (0 if r["enc"] else w["E"])
            k = r["done"] + max(0, min(m.steps - r["done"], int(elapsed // m.q(w, rep["n"])))) if elapsed > 0 else r["done"]
        self.lost_gpu_s += rep["n"] * self.fail[3]
        push(t + self.fail[3], "detect", i, (r, k))

    def _detect(self, reps, i, rk, t, q, push, start, go_down):
        policy, kind, T_restart, _ = self.fail
        m, N, rep = self.m, self.N, reps[i]
        r, k = rk
        rep["busy"] = None
        if policy in ("P1", "P2", "stock"):
            if r is not None:
                if policy == "P2":
                    r["done"], r["enc"], r["resume_extra"] = k, True, 0.1
                else:
                    self.lost_gpu_s += rep["n"] * (k * m.q(r["w"], rep["n"]))     # discarded work
                    r["done"], r["enc"] = 0, False
                q.appendleft(r)
            back = N if kind == "transient" else N - 1
            rep["n"] = N - 1 if kind == "hard" else N
            rep["up"] = True
            go_down(i, t, back_n=back)
            rep["n"] = back
        else:   # P3 / P3i: continue in place at SP=N-1
            rep["n"], rep["up"] = N - 1, True
            self.lost_gpu_s += (N - 1) * (m.S + m.C)
            if kind == "transient":
                rep["expand"] = True
            if r is not None:
                r["done"], r["enc"] = k, True
                start(i, r, t, extra=m.S + m.C)
            elif kind == "transient":
                go_down(i, t, back_n=N)   # idle right now: restart to SP=N immediately


def one(args):
    mix, scaling, R, N, rho, seed, fail = args
    cal = calibrate()
    m = Model(cal, mix, scaling)
    s = Sim(m, R, N, rho, seed, fail)
    reqs = s.run()
    viol = sum(1 for r in reqs if r["fin"] - r["t"] > SLO_MULT * r["ref"])
    lat = sum(r["fin"] - r["t"] for r in reqs)
    return {"n": len(reqs), "viol": viol, "lat": lat, "lost_gpu_s": s.lost_gpu_s,
            "busy_fail": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--mix", default="video", choices=sorted(MIXES))
    ap.add_argument("--scaling", default="amdahl", choices=["amdahl", "perfect"])
    ap.add_argument("--configs", default="8x4,16x2", help="RxN list, same 32 GPUs by default")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--md", default=str(RES / "cluster_oracle.md"))
    ap.add_argument("--csv", default=str(RES / "cluster_oracle.csv"))
    a = ap.parse_args()
    cal = calibrate()
    m = Model(cal, a.mix, a.scaling)
    configs = [tuple(int(x) for x in c.split("x")) for c in a.configs.split(",")]
    jobs, keys = [], []
    for R, N in configs:
        for rho in RHOS:
            for sd in range(a.seeds):
                jobs.append((a.mix, a.scaling, R, N, rho, sd, None)); keys.append((R, N, rho, None, None, "nofail", sd))
                jobs.append((a.mix, a.scaling, R, N, rho, sd, ("stock", "hard", 40.0, 600.0))); keys.append((R, N, rho, 40.0, "hard", "stock", sd))
                for T in RESTARTS:
                    for kind in KINDS:
                        for p in POLICIES:
                            jobs.append((a.mix, a.scaling, R, N, rho, sd, (p, kind, T, cal["Td"])))
                            keys.append((R, N, rho, T, kind, p, sd))
    with Pool(a.procs) as pool:
        outs = pool.map(one, jobs, chunksize=16)
    base = {(k[0], k[1], k[2], k[6]): o for k, o in zip(keys, outs) if k[5] == "nofail"}
    agg = {}
    for k, o in zip(keys, outs):
        if k[5] == "nofail":
            continue
        b = base[(k[0], k[1], k[2], k[6])]
        d = agg.setdefault(k[:6], {"viol": [], "lat": [], "gpu": [], "n": []})
        d["viol"].append(o["viol"] - b["viol"]); d["lat"].append(o["lat"] - b["lat"]); d["gpu"].append(o["lost_gpu_s"]); d["n"].append(b["n"])
    mean = lambda xs: sum(xs) / len(xs)
    rows = []
    for (R, N, rho, T, kind, p), d in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0, kv[0][4] or "", kv[0][5])):
        lam_h = mean(d["n"]) / ((WARMUP_S + WINDOW_S) / 3600)            # requests per hour, whole cluster
        sd = (sum((x - mean(d["viol"])) ** 2 for x in d["viol"]) / max(1, len(d["viol"]) - 1)) ** 0.5
        row = {"R": R, "N": N, "rho": rho, "T_restart_s": T, "kind": kind, "policy": p,
               "extra_viol_per_event": mean(d["viol"]), "se_viol": sd / len(d["viol"]) ** 0.5, "extra_req_s_per_event": mean(d["lat"]),
               "healthy_gpu_s_lost": mean(d["gpu"]), "req_per_hour": lam_h}
        for g in GPU_FAIL_PER_DAY:
            ev_h = R * N * g / 24
            row[f"fleet_viol_frac@{g:g}"] = row["extra_viol_per_event"] * ev_h / lam_h
        rows.append(row)
    write(rows, a, m, cal)
    return 0


def write(rows, a, m, cal):
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    get = {(r["R"], r["N"], r["rho"], r["T_restart_s"], r["kind"], r["policy"]): r for r in rows}
    md = ["# Cluster oracle: one GPU failure under restart, migration and in-place degradation", "",
          f"Mix `{a.mix}` ({', '.join(f'{s} [{src}]' for s, src in m.src.items())}), {m.steps} steps, SP scaling `{a.scaling}`, "
          f"{a.seeds} seeds per cell, SLO = {SLO_MULT:g} x failure-free service time, detection {cal['Td']:.1f} s for every policy, "
          f"switch {m.S:.2f} s + first use {m.C:.2f} s for ours, first-request warm-up {m.W_first:.1f} s after every restart.", "",
          "Cells: extra SLO violations per failure event versus the same arrivals without a failure (mean +- standard error over seeds; "
          "lower is better). Healthy GPU-seconds lost: survivors sitting out a restart or detection, discarded work, warm-up / switch.", ""]
    configs = sorted({(r["R"], r["N"]) for r in rows})
    for R, N in configs:
        for kind in KINDS:
            md += [f"## {R} replicas x SP={N}, {kind} failure", "",
                   "| rho | T_restart | P1 retry+restart | P2 migrate+restart | P3 ours | P3i ours-idle | ours / best baseline | healthy GPU-s lost: best baseline / ours |",
                   "|---|---|---|---|---|---|---|---|"]
            for rho in RHOS:
                for T in RESTARTS:
                    rr = {p: get[(R, N, rho, T, kind, p)] for p in POLICIES}
                    c = {p: rr[p]["extra_viol_per_event"] for p in POLICIES}
                    se = {p: rr[p]["se_viol"] for p in POLICIES}
                    bbp = min(("P1", "P2"), key=lambda p: c[p]); oop = min(("P3", "P3i"), key=lambda p: c[p])
                    best_b, ours = c[bbp], c[oop]
                    ratio = "n/a" if best_b <= 0.05 else f"{ours / best_b:.2f}"
                    cell = lambda p: f"{c[p]:.2f} +- {se[p]:.2f}"
                    md.append(f"| {rho} | {T:.0f} s | {cell('P1')} | {cell('P2')} | {cell('P3')} | {cell('P3i')} | {ratio} | "
                              f"{rr[bbp]['healthy_gpu_s_lost']:.0f} / {rr[oop]['healthy_gpu_s_lost']:.0f} |")
            md.append("")
        st = [get[(R, N, rho, 40.0, "hard", "stock")]["extra_viol_per_event"] for rho in RHOS]
        md += [f"Context, {R} x SP={N}: stock-like 600 s detection then retry+restart(40 s): extra violations per event "
               + ", ".join(f"rho {rho}: {v:.2f}" for rho, v in zip(RHOS, st)) + " (optimistic: real stock never evicts).", ""]
    # fleet view at the measured restart time and a slow one
    md += ["## Fleet view: fraction of all requests that violate the SLO because of failures", "",
           "Per-GPU failure rate per day: 4.7e-4 (order of magnitude of Llama 3 pre-training interruptions), x10, x100 "
           "(frequent transient hangs).", "",
           "| config | kind | rho | T_restart | best baseline @4.7e-4 | ours @4.7e-4 | best baseline @4.7e-2 | ours @4.7e-2 |",
           "|---|---|---|---|---|---|---|---|"]
    for R, N in configs:
        for kind in KINDS:
            for rho in (0.5, 0.7, 0.9):
                for T in (40.0, 900.0):
                    rs = {p: get[(R, N, rho, T, kind, p)] for p in POLICIES}
                    bb = min(("P1", "P2"), key=lambda p: rs[p]["extra_viol_per_event"]); oo = min(("P3", "P3i"), key=lambda p: rs[p]["extra_viol_per_event"])
                    md.append(f"| {R}x{N} | {kind} | {rho} | {T:.0f} s | {rs[bb]['fleet_viol_frac@0.00047']:.1e} | {rs[oo]['fleet_viol_frac@0.00047']:.1e} | "
                              f"{rs[bb]['fleet_viol_frac@0.047']:.1e} | {rs[oo]['fleet_viol_frac@0.047']:.1e} |")
    md.append("")
    # pre-registered gates
    cells = [(R, N, rho, T, kind) for R, N in configs if N >= 3 for kind in KINDS for rho in (0.5, 0.7, 0.9) for T in RESTARTS]
    better2, close = 0, 0
    for R, N, rho, T, kind in cells:
        c = {p: get[(R, N, rho, T, kind, p)]["extra_viol_per_event"] for p in POLICIES}
        bb, oo = min(c["P1"], c["P2"]), min(c["P3"], c["P3i"])
        if bb > 0.05 and oo <= 0.5 * bb:
            better2 += 1
        if bb <= 0.05 or oo >= 0.8 * bb:
            close += 1
    n = len(cells)
    hw = [get[(R, N, rho, 40.0, kind, p)]["fleet_viol_frac@0.00047"] for R, N in configs if N >= 3
          for kind in KINDS for rho in (0.5, 0.7, 0.9) for p in ("P1", "P2")]
    verdict = "KILL" if close >= 0.8 * n else "GO" if better2 >= 0.5 * n else "GRAY"
    md += ["## Gate (fixed before the run; SP>=3 configs, rho 0.5-0.9, every restart time, both kinds)", "",
           f"- cells where ours cuts the best baseline's extra violations by >= 2x: {better2}/{n}",
           f"- cells where ours is within 20% of the best baseline (or the baseline loses < 0.05 requests): {close}/{n}",
           f"- verdict: **{verdict}** (KILL if >= 80% of cells are close; GO if >= 50% are >= 2x better; else GRAY)",
           f"- significance at hardware failure rates (4.7e-4/GPU-day, measured 40 s restart): best-baseline failure-attributable "
           f"violation fraction up to {max(hw):.1e} of requests (operationally negligible if well below a 1e-3 SLO budget)", ""]
    Path(a.md).write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
