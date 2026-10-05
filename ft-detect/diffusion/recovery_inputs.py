"""Collect the measured inputs of the recovery cost model into results/recovery_inputs.json.

Sources (all produced earlier by this harness):
  results/char_*.csv, *_stages.csv      per-workload warm step time and VAE-decode time at the SP degree they ran
  runs/traj_*_sp1_ref_*/logs/steps_rank0.jsonl  SP=1 step and decode times for the portability workloads
  runs/traj_*_sp2_save_*/logs/steps_rank0.jsonl SP=2 step and decode times for the same workloads
  results/inventory_*.csv               per-step dynamic state size and pinned-copy cost
  results/startup_decomp.md             rank restart decomposition (spawn+import, load, warmup, ready)
  results/containment_B.csv             process exit / teardown time
  results/gpu_bench_idle.json, runs/*/external_bench_during_hang.json   GPU memory headroom

    python recovery_inputs.py            # -> results/recovery_inputs.json (+ prints the table)
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read_jsonl  # noqa: E402

RES = HERE / "results"
RUNS = HERE / "runs"


def _shape(r):
    nf = r.get("nf") or 1
    return f"{r.get('w')}x{r.get('h')}" + (f"x{int(nf)}f" if nf and int(nf) > 1 else "")


def steps_and_decode(run_dir: Path):
    """median warm step (skip the first step of each request) and median decode per shape in one run."""
    out = {}
    recs = read_jsonl(run_dir / "logs" / "steps_rank0.jsonl")
    by_req = {}
    for r in recs:
        if r.get("ev") == "ds":
            by_req.setdefault((r.get("rid"), _shape(r)), []).append((r["t_ns"] - r["t0_ns"]) / 1e6)
    for (rid, sh), durs in by_req.items():
        warm = durs[1:] or durs
        out.setdefault(sh, {"step_ms": [], "decode_ms": [], "n_steps": []})
        out[sh]["step_ms"].extend(warm)
        out[sh]["n_steps"].append(len(durs))
    for r in recs:
        if r.get("ev") == "stage" and "decod" in str(r.get("name", "")).lower():
            sh = _shape(r)
            out.setdefault(sh, {"step_ms": [], "decode_ms": [], "n_steps": []})
            out[sh]["decode_ms"].append((r["t_ns"] - r["t0_ns"]) / 1e6)
    med = lambda xs: float(pd.Series(xs).median()) if xs else None
    return {sh: {"step_ms": med(v["step_ms"]), "decode_ms": med(v["decode_ms"]), "n_steps": med(v["n_steps"])} for sh, v in out.items()}


def main():
    inputs = {"workloads": {}, "models": {}, "notes": []}
    # 1) SP=2 quanta from characterization CSVs (+ stage CSVs for decode)
    for csv in glob.glob(str(RES / "char_*.csv")):
        if csv.endswith("_stages.csv"):
            continue
        df = pd.read_csv(csv)
        st = Path(csv.replace(".csv", "_stages.csv"))
        sdf = pd.read_csv(st) if st.exists() else pd.DataFrame()
        for _, r in df.iterrows():
            key = f"{r['model']}|{r['shape']}"
            w = inputs["workloads"].setdefault(key, {"model": r["model"], "shape": r["shape"], "pixels_frames": float(r["pixels_frames"])})
            w["q_sp2_ms"] = float(r.get("warm_q_ms") if pd.notna(r.get("warm_q_ms")) else r["step_p50_ms"])
            w["step_p50_sp2_ms"] = float(r["step_p50_ms"])
            if not sdf.empty:
                d = sdf[(sdf["model"] == r["model"]) & (sdf["shape"] == r["shape"]) & (sdf["stage"].astype(str).str.contains("decod", case=False))]
                if not d.empty:
                    w["decode_sp2_ms"] = float(d["p50_ms"].iloc[0])
                e = sdf[(sdf["model"] == r["model"]) & (sdf["shape"] == r["shape"]) & (sdf["stage"].astype(str).str.contains("TextEncod", case=False))]
                if not e.empty:
                    w["encode_ms"] = float(e["p50_ms"].iloc[0])
    # 2) SP=1 and SP=2 from portability runs
    for d in sorted(RUNS.glob("traj_*")):
        m = re.match(r"traj_(.+)_(sp1_ref|sp1_ref2|sp2_save)_\d{8}-\d{6}$", d.name)
        if not m:
            continue
        tag, cfg = m.group(1), m.group(2)
        model = None
        try:
            model = json.loads((d / "launch_env.json").read_text()).get("model")
        except Exception:
            pass
        for sh, v in steps_and_decode(d).items():
            key = f"{model}|{sh}"
            w = inputs["workloads"].setdefault(key, {"model": model, "shape": sh})
            sp = "sp1" if cfg.startswith("sp1") else "sp2"
            if v["step_ms"]:
                w.setdefault(f"q_{sp}_ms_samples", []).append(v["step_ms"])
                w[f"q_{sp}_ms"] = float(pd.Series(w[f"q_{sp}_ms_samples"]).median())
            if v["decode_ms"]:
                w.setdefault(f"decode_{sp}_ms_samples", []).append(v["decode_ms"])
                w[f"decode_{sp}_ms"] = float(pd.Series(w[f"decode_{sp}_ms_samples"]).median())
    # 3) per-model: SP1/SP2 ratio (from workloads that have both), restart decomposition, state sizes
    ratios = {}
    for key, w in inputs["workloads"].items():
        if w.get("q_sp1_ms") and w.get("q_sp2_ms"):
            ratios.setdefault(w["model"], []).append(w["q_sp1_ms"] / w["q_sp2_ms"])
        if w.get("decode_sp1_ms") and w.get("decode_sp2_ms"):
            ratios.setdefault(w["model"] + "|decode", []).append(w["decode_sp1_ms"] / w["decode_sp2_ms"])
    sd = RES / "startup_decomp.md"
    restart = {}
    if sd.exists():
        for line in sd.read_text().splitlines():
            if not line.startswith("| ") or "T_spawn" in line or line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 8 or not cells[1]:
                continue
            run, model = cells[0], cells[1]
            try:
                spawn, load, warm, ready = float(cells[2]), float(cells[5]), float(cells[6]) if cells[6] else None, float(cells[7]) if cells[7] else None
            except ValueError:
                continue
            if ready is None:
                continue
            r = restart.setdefault(model, {"spawn_import_s": [], "load_s": [], "warmup_s": [], "ready_s": []})
            r["spawn_import_s"].append(spawn); r["load_s"].append(load); r["ready_s"].append(ready)
            if warm is not None:
                r["warmup_s"].append(warm)
    for model, r in restart.items():
        inputs["models"].setdefault(model, {})["restart"] = {k: float(pd.Series(v).median()) for k, v in r.items() if v}
    for model, rs in ratios.items():
        if "|decode" in model:
            inputs["models"].setdefault(model.split("|")[0], {})["decode_sp1_over_sp2"] = float(pd.Series(rs).median())
        else:
            inputs["models"].setdefault(model, {})["q_sp1_over_sp2"] = float(pd.Series(rs).median())
    for csv in glob.glob(str(RES / "inventory_*.csv")):
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            key = f"{r['model']}|{r['shape']}"
            w = inputs["workloads"].setdefault(key, {"model": r["model"], "shape": r["shape"]})
            w["state_MiB"] = float(r["S_state_MiB"]); w["t_pin_ms"] = float(r["T_pin_ms"])
            w["dynamic_state_MiB"] = float(r["latents_MiB"]) + float(r["solver_history_MiB"])
    cb = RES / "containment_B.csv"
    if cb.exists():
        df = pd.read_csv(cb)
        ex = df[df["abort_mode"] == "exit"]
        if not ex.empty:
            inputs["fence_exit_s"] = {"min": float((ex["rank_dead_at"] - ex["t_deadline_miss"]).min()),
                                      "max": float((ex["rank_dead_at"] - ex["t_deadline_miss"]).max())}
    gb = RES / "gpu_bench_idle.json"
    if gb.exists():
        j = json.loads(gb.read_text())
        inputs["gpu_total_MiB"] = j["runs"][0]["total_MiB"]
    for f in glob.glob(str(RUNS / "*" / "external_bench_during_hang.json")):
        j = json.loads(Path(f).read_text())
        inputs["gpu_free_during_hang_MiB"] = j["runs"][0]["free_MiB"]
        inputs["matmul_slowdown_during_hang"] = None
    inputs["notes"].append("q_sp1 for workloads without an SP=1 run is extrapolated with the model's median SP1/SP2 ratio (flagged in the model output).")
    for w in inputs["workloads"].values():
        for k in list(w):
            if k.endswith("_samples"):
                del w[k]
    out = RES / "recovery_inputs.json"
    out.write_text(json.dumps(inputs, indent=1))
    print(json.dumps({k: v for k, v in inputs.items() if k != "workloads"}, indent=1))
    print(pd.DataFrame(inputs["workloads"].values()).to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
