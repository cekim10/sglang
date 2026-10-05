"""Test 3 report: recovery timeline of the in-process SP2 -> SP1 failover and the output check.

Per run directory from run_contain.sh (MODE=ours):
  T_detect   fail_inject (rank 1 freezes, steps_rank1.jsonl) -> contain/deadline_miss (rank 0)
  T_abort    _abort_process_group() wall time
  T_switch   deadline_miss -> failed_over (abort + coordinator shrink + peer kill)
  T_recompute re-run of the failed step at SP=1
  T_added    latency of the failing request minus the SP=2 reference request in the same server
  after      latency of the next request, served by the surviving process at SP=1
  output     final latents: rel error ||ours - ref|| / ||ref|| against the SP=2 reference (same
             server, request 1) and the SP=1 reference (--sp1ref run); control = SP=1 vs SP=2 refs

    python contain_report.py runs/contain_ours_* [--sp1ref 'runs/contain_sp1ref_*'] [--md out.md] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read_jsonl  # noqa: E402


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _rel_err(a_path: Path, b_path: Path):
    try:
        import torch

        a = torch.load(a_path, map_location="cpu", weights_only=False)["latents"].float()
        b = torch.load(b_path, map_location="cpu", weights_only=False)["latents"].float()
        if a.shape != b.shape:
            return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
        return float(((a - b).norm() / b.norm()).item())
    except Exception as e:
        return f"n/a ({type(e).__name__})"


def analyze(run: Path, sp1ref: Path | None) -> dict:
    env = _load(run / "contain_env.json")
    r0 = read_jsonl(run / "logs" / "steps_rank0.jsonl")
    r1 = read_jsonl(run / "logs" / "steps_rank1.jsonl")
    inj = next((r for r in r1 if r.get("ev") == "fail_inject"), None)
    c = {r["phase"]: r for r in r0 if r.get("ev") == "contain"}
    miss, fo, rec = c.get("deadline_miss"), c.get("failed_over"), c.get("step_recomputed")
    row = {"run": run.name, "mode": env.get("mode"), "shape": env.get("shape"), "steps": env.get("steps"),
           "fail_step": env.get("fail_step"), "deadline_s": env.get("deadline_s"), "injected": inj is not None,
           "contained": miss is not None, "failed_over": fo is not None, "recomputed": rec is not None}
    if inj and miss:
        row["T_detect_s"] = (miss["t_ns"] - inj["t_ns"]) / 1e9
    if fo:
        row["T_abort_s"] = fo.get("t_abort_s"); row["T_reconfigure_s"] = fo.get("t_reconfigure_s")
        row["coordinators_shrunk"] = ",".join(fo.get("coordinators_shrunk", [])); row["peers_killed"] = fo.get("peers_killed")
        row["modules_sp_size_reset"] = fo.get("modules_sp_size_reset")
        if miss:
            row["T_switch_s"] = (fo["t_ns"] - miss["t_ns"]) / 1e9
    if rec:
        row["T_recompute_s"] = (rec["t_ns"] - rec["t0_ns"]) / 1e9
        # steps after the switch, at SP=1, in the same request
        later = [r for r in r0 if r.get("ev") == "ds" and r["t0_ns"] >= rec["t0_ns"]]
        if later:
            n_total = env.get("steps") or 0
            same_req = later[: max(0, n_total - int(env.get("fail_step", 0)))]
            if same_req:
                row["sp1_steps_in_failed_req"] = len(same_req)
                row["sp1_step_ms_median"] = sorted((r["t_ns"] - r["t0_ns"]) / 1e6 for r in same_req)[len(same_req) // 2]
    pre = [r for r in r0 if r.get("ev") == "ds" and (not miss or r["t_ns"] < miss["t_ns"])]
    if pre:
        warm = pre[1:] or pre
        row["sp2_step_ms_median"] = sorted((r["t_ns"] - r["t0_ns"]) / 1e6 for r in warm)[len(warm) // 2]
    counts = [r for r in r1 if r.get("ev") == "fail_req_count"]
    if counts:
        row["rank1_req_count"] = counts[-1].get("n")
    rs = read_jsonl(run / "rank_states.jsonl") if (run / "rank_states.jsonl").exists() else []
    for r in rs:
        row[f"rank1_{r['tag']}"] = r["states"].get("1")
    exc = [r for r in r0 if r.get("ev") in ("step_exception",) or (r.get("ev") == "contain" and "error" in r.get("phase", ""))]
    if exc:
        row["errors"] = "; ".join(str(r.get("error"))[:120] for r in exc[:3])
    for phase in ("ref", "fail", "after"):
        j = _load(run / f"resp_{phase}.json")
        if j:
            row[f"{phase}_latency_s"] = j.get("latency_s"); row[f"{phase}_ok"] = bool(j.get("sha256"))
            row[f"{phase}_status"] = j.get("job_status") or j.get("status")
            if j.get("error"):
                row[f"{phase}_error"] = str(j["error"])[:160]
    if row.get("fail_latency_s") and row.get("ref_latency_s"):
        row["T_added_s"] = row["fail_latency_s"] - row["ref_latency_s"]
    lat = run / "latents"
    if (lat / "fail_final_rank0.pt").exists() and (lat / "ref_final_rank0.pt").exists():
        row["relerr_vs_sp2ref"] = _rel_err(lat / "fail_final_rank0.pt", lat / "ref_final_rank0.pt")
    if sp1ref is not None:
        s1 = sp1ref / "latents" / "ref_final_rank0.pt"
        j = _load(sp1ref / "resp_ref.json")
        row["sp1ref_latency_s"] = j.get("latency_s"); row["sp1ref_run"] = sp1ref.name
        if s1.exists():
            if (lat / "fail_final_rank0.pt").exists():
                row["relerr_vs_sp1ref"] = _rel_err(lat / "fail_final_rank0.pt", s1)
            if (lat / "ref_final_rank0.pt").exists():
                row["control_sp1_vs_sp2"] = _rel_err(lat / "ref_final_rank0.pt", s1)
            if (lat / "after_final_rank0.pt").exists():
                row["after_vs_sp1ref"] = _rel_err(lat / "after_final_rank0.pt", s1)
    return row


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) >= 0.01 or v == 0 else f"{v:.2e}"
    return "" if v is None else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sp1ref", default=None, help="run dir or glob of MODE=sp1ref runs (latest is used)")
    ap.add_argument("--md", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    sp1 = None
    if a.sp1ref:
        cands = sorted(glob.glob(a.sp1ref)) or ([a.sp1ref] if Path(a.sp1ref).exists() else [])
        sp1 = Path(cands[-1]) if cands else None
    elif (HERE / "runs").exists():
        cands = sorted((HERE / "runs").glob("contain_sp1ref_*"))
        sp1 = cands[-1] if cands else None
    rows = [analyze(Path(r), sp1) for r in a.runs if Path(r).is_dir()]
    cols_timeline = ["run", "shape", "steps", "fail_step", "deadline_s", "T_detect_s", "T_abort_s", "T_switch_s", "T_recompute_s",
                     "sp2_step_ms_median", "sp1_step_ms_median", "ref_latency_s", "fail_latency_s", "T_added_s", "after_latency_s", "sp1ref_latency_s"]
    cols_output = ["run", "injected", "rank1_after_fail", "fail_ok", "after_ok", "relerr_vs_sp2ref", "relerr_vs_sp1ref", "control_sp1_vs_sp2", "after_vs_sp1ref",
                   "coordinators_shrunk", "peers_killed", "modules_sp_size_reset", "errors", "fail_error", "after_error"]
    out = ["# Test 3: in-process failure containment (SGLang Diffusion, Wan SP=2 -> SP=1)", "",
           "Timeline (s): T_detect = rank-1 freeze -> deadline miss on rank 0; T_switch = miss -> failed over "
           "(abort + coordinator shrink + peer kill); T_recompute = the failed step re-run at SP=1; "
           "T_added = failing request latency minus the SP=2 reference request in the same server.", ""]
    for title, cols in (("Recovery timeline", cols_timeline), ("Output check and state", cols_output)):
        out += [f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
        out.append("")
    out += ["Baselines measured earlier on this host (not re-run here): stock SGLang Diffusion never detects the "
            "frozen peer (Phase 2: 600.2 s to the torch watchdog, request lost, replica never evicted); "
            "kill -> cold restart of the rank is ~37 s to ready (startup_decomp.md) plus the lost request.",
            "Verdict rule: PASS if the failing request completes with relerr_vs_sp1ref within the control "
            "(SP=1 vs SP=2 references) and the next request is served by the same process at SP=1.", ""]
    txt = "\n".join(out)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True); Path(a.md).write_text(txt)
    if a.csv:
        import csv

        keys = sorted({k for r in rows for k in r})
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
