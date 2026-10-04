"""Phase 3b-1 comparison: final latents and outputs of the five portability configurations.

Reads results/traj_<tag>/{sp2_save,sp1_ref,sp1_ref2,sp1_full,sp1_lower}/{final_rank0.pt,resp.json,steps_rank0.jsonl}
and reports, against sp1_ref:
  exact            bit-identical final latents (torch.equal)
  max_abs / rel_l2 numerical distance of the final latents
  image_sha_equal  returned image/video bytes identical
  and per configuration the restore time, first resumed step time, remaining completion time
  (from the 'traj' records) and the end-to-end request latency.
The sp1_ref2 row is the run-to-run noise floor; sp2_save vs sp1_ref is the SP2<->SP1 control
(collective ordering) that resume-induced error must be read against.

    python trajectory_compare.py results/traj_<tag> [--md out.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read_jsonl  # noqa: E402

CFGS = ["sp2_save", "sp1_ref", "sp1_ref2", "sp1_full", "sp1_lower"]


def load_final(d: Path):
    import torch

    p = d / "final_rank0.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)["latents"].float()


def traj_timings(d: Path) -> dict:
    out = {}
    for r in read_jsonl(d / "steps_rank0.jsonl"):
        if r.get("ev") != "traj":
            continue
        ph = r.get("phase")
        if ph == "restored":
            out["t_restore_ms"] = round(r.get("t_load_ms", 0) + r.get("t_inject_ms", 0), 1)
            out["restore_notes"] = "; ".join(r.get("notes") or []) or None
            out["mode"] = r.get("mode")
        elif ph == "first_resumed_step":
            out["t_first_resumed_step_ms"] = round(r["t_step_ms"], 1)
        elif ph == "remaining_done":
            out["t_remaining_ms"] = round(r["t_remaining_ms"], 1)
        elif ph == "saved":
            out["t_save_ms"] = round(r.get("t_bundle_ms", 0) + r.get("t_save_ms", 0), 1)
            out["save_bytes"] = r.get("bytes")
        elif ph in ("restore_failed", "save_failed"):
            out["error"] = r.get("error")
    steps = [r for r in read_jsonl(d / "steps_rank0.jsonl") if r.get("ev") == "ds"]
    if steps:
        out["steps_executed"] = len(steps)
        out["warm_step_ms"] = round(sorted((r["t_ns"] - r["t0_ns"]) / 1e6 for r in steps)[len(steps) // 2], 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    import torch

    D = Path(a.dir)
    cfg = json.loads((D / "config.json").read_text()) if (D / "config.json").exists() else {}
    finals = {c: load_final(D / c) for c in CFGS if (D / c).exists()}
    resps = {c: (json.loads((D / c / "resp.json").read_text()) if (D / c / "resp.json").exists() else {}) for c in CFGS}
    ref = finals.get("sp1_ref")
    rows = []
    for c in CFGS:
        if not (D / c).exists():
            continue
        f = finals.get(c)
        row = {"config": c, "ok": f is not None and resps[c].get("sha256") is not None}
        if f is not None and ref is not None and f.shape == ref.shape:
            diff = (f - ref).abs()
            row.update({"exact_vs_sp1_ref": bool(torch.equal(f, ref)),
                        "max_abs": float(diff.max()), "rel_l2": float(diff.norm() / (ref.norm() + 1e-12)),
                        "mean_abs": float(diff.mean())})
        elif f is not None and ref is not None:
            row["shape_mismatch"] = f"{list(f.shape)} vs {list(ref.shape)}"
        row["image_sha_equal_sp1_ref"] = (resps[c].get("sha256") == resps.get("sp1_ref", {}).get("sha256")) if resps[c].get("sha256") else None
        row["latency_s"] = round(resps[c].get("latency_s", 0), 2) if resps[c] else None
        row.update(traj_timings(D / c))
        rows.append(row)
    cols = ["config", "ok", "exact_vs_sp1_ref", "max_abs", "rel_l2", "mean_abs", "image_sha_equal_sp1_ref", "latency_s",
            "steps_executed", "warm_step_ms", "t_save_ms", "save_bytes", "t_restore_ms", "t_first_resumed_step_ms", "t_remaining_ms",
            "restore_notes", "error"]
    lines = [f"Trajectory portability: {cfg}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else (f"{r[c]:.3e}" if isinstance(r.get(c), float) and c in ("max_abs", "rel_l2", "mean_abs") else str(r[c])) for c in cols) + " |")
    lines.append("")
    lines.append("Reading: `sp1_ref2` = run-to-run noise; `sp2_save` vs `sp1_ref` = SP2<->SP1 control (collective ordering); "
                 "`sp1_full` = resumed with the full solver history (the gate: should match sp1_ref to within the control); "
                 "`sp1_lower` = resumed with history reset (quality cost of a low-order restart).")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
