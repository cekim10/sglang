"""Phase 3a: resume-state inventory at a denoising-step boundary, per workload and per rank.

Input: 'inventory' records in run/logs/steps_rank*.jsonl (launch_wrapped_diff.py with FT_INVENTORY=1).
Per (model, shape):
  latents shape / bytes on each rank, and whether the ranks hold the SAME tensor
  (equal shape and checksum -> replicated: a surviving rank already has x_t) or a shard
  (did_sp_shard_latents / different checksum -> recovery needs the dead rank's half),
  scheduler class and solver history bytes (multi-step solvers carry extra latents),
  conditioning bytes (prompt / image embeds), generator state bytes,
  the resume bundle total (S_state) and the measured checkpoint cost on the serving rank
  (T_pin: device->pinned host, T_save: torch.save; serialized bytes),
  and the checkpoint cost relative to the workload's warm step time if ds records exist.

    python inventory.py <run_dir>... [--md out.md] [--csv out.csv]
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


def load(run_dirs):
    inv, steps = [], []
    for rd in run_dirs:
        rd = Path(rd)
        model = _model_of(rd)
        for p in sorted(rd.glob("logs/steps_rank*.jsonl")):
            rank = int(p.stem.replace("steps_rank", ""))
            for r in read_jsonl(p):
                if r.get("ev") == "inventory":
                    r = dict(r)
                    r.update({"run": rd.name, "model": model, "rank": rank, "shape": _shape(r)})
                    inv.append(r)
                elif r.get("ev") == "ds":
                    steps.append({"model": model, "shape": _shape(r), "dur_ms": (r["t_ns"] - r["t0_ns"]) / 1e6})
    return inv, pd.DataFrame(steps)


def _sum_bytes(d: dict, prefix: str) -> int:
    return int(sum(v for k, v in d.items() if k.startswith(prefix)))


def summarize(inv, steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_key = {}
    for r in inv:
        by_key.setdefault((r["model"], r["shape"], r.get("rid")), {})[r["rank"]] = r
    for (model, shape, rid), ranks in by_key.items():
        r0 = ranks.get(0) or next(iter(ranks.values()))
        lat = r0["tensors"].get("latents") or {}
        other = [ranks[k] for k in ranks if k != r0["rank"]]
        same = None
        if other:
            o = other[0]["tensors"].get("latents") or {}
            same = (o.get("shape") == lat.get("shape")) and (o.get("checksum") == lat.get("checksum"))
        bb = r0.get("bundle_bytes", {})
        ck = r0.get("checkpoint", {})
        rows.append({
            "model": model, "shape": shape, "rid": rid, "ranks_seen": len(ranks),
            "step_index": r0.get("step_index"), "n_steps": r0.get("num_inference_steps"),
            "latents_shape": "x".join(map(str, lat.get("shape", []))), "latents_dtype": lat.get("dtype"),
            "latents_MiB": round(lat.get("nbytes", 0) / 2**20, 3),
            "sharded_flag": r0.get("did_sp_shard_latents"),
            "latents_identical_across_ranks": same,
            "scheduler": r0.get("scheduler_class"),
            "solver_history_MiB": round(_sum_bytes(bb, "scheduler.") / 2**20, 3),
            "conditioning_MiB": round(_sum_bytes(bb, "req.") / 2**20, 3),
            "generator_state_bytes": sum(g.get("state_bytes", 0) for g in r0.get("generator", [])),
            "S_state_MiB": round(r0.get("bundle_total_bytes", 0) / 2**20, 3),
            "serialized_MiB": round(ck.get("serialized_bytes", 0) / 2**20, 3) if ck else None,
            "T_pin_ms": round(ck.get("t_pin_ms", 0), 2) if ck else None,
            "T_save_ms": round(ck.get("t_save_ms", 0), 2) if ck else None,
            "checkpoint_error": r0.get("checkpoint_error"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # one row per workload: median over requests
    num = ["latents_MiB", "solver_history_MiB", "conditioning_MiB", "generator_state_bytes", "S_state_MiB",
           "serialized_MiB", "T_pin_ms", "T_save_ms"]
    g = df.groupby(["model", "shape"])
    out = g.agg(requests=("rid", "nunique"), ranks_seen=("ranks_seen", "max"), step_index=("step_index", "first"),
                n_steps=("n_steps", "first"), latents_shape=("latents_shape", "first"), latents_dtype=("latents_dtype", "first"),
                sharded_flag=("sharded_flag", "max"), latents_identical_across_ranks=("latents_identical_across_ranks", "min"),
                scheduler=("scheduler", "first"), **{c: (c, "median") for c in num}).reset_index()
    out["recovery_source"] = out.apply(
        lambda r: "survivor holds full x_t" if r["latents_identical_across_ranks"] is True
        else ("sharded: needs peer's shard" if r["sharded_flag"] or r["latents_identical_across_ranks"] is False else "unknown (1 rank)"), axis=1)
    if not steps.empty:
        st = steps.groupby(["model", "shape"])["dur_ms"].median().rename("warm_step_ms").reset_index()
        out = out.merge(st, on=["model", "shape"], how="left")
        out["T_pin_over_step_pct"] = (100 * out["T_pin_ms"] / out["warm_step_ms"]).round(2)
    for c in num:
        if c in out:
            out[c] = out[c].round(3)
    return out


def md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_no inventory records (run with FT_INVENTORY=1)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    inv, steps = load([r for r in a.runs if Path(r).is_dir()])
    out = summarize(inv, steps)
    text = ("Resume-state inventory at the boundary after denoising step FT_INVENTORY_STEP (per workload, median over requests).\n\n"
            + md_table(out)
            + "\n\n`S_state` = latents + solver history + conditioning + generator state on the serving rank; `T_pin` = device to pinned host copy "
              "of that bundle; `T_save` = torch.save of the host copy. `recovery_source` says whether a surviving SP rank already holds the full "
              "latent (replicated) or only a shard.")
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
