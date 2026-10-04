"""Phase 3b-2 prerequisite: decompose the rank restart time from existing logs (no new mechanism).

Per run directory, from launch.json, logs/rank0.log, logs/steps_rank0.jsonl, logs/server.log, ready.json:
  T_spawn    launch_wrapped_diff.py started -> rank0 process alive (our '[ft-detect] rank0 pid' line)
  T_import   rank alive -> sglang modules imported (first 'patched' record in steps_rank0.jsonl)
  T_dist     import done -> NCCL / torch.distributed initialised ('using nccl==' line in rank0.log)
  T_load     dist init -> 'Worker 0: Scheduler loop started.' (model load, H2D, device setup; not separable in the logs)
  T_warmup   loop started -> /health 200 (ready.json written by launch_diff.sh; warmup requests)
  T_ready    launch -> /health 200 (sum)
Log-line timestamps have 1 s resolution ([MM-DD HH:MM:SS]); our own records are sub-ms.

    python startup_decomp.py runs/diff_*  [--md out.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read_jsonl  # noqa: E402

TS = re.compile(r"^\[(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\]")


def _wall_of_line(line: str, year: int) -> float | None:
    m = TS.match(line)
    if not m:
        return None
    mo, d, h, mi, s = map(int, m.groups())
    return dt.datetime(year, mo, d, h, mi, s).timestamp()


def _first_line(path: Path, rx: re.Pattern, year: int):
    if not path.exists():
        return None, None
    for line in open(path, errors="replace"):
        if rx.search(line):
            return _wall_of_line(line, year), line.strip()[:160]
    return None, None


def decompose(run: Path) -> dict:
    out = {"run": run.name}
    try:
        launch = json.loads((run / "launch.json").read_text())
    except Exception:
        return out | {"error": "no launch.json"}
    t_launch = launch["wall_launch"]
    year = dt.datetime.fromtimestamp(t_launch).year
    try:
        out["model"] = json.loads((run / "launch_env.json").read_text()).get("model")
    except Exception:
        pass
    rank_line = _first_line(run / "logs" / "rank0.log", re.compile(r"\[ft-detect\] rank0 .*wall=([\d.]+)"), year)[1]
    t_rank = float(re.search(r"wall=([\d.]+)", rank_line).group(1)) if rank_line else None
    patched = next((r for r in read_jsonl(run / "logs" / "steps_rank0.jsonl") if r.get("ev") == "patched"), None)
    t_import = patched["wall"] if patched else None
    t_nccl, _ = _first_line(run / "logs" / "rank0.log", re.compile(r"using nccl==|Setting distributed timeout|Initializing distributed"), year)
    t_loop, _ = _first_line(run / "logs" / "rank0.log", re.compile(r"Scheduler loop started"), year)
    t_ready = None
    if (run / "ready.json").exists():
        t_ready = json.loads((run / "ready.json").read_text()).get("wall_ready")
    if t_ready is None:
        t_ready, _ = _first_line(run / "logs" / "server.log", re.compile(r"fired up and ready"), year)

    def d(a, b):
        return None if a is None or b is None else round(b - a, 2)

    out.update({
        "T_spawn_s": d(t_launch, t_rank), "T_import_s": d(t_rank, t_import), "T_dist_s": d(t_import, t_nccl),
        "T_load_s": d(t_nccl, t_loop), "T_warmup_s": d(t_loop, t_ready), "T_ready_s": d(t_launch, t_ready),
        "note": "log timestamps are 1 s resolution; T_load bundles weight load, H2D and device setup",
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = [decompose(Path(r)) for r in a.runs if Path(r).is_dir()]
    cols = ["run", "model", "T_spawn_s", "T_import_s", "T_dist_s", "T_load_s", "T_warmup_s", "T_ready_s"]
    lines = ["Rank restart decomposition (seconds; from existing logs)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in cols) + " |")
    lines.append("\nT_load bundles weight load, host-to-device copy and device setup (not separable in these logs); "
                 "T_warmup is the server's own warmup requests until /health returns 200.")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
