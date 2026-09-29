"""Fault injector. Records t_inject (shared monotonic clock) then signals one TP rank.

    python inject.py --case A --rank 1      # SIGKILL rank 1
    python inject.py --case B --rank 1      # SIGSTOP rank 1 (hang: peers block in a collective)
    python inject.py --resume --rank 1      # SIGCONT rank 1 (cleanup for case B)

Safety: the target PID must come from run/pids.json and its parent must be the
recorded HTTP-server PID, so a stale pids.json can never signal someone else's
process. Refuses rank 0 unless --allow-rank0.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, load_pids, now_ns, run_dir, wall  # noqa: E402

CASES = {"A": signal.SIGKILL, "B": signal.SIGSTOP}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=sorted(CASES))
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--resume", action="store_true", help="send SIGCONT instead of injecting")
    ap.add_argument("--allow-rank0", action="store_true")
    ap.add_argument("--run-id", default=None, help="free-form label stored with the record")
    a = ap.parse_args()
    if not a.resume and a.case is None:
        ap.error("--case is required unless --resume")
    if a.rank == 0 and not a.allow_rank0:
        ap.error("rank 0 is the TP leader; pass --allow-rank0 if you really mean it")

    pids = load_pids()
    if pids is None:
        print(f"no pids.json in {run_dir()}", file=sys.stderr)
        return 2
    pid = pids.get("ranks", {}).get(str(a.rank))
    if pid is None:
        print(f"rank {a.rank} not in pids.json: {pids.get('ranks')}", file=sys.stderr)
        return 2
    try:
        p = psutil.Process(int(pid))
        ppid = p.ppid()
        cmd = " ".join(p.cmdline())
    except psutil.NoSuchProcess:
        print(f"rank {a.rank} pid {pid} does not exist", file=sys.stderr)
        return 2
    if ppid != int(pids["http_server"]):
        print(f"refusing: pid {pid} parent is {ppid}, expected http_server {pids['http_server']}", file=sys.stderr)
        return 3

    sig = signal.SIGCONT if a.resume else CASES[a.case]
    ev = JsonlWriter(run_dir() / "inject.jsonl")
    rec = {"ev": "resume" if a.resume else "inject", "case": None if a.resume else a.case, "rank": a.rank,
           "pid": int(pid), "signal": sig.name, "status_before": p.status(), "cmd": cmd[:120], "run_id": a.run_id}
    t = now_ns()
    os.kill(int(pid), sig)
    t2 = now_ns()
    rec["t_ns"] = t  # t_inject: immediately before the syscall
    rec["t_after_ns"] = t2
    rec["wall"] = wall()
    ev.write(rec)
    print(f"[inject] {rec['ev']} case={rec['case']} rank={a.rank} pid={pid} sig={sig.name} t_inject_ns={t}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
