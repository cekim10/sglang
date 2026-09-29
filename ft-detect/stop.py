"""Stop the replica started by launch.sh and clean up stopped/orphaned rank processes.

Only touches PIDs recorded in run/pids.json and their descendants. Never scans
for other users' processes.

    python stop.py            # SIGCONT every rank, then SIGKILL the whole tree
    python stop.py --graceful # SIGTERM the HTTP server first, wait, then SIGKILL leftovers
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, load_pids, run_dir  # noqa: E402


def _proc(pid):
    try:
        return psutil.Process(int(pid))
    except (psutil.NoSuchProcess, ValueError, TypeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graceful", action="store_true")
    ap.add_argument("--wait", type=float, default=30.0)
    args = ap.parse_args()

    pids = load_pids()
    if pids is None:
        print(f"no pids.json in {run_dir()}; nothing to stop", file=sys.stderr)
        return 0
    ev = JsonlWriter(run_dir() / "inject.jsonl")

    targets = []
    server = _proc(pids.get("http_server"))
    if server is not None:
        targets.append(server)
        try:
            targets.extend(server.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
    for rank, pid in pids.get("ranks", {}).items():
        p = _proc(pid)
        if p is not None and p not in targets:
            targets.append(p)
    detok = _proc(pids.get("detokenizer"))
    if detok is not None and detok not in targets:
        targets.append(detok)

    # Un-freeze anything SIGSTOPped so it can actually die.
    for p in targets:
        try:
            if p.status() == psutil.STATUS_STOPPED:
                p.send_signal(signal.SIGCONT)
                ev.write({"ev": "sigcont", "pid": p.pid})
        except psutil.NoSuchProcess:
            pass

    if args.graceful and server is not None and server.is_running():
        server.send_signal(signal.SIGTERM)
        ev.write({"ev": "sigterm", "pid": server.pid})
        gone, alive = psutil.wait_procs(targets, timeout=args.wait)
        targets = alive

    for p in targets:
        try:
            p.kill()
            ev.write({"ev": "sigkill", "pid": p.pid})
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(targets, timeout=args.wait)
    for p in alive:
        print(f"still alive after SIGKILL: {p.pid} {p.status()}", file=sys.stderr)
    ev.write({"ev": "stopped", "killed": [p.pid for p in gone], "alive": [p.pid for p in alive]})
    try:
        os.rename(run_dir() / "pids.json", run_dir() / "pids.stopped.json")
    except OSError:
        pass
    time.sleep(1)
    return 1 if alive else 0


if __name__ == "__main__":
    sys.exit(main())
