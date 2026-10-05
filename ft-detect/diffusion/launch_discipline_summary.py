"""Summarise one launch_discipline_probe rank-0 JSON: side-stream health before/during/after the
hang, launch-queue headroom (flood), and whether the escape hatches worked (disciplined)."""

from __future__ import annotations

import json
import sys


def main():
    d = json.load(open(sys.argv[1]))
    log = d["log"]
    st = d.get("state", {})
    t_hang = next((r["t"] for r in log if r["ev"] == "main.collective_enqueued"), None)
    t_abort = next((r["t"] for r in log if r["ev"] == "main.abort_result"), None)
    syncs = [r for r in log if r["ev"] == "side.sync_done"]
    begins = [r for r in log if r["ev"] == "side.launch_begin"]
    launched = {r["i"] for r in log if r["ev"] == "side.launch_done"}
    synced = {r["i"] for r in syncs}
    print(f"mode={st.get('mode')} exit={d.get('exit')} hang_enqueued_at={t_hang}s")
    def phase(r):
        if t_hang is None or r["t"] < t_hang:
            return "before"
        if t_abort is not None and r["t"] > t_abort:
            return "after_abort"
        return "during"
    for ph in ("before", "during", "after_abort"):
        ss = [r for r in syncs if phase(r) == ph]
        bb = [r for r in begins if phase(r) == ph]
        stuck = [r["i"] for r in bb if r["i"] not in launched]
        unsynced = [r["i"] for r in bb if r["i"] in launched and r["i"] not in synced]
        mm = sorted(r["matmul_ms"] for r in ss)
        print(f"  side-stream {ph:12s}: iterations started {len(bb):3d}, completed {len(ss):3d}, "
              f"matmul_ms median {mm[len(mm)//2] if mm else None}, blocked-in-launch {len(stuck)}, launched-but-never-synced {len(unsynced)}")
    for ev in ("main.collective_enqueued", "main.flood_first_launch_returned", "main.launch_blocked_or_slow"):
        for r in log:
            if r["ev"] == ev:
                print(f"  {r}")
    if st.get("mode") == "flood":
        print(f"  launches on the main stream before cudaLaunchKernel blocked: {st.get('launches_before_block')}")
    elif st.get("mode") == "disciplined":
        print(f"  abort: {st.get('abort')}")
        print(f"  main stream usable after abort: {st.get('main_stream_after_abort')}")
        for ev in ("main.deadline_miss", "main.post_abort_main_stream", "main.destroy_done", "main.destroy_error", "watchdog.exit"):
            for r in log:
                if r["ev"] == ev:
                    print(f"  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
