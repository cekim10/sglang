"""Steady-state gate: is the replica under stable load?

Reads run/requests.jsonl and compares the last two --window s windows:
  * each window has >= --min-reqs completed requests with a first token
  * p50 TTFT of the newest window is within --tol of the previous one
  * no request errors in the newest window
Exit 0 when steady, 1 otherwise. With --wait N it polls until steady or N s.

    python steady.py --wait 600
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import now_ns, read_jsonl, run_dir  # noqa: E402


def check(window: float, min_reqs: int, tol: float):
    recs = [r for r in read_jsonl(run_dir() / "requests.jsonl") if "rid" in r]
    t = now_ns()
    w_ns = int(window * 1e9)
    # load.py writes an early {"ev":"ft"} record when the first token arrives, so
    # requests that are still streaming are counted in the newest window too.
    got = [r for r in recs if r.get("ev") == "ft"]
    new = [r for r in got if r["first_token_ts"] >= t - w_ns]
    old = [r for r in got if t - 2 * w_ns <= r["first_token_ts"] < t - w_ns]

    def ttfts(rs):
        return [(r["first_token_ts"] - r["send_ts"]) / 1e6 for r in rs]

    tn, to = ttfts(new), ttfts(old)
    full = [r for r in recs if "ev" not in r]
    errs = sum(1 for r in full if r.get("error") and (r.get("last_chunk_ts") or r["send_ts"]) >= t - w_ns)
    p50n = statistics.median(tn) if tn else None
    p50o = statistics.median(to) if to else None
    ok = (
        len(tn) >= min_reqs and len(to) >= min_reqs and errs == 0
        and p50n is not None and p50o is not None and abs(p50n - p50o) <= tol * max(p50o, 1e-9)
    )
    return ok, {"n_new": len(tn), "n_old": len(to), "errors_new": errs, "p50_new_ms": p50n, "p50_old_ms": p50o}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--min-reqs", type=int, default=20)
    ap.add_argument("--tol", type=float, default=0.30)
    ap.add_argument("--wait", type=float, default=0.0)
    ap.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args()
    t0 = time.monotonic()
    while True:
        ok, info = check(a.window, a.min_reqs, a.tol)
        print(json.dumps({"steady": ok, **info}), flush=True)
        if ok:
            return 0
        if time.monotonic() - t0 >= a.wait:
            return 1
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
