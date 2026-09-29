"""Open-loop load generator (asyncio + httpx) against SGLang's native /generate.

Prompts are random token ids sent as `input_ids`, so input lengths are exact
without a tokenizer; `ignore_eos` pins output lengths. Every request appends one
record to run/requests.jsonl:
  rid, cls, input_len, max_new_tokens, send_ts, first_token_ts, last_token_ts,
  last_chunk_ts, out_tokens, status, error      (all *_ts are monotonic ns)

Profiles (request-class mix):
  mixed : 70% short (256-1024 in / 128-256 out), 20% medium (2k-4k in / 128-256 out),
          10% long (16k-32k in / 64-128 out)
  harsh : 30% long32k (32768 in / 64-128 out), 50% short, 20% medium,
          plus a burst: every --burst-period s, rate x --burst-factor for --burst-len s

Modes:
  --rate R            open loop, Poisson arrivals (or --arrival fixed) at R req/s
  --closed-loop N     N concurrent workers back-to-back (for saturation calibration)

    python load.py --rate 4 --duration 900 --profile mixed
    python load.py --closed-loop 128 --duration 180 --profile mixed   # calibration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, base_url, now_ns, run_dir  # noqa: E402

PROFILES = {
    "mixed": [
        ("short", 0.70, (256, 1024), (128, 256)),
        ("medium", 0.20, (2048, 4096), (128, 256)),
        ("long", 0.10, (16384, 32768), (64, 128)),
    ],
    "harsh": [
        ("short", 0.50, (256, 1024), (128, 256)),
        ("medium", 0.20, (2048, 4096), (128, 256)),
        ("long32k", 0.30, (32768, 32768), (64, 128)),
    ],
}


class Stats:
    def __init__(self):
        self.sent = 0
        self.ok = 0
        self.err = 0
        self.inflight = 0
        self.ttft_ms = []
        self.out_tokens = 0
        self.t_start = now_ns()

    def line(self):
        el = (now_ns() - self.t_start) / 1e9
        p50 = statistics.median(self.ttft_ms[-200:]) if self.ttft_ms else float("nan")
        return (f"[load] t={el:7.1f}s sent={self.sent} ok={self.ok} err={self.err} inflight={self.inflight} "
                f"ttft_p50(last200)={p50:.0f}ms out_tok/s={self.out_tokens / max(el, 1e-9):.0f}")


def pick_class(rng: random.Random, profile):
    r = rng.random()
    acc = 0.0
    for name, w, in_rng, out_rng in profile:
        acc += w
        if r <= acc:
            return name, in_rng, out_rng
    name, w, in_rng, out_rng = profile[-1]
    return name, in_rng, out_rng


async def one_request(client: httpx.AsyncClient, url: str, rid: str, cls: str, input_len: int,
                      max_new: int, vocab_lo: int, vocab_hi: int, rng: random.Random,
                      out: JsonlWriter, stats: Stats, req_timeout: float | None):
    payload = {
        "rid": rid,
        "input_ids": [rng.randint(vocab_lo, vocab_hi) for _ in range(input_len)],
        "sampling_params": {"max_new_tokens": max_new, "ignore_eos": True, "temperature": 0.0},
        "stream": True,
    }
    rec = {"rid": rid, "cls": cls, "input_len": input_len, "max_new_tokens": max_new,
           "send_ts": None, "first_token_ts": None, "last_token_ts": None, "last_chunk_ts": None,
           "out_tokens": 0, "status": None, "error": None, "done": False}
    stats.sent += 1
    stats.inflight += 1
    rec["send_ts"] = now_ns()
    timeout = httpx.Timeout(connect=10.0, read=req_timeout, write=30.0, pool=None)
    try:
        async with client.stream("POST", url, json=payload, timeout=timeout) as resp:
            rec["status"] = resp.status_code
            if resp.status_code != 200:
                body = await resp.aread()
                rec["error"] = f"http {resp.status_code}: {body[:200]!r}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    t = now_ns()
                    rec["last_chunk_ts"] = t
                    if data == "[DONE]":
                        rec["done"] = True
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        rec["error"] = f"stream error: {json.dumps(obj['error'])[:200]}"
                        break
                    if rec["first_token_ts"] is None:
                        rec["first_token_ts"] = t
                        # early record so steady.py sees TTFT before the request completes
                        out.write({"ev": "ft", "rid": rid, "cls": cls, "send_ts": rec["send_ts"], "first_token_ts": t})
                    rec["last_token_ts"] = t
                    mi = obj.get("meta_info") or {}
                    ct = mi.get("completion_tokens")
                    if isinstance(ct, int):
                        rec["out_tokens"] = ct
                    else:
                        rec["out_tokens"] += 1
                if rec["error"] is None and rec["first_token_ts"] is None:
                    rec["error"] = "stream ended with no tokens"
    except asyncio.CancelledError:
        # load.py was stopped while this request was still in flight (it was hung)
        rec["error"] = "cancelled: load generator stopped while request in flight"
        raise
    except Exception as e:  # httpx.ReadTimeout, ConnectError, RemoteProtocolError, ...
        rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        stats.inflight -= 1
        if rec["error"] is None and rec["status"] == 200 and not rec["done"]:
            rec["error"] = "stream closed before [DONE]"
        if rec["error"] is None and rec["status"] == 200:
            stats.ok += 1
            stats.ttft_ms.append((rec["first_token_ts"] - rec["send_ts"]) / 1e6)
            stats.out_tokens += rec["out_tokens"]
        else:
            stats.err += 1
        out.write(rec)


def rate_at(t_el: float, base: float, a) -> float:
    if a.burst_period > 0 and (t_el % a.burst_period) < a.burst_len and t_el >= a.burst_period:
        return base * a.burst_factor
    return base


async def open_loop(a, client, url, out, stats, rng, profile):
    tasks = set()
    t0 = time.monotonic()
    i = 0
    next_at = t0
    while True:
        el = time.monotonic() - t0
        if el >= a.duration:
            break
        r = rate_at(el, a.rate, a)
        gap = (rng.expovariate(r) if a.arrival == "poisson" else 1.0 / r) if r > 0 else 1.0
        next_at += gap
        delay = next_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        cls, in_rng, out_rng = pick_class(rng, profile)
        i += 1
        t = asyncio.create_task(one_request(
            client, url, f"ld-{a.tag}-{i}", cls, rng.randint(*in_rng), rng.randint(*out_rng),
            a.vocab_lo, a.vocab_hi, rng, out, stats, a.req_timeout))
        tasks.add(t)
        t.add_done_callback(tasks.discard)
    # Do not cancel in-flight requests: a hung request must be observed hanging.
    if tasks and a.drain_timeout > 0:
        await asyncio.wait(tasks, timeout=a.drain_timeout)


async def closed_loop(a, client, url, out, stats, rng, profile):
    t0 = time.monotonic()
    counter = [0]

    async def worker(w):
        while time.monotonic() - t0 < a.duration:
            cls, in_rng, out_rng = pick_class(rng, profile)
            counter[0] += 1
            await one_request(client, url, f"cl-{a.tag}-{w}-{counter[0]}", cls, rng.randint(*in_rng),
                              rng.randint(*out_rng), a.vocab_lo, a.vocab_hi, rng, out, stats, a.req_timeout)

    await asyncio.gather(*(worker(w) for w in range(a.closed_loop)))


async def reporter(stats, every):
    while True:
        await asyncio.sleep(every)
        print(stats.line(), flush=True)


async def amain(a):
    profile = PROFILES[a.profile]
    rng = random.Random(a.seed)
    out = JsonlWriter(run_dir() / a.out)
    stats = Stats()
    url = base_url().rstrip("/") + "/generate"
    out.write({"ev": "load_start", "mode": "closed" if a.closed_loop else "open", "rate": a.rate,
               "profile": a.profile, "duration": a.duration, "burst_period": a.burst_period,
               "burst_factor": a.burst_factor, "burst_len": a.burst_len, "url": url})
    limits = httpx.Limits(max_connections=a.max_conn, max_keepalive_connections=a.max_conn)
    rep = asyncio.create_task(reporter(stats, a.report_every))
    async with httpx.AsyncClient(limits=limits, http2=False) as client:
        if a.closed_loop:
            await closed_loop(a, client, url, out, stats, rng, profile)
        else:
            await open_loop(a, client, url, out, stats, rng, profile)
    rep.cancel()
    el = (now_ns() - stats.t_start) / 1e9
    ttft = sorted(stats.ttft_ms)
    summary = {
        "ev": "load_end", "elapsed_s": el, "sent": stats.sent, "ok": stats.ok, "err": stats.err,
        "req_per_s_ok": stats.ok / el if el else 0, "out_tok_per_s": stats.out_tokens / el if el else 0,
        "ttft_p50_ms": ttft[len(ttft) // 2] if ttft else None,
        "ttft_p99_ms": ttft[int(len(ttft) * 0.99)] if ttft else None,
    }
    out.write(summary)
    print("[load] " + json.dumps(summary), flush=True)
    if a.closed_loop:
        print(f"[load] saturation ~= {summary['req_per_s_ok']:.2f} req/s for profile '{a.profile}'; "
              f"70% -> --rate {0.7 * summary['req_per_s_ok']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=2.0, help="open-loop request rate (req/s)")
    ap.add_argument("--closed-loop", type=int, default=0, help="N concurrent workers (calibration mode)")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="mixed")
    ap.add_argument("--arrival", choices=["poisson", "fixed"], default="poisson")
    ap.add_argument("--burst-period", type=float, default=0.0, help="s between bursts (0 = none); harsh uses 300")
    ap.add_argument("--burst-factor", type=float, default=3.0)
    ap.add_argument("--burst-len", type=float, default=30.0)
    ap.add_argument("--req-timeout", type=float, default=None, help="per-request read timeout s (default: none)")
    ap.add_argument("--drain-timeout", type=float, default=600.0, help="wait this long for in-flight after duration")
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=30000)
    ap.add_argument("--max-conn", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="r")
    ap.add_argument("--out", default="requests.jsonl", help="relative to FT_RUN_DIR")
    ap.add_argument("--report-every", type=float, default=10.0)
    a = ap.parse_args()
    if a.profile == "harsh" and a.burst_period == 0.0:
        a.burst_period = 300.0
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
