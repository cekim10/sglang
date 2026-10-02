"""Open-loop image-generation load against SGLang Diffusion's /v1/images/generations.

No streaming exists, so each record's first_token_ts == last_token_ts == completion
time; analyze.py/steady.py therefore treat end-to-end latency as "TTFT". Records go
to run/requests.jsonl with the same fields as ../load.py plus size/steps.

Profiles (shape mix; steps per class):
  mixed : 60% small  512x512  @ 20 steps, 30% medium 1024x1024 @ 28 steps, 10% large 1536x1536 @ 40 steps
  harsh : 40% large 1536x1536 @ 50 steps, 40% medium, 20% small, plus a burst every --burst-period s
  video : (only if the served model is a video model) 100% 480p 33 frames @ 30 steps

    python load_diff.py --rate 0.08 --duration 1200 --profile mixed
    python load_diff.py --closed-loop 4 --duration 300 --profile mixed     # calibration
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import JsonlWriter, base_url, now_ns, run_dir  # noqa: E402

PROFILES = {
    "mixed": [("small", 0.60, (512, 512), 20), ("medium", 0.30, (1024, 1024), 28), ("large", 0.10, (1536, 1536), 40)],
    "harsh": [("small", 0.20, (512, 512), 20), ("medium", 0.40, (1024, 1024), 28), ("large", 0.40, (1536, 1536), 50)],
    "video": [("v480p", 1.0, (832, 480), 30)],
}
PROMPTS = [
    "a watercolor painting of a lighthouse at dawn", "macro photo of a dew-covered spider web",
    "isometric voxel city at night", "a red fox running through fresh snow", "a cup of coffee on a wooden table, soft light",
]


class Stats:
    def __init__(self):
        self.sent = self.ok = self.err = self.inflight = 0
        self.lat_ms = []
        self.t_start = now_ns()

    def line(self):
        el = (now_ns() - self.t_start) / 1e9
        p50 = statistics.median(self.lat_ms[-50:]) if self.lat_ms else float("nan")
        return f"[load] t={el:7.1f}s sent={self.sent} ok={self.ok} err={self.err} inflight={self.inflight} lat_p50(last50)={p50 / 1e3:.1f}s"


def pick(rng, profile):
    r, acc = rng.random(), 0.0
    for name, w, size, steps in profile:
        acc += w
        if r <= acc:
            return name, size, steps
    return profile[-1][0], profile[-1][2], profile[-1][3]


async def one(client, url, rid, cls, size, steps, rng, out, stats, a):
    payload = {"prompt": rng.choice(PROMPTS), "width": size[0], "height": size[1], "n": 1,
               "num_inference_steps": steps, "seed": rng.randint(0, 2**31 - 1), "response_format": "b64_json"}
    if a.video:
        payload["num_frames"] = a.video_frames
    rec = {"rid": rid, "cls": cls, "input_len": size[0] * size[1] // 1000, "max_new_tokens": steps,
           "width": size[0], "height": size[1], "steps": steps,
           "send_ts": None, "first_token_ts": None, "last_token_ts": None, "last_chunk_ts": None,
           "out_tokens": 0, "status": None, "error": None, "done": False}
    stats.sent += 1
    stats.inflight += 1
    rec["send_ts"] = now_ns()
    timeout = httpx.Timeout(connect=10.0, read=a.req_timeout, write=60.0, pool=None)
    try:
        r = await client.post(url, json=payload, timeout=timeout)
        t = now_ns()
        rec["status"] = r.status_code
        rec["last_chunk_ts"] = t
        if r.status_code == 200:
            try:
                body = r.json()
                n_img = len(body.get("data", []))
            except Exception:
                n_img = 0
            if n_img:
                rec["first_token_ts"] = rec["last_token_ts"] = t
                rec["out_tokens"] = steps
                rec["done"] = True
                out.write({"ev": "ft", "rid": rid, "cls": cls, "send_ts": rec["send_ts"], "first_token_ts": t})
            else:
                rec["error"] = f"200 with no image: {r.text[:160]!r}"
        else:
            rec["error"] = f"http {r.status_code}: {r.text[:200]!r}"
    except asyncio.CancelledError:
        rec["error"] = "cancelled: load generator stopped while request in flight"
        raise
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        stats.inflight -= 1
        if rec["error"] is None and rec["done"]:
            stats.ok += 1
            stats.lat_ms.append((rec["last_token_ts"] - rec["send_ts"]) / 1e6)
        else:
            stats.err += 1
        out.write(rec)


def rate_at(el, base, a):
    if a.burst_period > 0 and el >= a.burst_period and (el % a.burst_period) < a.burst_len:
        return base * a.burst_factor
    return base


async def open_loop(a, client, url, out, stats, rng, profile):
    tasks, t0, i, next_at = set(), time.monotonic(), 0, time.monotonic()
    while time.monotonic() - t0 < a.duration:
        r = rate_at(time.monotonic() - t0, a.rate, a)
        next_at += (rng.expovariate(r) if a.arrival == "poisson" else 1.0 / r) if r > 0 else 1.0
        d = next_at - time.monotonic()
        if d > 0:
            await asyncio.sleep(d)
        cls, size, steps = pick(rng, profile)
        i += 1
        t = asyncio.create_task(one(client, url, f"ld-{a.tag}-{i}", cls, size, steps, rng, out, stats, a))
        tasks.add(t)
        t.add_done_callback(tasks.discard)
    if tasks and a.drain_timeout > 0:
        await asyncio.wait(tasks, timeout=a.drain_timeout)


async def closed_loop(a, client, url, out, stats, rng, profile):
    t0, n = time.monotonic(), [0]

    async def worker(w):
        while time.monotonic() - t0 < a.duration:
            cls, size, steps = pick(rng, profile)
            n[0] += 1
            await one(client, url, f"cl-{a.tag}-{w}-{n[0]}", cls, size, steps, rng, out, stats, a)

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
    url = base_url().rstrip("/") + ("/v1/videos/generations" if a.video else "/v1/images/generations")
    out.write({"ev": "load_start", "mode": "closed" if a.closed_loop else "open", "rate": a.rate, "profile": a.profile,
               "duration": a.duration, "burst_period": a.burst_period, "burst_factor": a.burst_factor,
               "burst_len": a.burst_len, "url": url, "stack": "sglang-diffusion"})
    rep = asyncio.create_task(reporter(stats, a.report_every))
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=512, max_keepalive_connections=512)) as client:
        if a.closed_loop:
            await closed_loop(a, client, url, out, stats, rng, profile)
        else:
            await open_loop(a, client, url, out, stats, rng, profile)
    rep.cancel()
    el = (now_ns() - stats.t_start) / 1e9
    lat = sorted(stats.lat_ms)
    s = {"ev": "load_end", "elapsed_s": el, "sent": stats.sent, "ok": stats.ok, "err": stats.err,
         "req_per_s_ok": stats.ok / el if el else 0,
         "ttft_p50_ms": lat[len(lat) // 2] if lat else None, "ttft_p99_ms": lat[int(len(lat) * 0.99)] if lat else None}
    out.write(s)
    print("[load] " + json.dumps(s), flush=True)
    if a.closed_loop:
        print(f"[load] saturation ~= {s['req_per_s_ok']:.3f} req/s for profile '{a.profile}'; 70% -> --rate {0.7 * s['req_per_s_ok']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--closed-loop", type=int, default=0)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="mixed")
    ap.add_argument("--arrival", choices=["poisson", "fixed"], default="poisson")
    ap.add_argument("--burst-period", type=float, default=0.0, help="harsh defaults to 300")
    ap.add_argument("--burst-factor", type=float, default=3.0)
    ap.add_argument("--burst-len", type=float, default=60.0)
    ap.add_argument("--req-timeout", type=float, default=None, help="per-request read timeout s (default none: hangs are observed)")
    ap.add_argument("--drain-timeout", type=float, default=900.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-frames", type=int, default=33)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="r")
    ap.add_argument("--out", default="requests.jsonl")
    ap.add_argument("--report-every", type=float, default=15.0)
    a = ap.parse_args()
    if a.profile == "harsh" and a.burst_period == 0.0:
        a.burst_period = 300.0
    if a.profile == "video":
        a.video = True
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
