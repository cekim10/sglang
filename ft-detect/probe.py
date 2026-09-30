"""Sidecar probe. Every --interval s (default 0.1) it checks:
  (a) GET /health                       (blocks up to SGLANG_HEALTH_CHECK_TIMEOUT=20 s server-side)
  (b) GET /health_generate              (same handler in current SGLang; kept separate for the record)
  (c) POST /generate 1 token, short client timeout (--gen-timeout)   -> "gen1"
  (d) liveness + status of every PID in run/pids.json (running/sleeping/stopped/zombie/dead)
  (e) new lines in run/logs/*.log for detector strings (watchdog, NCCL, SIGQUIT, crash, abort)

Each HTTP probe kind keeps at most one request in flight, re-issued as soon as
the previous one returns, so a blocking /health does not stall the loop.
Every *state change* is one JSONL record in run/probe.jsonl with the shared
monotonic clock; heartbeats every --heartbeat s record current state + latency.

    python probe.py                # until Ctrl-C or --duration
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import signal
import sys
from pathlib import Path

import httpx
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, base_url, load_pids, logs_dir, now_ns, run_dir  # noqa: E402

# Strings found in Step 0 recon (sglang @ c7be3e935b) plus torch/NCCL runtime messages.
PATTERNS = {
    "watchdog_fire": re.compile(r"watchdog timeout", re.I),
    "watchdog_soft": re.compile(r"soft=True", re.I),
    "scheduler_exception": re.compile(r"Scheduler hit an exception"),
    "subprocess_crashed": re.compile(r"crashed with exit code"),
    "sigquit": re.compile(r"SIGQUIT received|Received sigquit", re.I),
    "kill_tree": re.compile(r"kill_process_tree called"),
    "health_failed": re.compile(r"Health check failed"),
    "pyspy_dump": re.compile(r"Pyspy dump|py-spy dump", re.I),
    "nccl_timeout": re.compile(r"Watchdog caught collective operation timeout|ProcessGroupNCCL.*(timeout|Timed out)|NCCL timeout", re.I),
    "nccl_error": re.compile(r"ncclSystemError|ncclRemoteError|ncclInternalError|ncclUnhandledCudaError|NCCL error", re.I),
    "torch_dist_error": re.compile(r"torch\.distributed\.DistBackendError|DistNetworkError|Connection reset by peer|Broken pipe"),
    # NOTE: "Abort request obj.rid=..." is the server logging a *client* disconnect; excluded on purpose.
    "abort": re.compile(r"std::terminate|Aborted \(core dumped\)|SIGABRT|ncclCommAbort|\baborted\b(?! request)|\babort(?!(ed)? request)\b", re.I),
    "client_disconnect": re.compile(r"Abort request obj\.rid=|disconnected from the client side"),
    "cuda_error": re.compile(r"CUDA error|illegal memory access|device-side assert", re.I),
    "scheduler_terminated": re.compile(r"terminated with"),
    "server_shutdown": re.compile(r"Shutting down|Draining requests and shutting down", re.I),
}
MAX_HITS_PER_KEY = 5


class LogTailer:
    def __init__(self, out: JsonlWriter):
        self.out = out
        self.offsets: dict[Path, int] = {}
        self.hits: dict[tuple, int] = {}

    def poll(self):
        d = logs_dir()
        if not d.exists():
            return
        for path in sorted(d.glob("*.log")):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            off = self.offsets.get(path, 0)
            if size <= off:
                continue
            with open(path, "rb") as f:
                f.seek(off)
                chunk = f.read(size - off)
            # keep a partial trailing line for the next poll
            nl = chunk.rfind(b"\n")
            if nl == -1:
                continue
            self.offsets[path] = off + nl + 1
            text = chunk[: nl + 1].decode("utf-8", errors="replace")
            for line in text.splitlines():
                for key, rx in PATTERNS.items():
                    if rx.search(line):
                        k = (path.name, key)
                        n = self.hits.get(k, 0)
                        if n < MAX_HITS_PER_KEY:
                            self.out.write({"ev": "log", "file": path.name, "pattern": key,
                                            "first": n == 0, "line": line[-400:]})
                        self.hits[k] = n + 1


class PidWatch:
    def __init__(self, out: JsonlWriter):
        self.out = out
        self.state: dict[str, str] = {}
        self.pids: dict[str, int] = {}
        self.loaded = False

    def load(self):
        pids = load_pids()
        if not pids:
            return
        self.pids = {"http_server": int(pids["http_server"])}
        for r, pid in pids.get("ranks", {}).items():
            self.pids[f"rank{r}"] = int(pid)
        if pids.get("detokenizer"):
            self.pids["detokenizer"] = int(pids["detokenizer"])
        self.loaded = True
        self.out.write({"ev": "pids_loaded", "pids": self.pids})

    def poll(self):
        if not self.loaded:
            self.load()
            if not self.loaded:
                return
        for name, pid in self.pids.items():
            try:
                p = psutil.Process(pid)
                st = p.status()
                if st == psutil.STATUS_ZOMBIE:
                    st = "zombie"
                elif st in (psutil.STATUS_STOPPED, psutil.STATUS_TRACING_STOP):
                    st = "stopped"
                else:
                    st = "alive"  # running/sleeping/disk-sleep flap every tick; not a state change we care about
            except psutil.NoSuchProcess:
                st = "dead"
            except psutil.AccessDenied:
                st = "access_denied"
            prev = self.state.get(name)
            if st != prev:
                self.state[name] = st
                self.out.write({"ev": "pid", "who": name, "pid": pid, "state": st, "prev": prev})


class HttpProbe:
    def __init__(self, kind: str, out: JsonlWriter, client: httpx.AsyncClient, timeout: float):
        self.kind = kind
        self.out = out
        self.client = client
        self.timeout = timeout
        self.task: asyncio.Task | None = None
        self.state: str | None = None
        self.last_latency_ms: float | None = None
        self.consecutive_bad = 0
        self.n = 0

    async def _one(self):
        url = base_url().rstrip("/")
        t0 = now_ns()
        try:
            if self.kind == "gen1":
                r = await self.client.post(
                    url + "/generate",
                    json={"input_ids": [1000], "sampling_params": {"max_new_tokens": 1, "ignore_eos": True}},
                    timeout=self.timeout,
                )
            else:
                r = await self.client.get(url + "/" + self.kind, timeout=self.timeout)
            st = str(r.status_code)
        except httpx.TimeoutException:
            st = "timeout"
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, OSError) as e:
            st = "conn_error:" + type(e).__name__
        except Exception as e:
            st = "error:" + type(e).__name__
        t1 = now_ns()
        self.n += 1
        self.last_latency_ms = (t1 - t0) / 1e6
        bad = st != "200"
        self.consecutive_bad = self.consecutive_bad + 1 if bad else 0
        if st != self.state:
            self.out.write({"ev": "probe", "kind": self.kind, "state": st, "prev": self.state,
                            "t_sent_ns": t0, "latency_ms": self.last_latency_ms, "n": self.n,
                            "consecutive_bad": self.consecutive_bad})
            self.state = st
        elif bad and self.consecutive_bad in (3, 10):
            self.out.write({"ev": "probe_sustained", "kind": self.kind, "state": st, "t_sent_ns": t0,
                            "latency_ms": self.last_latency_ms, "consecutive_bad": self.consecutive_bad})

    def kick(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._one())


async def amain(a):
    out = JsonlWriter(run_dir() / a.out)
    out.write({"ev": "probe_start", "interval": a.interval, "gen_timeout": a.gen_timeout,
               "health_timeout": a.health_timeout, "base_url": base_url()})
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    limits = httpx.Limits(max_connections=16)
    async with httpx.AsyncClient(limits=limits) as client:
        probes = [
            HttpProbe("health", out, client, a.health_timeout),
            HttpProbe("health_generate", out, client, a.health_timeout),
            HttpProbe("gen1", out, client, a.gen_timeout),
        ]
        pidw = PidWatch(out)
        tail = LogTailer(out)
        t_start = now_ns()
        last_hb = t_start
        while not stop.is_set():
            for p in probes:
                p.kick()
            try:
                pidw.poll()
                tail.poll()
            except Exception as e:
                out.write({"ev": "probe_error", "error": repr(e)})
            t = now_ns()
            if (t - last_hb) / 1e9 >= a.heartbeat:
                last_hb = t
                out.write({"ev": "heartbeat",
                           "probes": {p.kind: {"state": p.state, "latency_ms": p.last_latency_ms, "n": p.n} for p in probes},
                           "pids": dict(pidw.state)})
            if a.duration and (t - t_start) / 1e9 >= a.duration:
                break
            await asyncio.sleep(a.interval)
        out.write({"ev": "probe_end"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--gen-timeout", type=float, default=5.0, help="client timeout for the 1-token /generate probe")
    ap.add_argument("--health-timeout", type=float, default=60.0, help="client timeout for /health (server-side is 20 s)")
    ap.add_argument("--heartbeat", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until signal")
    ap.add_argument("--out", default="probe.jsonl")
    a = ap.parse_args()
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
