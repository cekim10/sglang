"""Shared helpers for the ft-detect harness.

One clock for every process on the host: time.monotonic_ns() (CLOCK_MONOTONIC on
Linux, system-wide, unaffected by NTP steps). Every record also carries a wall
clock stamp for cross-referencing against server logs, which print wall time.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent


def now_ns() -> int:
    return time.monotonic_ns()


def wall() -> float:
    return time.time()


def run_dir() -> Path:
    """Directory holding this run's logs. Overridable via FT_RUN_DIR."""
    return Path(os.environ.get("FT_RUN_DIR", HERE / "run")).resolve()


def logs_dir() -> Path:
    return run_dir() / "logs"


def ensure_run_dirs() -> Path:
    d = run_dir()
    (d / "logs").mkdir(parents=True, exist_ok=True)
    (d / "pids").mkdir(parents=True, exist_ok=True)
    return d


class JsonlWriter:
    """Append-only JSONL writer, one flush per record so a killed process loses nothing."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)

    def write(self, rec: Dict[str, Any]) -> None:
        if "t_ns" not in rec:
            rec["t_ns"] = now_ns()
        if "wall" not in rec:
            rec["wall"] = wall()
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def read_jsonl(path: Path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A record being written when the process died. Skip it.
                continue
    return out


def load_pids() -> Optional[Dict[str, Any]]:
    p = run_dir() / "pids.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def base_url() -> str:
    return os.environ.get("FT_BASE_URL", "http://127.0.0.1:%s" % os.environ.get("FT_PORT", "30000"))


def eprint(*a, **k):
    print(*a, file=sys.stderr, flush=True, **k)
