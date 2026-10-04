"""One deterministic generation request (fixed prompt, seed, shape, steps) and a record of the result.

Used by portability.sh so every configuration (SP=2 save, SP=1 reference, SP=1 resumed)
runs the identical request. Writes <out>: latency, HTTP status, sha256 of the returned
image bytes / video content, and the job id.

    python one_request.py --shape 832x480x81 --steps 9 --seed 1234 --out resp.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import base_url, now_ns  # noqa: E402

PROMPT = "a red fox running through fresh snow, cinematic lighting, high detail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, help="WxH or WxHxF")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=1800.0)
    a = ap.parse_args()
    dims = [int(x) for x in a.shape.lower().split("x")]
    w, h = dims[0], dims[1]
    frames = dims[2] if len(dims) > 2 else 1
    base = base_url().rstrip("/")
    rec = {"shape": a.shape, "steps": a.steps, "seed": a.seed, "prompt": a.prompt}
    t0 = now_ns()
    with httpx.Client(timeout=httpx.Timeout(connect=10, read=a.timeout, write=60, pool=None), headers={"Connection": "close"}) as c:
        if frames > 1:
            r = c.post(base + "/v1/videos", json={"prompt": a.prompt, "size": f"{w}x{h}", "n": 1, "num_frames": frames,
                                                   "num_inference_steps": a.steps, "seed": a.seed})
            rec["status"] = r.status_code
            if r.status_code == 200:
                job = r.json()
                rec["job_id"] = job.get("id")
                while job.get("status") not in ("completed", "failed"):
                    time.sleep(0.5)
                    job = c.get(f"{base}/v1/videos/{rec['job_id']}").json()
                rec["job_status"] = job.get("status")
                if job.get("status") == "completed":
                    content = c.get(f"{base}/v1/videos/{rec['job_id']}/content")
                    rec["content_status"] = content.status_code
                    rec["sha256"] = hashlib.sha256(content.content).hexdigest()
                    rec["bytes"] = len(content.content)
                else:
                    rec["error"] = str(job.get("error") or job)[:300]
            else:
                rec["error"] = r.text[:300]
        else:
            r = c.post(base + "/v1/images/generations", json={"prompt": a.prompt, "width": w, "height": h, "n": 1,
                                                               "num_inference_steps": a.steps, "seed": a.seed,
                                                               "response_format": "b64_json"})
            rec["status"] = r.status_code
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data and data[0].get("b64_json"):
                    raw = base64.b64decode(data[0]["b64_json"])
                    rec["sha256"] = hashlib.sha256(raw).hexdigest()
                    rec["bytes"] = len(raw)
                else:
                    rec["error"] = "no b64 image in response"
            else:
                rec["error"] = r.text[:300]
    rec["latency_s"] = (now_ns() - t0) / 1e9
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec))
    return 0 if rec.get("sha256") else 1


if __name__ == "__main__":
    sys.exit(main())
