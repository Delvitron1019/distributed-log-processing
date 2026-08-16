"""Generate synthetic web-server event logs.

The point of this file is reproducibility: anyone who clones the repo can
produce the same dataset the benchmarks were run on, without needing access to
real traffic logs.

The generated data deliberately looks like real traffic rather than uniform
noise, because uniform data hides the problem this pipeline exists to solve:

  * endpoint popularity follows a Zipf distribution, so a handful of routes
    carry most of the traffic. That creates partition skew, which is the
    interesting engineering problem here.
  * latency is log-normal with a long tail, so percentiles are meaningful and
    the mean is a bad summary.
  * error rates spike during defined incident windows rather than staying flat,
    so anomaly detection has something to find.
  * a small fraction of lines are malformed, because every real log corpus has
    them and a pipeline that assumes otherwise breaks in production.

Usage:
    python generate_logs.py --count 1000000 --out data/raw
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from datetime import datetime, timedelta

# Endpoints ordered most- to least-popular; Zipf weights are applied by index.
ENDPOINTS = [
    "/api/v1/feed", "/api/v1/search", "/api/v1/user/profile", "/api/v1/auth/login",
    "/api/v1/messages", "/api/v1/notifications", "/api/v1/upload", "/api/v1/orders",
    "/api/v1/orders/checkout", "/api/v1/recommendations", "/api/v1/settings",
    "/api/v1/analytics/events", "/health", "/metrics",
]

METHODS  = ["GET", "GET", "GET", "GET", "POST", "POST", "PUT", "DELETE"]
REGIONS  = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1", "ap-northeast-1"]
SERVICES = ["gateway", "feed-svc", "search-svc", "auth-svc", "media-svc", "orders-svc"]
AGENTS   = ["Chrome/122.0", "Safari/17.3", "Firefox/123.0", "Edge/121.0",
            "okhttp/4.12.0", "python-requests/2.31.0", "curl/8.5.0"]

# Windows (as a fraction through the run) where the error rate is elevated.
INCIDENTS = [(0.31, 0.36, "feed-svc"), (0.68, 0.71, "orders-svc")]


def zipf_weights(n: int, s: float = 1.1) -> list[float]:
    raw = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(raw)
    return [r / total for r in raw]


def pick_status(progress: float, service: str, rng: random.Random) -> int:
    """Status code, elevated toward errors inside an incident window."""
    in_incident = any(lo <= progress <= hi and svc == service for lo, hi, svc in INCIDENTS)
    roll = rng.random()
    if in_incident:
        if roll < 0.34: return rng.choice([500, 502, 503, 504])
        if roll < 0.42: return 429
    if roll < 0.010: return rng.choice([500, 502, 503])
    if roll < 0.020: return rng.choice([400, 401, 403, 404])
    if roll < 0.026: return 301
    return 200


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=1_000_000, help="number of log lines")
    ap.add_argument("--out", default="data/raw", help="output directory")
    ap.add_argument("--parts", type=int, default=8, help="split across N files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--malformed", type=float, default=0.004,
                    help="fraction of lines written deliberately broken")
    ap.add_argument("--gzip", action="store_true", help="write .jsonl.gz instead of .jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    weights = zipf_weights(len(ENDPOINTS))
    start = datetime(2026, 3, 1, 0, 0, 0)
    per_part = args.count // args.parts

    print(f"Generating {args.count:,} log lines across {args.parts} files -> {args.out}")
    t0 = time.time()
    written = malformed_written = 0

    for part in range(args.parts):
        suffix = "jsonl.gz" if args.gzip else "jsonl"
        path = os.path.join(args.out, f"events-{part:03d}.{suffix}")
        opener = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if args.gzip \
            else (lambda p: open(p, "w", encoding="utf-8", buffering=1 << 20))

        n = per_part if part < args.parts - 1 else args.count - per_part * (args.parts - 1)

        with opener(path) as fh:
            for i in range(n):
                progress = (part * per_part + i) / max(args.count, 1)

                # A few lines are written broken on purpose. A pipeline that
                # cannot survive these is not a pipeline.
                if rng.random() < args.malformed:
                    fh.write(rng.choice([
                        "{not valid json at all",
                        '{"ts": "2026-03-01T00:00:00Z", "endpoint": ',
                        "",
                        '{"ts": null, "status": "???", "latency_ms": "NaN"}',
                    ]) + "\n")
                    malformed_written += 1
                    written += 1
                    continue

                endpoint = rng.choices(ENDPOINTS, weights=weights, k=1)[0]
                service  = SERVICES[ENDPOINTS.index(endpoint) % len(SERVICES)]
                status   = pick_status(progress, service, rng)

                # Log-normal latency: most requests fast, a long tail that makes
                # p95 and p99 tell a different story from the mean.
                latency = rng.lognormvariate(3.4, 0.9)
                if status >= 500:
                    latency *= rng.uniform(2.0, 6.0)   # failures are slow

                ts = start + timedelta(seconds=(part * per_part + i) * 0.05,
                                       milliseconds=rng.randint(0, 999))

                fh.write(json.dumps({
                    "ts": ts.isoformat() + "Z",
                    "service": service,
                    "endpoint": endpoint,
                    "method": rng.choice(METHODS),
                    "status": status,
                    "latency_ms": round(latency, 2),
                    "bytes": rng.randint(180, 260_000),
                    "region": rng.choice(REGIONS),
                    "user_id": f"u{rng.randint(1, 120_000)}",
                    "user_agent": rng.choice(AGENTS),
                }) + "\n")
                written += 1

        print(f"  {os.path.basename(path)}  {n:,} lines")

    size_mb = sum(os.path.getsize(os.path.join(args.out, f))
                  for f in os.listdir(args.out)) / 1024 / 1024
    print(f"\nDone in {time.time() - t0:.1f}s — {written:,} lines, "
          f"{malformed_written:,} malformed ({malformed_written / written:.2%}), "
          f"{size_mb:.0f} MB on disk")


if __name__ == "__main__":
    main()
