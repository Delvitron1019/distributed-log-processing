"""Single-threaded pandas baseline computing the same aggregations.

This exists so the Spark numbers mean something. "Processes 1M logs" is a
description; "cut 4m12s to 38s" is a result, and you cannot claim the second
without running this first.

It is a fair comparison in the sense that it computes identical outputs. It is
an unfair comparison in the sense that pandas was never meant for this — which
is the point. The interesting question is where the crossover sits.

Run:
    python src/baseline.py --input data/raw --output data/agg_baseline
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import pandas as pd


def load(path: str) -> tuple[pd.DataFrame, int]:
    """Read every JSONL shard one line at a time, counting malformed rows."""
    frames, rejected = [], 0
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        # lines=True with errors would abort the whole file, so parse per line
        # and keep going — same tolerance the Spark reader has.
        rows = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    rejected += 1
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    rejected += 1
                    continue
                if rec.get("ts") is None or not isinstance(rec.get("status"), int):
                    rejected += 1
                    continue
                rows.append(rec)
        frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    df["is_error"] = (df["status"] >= 500).astype(int)
    df["is_client_error"] = ((df["status"] >= 400) & (df["status"] < 500)).astype(int)
    return df, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/raw")
    ap.add_argument("--output", default="data/agg_baseline")
    ap.add_argument("--window", default="1min")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    df, rejected = load(args.input)
    t_read = time.time() - t0
    print(f"  loaded {len(df):,} rows ({rejected:,} rejected) in {t_read:.1f}s")

    timings = {}

    t = time.time()
    endpoint = (df.groupby(["endpoint", "method"])
                  .agg(requests=("status", "size"),
                       errors=("is_error", "sum"),
                       mean_latency_ms=("latency_ms", "mean"),
                       p50_ms=("latency_ms", lambda s: s.quantile(0.50)),
                       p95_ms=("latency_ms", lambda s: s.quantile(0.95)),
                       p99_ms=("latency_ms", lambda s: s.quantile(0.99)),
                       mb_served=("bytes", lambda s: s.sum() / 1024 / 1024))
                  .reset_index())
    endpoint["error_rate_pct"] = 100 * endpoint["errors"] / endpoint["requests"]
    timings["endpoint_stats"] = round(time.time() - t, 2)

    t = time.time()
    timeline = (df.set_index("ts")
                  .groupby([pd.Grouper(freq=args.window), "service"])
                  .agg(requests=("status", "size"), errors=("is_error", "sum"))
                  .reset_index())
    timeline["error_rate_pct"] = 100 * timeline["errors"] / timeline["requests"]
    timings["error_timeline"] = round(time.time() - t, 2)

    t = time.time()
    health = (df.groupby(["service", "region"])
                .agg(requests=("status", "size"),
                     server_errors=("is_error", "sum"),
                     client_errors=("is_client_error", "sum"),
                     p95_ms=("latency_ms", lambda s: s.quantile(0.95)))
                .reset_index())
    timings["service_health"] = round(time.time() - t, 2)

    t = time.time()
    users = (df.groupby("user_id")
               .agg(requests=("status", "size"),
                    errors=("is_error", "sum"),
                    mb=("bytes", lambda s: s.sum() / 1024 / 1024))
               .reset_index()
               .sort_values("requests", ascending=False)
               .head(1000))
    timings["top_users"] = round(time.time() - t, 2)

    for name, frame in [("endpoint_stats", endpoint), ("error_timeline", timeline),
                        ("service_health", health), ("top_users", users)]:
        frame.to_parquet(os.path.join(args.output, f"{name}.parquet"), index=False)
        print(f"  {name:<16} {timings[name]:>6.2f}s")

    elapsed = time.time() - t0
    summary = {
        "engine":          "pandas (single-threaded)",
        "rows_parsed":     len(df),
        "rows_rejected":   rejected,
        "read_seconds":    round(t_read, 2),
        "total_seconds":   round(elapsed, 2),
        "rows_per_second": int(len(df) / elapsed) if elapsed else 0,
        "stage_seconds":   timings,
    }
    with open(os.path.join(args.output, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
