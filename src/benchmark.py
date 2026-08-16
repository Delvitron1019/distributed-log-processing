"""Run the pipeline across core counts and emit a comparison table.

This is the file that turns the project from "I used Spark" into "here is what
Spark bought, and where it stopped buying anything". Scaling is never linear —
finding the point where it flattens, and being able to explain why, is the
interesting part.

Run:
    python src/benchmark.py --input data/raw --cores 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run_spark(input_dir: str, cores: str, shuffle_partitions: int) -> dict:
    out = tempfile.mkdtemp(prefix=f"bench_{cores}_")
    cmd = [sys.executable, os.path.join(HERE, "pipeline.py"),
           "--input", input_dir, "--output", out,
           "--cores", cores, "--shuffle-partitions", str(shuffle_partitions)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    summary_path = os.path.join(out, "run_summary.json")
    if proc.returncode != 0 or not os.path.exists(summary_path):
        shutil.rmtree(out, ignore_errors=True)
        raise RuntimeError(f"spark run failed (cores={cores})\n{proc.stderr[-2000:]}")

    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    summary["wall_seconds"] = round(wall, 2)   # includes JVM startup
    shutil.rmtree(out, ignore_errors=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--cores", nargs="+", default=["1", "2", "4", "8", "16"])
    ap.add_argument("--shuffle-partitions", type=int, default=16)
    ap.add_argument("--baseline", default=os.path.join(ROOT, "data", "agg_baseline",
                                                      "run_summary.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "benchmark.json"))
    args = ap.parse_args()

    results = []

    if os.path.exists(args.baseline):
        with open(args.baseline, encoding="utf-8") as fh:
            base = json.load(fh)
        results.append({"engine": "pandas (1 thread)", "cores": 1,
                        "seconds": base["total_seconds"],
                        "rows_per_second": base["rows_per_second"]})
        print(f"  pandas baseline           {base['total_seconds']:>7.2f}s")

    for c in args.cores:
        s = run_spark(args.input, c, args.shuffle_partitions)
        results.append({"engine": f"Spark local[{c}]", "cores": int(c),
                        "seconds": s["total_seconds"],
                        "wall_seconds": s["wall_seconds"],
                        "rows_per_second": s["rows_per_second"],
                        "stage_seconds": s["stage_seconds"]})
        print(f"  Spark local[{c:<2}]            {s['total_seconds']:>7.2f}s   "
              f"({s['rows_per_second']:,} rows/s)")

    # Speedups are quoted against the pandas baseline, since that is the
    # honest "what did this replace" comparison.
    if results and results[0]["engine"].startswith("pandas"):
        ref = results[0]["seconds"]
        for r in results:
            r["speedup_vs_pandas"] = round(ref / r["seconds"], 2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"rows": results}, fh, indent=2)

    print(f"\n{'Engine':<22}{'Seconds':>10}{'Rows/s':>14}{'Speedup':>10}")
    print("-" * 56)
    for r in results:
        print(f"{r['engine']:<22}{r['seconds']:>10.2f}{r['rows_per_second']:>14,}"
              f"{r.get('speedup_vs_pandas', 1):>9.2f}x")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
