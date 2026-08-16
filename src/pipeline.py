"""Spark pipeline: ingest raw event logs, aggregate, write analytics tables.

Four aggregations, chosen because each stresses the engine differently:

  1. endpoint_stats   — groupBy with percentiles. Wide shuffle, skewed keys.
  2. error_timeline   — time-windowed error rate. Finds the incident windows.
  3. service_health   — per-service rollup. Small output, cheap.
  4. top_users        — high-cardinality groupBy. The one that actually hurts.

Run:
    python src/pipeline.py --input data/raw --output data/agg --shuffle-partitions 16
"""

from __future__ import annotations

import argparse
import json
import os
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (IntegerType, LongType, StringType, StructField,
                               StructType, DoubleType)

# An explicit schema rather than inferring it. Inference costs an extra full
# pass over the data, and it silently changes types when a new file arrives
# with a null column — the classic way a working pipeline breaks overnight.
LOG_SCHEMA = StructType([
    StructField("ts",         StringType(),  True),
    StructField("service",    StringType(),  True),
    StructField("endpoint",   StringType(),  True),
    StructField("method",     StringType(),  True),
    StructField("status",     IntegerType(), True),
    StructField("latency_ms", DoubleType(),  True),
    StructField("bytes",      LongType(),    True),
    StructField("region",     StringType(),  True),
    StructField("user_id",    StringType(),  True),
    StructField("user_agent", StringType(),  True),
])


def build_session(shuffle_partitions: int, cores: str, name: str) -> SparkSession:
    return (SparkSession.builder
            .appName(name)
            .master(f"local[{cores}]")
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def read_logs(spark: SparkSession, path: str):
    """Read raw JSONL, separating good rows from malformed ones.

    PERMISSIVE mode plus an explicit corrupt-record column means bad lines are
    counted and quarantined rather than silently dropped. Knowing your reject
    rate is the difference between a pipeline you can trust and one you hope
    about.
    """
    schema = StructType(LOG_SCHEMA.fields + [StructField("_corrupt", StringType(), True)])

    # Count physical lines separately. Spark's JSON reader silently skips blank
    # lines — it never emits a record for them, so they are invisible to the
    # corrupt-record path. The pandas baseline counts them as rejects, and the
    # two totals disagreed by ~1,000 until this was added. Reconciling reader
    # counts against raw line counts is the only way to know nothing vanished.
    physical_lines = spark.read.text(path).count()

    raw = (spark.read
           .option("mode", "PERMISSIVE")
           .option("columnNameOfCorruptRecord", "_corrupt")
           .schema(schema)
           .json(path))

    # Cache: every downstream aggregation scans this, and without caching Spark
    # re-reads and re-parses the JSON for each one.
    raw.cache()

    bad  = raw.filter(F.col("_corrupt").isNotNull() | F.col("ts").isNull() | F.col("status").isNull())
    good = (raw.filter(F.col("_corrupt").isNull() & F.col("ts").isNotNull() & F.col("status").isNotNull())
               .drop("_corrupt")
               .withColumn("ts", F.to_timestamp("ts"))
               .withColumn("is_error", (F.col("status") >= 500).cast("int"))
               .withColumn("is_client_error",
                           ((F.col("status") >= 400) & (F.col("status") < 500)).cast("int")))
    return good, bad, physical_lines


def endpoint_stats(df):
    """Per-endpoint volume, error rate, and latency percentiles.

    percentile_approx rather than an exact percentile: exact requires a full
    sort per group, approximate is a single pass with bounded error. On skewed
    traffic that is the difference between seconds and minutes.
    """
    return (df.groupBy("endpoint", "method")
              .agg(F.count("*").alias("requests"),
                   F.sum("is_error").alias("errors"),
                   F.round(F.avg("latency_ms"), 2).alias("mean_latency_ms"),
                   F.round(F.expr("percentile_approx(latency_ms, 0.50)"), 2).alias("p50_ms"),
                   F.round(F.expr("percentile_approx(latency_ms, 0.95)"), 2).alias("p95_ms"),
                   F.round(F.expr("percentile_approx(latency_ms, 0.99)"), 2).alias("p99_ms"),
                   F.round(F.sum("bytes") / 1024 / 1024, 1).alias("mb_served"))
              .withColumn("error_rate_pct",
                          F.round(100 * F.col("errors") / F.col("requests"), 3))
              .orderBy(F.desc("requests")))


def error_timeline(df, window: str = "1 minute"):
    """Error rate per service per time window — this is what surfaces incidents."""
    return (df.groupBy(F.window("ts", window).alias("w"), "service")
              .agg(F.count("*").alias("requests"),
                   F.sum("is_error").alias("errors"))
              .withColumn("error_rate_pct",
                          F.round(100 * F.col("errors") / F.col("requests"), 2))
              .select(F.col("w.start").alias("window_start"), "service",
                      "requests", "errors", "error_rate_pct")
              .orderBy(F.desc("error_rate_pct")))


def service_health(df):
    return (df.groupBy("service", "region")
              .agg(F.count("*").alias("requests"),
                   F.sum("is_error").alias("server_errors"),
                   F.sum("is_client_error").alias("client_errors"),
                   F.round(F.expr("percentile_approx(latency_ms, 0.95)"), 2).alias("p95_ms"))
              .orderBy(F.desc("requests")))


def top_users(df, limit: int = 1000):
    """High-cardinality groupBy — ~120k distinct keys. The expensive one."""
    return (df.groupBy("user_id")
              .agg(F.count("*").alias("requests"),
                   F.sum("is_error").alias("errors"),
                   F.round(F.sum("bytes") / 1024 / 1024, 2).alias("mb"))
              .orderBy(F.desc("requests"))
              .limit(limit))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="data/raw")
    ap.add_argument("--output", default="data/agg")
    ap.add_argument("--shuffle-partitions", type=int, default=16)
    ap.add_argument("--cores", default="*", help='Spark local cores, e.g. "4" or "*"')
    ap.add_argument("--window", default="1 minute")
    args = ap.parse_args()

    spark = build_session(args.shuffle_partitions, args.cores, "log-pipeline")
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.time()
    good, bad, physical_lines = read_logs(spark, args.input)

    total_good = good.count()
    total_bad  = bad.count()
    # Lines the JSON reader never emitted a record for at all (blank lines).
    skipped_by_reader = physical_lines - (total_good + total_bad)
    t_read = time.time() - t0

    os.makedirs(args.output, exist_ok=True)
    timings = {}

    for name, frame in [
        ("endpoint_stats", endpoint_stats(good)),
        ("error_timeline", error_timeline(good, args.window)),
        ("service_health", service_health(good)),
        ("top_users",      top_users(good)),
    ]:
        t = time.time()
        frame.write.mode("overwrite").parquet(os.path.join(args.output, name))
        timings[name] = round(time.time() - t, 2)
        print(f"  {name:<16} {timings[name]:>6.2f}s")

    elapsed = time.time() - t0

    summary = {
        "physical_lines":    physical_lines,
        "rows_parsed":       total_good,
        "rows_rejected":     total_bad,
        "skipped_by_reader": skipped_by_reader,
        "unaccounted":       physical_lines - total_good - total_bad - skipped_by_reader,
        "reject_rate_pct":   round(100 * (total_bad + skipped_by_reader) / max(physical_lines, 1), 3),
        "read_seconds":      round(t_read, 2),
        "total_seconds":     round(elapsed, 2),
        "rows_per_second":   int(total_good / elapsed) if elapsed else 0,
        "shuffle_partitions": args.shuffle_partitions,
        "cores":             args.cores,
        "stage_seconds":     timings,
    }

    with open(os.path.join(args.output, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + json.dumps(summary, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
