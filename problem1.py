from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, rand, count
import sys

def main(input_path: str, output_path: str):
    spark = SparkSession.builder.appName("Problem1_LogLevelDistribution").getOrCreate()

    # === 1. Read all text files recursively ===
    print(f"Reading log files from: {input_path}")
    logs_df = spark.read.text(f"{input_path}/**")
    total_lines = logs_df.count()

    # === 2. Extract log level using regex ===
    # Typical Spark log line format: "17/03/29 10:04:41 INFO ApplicationMaster: ..."
    pattern = r"\b(INFO|WARN|ERROR|DEBUG)\b"
    logs_with_level = logs_df.withColumn("log_level", regexp_extract(col("value"), pattern, 1))

    # Filter only rows that actually contain log levels
    logs_filtered = logs_with_level.filter(col("log_level") != "")
    matched_lines = logs_filtered.count()

    # === 3. Count occurrences of each log level ===
    level_counts = logs_filtered.groupBy("log_level").agg(count("*").alias("count")).orderBy("log_level")

    # === 4. Take 10 random samples ===
    samples = logs_filtered.orderBy(rand()).limit(10).select(col("value").alias("log_entry"), "log_level")

    # === 5. Write results ===
    (level_counts
        .coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(f"{output_path}/problem1_counts.csv"))

    (samples
        .coalesce(1)
        .write.mode("overwrite")
        .option("header", "true")
        .csv(f"{output_path}/problem1_sample.csv"))

    # === 6. Summary text ===
    levels = [r["log_level"] for r in level_counts.collect()]
    counts = [r["count"] for r in level_counts.collect()]
    total_with_level = sum(counts)
    summary_lines = [
        f"Total log lines processed: {total_lines:,}",
        f"Total lines with log levels: {matched_lines:,}",
        f"Unique log levels found: {len(levels)}",
        "",
        "Log level distribution:",
    ]
    for lvl, cnt in zip(levels, counts):
        pct = cnt / matched_lines * 100
        summary_lines.append(f"  {lvl:<6}: {cnt:>10,} ({pct:5.2f}%)")

    summary_text = "\n".join(summary_lines)

    (spark.sparkContext.parallelize([summary_text], 1)
        .saveAsTextFile(f"{output_path}/problem1_summary.txt"))

    print("\n=== Summary ===")
    print(summary_text)

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit problem1.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    main(input_path, output_path)