#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, round as spark_round

# ---------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Run log analysis on Spark cluster")
    p.add_argument("--net-id", required=True, help="Your Georgetown NetID (used for S3 paths)")
    return p.parse_args()


# ---------------------------------------------------------
# Main Spark Analysis
# ---------------------------------------------------------
def run_spark(master_url: str, net_id: str):
    print(f"Running Spark job on: {master_url}")

    spark = (
            SparkSession.builder
            .appName("Problem1_LogLevelDist")

            # Cluster Configuration
            .master(master_url)  # Connect to Spark cluster

            # Memory Configuration
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.driver.maxResultSize", "2g")

            # Executor Configuration
            .config("spark.executor.cores", "2")
            .config("spark.cores.max", "6")  # Use all available cores across cluster

            # S3 Configuration - Use S3A for AWS S3 access
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

            # Performance settings for cluster execution
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")

            # Serialization
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

            # Arrow optimization for Pandas conversion
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")

            .getOrCreate()
        )

    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------
    start_time = time.time()
    input_path = f"s3a://{net_id}-assignment-spark-cluster-logs/data/"
    output_dir = os.path.expanduser("~/spark-cluster")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Reading raw log data from: {input_path}")

    print("\nReading raw log data...")
    print("=" * 60)
    print(f"Input path: {input_path}")
    df = spark.read.text(input_path + "/**/*")
    df = df.withColumnRenamed("value", "log_entry")
    df.show(5, truncate=False)


    # -----------------------------------------------------
    # Analysis
    # -----------------------------------------------------
    print("Computing trip counts per passenger count...")
    counts_df = df.groupBy("passenger_count").count().orderBy("passenger_count")
    counts_out = os.path.join(output_dir, "problem1_counts.csv")
    counts_df.coalesce(1).write.mode("overwrite").option("header", True).csv(counts_out)

    print("Sampling 100 trips for inspection...")
    sample_df = df.sample(fraction=0.001).limit(100)
    sample_out = os.path.join(output_dir, "problem1_sample.csv")
    sample_df.coalesce(1).write.mode("overwrite").option("header", True).csv(sample_out)

    print("Computing average fare by passenger count...")
    summary_df = (
        df.groupBy("passenger_count")
        .agg(spark_round(avg("fare_amount"), 2).alias("avg_fare"),
             spark_round(avg("trip_distance"), 2).alias("avg_distance"))
        .orderBy("passenger_count")
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    summary_out = os.path.join(output_dir, "problem1_summary.txt")

    # Convert to Pandas for easy writing
    pdf = summary_df.toPandas()
    with open(summary_out, "w") as f:
        f.write("Summary\n")
        f.write("=====================\n")
        f.write(f"Total rows: {df.count()}\n\n")
        f.write(pdf.to_string(index=False))
        f.write("\n")

    print("Problem 1 complete!")
    print(f"Outputs written to: {output_dir}")

    spark.stop()


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    run_spark(args.master, args.net_id)