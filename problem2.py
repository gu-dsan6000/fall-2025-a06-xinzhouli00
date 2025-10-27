#!/usr/bin/env python3
import argparse, os, re, shutil
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, min as spark_min, max as spark_max

# -----------------------
# Argument Parsing
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("master", nargs="?", default=None)
    p.add_argument("--net-id", required=False, default="testnet")
    p.add_argument("--skip-spark", action="store_true")
    return p.parse_args()


# -----------------------
# Utility Functions
# -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clean_out(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


# -----------------------
# Spark Main Analysis
# -----------------------
def run_spark(master_url: str, net_id: str):
    print(f"Running Spark job on: {master_url}")

    spark = (
        SparkSession.builder
        .appName(f"Problem2_ClusterUsage_{net_id}")
        .master(master_url)
        .config(
            "spark.driver.extraClassPath",
            "/home/ubuntu/spark/jars/hadoop-aws-3.4.1.jar:/home/ubuntu/spark/jars/hadoop-common-3.4.1.jar"
        )
    .config("spark.jars.packages",
        ",".join([
            "org.apache.hadoop:hadoop-aws:3.3.6",
            "software.amazon.awssdk:bundle:2.25.64",
            "software.amazon.awssdk:auth:2.25.64",
            "software.amazon.awssdk:core:2.25.64",
            "software.amazon.awssdk:sdk-core:2.25.64",
            "software.amazon.awssdk:sts:2.25.64",
            "software.amazon.awssdk:s3:2.25.64",
            "software.amazon.awssdk:aws-core:2.25.64",
            "software.amazon.awssdk:protocol-core:2.25.64",
            "software.amazon.awssdk:aws-xml-protocol:2.25.64",
            "software.amazon.awssdk:aws-query-protocol:2.25.64",
            "software.amazon.awssdk:url-connection-client:2.25.64",
            "com.fasterxml.woodstox:woodstox-core:6.4.0",
            "org.codehaus.woodstox:stax2-api:4.2.1"
        ])
    ) \
    .config("spark.executor.extraClassPath", "/home/ubuntu/.ivy2.5.2/jars/*") \
    .config("spark.driver.extraClassPath", "/home/ubuntu/.ivy2.5.2/jars/*") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .config("spark.hadoop.fs.s3a.connection.maximum", "100") \
    .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000") \
    .config("spark.hadoop.fs.s3a.attempts.maximum", "20") \
    .config("spark.hadoop.fs.s3a.retry.limit", "15")
    .config(
        "spark.driver.extraClassPath",
        "/home/ubuntu/spark/jars/woodstox-core-asl-4.4.1.jar"
    )
    .config(
        "spark.executor.extraClassPath",
        "/home/ubuntu/spark/jars/woodstox-core-asl-4.4.1.jar"
    )

        # --- ☁️ S3A Configuration ---
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.InstanceProfileCredentialsProvider,"
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"
        )

        # --- 🩹 Fix Hadoop '60s' parsing issue ---
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "20")
        .config("spark.hadoop.fs.s3a.retry.limit", "15")
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")

        # --- ⚙️ Performance tuning ---
        .config("spark.hadoop.fs.s3a.paging.maximum", "1000")
        .config("spark.hadoop.fs.s3a.threads.max", "64")
        .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
        .config("spark.hadoop.fs.s3a.multipart.threshold", "209715200")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")

        # --- 🧠 Serialization & execution ---
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")

        # --- 🪶 Cluster behavior ---
        .config("spark.speculation", "true")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.default.parallelism", "8")

        # --- 🧭 Logging ---
        .config("spark.ui.showConsoleProgress", "true")

        .getOrCreate()
    )

    # --- Determine input/output paths ---
    if "spark://" in master_url:
        input_path = f"s3a://{net_id}-assignment-spark-cluster-logs/data/"
        output_path = f"s3a://{net_id}-assignment-spark-cluster-logs/output/"
    else:
        input_path = "data/raw/"
        output_path = "data/output/"
        ensure_dir(output_path)

    print(f"Reading logs from: {input_path}")

    # --- Read logs from S3 ---
    logs = spark.read.option("recursiveFileLookup", "true").text(input_path)

    # --- Extract fields ---
    logs = (
        logs.withColumn("cluster_id", regexp_extract(col("input_file_name()"), r"application_(\d+)_", 1))
            .withColumn("application_id", regexp_extract(col("input_file_name()"), r"(application_\d+_\d+)", 1))
            .filter(col("application_id") != "")
    )

    # --- Extract timestamps ---
    pattern_time = r"(\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})"
    logs = logs.withColumn("timestamp_str", regexp_extract(col("value"), pattern_time, 1))
    logs = logs.filter(col("timestamp_str") != "")

    # --- Compute per-application start and end times ---
    app_times = (
        logs.groupBy("cluster_id", "application_id")
            .agg(spark_min("timestamp_str").alias("start_str"),
                 spark_max("timestamp_str").alias("end_str"))
    )

    # --- Convert to Pandas ---
    pdf = app_times.toPandas()
    pdf["start_time"] = pd.to_datetime(pdf["start_str"], format="%y/%m/%d %H:%M:%S", errors="coerce")
    pdf["end_time"]   = pd.to_datetime(pdf["end_str"], format="%y/%m/%d %H:%M:%S", errors="coerce")

    # --- Sort & assign application numbers ---
    pdf = pdf.sort_values(["cluster_id", "start_time"])
    pdf["app_number"] = pdf.groupby("cluster_id").cumcount().add(1).astype(str).str.zfill(4)

    # --- Write outputs ---
    timeline_path = os.path.join(output_path, "problem2_timeline.csv")
    cluster_summary_path = os.path.join(output_path, "problem2_cluster_summary.csv")
    stats_path = os.path.join(output_path, "problem2_stats.txt")

    print(f"Saving results to: {output_path}")

    pdf[["cluster_id", "application_id", "app_number", "start_time", "end_time"]].to_csv(timeline_path, index=False)

    cluster_summary = (
        pdf.groupby("cluster_id")
           .agg(num_applications=("application_id", "count"),
                cluster_first_app=("start_time", "min"),
                cluster_last_app=("end_time", "max"))
           .reset_index()
    )
    cluster_summary.to_csv(cluster_summary_path, index=False)

    # --- Write text summary ---
    total_clusters = cluster_summary.shape[0]
    total_apps = pdf.shape[0]
    avg_apps = total_apps / total_clusters if total_clusters else 0

    lines = [
        f"Total unique clusters: {total_clusters}",
        f"Total applications: {total_apps}",
        f"Average applications per cluster: {avg_apps:.2f}",
        "",
        "Most heavily used clusters:"
    ]
    top_clusters = cluster_summary.sort_values("num_applications", ascending=False).head(5)
    for _, r in top_clusters.iterrows():
        lines.append(f"  Cluster {r['cluster_id']}: {r['num_applications']} applications")

    with open(stats_path, "w") as f:
        f.write("\n".join(lines))

    spark.stop()
    print("✅ Spark job completed successfully.")
    return timeline_path, cluster_summary_path, stats_path


# -----------------------
# Visualization
# -----------------------
def generate_visualizations(timeline_csv, cluster_csv, output_dir):
    sns.set_theme(style="whitegrid")
    timeline = pd.read_csv(timeline_csv, parse_dates=["start_time", "end_time"])
    clusters = pd.read_csv(cluster_csv)

    # --- Bar Chart ---
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="cluster_id", y="num_applications", data=clusters, palette="crest")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", label_type="edge", fontsize=8)
    plt.title("Number of Applications per Cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "problem2_bar_chart.png"))
    plt.close()

    # --- Duration Distribution ---
    largest_cluster = clusters.sort_values("num_applications", ascending=False).iloc[0]["cluster_id"]
    largest_df = timeline.query("cluster_id == @largest_cluster").copy()
    largest_df["duration_min"] = (largest_df["end_time"] - largest_df["start_time"]).dt.total_seconds() / 60

    plt.figure(figsize=(7, 5))
    sns.histplot(largest_df["duration_min"], kde=True, log_scale=True)
    plt.xlabel("Application Duration (minutes, log scale)")
    plt.ylabel("Count")
    plt.title(f"Duration Distribution – Cluster {largest_cluster} (n={len(largest_df)})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "problem2_density_plot.png"))
    plt.close()


# -----------------------
# Main Entry Point
# -----------------------
def main():
    args = parse_args()
    output_dir = "data/output/"
    ensure_dir(output_dir)

    if args.skip_spark:
        print("Skipping Spark – using existing CSV files for visualization…")
        generate_visualizations(
            "data/output/problem2_timeline.csv",
            "data/output/problem2_cluster_summary.csv",
            output_dir
        )
        print("✅ Visualizations regenerated successfully.")
    else:
        print("Running full Spark analysis … this may take 10–20 minutes on the cluster.")
        timeline_csv, cluster_csv, stats_txt = run_spark(args.master, args.net_id)
        generate_visualizations(timeline_csv, cluster_csv, output_dir)
        print("✅ Problem 2 complete! All outputs written to data/output/")

if __name__ == "__main__":
    main()