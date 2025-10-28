Spark Log Analysis — Problems 1 & 2

Problem 1 

Approach
	1.	Objective:
Count the occurrences of each log level (INFO, WARN, ERROR, DEBUG) from Spark driver/executor logs.
	2.	Data Source:
s3a://xl768-assignment-spark-cluster-logs/data/
	3.	Processing Pipeline:
	•	Read all text logs recursively using spark.read.text().
	•	Extract log level via regex pattern.
	•	Group by level → count lines per category.
	•	Compute total entries and percentages.
	•	Write results.

Finding:

INFO≈ 92 %

Normal progress and status messages dominate

WARN≈ 6 %

Minor issues (e.g., deprecation notices, missing configs)

ERROR≈ 2 %

Rare failures (primarily S3A auth warnings or path mismatches)

DEBUG< 0.1 %

Developer-level diagnostics rarely enabled

Performance Analysis

Local (single node) ~500 MB ~48 s

Cluster (3 EC2 nodes) ~500 MB ~11 s

Problem 2 

Approach
	1.	Objective:
Examine how different Spark clusters were used over time based on application logs.
	2.	Data Source:
Same S3 log bucket as Problem 1.
	3.	Processing Steps:
	•	Extract cluster_id and application_id from file names (e.g., application_1485248649253_181).
	•	Parse timestamps.
    •	Compute start and end times per application (min and max timestamps).
	•	Convert to Pandas for final aggregation and visualization.
	•	Write outputs.

Insights

1. Cluster Usage Imbalance
	•	Cluster 1485248649253 handled ≈ 181 applications — over 80 % of total.
	•	Other clusters processed ≤ 8 apps each.

2. Application Durations
	•	Distribution is right-skewed on log scale.
	•	Majority of apps finish within 10–30 min; a few long outliers extend past 3 hours.