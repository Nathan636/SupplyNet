from pyspark.sql import SparkSession

# ================================================================
# Initialize Spark Session
# ================================================================
spark = SparkSession.builder \
    .appName("SupplyChainAnalytics") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

print("\n Spark session started successfully!\n")

# ================================================================
#  Load Cleaned Data from HDFS (Parquet)
# ================================================================
input_path = "hdfs://localhost:9000/user/hadoop/supply_chain_processed/"
df = spark.read.parquet(input_path)

print(f" Loaded cleaned data from: {input_path}\n")
df.show(5)

df.createOrReplaceTempView("supply_chain")

# ================================================================
# Supplier Performance Analysis
# ================================================================
print("\n Supplier Performance (avg delay & avg cost):")
spark.sql("""
    SELECT supplier_name,
           ROUND(AVG(delivery_delay_days), 2) AS avg_delay,
           ROUND(AVG(shipping_cost), 2) AS avg_shipping_cost,
           COUNT(*) AS total_orders
    FROM supply_chain
    GROUP BY supplier_name
    ORDER BY avg_delay DESC
""").show(10)

# ================================================================
#  Region-wise Delay & Cost Trends
# ================================================================
print("\n Regional Delivery Delay:")
spark.sql("""
    SELECT customer_age_group,
           ROUND(AVG(delivery_delay_days), 2) AS avg_delay,
           ROUND(AVG(shipping_cost), 2) AS avg_shipping_cost
    FROM supply_chain
    GROUP BY customer_age_group
    ORDER BY avg_delay DESC
""").show(10)

# ================================================================
#  Product Type vs Cost & Delay
# ================================================================
print("\n Product Type vs Cost & Delay:")
spark.sql("""
    SELECT product_type,
           ROUND(AVG(delivery_delay_days), 2) AS avg_delay,
           ROUND(AVG(price), 2) AS avg_price,
           ROUND(AVG(shipping_cost), 2) AS avg_ship_cost
    FROM supply_chain
    GROUP BY product_type
    ORDER BY avg_delay DESC
""").show(10)

# ================================================================
#  On-Time Delivery Rate
# ================================================================
print("\n On-Time Delivery Rate:")
spark.sql("""
    SELECT supplier_name,
           ROUND(SUM(CASE WHEN delivery_delay_days = 0 THEN 1 ELSE 0 END)*100.0/COUNT(*), 2) AS on_time_percentage
    FROM supply_chain
    GROUP BY supplier_name
    ORDER BY on_time_percentage DESC
""").show(10)

# ================================================================
# Save Analytics Results Back to HDFS
# ================================================================
supplier_perf = spark.sql("""
    SELECT supplier_name,
           ROUND(AVG(delivery_delay_days), 2) AS avg_delay,
           ROUND(AVG(shipping_cost), 2) AS avg_shipping_cost,
           COUNT(*) AS total_orders
    FROM supply_chain
    GROUP BY supplier_name
""")

output_path = "hdfs://localhost:9000/user/hadoop/supply_chain_analytics/supplier_performance"
supplier_perf.write.mode("overwrite").parquet(output_path)

print(f"\n Saved analytics output to: {output_path}")

# ================================================================
#  Stop Spark Session
# ================================================================
spark.stop()
print("\n Analytics job completed successfully!\n")
