from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when, round

#  Initialize Spark Session
spark = SparkSession.builder \
    .appName("SupplyChainDataCleaning") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

print("\n Spark session started successfully!\n")

#  Load Raw Data from HDFS
input_path = "hdfs://localhost:9000/user/hadoop/supply_chain_input/supply_chain_data.txt"
print(f" Reading data from: {input_path}\n")

df = spark.read.csv(input_path, header=True, inferSchema=True)
print(" Data successfully loaded. Showing sample rows:")
df.show(5)

#  Data Cleaning + Transformation
print("\n Cleaning data and calculating delivery delay...\n")

cleaned_df = (
    df.select([trim(col(c)).alias(c) for c in df.columns])  # Trim spaces
      # Convert numeric columns safely
      .withColumn("lead_time_days", col("lead_time_days").cast("int"))
      .withColumn("shipping_time_days", col("shipping_time_days").cast("int"))
      .withColumn("price", col("price").cast("double"))
      .withColumn("shipping_cost", col("shipping_cost").cast("double"))
      # Calculate delivery delay
      .withColumn(
          "delivery_delay_days",
          when(col("shipping_time_days") > col("lead_time_days"),
               col("shipping_time_days") - col("lead_time_days"))
          .otherwise(0)
      )
      # Clean negative prices or costs
      .withColumn("price", when(col("price") < 0, 0).otherwise(col("price")))
      .withColumn("shipping_cost", when(col("shipping_cost") < 0, 0).otherwise(col("shipping_cost")))
      # Fill missing supplier names
      .withColumn("supplier_name", when(col("supplier_name").isNull(), "Unknown").otherwise(col("supplier_name")))
      # Handle missing coordinates
      .withColumn("customer_location_lat", when(col("customer_location_lat").isNull(), 0).otherwise(col("customer_location_lat")))
      .withColumn("customer_location_lon", when(col("customer_location_lon").isNull(), 0).otherwise(col("customer_location_lon")))
      .withColumn("supplier_location_lat", when(col("supplier_location_lat").isNull(), 0).otherwise(col("supplier_location_lat")))
      .withColumn("supplier_location_lon", when(col("supplier_location_lon").isNull(), 0).otherwise(col("supplier_location_lon")))
)

print(" Cleaned Data Preview:")
cleaned_df.select("product_type", "lead_time_days", "shipping_time_days", "delivery_delay_days", "price").show(10)

# Write Processed Data to HDFS (Parquet Format)
output_path = "hdfs://localhost:9000/user/hadoop/supply_chain_processed/"
cleaned_df.write.mode("overwrite").parquet(output_path)
print(f" Cleaned data successfully saved to: {output_path}")

#  Run Simple Spark SQL Query
cleaned_df.createOrReplaceTempView("supply_chain")

print("\n Average Delivery Delay by Product Type:")
spark.sql("""
    SELECT product_type, 
           ROUND(AVG(delivery_delay_days), 2) AS avg_delay,
           ROUND(AVG(price), 2) AS avg_price
    FROM supply_chain
    GROUP BY product_type
    ORDER BY avg_delay DESC
""").show(10)

spark.stop()
print("\n Job completed successfully!\n")

