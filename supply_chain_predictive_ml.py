from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# -----------------------------
#  Spark Session
# -----------------------------
spark = SparkSession.builder \
    .appName("SupplyChainPredictiveAnalytics_Optimized") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# -----------------------------
# Read Data
# -----------------------------
input_path = "hdfs://localhost:9000/user/hadoop/supply_chain_processed/"
df = spark.read.parquet(input_path)

# -----------------------------
# Numeric Conversion
# -----------------------------
numeric_cols = [
    "price", "shipping_cost", "lead_time_days", "order_quantity",
    "stock_level", "manufacturing_cost", "manufacturing_lead_time_days",
    "production_volume", "defect_rate_percentage"
]

# Convert numeric columns safely
for c in numeric_cols:
    df = df.withColumn(c, when(col(c).rlike("^[0-9.]+$"), col(c).cast(DoubleType())).otherwise(0.0))

df = df.withColumn("label", when(col("delivery_delay_days").rlike("^[0-9.]+$"),
                                 col("delivery_delay_days").cast(DoubleType())).otherwise(0.0))

# -----------------------------
# Features
# -----------------------------
categorical_cols = ["product_type", "supplier_name", "transportation_mode"]
feature_cols = numeric_cols

# StringIndexer for categorical columns
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep") for c in categorical_cols]

# VectorAssembler
assembler = VectorAssembler(
    inputCols=[f"{c}_indexed" for c in categorical_cols] + feature_cols,
    outputCol="features"
)

# -----------------------------
# Train/Test Split
# -----------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()

# -----------------------------
# Random Forest Model
# -----------------------------
rf = RandomForestRegressor(featuresCol="features", labelCol="label", seed=42)

param_grid = (ParamGridBuilder()
              .addGrid(rf.numTrees, [50])  # fast training
              .addGrid(rf.maxDepth, [10])
              .build())

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")

pipeline = Pipeline(stages=indexers + [assembler, rf])

cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator,
                    numFolds=2,       # faster
                    parallelism=2)

print("\n⏳ Training Random Forest...")
cv_model = cv.fit(train_df)

predictions = cv_model.transform(test_df)
rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2").evaluate(predictions)
print(f" Random Forest Model: RMSE={rmse:.2f}, R²={r2:.2f}")

best_model_pipeline = cv_model.bestModel

# -----------------------------
# Save Model
# -----------------------------
model_output_path = "hdfs://localhost:9000/user/hadoop/supply_chain_models/RandomForest_predictor"
best_model_pipeline.write().overwrite().save(model_output_path)
print(f" Model saved to: {model_output_path}")

# -----------------------------
# Step 7: Feature Importance
# -----------------------------
feature_importances = best_model_pipeline.stages[-1].featureImportances
all_features = [f"{c}_indexed" for c in categorical_cols] + feature_cols
importance_list = sorted(zip(all_features, feature_importances), key=lambda x: x[1], reverse=True)
print("\n Feature Importances:")
for f, imp in importance_list:
    print(f"{f}: {imp:.4f}")

# -----------------------------
#  Predict New Data
# -----------------------------
new_data = spark.createDataFrame([
    ("Electronics", "Supplier A", "Air", 1200, 350, 5, 100, 80, 450, 30, 5000, 1.2)
], [
    "product_type", "supplier_name", "transportation_mode", "price", "shipping_cost",
    "lead_time_days", "order_quantity", "stock_level", "manufacturing_cost",
    "manufacturing_lead_time_days", "production_volume", "defect_rate_percentage"
])

predicted = best_model_pipeline.transform(new_data)
predicted.select("prediction").show()

