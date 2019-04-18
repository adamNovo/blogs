// Databricks notebook source
// source: https://courses.edx.org/courses/course-v1:Microsoft+DAT202.3x+3T2018
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator

// COMMAND ----------

val csv = spark.read.option("inferSchema", "true").option("header", "true")
  .csv("/FileStore/tables/flights.csv")
csv.show(5)

// COMMAND ----------

val data = csv.select($"DayofMonth", $"DayOfWeek", $"Carrier", $"OriginAirportID", $"DestAirportID", $"DepDelay", $"ArrDelay".alias("label"))
data.show(5)

// COMMAND ----------

val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1).withColumnRenamed("label", "trueLabel")
println("Train # rows: " + train.count() + ", test # rows: " + test.count())

// COMMAND ----------

val features = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
val lr = new LinearRegression().setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxIter(10)
  .setRegParam(0.3)
val pipeline = new Pipeline().setStages(Array(features, lr))
val model = pipeline.fit(train)

// COMMAND ----------

val predicted = model.transform(test)
predicted.select("features", "prediction", "trueLabel").show(200, truncate=false)

// COMMAND ----------

predicted.createOrReplaceTempView("predictedView")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT prediction, trueLabel from predictedView

// COMMAND ----------

val evaluator = new RegressionEvaluator().setLabelCol("trueLabel").setPredictionCol("prediction")
val rmse = evaluator.evaluate(predicted)

// COMMAND ----------


