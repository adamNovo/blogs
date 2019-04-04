// Databricks notebook source
// source: https://courses.edx.org/courses/course-v1:Microsoft+DAT202.3x+3T2018
import org.apache.spark.sql.Encoders
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS

// COMMAND ----------

val flights = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/flights.csv")
flights.show(5)

// COMMAND ----------

val data = flights.select($"DayofMonth", $"DayOfWeek", $"OriginAirportID", $"DestAirportID", $"DepDelay",
                         ($"ArrDelay" > 15).cast("Int").alias("Late"))
data.show(5)

// COMMAND ----------

val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
println("Train rows: " + train.count() + ", test rows: " + test.count())

// COMMAND ----------

train.show(5, truncate=false)

// COMMAND ----------

val assembler = new VectorAssembler().setInputCols(
  Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).
  setOutputCol("features")
val training = assembler.transform(train).select($"features", $"Late".alias("label"))
training.show(5, truncate=false)

// COMMAND ----------

val lr = new LogisticRegression().setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxIter(5)
  .setRegParam(0.3)
val model = lr.fit(training)

// COMMAND ----------

val testing = assembler.transform(train)
  .select($"features", $"Late".alias("trueLabel"))
testing.show(5, truncate=false)

// COMMAND ----------

val prediction = model.transform(testing)
prediction.show(5)

// COMMAND ----------

prediction.select("features", "prediction", "trueLabel")
  .show(100)

// COMMAND ----------

prediction.filter("prediction = 1.0")
  .select("prediction", "trueLabel", "probability")
  .show(10, truncate=false)

// COMMAND ----------

prediction.filter("prediction = 0.0")
  .select("prediction", "trueLabel", "probability")
  .show(10, truncate=false)
