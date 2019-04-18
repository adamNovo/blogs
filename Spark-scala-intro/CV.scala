// Databricks notebook source
// source: https://courses.edx.org/courses/course-v1:Microsoft+DAT202.3x+3T2018
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

// COMMAND ----------

val csv = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/flights.csv")
csv.show(5)

// COMMAND ----------

val data = csv.select($"DayofMonth", $"DayOfWeek", $"Carrier", $"OriginAirportID", $"DestAirportID", $"DepDelay", ($"ArrDelay" > 15).cast("Double").alias("label"))
data.show(5)

// COMMAND ----------

val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1).withColumnRenamed("label", "trueLabel")
println("Train # rows: " + train.count() + ", test # rows: " + test.count())

// COMMAND ----------

val features = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")
val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
val pipeline = new Pipeline().setStages(Array(features, lr))

// COMMAND ----------

val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.3, 0.01))
  .addGrid(lr.maxIter, Array(10, 5))
  .addGrid(lr.threshold, Array(0.35, 0.3))
  .build()
val cv = new CrossValidator().setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
val model = cv.fit(train)

// COMMAND ----------

val predicted = model.transform(test)
predicted.select("features", "probability", "prediction", "trueLabel").show(200, truncate=false)

// COMMAND ----------

// confusion matrix
val tp = predicted.filter("prediction == 1 AND trueLabel == 1").count.toFloat
val fp = predicted.filter("prediction == 1 AND trueLabel == 0").count.toFloat
val tn = predicted.filter("prediction == 0 AND trueLabel == 0").count.toFloat
val fn = predicted.filter("prediction == 0 AND trueLabel == 1").count.toFloat
val precicision = tp / (tp + fp)
val recall = tp / (tp + fn)

// COMMAND ----------

val evaluator = new BinaryClassificationEvaluator()
  .setRawPredictionCol("prediction")
  .setLabelCol("trueLabel")
  .setMetricName("areaUnderROC")
evaluator.evaluate(predicted.select("prediction", "trueLabel"))

// COMMAND ----------


