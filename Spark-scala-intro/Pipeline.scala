// Databricks notebook source
// source: https://courses.edx.org/courses/course-v1:Microsoft+DAT202.3x+3T2018
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler}
import org.apache.spark.ml.classification.DecisionTreeClassifier

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

val strIdx = new StringIndexer().setInputCol("Carrier").setOutputCol("CarrierIdx")
val catVec = new VectorAssembler().setInputCols(Array("CarrierIdx", "DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID")).setOutputCol("catFeatures")
val catIdx = new VectorIndexer().setInputCol(catVec.getOutputCol).setOutputCol("idxCatFeatures")
val numVec = new VectorAssembler().setInputCols(Array("DepDelay")).setOutputCol("numFeatures")
val minMax = new MinMaxScaler().setInputCol(numVec.getOutputCol).setOutputCol("normFeatures")
val featuresVec = new VectorAssembler().setInputCols(Array("idxCatFeatures", "normFeatures")).setOutputCol("features")
val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
val pipeline = new Pipeline().setStages(Array(strIdx, catVec, catIdx, numVec, minMax, featuresVec, dt))
val model = pipeline.fit(train)

// COMMAND ----------

val predicted = model.transform(test)
predicted.select("features", "prediction", "trueLabel").show(200, truncate=false)

// COMMAND ----------


