// Databricks notebook source
// source: https://courses.edx.org/courses/course-v1:Microsoft+DAT202.3x+3T2018
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, StopWordsRemover}

// COMMAND ----------

val csv = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/tweets.csv")
csv.show(5)

// COMMAND ----------

val data = csv.select($"SentimentText", $"Sentiment".cast("Int").alias("label"))
data.show(5, truncate=false)

// COMMAND ----------

val splits = data.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1).withColumnRenamed("label", "trueLabel")
println("Train len " + train.count() + ", test len " + test.count())

// COMMAND ----------

val tokenizer = new Tokenizer().setInputCol("SentimentText").setOutputCol("SentimentWords")
val swr = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("MeaningfulWords")
val hashTF = new HashingTF().setInputCol(swr.getOutputCol).setOutputCol("features")
val lr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setMaxIter(10)
val pipeline = new Pipeline().setStages(Array(tokenizer, swr, hashTF, lr))
val model = pipeline.fit(train)

// COMMAND ----------

val prediction = model.transform(test)
prediction.select("MeaningfulWords", "features", "prediction", "trueLabel").show(100, truncate=false)

// COMMAND ----------


