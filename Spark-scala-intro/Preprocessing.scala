// Databricks notebook source
// source: https://courses.edx.org/courses/course-v1:Microsoft+DAT202.3x+3T2018/courseware/106c20c11a1d4fe29a2b32e95a4b43ae/73ba3e617d6d49c3816a152437e7ca95/2
import org.apache.spark.sql.Encoders

// COMMAND ----------

// Load data using explicit schema
case class flight(dayofMonth: Int, dayOfWeek: Int, Carrier: String, OriginAirportId: Int, DestAirportId: Int,
                 DepDelay: Int, ArrDelay: Int)
val flightSchema = Encoders.product[flight].schema
val flights = spark.read.schema(flightSchema).option("header", "true").csv("/FileStore/tables/raw_flight_data-eb9f8.csv")
flights.show(5)

// COMMAND ----------

flights.show(50)

// COMMAND ----------

flights.count()

// COMMAND ----------

flights.describe().select("summary", "DepDelay", "ArrDelay").show(truncate=false)

// COMMAND ----------

// find duplicates
flights.count() - flights.dropDuplicates().count()

// COMMAND ----------

// find na
flights.count() - flights.dropDuplicates().na.drop("any", Array("ArrDelay", "depDelay")).count()

// COMMAND ----------

val data = flights.dropDuplicates().na.fill(0, Array("ArrDelay", "depDelay"))
data.count()

// COMMAND ----------

data.describe().select("summary", "DepDelay", "ArrDelay").show(truncate=false)

// COMMAND ----------

data.stat.corr("DepDelay", "ArrDelay")

// COMMAND ----------

data.createOrReplaceTempView("dataView")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT DepDelay, ArrDelay 
// MAGIC FROM dataView

// COMMAND ----------


