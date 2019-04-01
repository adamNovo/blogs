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

val airports = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/airports.csv")
airports.show(5)

// COMMAND ----------

airports.show(50, truncate=false)

// COMMAND ----------

airports.printSchema

// COMMAND ----------

val cities = airports.select("city", "name")
cities.show(50, truncate=false)

// COMMAND ----------

val airportsByOrigin = flights.join(airports, $"OriginAirportId" === $"airport_id")
  .groupBy("city").count()
airportsByOrigin.show(truncate=false)

// COMMAND ----------

airports.createOrReplaceTempView("airportView")

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT state, count(state) AS state_count FROM airportView 
// MAGIC GROUP BY state 
// MAGIC ORDER BY state_count DESC;

// COMMAND ----------


