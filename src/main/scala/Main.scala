import java.io.{BufferedWriter, File, FileWriter}

import io.circe.syntax.EncoderOps
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SparkSession, functions}

import scala.collection.mutable
import scala.math.sqrt

object Main {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:/Users/Alexey/Downloads/hadoop-3.3.0/hadoop-3.3.0/")

    val spark = SparkSession.builder.appName("BigDataHW2")
      .config("spark.master", "local")
      .getOrCreate()

    val coursesId = Array(273, 13, 54, 809, 1441, 2009)

    val dataset = spark.read.json("src/main/data/dataset.json")
    val preparedDataset = dataset
      .withColumn("desc", regexp_replace(col("desc"), "[^\\w\\sа-яА-ЯёЁ]", ""))
      .withColumn("desc", lower(trim(regexp_replace(col("desc"), "\\s+", " "))))
      .where(functions.length(col("desc")) > 0)

    val tokenizer = new Tokenizer().setInputCol("desc").setOutputCol("words")
    val tokenizedDataset = tokenizer.transform(preparedDataset)

    val tf = new HashingTF().setInputCol("words").setOutputCol("tfFeatures").setNumFeatures(10000)
    val tfDataset = tf.transform(tokenizedDataset)

    val idf = new IDF().setInputCol("tfFeatures").setOutputCol("idfFeatures")
    val idfModel = idf.fit(tfDataset)
    val idfDataset = idfModel.transform(tfDataset)

    val toDense = udf((v: Vector) => v.toDense)

    val denseDataset = idfDataset.withColumn("denseFeatures", toDense(col("idfFeatures")))

    val norm = (vector: Array[Double]) => sqrt(vector.map(el => el * el).sum)
    val correlation = udf { (v1: Vector, v2: Vector) =>
      val arr1 = v1.toArray
      val arr2 = v2.toArray
      val scalar = arr1.zip(arr2).map(el => el._1 * el._2).sum
      scalar / (norm(arr1) * norm(arr2))
    }

    val coursesDataset = denseDataset
      .filter(col("id").isin(coursesId: _*))
      .select(
        col("id").alias("targetId"),
        col("denseFeatures").alias("targetDense"),
        col("lang").alias("targetLang")
      )

    val mergedDataset = denseDataset.join(broadcast(coursesDataset),
      col("id") =!= col("targetId") && col("lang") === col("targetLang"))
      .withColumn("correlation", correlation(col("targetDense"), col("denseFeatures")))

    val result = mergedDataset
      .withColumn("correlation", when(col("correlation").isNaN, 0).otherwise(col("correlation")))
      .withColumn("rate", row_number().over(
        Window.partitionBy(col("targetId"))
          .orderBy(col("correlation").desc, col("name").asc, col("id").asc)))
      .filter(col("rate").between(1, 10))

    val resultArray = result
      .groupBy(col("targetId"))
      .agg(concat_ws(",", collect_list("id")).alias("ids"))
      .select(col("targetId"), col("ids"))
      .collect()

    val map: mutable.Map[String, Array[Int]] = mutable.Map[String, Array[Int]]()
    resultArray.foreach(el => {
      map.put(el(0).toString, el(1).toString.split(",").map(el => el.toInt))
    })

    writeToFile(map.asJson.toString(), "src/main/data/result.json")
  }

  def writeToFile(string: String, path: String): Unit = {
    val file = new File(path)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(string)
    bw.close()
  }
}

