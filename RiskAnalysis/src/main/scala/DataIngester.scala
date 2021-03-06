import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.log4j.{Level, Logger}

class DataIngester {

  def run(sparkSession: SparkSession){

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val df: DataFrame = readData(sparkSession)

    val newDf: DataFrame = df.drop("Product_Info_2")

    val null_thresh = df.count()*0.6

    val to_keep = newDf.columns.filter(c => newDf.agg(
      sum(when(newDf(c) === 0 || newDf(c).isNull, 1).otherwise(0)).alias(c)).first().getLong(0) <= null_thresh
    )

    val clean = df.select(to_keep.head, to_keep.tail: _*)

    val finalDf= clean.na.fill(clean.columns.zip(clean.select(clean.columns.map(mean(_)): _*).first.toSeq).toMap)

    finalDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", true).save("Trimed")
  }

  def readData(sparkSession: SparkSession): DataFrame = {
    sparkSession.read.option("header","true").option("inferSchema","true")
      .option("delimeter","\t").csv("train.csv")
  }

  def readCleanedData(sparkSession: SparkSession): DataFrame = {
    sparkSession.read
      .option("header","true")
      .option("inferSchema","true")
      .format("csv")
      .load("Trimed/IntermediateData.csv")
  }

  def hasColumn(df:DataFrame, colName: String): Boolean = {
    df.columns.contains(colName)
  }

  def checkColumns(df:DataFrame): Boolean = {

    df.columns.isEmpty

  }
}