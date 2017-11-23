import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{sum, col, when, count, isnan}
import org.apache.spark.sql.functions.udf
import org.apache.log4j._
import org.apache.spark.rdd.RDD

import scala.util.{Success, Try}


//class DataIngester(sc: SparkContext) {
//  val rdd = sc.textFile("train.csv").toJavaRDD()
//  val result = rdd.count()
//  println(result)
//}


object DataIngester {

  def main(args: Array[String]) {
    def countNulls = {udf((v: Any) => if (v == null || v == "") 1 else 0)}

    //    val spark = SparkSession.builder().appName("Risk Analysis").master("local[*]").getOrCreate()
    //    // Set the log level to only print errors
    //
    //    Logger.getLogger("org").setLevel(Level.ERROR)
    //
    //    val pathFile = "train.csv"
    //
    //    val header: Row = spark.read.csv(pathFile).head()
    //    val data: RDD[String] = spark.read.option("header", true).csv(pathFile)
    //      .rdd.map(_.toString().replace("]","").replace("[","")).map(r => {
    //      val tmp: String = r.split(",").take(3).mkString(",")
    ////      r.split(",").foreach(println(_))
    //      val remain: String = r.split(",").drop(3).filter(
    //        x => {
    //          Try(x.toDouble) match {
    //        case Success(x) =>  x*0.6 != 0
    //
    //        case _ => false
    //      }
    //        }).mkString(",")
    //      tmp + "," + remain
    //    }
    //    )
    //    import spark.implicits._
    ////    val headerString = spark.createDataset(Array(header.toString()))
    //    // headerString.rdd.union(data).repartition(1).saveAsTextFile("testout")
    //    data.saveAsTextFile("testout")

    val spark = SparkSession.builder().master("local[*]").getOrCreate()
    val df = spark.read.option("header","true").option("inferSchema","true").csv("/Users/mtiburu/Downloads/try.csv")
    //    val newValue = df.na.replace(df.columns, Map("" -> "0"))
    //    df.select(df.columns.map(c => sum(col(c).isNull.cast("int" +
    //      "")).alias(c)): _*).show
    //    var countCol: org.apache.spark.sql.Column = null
    //    df.columns.foreach(c => {
    //      if (countCol == null || countNulls == "") countCol = countNulls(col(c))
    //      else countCol = countCol + countNulls(col(c))
    //    });
    //
    //    println(countCol)


    val isNaN = udf((value : Float) => {
      if (value.equals(Float.NaN) || value == null) true else false })

    val result = df.filter(isNaN(df("Product_Info_3"))).count()
    println(result)
    df.write.format("csv").option("header","true").save("output")




  }
}