import org.apache.spark.{SparkConf, SparkContext}

class DataIngester(sc: SparkContext) {
  val rdd = sc.textFile("train.csv").toJavaRDD()
  val result = rdd.count()
  println(result)
}


object DataIngester {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("RiskAnalysis").setMaster("local")
    val sc = new SparkContext(conf)
    val Ingester = new DataIngester(sc)
  }
}