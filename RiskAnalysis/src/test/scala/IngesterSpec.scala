import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, FunSuite}


class IngesterSpec extends FunSuite with BeforeAndAfter {

   var sparkSession : SparkSession = _

  before {
    sparkSession = SparkSession
      .builder()
      .config("spark.master", "local")
      .appName("Unit Testing Risk Analysis Application")
      .getOrCreate()
  }

  test("data read"){
    val df: DataFrame = new DataIngester().readData(sparkSession)
    assert(df.count() === 59381)
  }

  after {
    if (sparkSession != null) {
      sparkSession.stop()
    }
  }

}