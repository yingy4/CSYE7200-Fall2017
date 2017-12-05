import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, FunSuite}


class IngesterSpec extends FunSuite with BeforeAndAfter {

   var sparkSession : SparkSession = _
  var pcaDataFrame: DataFrame = _

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
  test("Validate column"){
    pcaDataFrame = new DataIngester().readCleanedData(sparkSession)

    assert(pcaDataFrame.columns.size === 73)
  }
  test("Does Column exists"){
    val df = new DataIngester().hasColumn(pcaDataFrame, "Product_Info_2")

    assert(df === false)
  }
  test("Is any column empty"){
    val df = new DataIngester().checkColumns(pcaDataFrame)

    assert(df === false)
  }
  after {
    if (sparkSession != null) {
      sparkSession.stop()
    }
  }

}