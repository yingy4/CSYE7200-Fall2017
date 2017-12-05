import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfter, FlatSpec, FunSuite}


class AlgorithmsAnalysisSpec extends FunSuite with BeforeAndAfter {

  var sparkSession : SparkSession = _
  var dataFrame: DataFrame = _
  var pcaDataFrame: DataFrame = _

  before {
    sparkSession = SparkSession
      .builder()
      .config("spark.master", "local")
      .appName("Unit Testing Risk Analysis Application")
      .getOrCreate()
  }

  test("PCA Read"){
    pcaDataFrame = new PrincipalComponentAnalysis(sparkSession).readCleanedData(sparkSession)
    assert(pcaDataFrame.count() === 59381)
  }

  after {
    if (sparkSession != null) {
      sparkSession.stop()
    }
  }

}