import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.scalatest.{BeforeAndAfter, FunSuite}
import org.apache.spark.sql.functions._

class AlgorithmsAnalysisSpec extends FunSuite with BeforeAndAfter {

  var sparkSession : SparkSession = _
  var dataFrame: DataFrame = _
  var pcaDataFrame: DataFrame = _
  var countBadData: Integer = 0

  val spark: SparkSession = sparkSession

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

  test("Linear Regression Normalize Data"){
    val columns = Array("prediction", "label", "features")
    val df = sparkSession.createDataFrame(List(
      (8.3,5, Seq(0,2,5)),
      (8.4,4, Seq(1,20,5)),
      (8.1,0, Seq(1,20,5))
    ))

    val renamedDF = df.toDF(columns: _*)

    val lr = new LinearRegressionAnalysis(pcaDataFrame)
    val finalDf = lr.normalizeData(renamedDF)

    finalDf.collect().foreach(t => if (t.getDouble(0) > 8.0) countBadData += 1)

    // after normalizing data we would expect to have zero outliers !
    assert(countBadData === 0)
  }

  after {
    if (sparkSession != null) {
      sparkSession.stop()
    }
  }

}