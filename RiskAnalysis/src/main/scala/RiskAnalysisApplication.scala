import org.apache.spark.ml.Model
import org.apache.spark.sql.{DataFrame, SparkSession}

class RiskAnalysisApplication(sparkSession: SparkSession) {
//  val cleanedData: Unit = new DataIngester().run(sparkSession)
  val pcaDF: DataFrame = new PrincipalComponentAnalysis(sparkSession).run()
//  val linearRegressionModel: Model[_] = new LinearRegression(pcaDF).run()
  val XGBoostModel: Unit = new XGBoost(pcaDF).run()
}

object RiskAnalysisApplication {
  def main(args: Array[String]) {
    val spark: SparkSession = SparkSession
                              .builder()
                              .config("spark.master", "local")
                              .appName("Risk Analysis PCA")
                              .getOrCreate()
    val ingester = new RiskAnalysisApplication(spark)
  }
}