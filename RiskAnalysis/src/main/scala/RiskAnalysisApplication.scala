import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}

class RiskAnalysisApplication(sparkSession: SparkSession) {
  /**
    * Commenting out the DataCleaning process b/c we don't want to keep generating a clean data file on every run
    */
//    val cleanedData: Unit = new DataIngester().run(sparkSession)
  val pcaDF: DataFrame = new PrincipalComponentAnalysis(sparkSession).run()
  val linearRegressionModel: Unit = new LinearRegressionAnalysis(pcaDF).run()
  val decisionTreeModel: Unit = new DecisionTreeAnalysis(pcaDF).run()
  val logisticRegressionModel: Unit = new LogisticRegressionAnalysis(pcaDF).run()
}

object RiskAnalysisApplication {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    val spark: SparkSession = SparkSession
                              .builder()
                              .config("spark.master", "local")
                              .appName("Risk Analysis Application")
                              .getOrCreate()
    val ingester = new RiskAnalysisApplication(spark)
  }
}