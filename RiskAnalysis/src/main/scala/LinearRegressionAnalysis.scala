import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

class LinearRegressionAnalysis(dataFrame: DataFrame) {
  def run(): Unit = {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    //split dataFrame into training and test data
    val Array(training,test) = dataFrame.randomSplit(Array(0.70, 0.30))
    val lr = new LinearRegression().setMaxIter(200)

    val paramGrid: Array[ParamMap] = new ParamGridBuilder().build()

    val trainValidationSplit: TrainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    val model: TrainValidationSplitModel = trainValidationSplit.fit(training)

    val deNormData = model.transform(test).select("features", "label", "prediction")

    // normalize data to get integers
    val normalizedData: DataFrame = normalizeData(deNormData)

    println("RMSE Linear Regression: ")
    print(model.bestModel.asInstanceOf[LinearRegressionModel].summary.rootMeanSquaredError)
  }

  def normalizeData(dataFrame: DataFrame): DataFrame = {
    dataFrame.withColumn("prediction", when(dataFrame("prediction") >= 8.0, 8.0).otherwise(floor(dataFrame("prediction"))))
  }

}