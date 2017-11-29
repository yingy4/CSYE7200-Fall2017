import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.DataFrame

class LinearRegressionAnalysis(dataFrame: DataFrame) {
  def run(): Unit = {
    //split dataFrame into training and test data
    val Array(training,test) = dataFrame.randomSplit(Array(0.70, 0.30))
    val lr = new LinearRegression().setMaxIter(100)

    val paramGrid: Array[ParamMap] = new ParamGridBuilder().build()

    val trainValidationSplit: TrainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    val model: TrainValidationSplitModel = trainValidationSplit.fit(training)

    model.transform(test).select("features", "label", "prediction").show()
  }
}