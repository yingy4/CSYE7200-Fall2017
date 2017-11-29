import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}

class LinearRegression(dataFrame: DataFrame) {
//  def run(): = {
//    //split dataFrame into training and test data
//    val Array(training,test) = dataFrame.randomSplit(Array(0.70, 0.30))
//    test.show(4, truncate = false)
//    val lr = new LinearRegression()
//
//    val paramGrid: Array[ParamMap] = new ParamGridBuilder().build()
//
//    val trainValidationSplit: TrainValidationSplit = new TrainValidationSplit()
//      .setEstimator(lr)
//      .setEvaluator(new RegressionEvaluator())
//      .setEstimatorParamMaps(paramGrid)
//      .setTrainRatio(0.8)
//
//    val model: TrainValidationSplitModel = trainValidationSplit.fit(training)
//
//    model.transform(test).select("features", "label", "prediction")
//  }
}