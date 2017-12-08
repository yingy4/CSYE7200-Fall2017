import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.DataFrame

class LogisticRegressionAnalysis(dataFrame:DataFrame) {

 def run():Unit = {
  val rootLogger = Logger.getRootLogger()
  rootLogger.setLevel(Level.ERROR)

   val splits = dataFrame.randomSplit(Array(0.7, 0.3), seed = 11L)
   val trainging = splits(0).cache()
   val test = splits(1)

   val binaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol("label")
     .setRawPredictionCol("rawPrediction")
   binaryClassificationEvaluator.setMetricName("areaUnderROC")
   val regressionEvaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction")
   regressionEvaluator.setMetricName("rmse")

//   logistic Regression
   val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
     .setFeaturesCol("features").setLabelCol("label")
   val lrModel = lr.fit(trainging)
   val predictions = lrModel.transform(trainging)
   val areaTraining = binaryClassificationEvaluator.evaluate(predictions)

   println("Area under ROC using Logistic Regression on training data = " + areaTraining)

   val predictionTest = lrModel.transform(test)
   val areaTest = binaryClassificationEvaluator.evaluate(predictionTest)
   println("Area under ROC using Logistic Regression on test data = " + areaTest)

   val rmseLR = regressionEvaluator.evaluate(predictionTest)

   println("Root Mean Squared Error (RMSE) Logistic Regression on test data = " + rmseLR)


 }
}