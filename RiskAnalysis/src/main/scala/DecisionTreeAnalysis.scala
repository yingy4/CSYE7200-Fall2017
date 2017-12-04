import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.sql.DataFrame

class DecisionTreeAnalysis(dataFrame: DataFrame) {
  def run(): Unit = {

    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures")
      .setMaxCategories(4).fit(dataFrame)

    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))

    val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")


    val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))

    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)

    predictions.select("prediction", "label", "features").show(8)

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)

    println("Root Mean Square Error (RMSE) on test data = "+ rmse)


//    val treeModel = model.stages(2).asInstanceOf[DecisionTreeRegressionModel]
//
//    println("Learned classification tree model:\n" + treeModel.toDebugString)

  }
}