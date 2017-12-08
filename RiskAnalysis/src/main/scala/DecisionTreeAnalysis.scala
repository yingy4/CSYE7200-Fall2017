import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.DataFrame

class DecisionTreeAnalysis(dataFrame: DataFrame) {
  def run(): Unit = {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    /*
    Automatically identify categorical features, and index them.
     Here, we treat features with > 4 distinct values as continuous.

      */
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures")
      .setMaxCategories(4).fit(dataFrame)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(featureIndexer, dt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    println("\nDecision Tree Model:")

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(10)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)




    println("Root Mean Square Error (RMSE) on test data = "+ rmse)

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)
    println
    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    val pipe = new Pipeline().setStages(Array(featureIndexer, rf))

    val newModel = pipe.fit(trainingData)

    val newPredictions = newModel.transform(testData)

    // Select example rows to display.
    newPredictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluate = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val nrmse = evaluate.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + nrmse)
    //Random forest selection
    val rfModel = newModel.stages(1).asInstanceOf[RandomForestRegressionModel]

    println("Learned regression forest model:\n" + rfModel.toDebugString)

  }
}