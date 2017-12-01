import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.DataFrame

class DecisionTreeAnalyses(dataFrame: DataFrame) {
  def run(): Unit = {
    val splitSeed = 5043
    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3), splitSeed)


    val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(20).setFeatureSubsetStrategy("auto").setSeed(5043)

    val model = classifier.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.show
/*
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "variance"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
      maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean()
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression tree model:\n" + model.toDebugString)

*/


  }

}
