import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.DataFrame

class DecisionTreeAnalyses(dataFrame: DataFrame) {
  def run(): Unit = {
    val splitSeed = 5043
    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3), splitSeed)


    val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(20).setFeatureSubsetStrategy("auto").setSeed(5043)

    val model = classifier.fit(trainingData)

    val predictions = model.transform(testData)

    predictions.show
  }

}
