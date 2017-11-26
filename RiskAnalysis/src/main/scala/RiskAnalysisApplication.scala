import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.functions._

class RiskAnalysisApplication(sparkSession: SparkSession) {
  val data: DataFrame = sparkSession.read
                  .option("header","true")
                  .option("inferSchema","true")
                  .format("csv")
                  .load("IntermediateData.csv")
//  data.printSchema()

  val colNames: Array[String] = Array("Product_Info_4", "Ins_Age", "BMI", "Employment_Info_1",
    "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2",
    "Family_Hist_4", "Id", "Medical_History_1", "Medical_Keyword_1", "Medical_Keyword_2",
    "Medical_Keyword_3"
  )

  // Use VectorAssembler to convert the input columns of the cancer data
  // to a single output column of an array called "features"
  // Set the input columns from which we are supposed to read the values.
  val assembler: VectorAssembler = new VectorAssembler().setInputCols(colNames).setOutputCol("featuresIntermediate")

  // Use the assembler to transform our DataFrame to a single column: features
  val output: DataFrame = assembler.transform(data).select("featuresIntermediate")

  // Use StandardScaler on the data
  val scaler: StandardScaler = new StandardScaler()
    .setInputCol("featuresIntermediate")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)
    .setWithMean(false)

  // Compute summary statistics by fitting the StandardScaler.
  val scalerModel: StandardScalerModel = scaler.fit(output)

  // Normalize each feature to have unit standard deviation.
  val scaledData: DataFrame = scalerModel.transform(output)

  // Create a new PCA() object that will take in the scaledFeatures
  // and output the pcs features, use 4 principal components
  // Then fit this to the scaledData
  val pca: PCAModel = new PCA()
    .setInputCol("scaledFeatures")
    .setOutputCol("features")
    .setK(4)
    .fit(scaledData)

  // Once your pca has been created and fit, transform the scaledData
  // Call this new dataframe pcaDF
  val pcaDF: DataFrame = pca.transform(scaledData)

  // Show the new pcaFeatures
  val result: DataFrame = pcaDF.select("features")

  /*
   * In order to join these two data frames we need to add a unique index to each of these DF's
   */
  val dataWithUniqueId: DataFrame = data.withColumn("row_id_1", monotonically_increasing_id())
  val resultWithUniqueId: DataFrame = result.withColumn("row_id_2", monotonically_increasing_id())

  // this DF contains the response column
  val finalDF: DataFrame = dataWithUniqueId.as("df1").join(resultWithUniqueId.as("df2"), dataWithUniqueId("row_id_1") === resultWithUniqueId("row_id_2"), "inner")
                           .select("df2.features", "df1.response")

  val testDF = finalDF.toDF("features", "label")

  val Array(training,test) = testDF.randomSplit(Array(0.70, 0.30))
  test.show(4, truncate = false)
  val lr = new LinearRegression()

  val paramGrid: Array[ParamMap] = new ParamGridBuilder().build()

  val trainValidationSplit: TrainValidationSplit = new TrainValidationSplit()
    .setEstimator(lr)
    .setEvaluator(new RegressionEvaluator())
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.8)

  val model: TrainValidationSplitModel = trainValidationSplit.fit(training)

  model.transform(test).select("features", "label", "prediction").show()

  //TODO: Extract training summary from here
  model.bestModel
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