import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.Vectors

class RiskAnalysisApplication(sparkSession: SparkSession) {
  val data: DataFrame = sparkSession.read
                  .option("header","true")
                  .option("inferSchema","true")
                  .format("csv")
                  .load("IntermediateData.csv")
  data.printSchema()

  val colNames: Array[String] = Array("Product_Info_4", "Ins_Age", "BMI", "Employment_Info_1",
    "Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Family_Hist_2",
    "Family_Hist_4", "Id", "Medical_History_1", "Medical_Keyword_1", "Medical_Keyword_2",
    "Medical_Keyword_3"
  )

  // Use VectorAssembler to convert the input columns of the cancer data
  // to a single output column of an array called "features"
  // Set the input columns from which we are supposed to read the values.
  val assembler: VectorAssembler = new VectorAssembler().setInputCols(colNames).setOutputCol("features")

  // Use the assembler to transform our DataFrame to a single column: features
  val output: DataFrame = assembler.transform(data).select("features")

  // Use StandardScaler on the data
  val scaler: StandardScaler = new StandardScaler()
    .setInputCol("features")
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
    .setOutputCol("pcaFeatures")
    .setK(4)
    .fit(scaledData)

  // Once your pca has been created and fit, transform the scaledData
  // Call this new dataframe pcaDF
  val pcaDF: DataFrame = pca.transform(scaledData)

  // Show the new pcaFeatures
  val result: DataFrame = pcaDF.select("pcaFeatures")
  result.show()

//  result.write
//        .format("com.databricks.spark.csv")
//        .option("header", "true")
//        .save("IntermediateTestdata.csv")
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