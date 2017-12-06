import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.{DataFrame, SparkSession}

class PrincipalComponentAnalysis(sparkSession: SparkSession) {
  def run(): DataFrame = {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val data: DataFrame = readCleanedData(sparkSession)

    val colNames: Array[String] = Array("Id", "Product_Info_1", "Product_Info_3", "Product_Info_4", "Product_Info_5", "Ins_Age", "Ht", "Wt",
      "BMI", "Employment_Info_1", "Employment_Info_2", "Employment_Info_3", "Employment_Info_6", "InsuredInfo_1", "InsuredInfo_2",
      "InsuredInfo_3", "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Family_Hist_1", "Family_Hist_2", "Family_Hist_3",
      "Family_Hist_4", "Medical_History_1", "Medical_History_2", "Medical_History_3", "Medical_History_4", "Medical_History_6", "Medical_History_9"
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

    finalDF.toDF("features", "label")
  }

  def readCleanedData(sparkSession: SparkSession): DataFrame = {
    sparkSession.read
      .option("header","true")
      .option("inferSchema","true")
      .format("csv")
      .load("Trimed/IntermediateData.csv")
  }
}
