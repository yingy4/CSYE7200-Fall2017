import org.apache.spark.ml.classification.LogisticRegression

import org.apache.spark.sql.DataFrame

class LogisticRegressionAnalysis(dataFrame:DataFrame) {

 def run():Unit = {
  val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

   val lrModel = lr.fit(dataFrame)

   println(s"Coefficients: ${lrModel.coefficientMatrix} Intercept: ${lrModel.interceptVector}")

   val mlr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
     .setFamily("multinomial")

   val mlrModel = mlr.fit(dataFrame)
   println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}" +
     s"Multinomial intercepts: ${mlrModel.interceptVector}")

 }
}
