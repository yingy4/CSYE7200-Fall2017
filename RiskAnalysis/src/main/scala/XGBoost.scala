import org.apache.spark.sql.DataFrame

class XGBoost(dataFrame: DataFrame) {
  def run() = {
    dataFrame.show(10, false)
  }
}
