package org.apache.spark.ml.bayesian

import org.apache.spark.ml.regression.{GaussianProcessRegression, GaussianProcessRegressionModel}
import org.apache.spark.ml.linalg.Vector

class UtilityFunction(kind: String, kappa: Double, xi: Double) {

  def upperConfidenceBound(x: Vector, gpModel: GaussianProcessRegressionModel): Double = {
    gpModel.predict(x)
  }

}
