package org.apache.spark.ml.bayesian

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.GaussianProcessRegressionModel

class UtilityFunction(kind: String, kappa: Double, xi: Double) {

  def upperConfidenceBound(x: Vector, gpModel: GaussianProcessRegressionModel): Double = {
    gpModel.predict(x)
  }

  def expectedImprovement(x: Vector, gpModel: GaussianProcessRegressionModel): Double = ???
}
