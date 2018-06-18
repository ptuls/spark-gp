package org.apache.spark.ml.regression.kernel

import breeze.linalg.{DenseMatrix => BDM, all}
import breeze.numerics.abs
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.FunSuite

class DotProductKernelTest extends FunSuite {
  private val dataset =
    Array(Array(1d, 2d), Array(2d, 3d), Array(5d, 7d)).map(Vectors.dense)

  test(
    "being called after `setTrainingVector`," +
      " `trainingKernel` should return the correct kernel matrix") {
    val dot = new DotProductKernel(0.2)
    dot.setTrainingVectors(dataset)

    val correctKernelMatrix =
      BDM((5.040000e+00, 8.040000e+00, 19.040000e+00),
          (8.040000e+00, 13.040000e+00, 31.040000e+00),
          (19.040000e+00, 31.040000e+00, 74.040000e+00))

    assert(all(abs(dot.trainingKernel() - correctKernelMatrix) <:< 1e-4))
  }

  private def computationalDerivative(sigma: Double, h: Double): BDM[Double] = {
    val left = new DotProductKernel(sigma - h)
    val right = new DotProductKernel(sigma + h)

    left.setTrainingVectors(dataset)
    right.setTrainingVectors(dataset)
    (right.trainingKernel() - left.trainingKernel()) / (2 * h)
  }

  test(
    "being called after `setTrainingVector`," +
      " `derivative` should return the correct kernel matrix derivative") {
    val dot = new DotProductKernel(0.2)
    dot.setTrainingVectors(dataset)

    val analytical = dot.trainingKernelAndDerivative()._2.reduce(_ + _)
    val computational = computationalDerivative(0.2, 1e-10)

    assert(all(abs(analytical - computational) <:< 1e-3))
  }

}
