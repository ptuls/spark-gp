package org.apache.spark.ml.regression.kernel

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import breeze.numerics._
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Implements traditional RBF kernel `k(x_i, k_j) = exp(||x_i - x_j||^2 / sigma^2)`
  *
  * @param sigma
  * @param lower
  * @param upper
  */
class RBFKernel(sigma: Double,
                private val lower: Double = 1e-6,
                private val upper: Double = inf)
    extends Kernel {
  override var hyperparameters: BDV[Double] = BDV[Double](sigma)

  private def getSigma() = hyperparameters(0)

  private var squaredDistances: Option[BDM[Double]] = None

  var trainOption: Option[Array[Vector]] = None

  def this() = this(1)

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = {
    (BDV[Double](lower), BDV[Double](upper))
  }

  override def setTrainingVectors(vectors: Array[Vector]): this.type = {
    trainOption = Some(vectors)
    val sqd = BDM.zeros[Double](vectors.length, vectors.length)

    var i = 0
    while (i < vectors.length) {
      var j = 0
      while (j <= i) {
        val dist = Vectors.sqdist(vectors(i), vectors(j))
        sqd(i, j) = dist
        sqd(j, i) = dist
        j += 1
      }
      i += 1
    }

    squaredDistances = Some(sqd)
    this
  }

  override def trainingKernel(): BDM[Double] = {
    val result = squaredDistances.getOrElse(
      throw new TrainingVectorsNotInitializedException) / (-2d * sqr(
      getSigma()))
    exp.inPlace(result)
    result
  }

  override def trainingKernelAndDerivative()
    : (BDM[Double], Array[BDM[Double]]) = {
    val sqd = squaredDistances.getOrElse(
      throw new TrainingVectorsNotInitializedException)

    val kernel = trainingKernel()
    val derivative = sqd *:* kernel
    derivative /= cube(getSigma())

    (kernel, Array(derivative))
  }

  override def crossKernel(test: Array[Vector]): BDM[Double] = {
    val train =
      trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    val result = BDM.zeros[Double](test.length, train.length)

    var i = 0
    while (i < test.length) {
      var j = 0
      while (j < train.length) {
        result(i, j) = Vectors.sqdist(test(i), train(j)) / (-2d * sqr(
          getSigma()))
        j += 1
      }
      i += 1
    }

    exp.inPlace(result)

    result
  }

  override def trainingKernelDiag(): Array[Double] = {
    trainOption
      .getOrElse(throw new TrainingVectorsNotInitializedException)
      .map(_ => 1d)
  }

  private def sqr(x: Double) = x * x

  private def cube(x: Double) = x * x * x
}
