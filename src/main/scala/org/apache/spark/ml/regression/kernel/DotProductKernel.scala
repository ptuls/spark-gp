package org.apache.spark.ml.regression.kernel

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import breeze.numerics._
import org.apache.spark.ml.linalg.Vector

class DotProductKernel(sigma: Double,
                       private val lower: Double = 1e-6,
                       private val upper: Double = inf)
    extends Kernel {

  override var hyperparameters: BDV[Double] = BDV[Double](sigma)

  private def getSigma: Double = hyperparameters(0)

  private var dotProducts: Option[BDM[Double]] = None

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
        val product = vectors(i).asBreeze dot vectors(j).asBreeze + getSigma
        sqd(i, j) = product
        sqd(j, i) = product
        j += 1
      }
      i += 1
    }

    dotProducts = Some(sqd)
    this
  }

  override def trainingKernel(): BDM[Double] = {
    dotProducts.getOrElse(throw new TrainingVectorsNotInitializedException)
  }

  // Note: derivative with respect to sigma
  override def trainingKernelAndDerivative()
    : (BDM[Double], Array[BDM[Double]]) = {
    val kernel = trainingKernel()
    val derivative = 2 * sqrt(getSigma) * BDM.ones(kernel.rows, kernel.cols)

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
        result(i, j) = test(i).asBreeze dot train(j).asBreeze + getSigma
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
}
