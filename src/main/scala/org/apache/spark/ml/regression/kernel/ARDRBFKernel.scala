package org.apache.spark.ml.regression.kernel

import breeze.linalg.{
  norm,
  DenseMatrix => BDM,
  DenseVector => BDV,
  Vector => BV
}
import breeze.numerics._
import org.apache.spark.ml.linalg.Vector

class ARDRBFKernel(override var hyperparameters: BDV[Double],
                   private val lower: BDV[Double],
                   private val upper: BDV[Double])
    extends Kernel {

  def this(hyperparameters: BDV[Double]) =
    this(hyperparameters, hyperparameters * 0d, hyperparameters * inf)

  def this(p: Int, beta: Double = 1, lower: Double = 0, upper: Double = inf) =
    this(BDV.zeros[Double](p) + beta,
         BDV.zeros[Double](p) + lower,
         BDV.zeros[Double](p) + upper)

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) =
    (lower, upper)

  override var trainOption: Option[Array[Vector]] = _

  override def setTrainingVectors(vectors: Array[Vector]): this.type = {
    trainOption = Some(vectors)
    this
  }

  private def kernelElement(a: BV[Double], b: BV[Double]): Double = {
    val weightedDistance = norm((a - b) *:* hyperparameters)
    exp(-weightedDistance * weightedDistance)
  }

  override def trainingKernelDiag(): Array[Double] = {
    val train =
      trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    train.map(_ => 1d)
  }

  override def trainingKernel(): BDM[Double] = {
    val train =
      trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)

    val result = BDM.zeros[Double](train.length, train.length)

    var i = 0
    while (i < train.length) {
      var j = 0
      while (j <= i) {
        val k = kernelElement(train(i).asBreeze, train(j).asBreeze)
        result(i, j) = k
        result(j, i) = k
        j += 1
      }
      i += 1
    }

    result
  }

  override def trainingKernelAndDerivative()
    : (BDM[Double], Array[BDM[Double]]) = {
    val train =
      trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    val K = trainingKernel()
    val minus2Kernel = -2d * K
    val result = Array.fill[BDM[Double]](hyperparameters.length)(
      BDM.zeros[Double](train.length, train.length))

    var i = 0
    while (i < train.length) {
      var j = 0
      while (j <= i) {
        val diff = train(i).asBreeze - train(j).asBreeze
        diff :*= diff
        diff :*= hyperparameters
        val betaXi_Xj = diff
        for (k <- 0 until hyperparameters.length) {
          result(k)(i, j) = betaXi_Xj(k)
          result(k)(j, i) = betaXi_Xj(k)
        }
      }
    }

    (K, result.map(derivative => derivative *:* minus2Kernel))
  }

  override def crossKernel(test: Array[Vector]): BDM[Double] = {
    val train =
      trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    val result = BDM.zeros[Double](test.length, train.length)

    for (testIndx <- test.indices; trainIndex <- train.indices)
      result(testIndx, trainIndex) =
        kernelElement(train(trainIndex).asBreeze, test(testIndx).asBreeze)

    result
  }
}
