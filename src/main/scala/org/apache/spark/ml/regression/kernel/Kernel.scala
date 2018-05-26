package org.apache.spark.ml.regression.kernel

import breeze.linalg.{
  Transpose,
  DenseMatrix => BDM,
  DenseVector => BDV,
  Vector => BV
}
import org.apache.spark.ml.linalg.Vector

/**
  * Trait defining the covariance function `k` of a Gaussian Process
  *
  * The kernel should be differentiable with respect to the hyperparemeters
  *
  */
trait Kernel extends Serializable {

  /**
    * A vector of hyperparameters of the kernel
    */
  var hyperparameters: BDV[Double]

  /**
    * Stores some portion of the training sample
    */
  var trainOption: Option[Array[Vector]]

  /**
    * Setter
    *
    * @param value
    * @return this
    */
  def setHyperparameters(value: BDV[Double]): this.type = {
    hyperparameters = value
    this
  }

  /**
    *
    * @return boundaries lower and upper: lower <= hyperparameter <= upper (inequalities are element-wise).
    */
  def hyperparameterBoundaries: (BDV[Double], BDV[Double])

  /**
    *
    * @param vectors
    * @return this
    */
  def setTrainingVectors(vectors: Array[Vector]): this.type

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @return a matrix K such that `K_{ij} = k(trainingVectros(i), trainingVectros(j))`
    */
  def trainingKernel(): BDM[Double]

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @return the diagonal of the matrix which would be returned by `trainingKernel`.
    */
  def trainingKernelDiag(): Array[Double]

  /**
    *`setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @return a pair of a kernel K (returned by `trainingKernel`)
    *         and an array of partial derivatives \partial K / \partial hyperparameters(i)
    */
  def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]])

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @param test vectors (typically, those we want a prediction for)
    * @return a matrix K of size `test.size * trainVectors.size` such that K_{ij} = k(test(i), trainVectors(j))
    */
  def crossKernel(test: Array[Vector]): BDM[Double]

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @param test a vector (typically, the one we want a prediction for)
    * @return a row-vector K such that K_{i} = k(test, trainVectors(i))
    */
  def crossKernel(test: Vector): Transpose[BDV[Double]] = {
    val k = crossKernel(Array(test))
    k(0, ::)
  }
}

class TrainingVectorsNotInitializedException
    extends Exception("setTrainingVectors method should have been called first")
