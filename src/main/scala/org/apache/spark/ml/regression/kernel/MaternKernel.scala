package org.apache.spark.ml.regression.kernel

import breeze.numerics.Bessel
import org.apache.spark.ml.linalg

class MaternKernel(sigma: Double, order: Double, phi: Double, distance: Double => Double) extends Kernel {/**
  * A vector of hyperparameters of the kernel
  */
override var hyperparameters = _
  /**
    * Stores some portion of the training sample
    */
  override var trainOption = _

  /**
    *
    * @return boundaries lower and upper: lower <= hyperparameter <= upper (inequalities are element-wise).
    */
  override def hyperparameterBoundaries = ???

  /**
    *
    * @param vectors
    * @return this
    */
  override def setTrainingVectors(vectors: Array[linalg.Vector]) = ???

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    *
    * @return a matrix K such that `K_{ij} = k(trainingVectros(i), trainingVectros(j))`
    */
  override def trainingKernel() = ???

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    *
    * @return the diagonal of the matrix which would be returned by `trainingKernel`.
    */
override def trainingKernelDiag() = ???

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    *
    * @return a pair of a kernel K (returned by `trainingKernel`)
    *         and an array of partial derivatives \partial K / \partial hyperparameters(i)
    */
  override def trainingKernelAndDerivative() = ???

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    *
    * @param test vectors (typically, those we want a prediction for)
    * @return a matrix K of size `test.size * trainVectors.size` such that K_{ij} = k(test(i), trainVectors(j))
    */
  override def crossKernel(test: Array[linalg.Vector]) = ???
}
