package org.apache.spark.ml.regression

import breeze.linalg.{inv, logdet, DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.optimize.{DiffFunction, LBFGSB}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.regression.kernel.{Kernel, RBFKernel}
import org.apache.spark.ml.util.{Identifiable, Instrumentation}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.mutable

private[regression] trait GaussianProcessRegressionParams
    extends PredictorParams
    with HasMaxIter
    with HasTol
    with HasAggregationDepth
    with HasSeed {

  final val activeSetProvider = new Param[ActiveSetProvider](
    this,
    "activeSetProvider",
    "the class which provides the active set used by Projected Process Approximation")

  final val kernel = new Param[() => Kernel](
    this,
    "kernel",
    "function of no arguments which returns " +
      "the kernel of the prior Gaussian Process")

  final val datasetSizeForExpert = new IntParam(
    this,
    "datasetSizeForExpert",
    "The number of data points fed to each expert. " +
      "Time and space complexity of training quadratically grows with it.")

  final val sigma2 = new DoubleParam(
    this,
    "sigma2",
    "The variance of noise in the inputs. The value is added to the diagonal of the " +
      "kernel Matrix. Also prevents numerical issues associated with inversion " +
      "of a computationally singular matrix "
  )

  final val activeSetSize = new IntParam(
    this,
    "activeSetSize",
    "Number of latent functions to project the process onto. " +
      "The size of the produced model and prediction complexity " +
      "linearly depend on this value."
  )

  def setActiveSetProvider(value: ActiveSetProvider): this.type =
    set(activeSetProvider, value)
  setDefault(activeSetProvider -> RandomActiveSetProvider)

  def setDatasetSizeForExpert(value: Int): this.type =
    set(datasetSizeForExpert, value)
  setDefault(datasetSizeForExpert -> 100)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setSigma2(value: Double): this.type = set(sigma2, value)
  setDefault(sigma2 -> 1e-3)

  def setKernel(value: () => Kernel): this.type = set(kernel, value)
  setDefault(kernel -> (() => new RBFKernel()))

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setActiveSetSize(value: Int): this.type = set(activeSetSize, value)
  setDefault(activeSetSize -> 100)

  def setSeed(value: Long): this.type = set(seed, value)
}

/**
  * Gaussian Process Regression.
  *
  * Fitting of hyperparameters and prediction for GPR is infeasible for large datasets due to
  * high computational complexity O(N^3^).
  *
  * This implementation relies on the Bayesian Committee Machine proposed in [2] for fitting and on
  * Projected Process Approximation for prediction Chapter 8.3.4 [1].
  *
  * This way the linear complexity in sample size is achieved for fitting,
  * while prediction complexity doesn't depend on it.
  *
  * [1] Carl Edward Rasmussen and Christopher K. I. Williams. 2005. Gaussian Processes for Machine Learning
  * (Adaptive Computation and Machine Learning). The MIT Press.
  *
  * [2] Marc Peter Deisenroth and Jun Wei Ng. 2015. Distributed Gaussian processes.
  * In Proceedings of the 32nd International Conference on International Conference on Machine Learning
  * Volume 37 (ICML'15), Francis Bach and David Blei (Eds.), Vol. 37. JMLR.org 1481-1490.
  *
  */
class GaussianProcessRegression(override val uid: String)
    extends Regressor[Vector,
                      GaussianProcessRegression,
                      GaussianProcessRegressionModel]
    with GaussianProcessRegressionParams
    with GaussianProcessRegressionHelper
    with Logging {

  def this() = this(Identifiable.randomUID("gaussProcessReg"))

  override protected def train(
      dataset: Dataset[_]): GaussianProcessRegressionModel = {
    val instr = Instrumentation.create(this, dataset)

    val points: RDD[LabeledPoint] = getPoints(dataset).cache()

    val expertLabelsAndKernels: RDD[(BDV[Double], Kernel)] =
      groupForExperts(points)
        .map { chunk =>
          val (labels, trainingVectors) =
            chunk.map(lp => (lp.label, lp.features)).toArray.unzip
          (BDV(labels: _*), $(kernel)().setTrainingVectors(trainingVectors))
        }
        .cache()

    instr.log("Optimising the kernel hyperparameters")
    val optimalHyperparameters =
      optimizeHyperparameters(expertLabelsAndKernels, $(sigma2))
    instr.log("Optimal hyperparameter values: " + optimalHyperparameters)

    expertLabelsAndKernels.foreach(
      _._2.setHyperparameters(optimalHyperparameters))

    val activeSet = $(activeSetProvider)($(activeSetSize),
                                         expertLabelsAndKernels,
                                         points,
                                         $(kernel),
                                         optimalHyperparameters,
                                         $(sigma2),
                                         $(seed))

    points.unpersist()

    val (matrixKmnKnm, vectorKmny) =
      getMatrixKmnKnmAndVectorKmny(expertLabelsAndKernels, activeSet)

    expertLabelsAndKernels.unpersist()

    val optimalKernel = $(kernel)()
      .setTrainingVectors(activeSet)
      .setHyperparameters(optimalHyperparameters)

    // inv(sigma^2 K_mm + K_mn * K_nm) * K_mn * y
    val magicVector = getMagicVector(optimalKernel,
                                     $(sigma2),
                                     matrixKmnKnm,
                                     vectorKmny,
                                     activeSet,
                                     optimalHyperparameters)

    val model =
      new GaussianProcessRegressionModel(uid, magicVector, optimalKernel)
    instr.logSuccess(model)
    model
  }

  private def getPoints(dataset: Dataset[_]) = {
    dataset.select(col($(labelCol)), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }
  }

  private def optimizeHyperparameters(
      expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
      sigma2: Double) = {
    val f = new DiffFunction[BDV[Double]] with Serializable {
      private val cache = mutable.HashMap[BDV[Double], (Double, BDV[Double])]()

      override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
        cache.getOrElseUpdate(x, calculateNoMemory(x))
      }

      private def calculateNoMemory(x: BDV[Double]): (Double, BDV[Double]) = {
        expertLabelsAndKernels.treeAggregate((0d, BDV.zeros[Double](x.length)))(
          {
            case (u, (y, k)) =>
              k.setHyperparameters(x)
              val (likelihood, gradient) = likelihoodAndGradient(y, k, sigma2)
              (u._1 + likelihood, u._2 += gradient)
          }, {
            case (u, v) =>
              (u._1 + v._1, u._2 += v._2)
          }
        )
      }
    }

    val x0 = $(kernel)().hyperparameters
    val (lower, upper) = $(kernel)().hyperparameterBoundaries
    val solver =
      new LBFGSB(lower, upper, maxIter = $(maxIter), tolerance = $(tol))

    solver.minimize(f, x0)
  }

  private def likelihoodAndGradient(y: BDV[Double],
                                    kernel: Kernel,
                                    sigma2: Double): (Double, BDV[Double]) = {
    val (k, derivative) = kernel.trainingKernelAndDerivative()
    regularizeMatrix(k, sigma2)
    val Kinv = inv(k)
    val alpha = Kinv * y
    val likelihood = 0.5 * (y.t * alpha) + 0.5 * logdet(k)._2

    val alphaAlphaTMinusKinv = alpha * alpha.t
    alphaAlphaTMinusKinv -= Kinv

    val gradient = derivative.map(derivative =>
      -0.5 * traceOfProduct(alphaAlphaTMinusKinv, derivative))
    (likelihood, BDV(gradient: _*))
  }

  private def traceOfProduct(a: BDM[Double], b: BDM[Double]) =
    BDV(a.data).t * BDV(b.data)

  private def groupForExperts(points: RDD[LabeledPoint]) = {
    val numberOfExperts =
      Math.round(points.count().toDouble / $(datasetSizeForExpert))
    points.zipWithIndex
      .map {
        case (instance, index) =>
          (index % numberOfExperts, instance)
      }
      .groupByKey()
      .map(_._2)
  }

  override def copy(extra: ParamMap): GaussianProcessRegression =
    defaultCopy(extra)
}

class GaussianProcessRegressionModel private[regression] (
    override val uid: String,
    val magicVector: BDV[Double],
    val kernel: Kernel)
    extends RegressionModel[Vector, GaussianProcessRegressionModel] {

  override def predict(features: Vector): Double = {
    kernel.crossKernel(features) * magicVector
  }

  def predictWithStdDev(features: Vector): Array[Double] = {
    val mean = predict(features)
    val stdDev = kernel.crossKernel(features) * magicVector
    Array(mean, stdDev)
  }

  override def copy(extra: ParamMap): GaussianProcessRegressionModel = {
    val newModel = copyValues(
      new GaussianProcessRegressionModel(uid, magicVector, kernel),
      extra)
    newModel.setParent(parent)
  }
}

trait GaussianProcessRegressionHelper {
  class NotPositiveDefiniteException
      extends Exception(
        "Some matrix which is supposed to be " +
          "positive definite is not. This probably happened due to `sigma2` parameter being too small." +
          " Try to gradually increase it.")

  /**
    * Does some matrix multiplications in a distributed manner.
    *
    * @param expertLabelsAndKernels
    * @param activeSet
    * @return (K_mn * K_nm, K_mn * y)
    */
  def getMatrixKmnKnmAndVectorKmny(
      expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
      activeSet: Array[Vector]): (BDM[Double], BDV[Double]) = {
    val activeSetSize = activeSet.length
    val activeSetBC = expertLabelsAndKernels.sparkContext.broadcast(activeSet)

    expertLabelsAndKernels.treeAggregate(
      BDM.zeros[Double](activeSetSize, activeSetSize),
      BDV.zeros[Double](activeSetSize))(
      {
        case (u, (y, k)) =>
          val kernelMatrix = k.crossKernel(activeSetBC.value)
          u._1 += kernelMatrix * kernelMatrix.t
          u._2 += kernelMatrix * y
          u
      }, {
        case (u, v) =>
          u._1 += v._1
          u._2 += v._2
          u
      }
    )
  }

  /**
    * Computes inv(sigma^2^ K_mm + K_mn * K_nm) * K_mn * y
    *
    * @param kernel Should be fully initialized: both `setHyperparameters` and `setTrainingVectors`
    *               should called beforehand
    * @param sigma2
    * @param matrixKmnKnm
    * @param vectorKmny
    * @param activeSet
    * @param optimalHyperparameter
    * @return
    */
  def getMagicVector(kernel: Kernel,
                     sigma2: Double,
                     matrixKmnKnm: BDM[Double],
                     vectorKmny: BDV[Double],
                     activeSet: Array[Vector],
                     optimalHyperparameter: BDV[Double]): BDV[Double] = {
    val Kmm = kernel.trainingKernel()
    regularizeMatrix(Kmm, sigma2)

    val positiveDefiniteMatrix = sigma2 * Kmm + matrixKmnKnm // sigma^2 K_mm + K_mn * K_nm
    assertSymPositiveDefinite(positiveDefiniteMatrix)
    positiveDefiniteMatrix \ vectorKmny
  }

  protected def regularizeMatrix(matrix: BDM[Double],
                                 regularization: Double): Unit = {
    var i = 0
    while (i < matrix.cols) {
      matrix(i, i) += regularization
      i += 1
    }
  }

  protected def assertSymPositiveDefinite(matrix: BDM[Double]): Unit = {
    if (any(eigSym(matrix).eigenvalues <:< 0d))
      throw new NotPositiveDefiniteException
  }
}
