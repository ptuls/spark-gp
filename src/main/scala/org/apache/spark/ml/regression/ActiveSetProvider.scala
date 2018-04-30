package org.apache.spark.ml.regression

import breeze.linalg.{inv, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.kernel.Kernel
import org.apache.spark.rdd.RDD

trait ActiveSetProvider extends Serializable {
  def apply(activeSetSize: Int,
            expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
            points: RDD[LabeledPoint],
            kernel: () => Kernel,
            optimalHyperparameter: BDV[Double],
            sigma2: Double,
            seed : Long) : Array[Vector]
}

/**
  * Chooses the active set uniformly
  */
object RandomActiveSetProvider extends ActiveSetProvider {
  override def apply(activeSetSize: Int,
                     expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
                     points: RDD[LabeledPoint],
                     kernel: () => Kernel,
                     optimalHyperparameter: BDV[Double],
                     sigma2: Double,
                     seed: Long): Array[Vector] =
    points.takeSample(withReplacement = false, activeSetSize, seed).map(_.features)
}

/**
  * Implements the approach suggested in
  * Fast Forward Selection to Speed Up Sparse Gaussian Process Regression by Seeger et al. 2003
  *
  */
object GreedilyOptimizingActiveSetProvider extends ActiveSetProvider with GaussianProcessRegressionHelper {
  def apply(activeSetSize: Int,
            expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
            points: RDD[LabeledPoint],
            kernel: () => Kernel,
            optimalHyperparameter: BDV[Double],
            sigma2: Double,
            seed : Long) = {
    var activeSet = points.takeSample(withReplacement = false, 1, seed).map(_.features)

    while (activeSet.length <  activeSetSize) {
      val Kmm = kernel().setHyperparameters(optimalHyperparameter).setTrainingVectors(activeSet).trainingKernel()
      regularizeMatrix(Kmm, sigma2)
      val activeSetBC = points.sparkContext.broadcast(activeSet)
      val next = getNext(Kmm, expertLabelsAndKernels, activeSetBC, sigma2)
      activeSet :+= next
      activeSetBC.destroy()
    }

    activeSet
  }

  private def getNext(Kmm : BDM[Double],
                        expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
                        activeSetBC: Broadcast[Array[Vector]],
                        sigma2: Double) = {
    val KinvBC = expertLabelsAndKernels.sparkContext.broadcast(inv(Kmm))

    val labels2crossKernels = expertLabelsAndKernels.map{ case (y, k) =>
      (y, k.crossKernel(activeSetBC.value), k.trainingKernelDiag(), k.trainOption.get)
    }.cache()

    val (matrixKmnKnm, vectorKmny)  = labels2crossKernels
      .map {case(y, crossKernel, _, _) => (crossKernel * crossKernel.t, crossKernel * y) }
      .reduce((a,b) => (a._1 + b._1, a._2 + b._2))

    val positiveDefiniteMatrix = sigma2 * Kmm + matrixKmnKnm  // sigma^2 K_mm + K_mn * K_nm
    assertSymPositiveDefinite(positiveDefiniteMatrix)
    val invPDMBC = expertLabelsAndKernels.sparkContext.broadcast(inv(positiveDefiniteMatrix))

    val magicVector = positiveDefiniteMatrix \ vectorKmny

    val likelihood = vectorKmny.t * magicVector

    val deltas2vectors = labels2crossKernels.map {case (y, crossKernel, kernelDiag, vectors) =>
      val deltas = (0 until crossKernel.cols).map{ i =>
        val column = crossKernel(::, i)
        val yi = y(i)
        val Kii = kernelDiag(i)
        val pi = column.t * KinvBC.value * column
        val qi = column.t * invPDMBC.value * column
        val mui = column.t * magicVector
        val sigma = math.sqrt(sigma2)

        val li = math.sqrt(Kii - pi)
        val ksii = 1d/ (sqr(sigma/li) + 1 - qi)
        val kappai = ksii * (1 + 2 * sqr(sigma/li))

        -math.log(sigma/li) - (math.log(ksii) + ksii * (1-kappai) / sigma2 * sqr(yi - mui) - kappai + 2) / 2
      }
      val i = deltas.indices.maxBy(deltas)
      (deltas(i), vectors(i))
    }

    val score2vector = deltas2vectors.filter(!_._1.isNaN).max()(new Ordering[(Double, Vector)] {
      override def compare(x: (Double, Vector), y: (Double, Vector)): Int = Ordering[Double].compare(x._1, y._1)
    })

    labels2crossKernels.unpersist()
    score2vector._2
  }

  private def sqr(x : Double) = x * x
}