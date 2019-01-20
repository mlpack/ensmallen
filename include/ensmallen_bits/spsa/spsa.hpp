/**
 * @file spsa.hpp
 * @author N Rajiv Vaidyanathan
 * @author Marcus Edel
 *
 * SPSA (Simultaneous perturbation stochastic approximation) method for
 * faster convergence.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SPSA_SPSA_HPP
#define ENSMALLEN_SPSA_SPSA_HPP

namespace ens {

/**
 * Implementation of the SPSA method. The SPSA algorithm approximates the
 * gradient of the function by finite differences along stochastic directions.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Spall1998,
 *   author  = {Spall, J. C.},
 *   title   = {An Overview of the Simultaneous Perturbation Method for
 *              Efficient Optimization},
 *   journal = {Johns Hopkins APL Technical Digest},
 *   volume  = {19},
 *   number  = {4},
 *   pages   = {482--492},
 *   year    = {1998}
 * }
 * @endcode
 *
 * SPSA can optimize differentiable separable functions.  For more details,
 * see the documentation on function types included with this distribution or on
 * the ensmallen website.
 */
class SPSA
{
 public:
  /**
   * Construct the SPSA optimizer with the given function and parameters.  The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param alpha Scaling exponent for the step size.
   * @param batchSize Batch size to use for each step.
   * @param gamma Scaling exponent for evaluation step size.
   * @param stepSize Scaling parameter for step size (named as 'a' in the paper).
   * @param evaluationStepSize Scaling parameter for evaluation step size (named as 'c' in the paper).
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   */
  SPSA(const double alpha = 0.602,
       const size_t batchSize = 32,
       const double gamma = 0.101,
       const double stepSize = 0.16,
       const double evaluationStepSize = 0.3,
       const size_t maxIterations = 100000,
       const double tolerance = 1e-5,
       const bool shuffle = true);

  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);

  //! Get the scaling exponent for the step size.
  double Alpha() const { return alpha; }
  //! Modify the scaling exponent for the step size.
  double& Alpha() { return alpha; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the scaling exponent for evaluation step size.
  double Gamma() const { return gamma; }
  //! Modify the scaling exponent for evaluation step size.
  double& Gamma() { return gamma; }

  //! Get the scaling parameter for step size.
  double StepSize() const { return stepSize; }
  //! Modify the scaling parameter for step size.
  double& StepSize() { return stepSize; }

  //! Get the scaling parameter for step size.
  double EvaluationStepSize() const { return evaluationStepSize; }
  //! Modify the scaling parameter for step size.
  double& EvaluationStepSize() { return evaluationStepSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

 private:
  //! Scaling exponent for the step size.
  double alpha;

  //! The batch size for processing.
  size_t batchSize;

  //! Scaling exponent for evaluation step size.
  double gamma;

  //! Scaling parameter for step size.
  double stepSize;

  //! Scaling parameter for step size.
  double evaluationStepSize;

  //! Control the amount of gradient update.
  double Ak;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;
};

} // namespace ens

// Include implementation.
#include "spsa_impl.hpp"

#endif
