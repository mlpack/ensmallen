/**
 * @file fista.hpp
 * @author Ryan Curtin
 *
 * An implementation of FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FISTA_FISTA_HPP
#define ENSMALLEN_FISTA_FISTA_HPP

#include "../fbs/l1_penalty.hpp"
#include "../fbs/l1_constraint.hpp"

namespace ens {

/**
 * FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) is a proximal
 * gradient optimization technique for optimizing a function of the form
 *
 *   h(x) = f(x) + g(x)
 *
 * where f(x) is a differentiable function and g(x) is an arbitrary
 * non-differentiable function.  In such a situation, standard gradient descent
 * techniques cannot work because of the non-differentiability of g(x).  To work
 * around this, FISTA takes a _forward step_ that is just a gradient descent
 * step on f(x), and then a _backward step_ that is the _proximal operator_
 * corresponding to g(x).  This continues until convergence.
 *
 * This implementation of FISTA allows specification of the backward step (or
 * proximal operator) via the `BackwardStepType` template parameter.  When using
 * FBS, the differentiable `FunctionType` given to `Optimize()` should be f(x),
 * *not* the combined function h(x).  g(x) should be specified by the choice of
 * `BackwardStepType` (e.g. `L1Penalty` or `L1Maximum`).  The `Optimize()`
 * function will then return optimized coordinates for h(x), not f(x).
 *
 * For more information, see the following paper:
 *
 * ```
 * @article{beck2009fast,
 *   title={A fast iterative shrinkage-thresholding algorithm for linear inverse
 *       problems},
 *   author={Beck, Amir and Teboulle, Marc},
 *   journal={SIAM Journal On Imaging Sciences},
 *   volume={2},
 *   number={1},
 *   pages={183--202},
 *   year={2009},
 *   publisher={SIAM}
 * }
 * ```
 */
template<typename BackwardStepType = L1Penalty>
class FISTA
{
 public:
  /**
   * Construct the FISTA optimizer with the given options, using a
   * default-constructed BackwardStepType.
   */
  FISTA(const size_t maxIterations = 10000,
        const double tolerance = 1e-10,
        const size_t maxLineSearchSteps = 50,
        const double stepSizeAdjustment = 2.0,
        const bool estimateStepSize = true,
        const size_t estimateTrials = 10,
        const double maxStepSize = 0.001);

  /**
   * Construct the FISTA optimizer with the given options.
   */
  FISTA(BackwardStepType backwardStepType,
        const size_t maxIterations = 10000,
        const double tolerance = 1e-10,
        const size_t maxLineSearchSteps = 50,
        const double stepSizeAdjustment = 2.0,
        const bool estimateStepSize = true,
        const size_t estimateTrials = 10,
        const double maxStepSize = 0.001);

  /**
   * Optimize the given function using FISTA.  The given starting
   * point will be modified to store the finishing point of the algorithm,
   * the final objective value is returned.
   *
   * The FunctionType template class must provide the following functions:
   *
   *   double Evaluate(const arma::mat& coordinates);
   *   void Gradient(const arma::mat& coordinates,
   *                 arma::mat& gradient);
   *
   * @tparam FunctionType Type of function to be optimized.
   * @tparam MatType Type of objective matrix.
   * @tparam GradType Type of gradient matrix (default is MatType).
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to be optimized.
   * @param iterate Input with starting point, and will be modified to save
   *                the output optimial solution coordinates.
   * @param callbacks Callback functions.
   * @return Objective value at the final solution.
   */
  template<typename FunctionType, typename MatType, typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsMatrixType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(FunctionType& function,
           MatType& iterate,
           CallbackTypes&&... callbacks);

  //! Forward the MatType as GradType.
  template<typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(FunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks)
  {
    return Optimize<FunctionType, MatType, MatType,
        CallbackTypes...>(function, iterate,
        std::forward<CallbackTypes>(callbacks)...);
  }

  //! Get the backward step object.
  const BackwardStepType& BackwardStep() const { return backwardStep; }
  //! Modify the backward step object.
  BackwardStepType& BackwardStep() { return backwardStep; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance on the gradient norm for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance on the gradient norm for termination.
  double& Tolerance() { return tolerance; }

  //! Get the maximum number of line search steps.
  size_t MaxLineSearchSteps() const { return maxLineSearchSteps; }
  //! Modify the maximum number of line search steps.
  size_t& MaxLineSearchSteps() { return maxLineSearchSteps; }

  //! Get the step size adjustment parameter.
  double StepSizeAdjustment() const { return stepSizeAdjustment; }
  //! Modify the step size adjustment parameter.
  double& StepSizeAdjustment() { return stepSizeAdjustment; }

  //! Get whether or not to estimate the initial step size.
  bool EstimateStepSize() const { return estimateStepSize; }
  //! Modify whether or not to estimate the initial step size.
  bool& EstimateStepSize() { return estimateStepSize; }

  //! Get the number of trials to use for Lipschitz constant estimation.
  size_t EstimateTrials() const { return estimateTrials; }
  //! Modify the number of trials to use for Lipschitz constant estimation.
  size_t& EstimateTrials() { return estimateTrials; }

  //! Get the maximum step size.  If Optimize() has been called, this will
  //! contain the estimated maximum step size value.
  double MaxStepSize() const { return maxStepSize; }
  //! Modify the step size (ignored if EstimateStepSize() is true).
  double& MaxStepSize() { return maxStepSize; }

 private:
  //! Utility function: fill with random values.
  template<typename MatType>
  static void RandomFill(MatType& x,
                         const size_t rows,
                         const size_t cols,
                         const typename MatType::elem_type maxVal);

  template<typename eT>
  static void RandomFill(arma::SpMat<eT>& x,
                         const size_t rows,
                         const size_t cols,
                         const eT maxVal);

  template<typename FunctionType, typename MatType>
  void EstimateLipschitzStepSize(FunctionType& f, const MatType& x);

  //! The instantiated backward step object.
  BackwardStepType backwardStep;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! The maximum number of line search trials.
  size_t maxLineSearchSteps;

  //! The step size adjustment parameter for the line search.
  double stepSizeAdjustment;

  //! Whether or not to try and estimate the initial step size.
  bool estimateStepSize;

  //! Number of trials to use for initial step size estimation.
  size_t estimateTrials;

  //! The maximum step size to use (estimated if estimateStepSize is true).
  double maxStepSize;
};

} // namespace ens

// Include implementation.
#include "fista_impl.hpp"

#endif
