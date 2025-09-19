/**
 * @file fbs.hpp
 * @author Ryan Curtin
 *
 * An implementation of Forward-Backward Splitting (FBS).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FBS_FBS_HPP
#define ENSMALLEN_FBS_FBS_HPP

#include "l1_penalty.hpp"
#include "l1_constraint.hpp"

namespace ens {

/**
 * Forward-Backward Splitting is a proximal gradient optimization technique for
 * optimizing a function of the form
 *
 *   h(x) = f(x) + g(x)
 *
 * where f(x) is a differentiable function and g(x) is an arbitrary
 * non-differentiable function.  In such a situation, standard gradient descent
 * techniques cannot work because of the non-differentiability of g(x).  To work
 * around this, FBS takes a _forward step_ that is just a gradient descent step
 * on f(x), and then a _backward step_ that is the _proximal operator_
 * corresponding to g(x).  This continues until convergence.
 *
 * This implementation of FBS allows specification of the backward step (or
 * proximal operator) via the `BackwardStepType` template parameter.  When using
 * FBS, the differentiable `FunctionType` given to `Optimize()` should be f(x),
 * *not* the combined function h(x).  g(x) should be specified by the choice of
 * `BackwardStepType` (e.g. `L1Penalty` or `L1Maximum`).  The `Optimize()`
 * function will then return optimized coordinates for h(x), not f(x).
 *
 * For more information, see the following paper:
 *
 * ```
 * @article{goldstein2014field,
 *   title={A field guide to forward-backward splitting with a FASTA
 *       implementation},
 *   author={Goldstein, Tom and Studer, Christoph and Baraniuk, Richard},
 *   journal={arXiv preprint arXiv:1411.3406},
 *   year={2014}
 * }
 * ```
 */
template<typename BackwardStepType = L1Penalty>
class FBS
{
 public:
  /**
   * Construct the FBS optimizer with the given options, using a
   * default-constructed BackwardStepType.
   */
  FBS(const double stepSize = 0.001,
      const size_t maxIterations = 10000,
      const double tolerance = 1e-10);

  /**
   * Construct the FBS optimizer with the given options.
   */
  FBS(BackwardStepType backwardStepType,
      const double stepSize = 0.001,
      const size_t maxIterations = 10000,
      const double tolerance = 1e-10);

  /**
   * Optimize the given function using FBS.  The given starting
   * point will be modified to store the finishing point of the algorithm,
   * the final objective value is returned.
   *
   * FunctionType template class must provide the following functions:
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

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

 private:
  //! The instantiated backward step object.
  BackwardStepType backwardStep;

  //! The step size for FBS steps.
  double stepSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;
};

} // namespace ens

// Include implementation.
#include "fbs_impl.hpp"

#endif
