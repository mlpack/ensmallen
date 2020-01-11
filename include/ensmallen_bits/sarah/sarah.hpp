/**
 * @file sarah.hpp
 * @author Marcus Edel
 *
 * StochAstic Recusive gRadient algoritHm (SARAH).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SARAH_SARAH_HPP
#define ENSMALLEN_SARAH_SARAH_HPP

#include "sarah_update.hpp"
#include "sarah_plus_update.hpp"

namespace ens {

/**
 * StochAstic Recusive gRadient algoritHm (SARAH). is a variance reducing
 * stochastic recursive gradient algorithm for minimizing a function
 * which can be expressed as a sum of other functions.
 *
 * For more information, see the following.
 *
 * @code
 * @article{Nguyen2017,
 *   author  = {{Nguyen}, L.~M. and {Liu}, J. and {Scheinberg},
 *              K. and {Tak{\'a}{\v c}}, M.},
 *   title   = {SARAH: A Novel Method for Machine Learning Problems Using
 *              Stochastic Recursive Gradient},
 *   journal = {ArXiv e-prints},
 *   url     = {https://arxiv.org/abs/1703.00102}
 *   year    = 2017,
 * }
 * @endcode
 *
 * SARAH can optimize differentiable separable functions.  For more details, see
 * the documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam UpdatePolicyType update policy used by SARAHType during the iterative
 *    update process.
 */
template<typename UpdatePolicyType = SARAHUpdate>
class SARAHType
{
 public:
  /**
   * Construct the SARAH optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Batch size to use for each step.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param innerIterations The number of inner iterations allowed (0 means
   *    n / batchSize). Note that the full gradient is only calculated in
   *    the outer iteration.
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   * @param updatePolicy Instantiated update policy used to adjust the given
   *     parameters.
   * @param exactObjective Calculate the exact objective (Default: estimate the
   *        final objective obtained on the last pass over the data).
   */
  SARAHType(const double stepSize = 0.01,
            const size_t batchSize = 32,
            const size_t maxIterations = 1000,
            const size_t innerIterations = 0,
            const double tolerance = 1e-5,
            const bool shuffle = true,
            const UpdatePolicyType& updatePolicy = UpdatePolicyType(),
            const bool exactObjective = false);

  /**
   * Optimize the given function using SARAH. The given starting point will be
   * modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam SeparableFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename SeparableFunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(SeparableFunctionType& function,
           MatType& iterate,
           CallbackTypes&&... callbacks);

  //! Forward the MatType as GradType.
  template<typename SeparableFunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(SeparableFunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks)
  {
    return Optimize<SeparableFunctionType, MatType, MatType,
        CallbackTypes...>(function, iterate,
        std::forward<CallbackTypes>(callbacks)...);
  }

  //! Get the step size.
  double StepSize() const { return stepSize; }
  //! Modify the step size.
  double& StepSize() { return stepSize; }

  //! Get the batch size.
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size.
  size_t& BatchSize() { return batchSize; }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the maximum number of iterations (0 indicates default n / b).
  size_t InnerIterations() const { return innerIterations; }
  //! Modify the maximum number of iterations (0 indicates default n / b).
  size_t& InnerIterations() { return innerIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

  //! Get whether or not the actual objective is calculated.
  bool ExactObjective() const { return exactObjective; }
  //! Modify whether or not the actual objective is calculated.
  bool& ExactObjective() { return exactObjective; }

  //! Get the update policy.
  const UpdatePolicyType& UpdatePolicy() const { return updatePolicy; }
  //! Modify the update policy.
  UpdatePolicyType& UpdatePolicy() { return updatePolicy; }

 private:
  //! The step size for each example.
  double stepSize;

  //! The batch size for processing.
  size_t batchSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The maximum number of allowed inner iterations per epoch.
  size_t innerIterations;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;

  //! Controls whether or not the actual Objective value is calculated.
  bool exactObjective;

  //! The update policy used to update the parameters in each iteration.
  UpdatePolicyType updatePolicy;
};

// Convenience typedefs.

/**
 * Standard stochastic variance reduced gradient.
 */
using SARAH = SARAHType<SARAHUpdate>;

/**
 * Stochastic variance reduced gradient with Barzilai-Borwein.
 */
using SARAH_Plus = SARAHType<SARAHPlusUpdate>;

} // namespace ens

// Include implementation.
#include "sarah_impl.hpp"

#endif
