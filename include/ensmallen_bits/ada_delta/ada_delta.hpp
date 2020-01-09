/**
 * @file ada_delta.hpp
 * @author Ryan Curtin
 * @author Vasanth Kalingeri
 * @author Abhinav Moudgil
 *
 * Implementation of the AdaDelta optimizer. AdaDelta is an optimizer that
 * dynamically adapts over time using only first order information.
 * Additionally, AdaDelta requires no manual tuning of a learning rate.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_DELTA_ADA_DELTA_HPP
#define ENSMALLEN_ADA_DELTA_ADA_DELTA_HPP

#include <ensmallen_bits/sgd/sgd.hpp>
#include "ada_delta_update.hpp"

namespace ens {

/**
 * AdaDelta is an optimizer that uses two ideas to improve upon the two main
 * drawbacks of the Adagrad method:
 *
 *  - Accumulate Over Window
 *  - Correct Units with Hessian Approximation
 *
 * For more information, see the following.
 *
 * @code
 * @article{Zeiler2012,
 *   author  = {Matthew D. Zeiler},
 *   title   = {{ADADELTA:} An Adaptive Learning Rate Method},
 *   journal = {CoRR},
 *   year    = {2012}
 * }
 * @endcode
 *
 * AdaDelta can optimize differentiable separable functions.  For more details,
 * see the documentation on function types included with this distribution or on
 * the ensmallen website.
 */
class AdaDelta
{
 public:
  /**
   * Construct the AdaDelta optimizer with the given function and parameters.
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand. The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in one step.
   * @param rho Smoothing constant.
   * @param epsilon Value used to initialise the mean squared gradient
   *        parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   * @param resetPolicy If true, parameters are reset before every Optimize
   *        call; otherwise, their values are retained.
   * @param exactObjective Calculate the exact objective (Default: estimate the
   *        final objective obtained on the last pass over the data).
   */
  AdaDelta(const double stepSize = 1.0,
           const size_t batchSize = 32,
           const double rho = 0.95,
           const double epsilon = 1e-6,
           const size_t maxIterations = 100000,
           const double tolerance = 1e-5,
           const bool shuffle = true,
           const bool resetPolicy = true,
           const bool exactObjective = false);

  /**
   * Optimize the given function using AdaDelta. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
   * objective value is returned. The SeparableFunctionType is checked for
   * API consistency at compile time.
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
           CallbackTypes&&... callbacks)
  {
    return optimizer.Optimize<SeparableFunctionType, MatType, GradType,
        CallbackTypes...>(function, iterate,
        std::forward<CallbackTypes>(callbacks)...);
  }

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
  double StepSize() const { return optimizer.StepSize(); }
  //! Modify the step size.
  double& StepSize() { return optimizer.StepSize(); }

  //! Get the batch size.
  size_t BatchSize() const { return optimizer.BatchSize(); }
  //! Modify the batch size.
  size_t& BatchSize() { return optimizer.BatchSize(); }

  //! Get the smoothing parameter.
  double Rho() const { return optimizer.UpdatePolicy().Rho(); }
  //! Modify the smoothing parameter.
  double& Rho() { return optimizer.UpdatePolicy().Rho(); }

  //! Get the value used to initialise the mean squared gradient parameter.
  double Epsilon() const { return optimizer.UpdatePolicy().Epsilon(); }
  //! Modify the value used to initialise the mean squared gradient parameter.
  double& Epsilon() { return optimizer.UpdatePolicy().Epsilon(); }

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return optimizer.MaxIterations(); }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return optimizer.MaxIterations(); }

  //! Get the tolerance for termination.
  double Tolerance() const { return optimizer.Tolerance(); }
  //! Modify the tolerance for termination.
  double& Tolerance() { return optimizer.Tolerance(); }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return optimizer.Shuffle(); }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return optimizer.Shuffle(); }

  //! Get whether or not the actual objective is calculated.
  bool ExactObjective() const { return optimizer.ExactObjective(); }
  //! Modify whether or not the actual objective is calculated.
  bool& ExactObjective() { return optimizer.ExactObjective(); }

  //! Get whether or not the update policy parameters
  //! are reset before Optimize call.
  bool ResetPolicy() const { return optimizer.ResetPolicy(); }
  //! Modify whether or not the update policy parameters
  //! are reset before Optimize call.
  bool& ResetPolicy() { return optimizer.ResetPolicy(); }

 private:
  //! The Stochastic Gradient Descent object with AdaDelta policy.
  SGD<AdaDeltaUpdate> optimizer;
};

} // namespace ens

// Include implementation.
#include "ada_delta_impl.hpp"

#endif
