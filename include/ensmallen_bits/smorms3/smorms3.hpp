/**
 * @file smorms3.hpp
 * @author Vivek Pal
 *
 * SMORMS3 i.e. squared mean over root mean squared cubed optimizer. It is a
 * hybrid of RMSprop, which estimates a safe and optimal distance based on
 * curvature and Yann LeCun’s method in "No more pesky learning rates".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SMORMS3_SMORMS3_HPP
#define ENSMALLEN_SMORMS3_SMORMS3_HPP

#include <ensmallen_bits/sgd/sgd.hpp>

#include "smorms3_update.hpp"

namespace ens {

/**
 * SMORMS3 is an optimizer that estimates a safe and optimal distance based on
 * curvature and normalizing the stepsize in the parameter space. It is a hybrid
 * of RMSprop and Yann LeCun’s method in "No more pesky learning rates".
 *
 * For more information, see the following.
 *
 * @code
 * @misc{Funk2015,
 *   author = {Simon Funk},
 *   title  = {RMSprop loses to SMORMS3 - Beware the Epsilon!},
 *   year   = {2015}
 *   url    = {http://sifter.org/~simon/journal/20150420.html}
 * }
 * @endcode
 *
 * SMORMS3 can optimize differentiable separable functions.  For more details,
 * see the documentation on function types included with this distribution or on
 * the ensmallen website.
 */
class SMORMS3
{
 public:
  /**
   * Construct the SMORMS3 optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process at each step.
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
  SMORMS3(const double stepSize = 0.001,
          const size_t batchSize = 32,
          const double epsilon = 1e-16,
          const size_t maxIterations = 100000,
          const double tolerance = 1e-5,
          const bool shuffle = true,
          const bool resetPolicy = true,
          const bool exactObjective = false);

  /**
   * Optimize the given function using SMORMS3. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
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
           CallbackTypes&&... callbacks)
  {
    // TODO: disallow sp_mat

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
  //! The Stochastic Gradient Descent object with SMORMS3Update update policy.
  SGD<SMORMS3Update> optimizer;
};

} // namespace ens

// Include implementation.
#include "smorms3_impl.hpp"

#endif
