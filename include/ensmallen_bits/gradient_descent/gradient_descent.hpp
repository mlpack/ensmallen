/**
 * @file gradient_descent.hpp
 * @author Sumedh Ghaisas
 *
 * Simple Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP
#define ENSMALLEN_GRADIENT_DESCENT_GRADIENT_DESCENT_HPP

namespace ens {

/**
 * Gradient Descent is a technique to minimize a function. To find a local
 * minimum of a function using gradient descent, one takes steps proportional
 * to the negative of the gradient of the function at the current point,
 * producing the following update scheme:
 *
 * \f[
 * A_{j + 1} = A_j + \alpha \nabla F(A)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size. \f$ F \f$
 * is the function being optimized. The algorithm continues until \f$ j
 * \f$ reaches the maximum number of iterations---or when an update produces
 * an improvement within a certain tolerance \f$ \epsilon \f$.  That is,
 *
 * \f[
 * | F(A_{j + 1}) - F(A_j) | < \epsilon.
 * \f]
 *
 * The parameter \f$\epsilon\f$ is specified by the tolerance parameter to the
 * constructor.
 *
 * GradientDescent can optimize differentiable functions.  For more details, see
 * the documentation on function types included with this distribution or on the
 * ensmallen website.
 */
class GradientDescent
{
 public:
  /**
   * Construct the Gradient Descent optimizer with the given function and
   * parameters.  The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored to the task
   * at hand.
   *
   * @param function Function to be optimized (minimized).
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   */
  GradientDescent(const double stepSize = 0.01,
                  const size_t maxIterations = 100000,
                  const double tolerance = 1e-5);

  /**
   * Optimize the given function using gradient descent.  The given starting
   * point will be modified to store the finishing point of the algorithm, and
   * the final objective value is returned.
   *
   * @tparam FunctionType Type of the function to optimize.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename FunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(FunctionType& function,
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

  /**
   * Assert all dimensions are numeric and optimize the given function using
   * gradient descent. The given starting point will be modified to store the
   * finishing point of the algorithm, and the final objective value is
   * returned.
   *
   * This overload is intended to be used primarily by the hyper-parameter
   * tuning module.
   *
   * @tparam FunctionType Type of the function to optimize.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param categoricalDimensions A vector of dimension information.  If a value
   *     is true, then that dimension is a categorical dimension.
   * @param numCategories Number of categories in each categorical dimension.
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename FunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(FunctionType& function,
           MatType& iterate,
           const std::vector<bool>& categoricalDimensions,
           const arma::Row<size_t>& numCategories,
           CallbackTypes&&... callbacks);

  //! Forward the MatType as GradType.
  template<typename FunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(
      FunctionType& function,
      MatType& iterate,
      const std::vector<bool>& categoricalDimensions,
      const arma::Row<size_t>& numCategories,
      CallbackTypes&&... callbacks)
  {
    return Optimize<FunctionType, MatType, MatType,
        CallbackTypes...>(function, iterate, categoricalDimensions,
        numCategories, std::forward<CallbackTypes>(callbacks)...);
  }

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
  //! The step size for each example.
  double stepSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;
};

} // namespace ens

#include "gradient_descent_impl.hpp"

#endif
