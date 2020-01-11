/**
 * @file iqn.hpp
 * @author Marcus Edel
 *
 * Definition of an incremental Quasi-Newton with local superlinear
 * convergence rate as proposed by A. Mokhtari et al. in "IQN: An Incremental
 * Quasi-Newton Method with Local Superlinear Convergence Rate".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_IQN_IQN_HPP
#define ENSMALLEN_IQN_IQN_HPP

namespace ens {

/**
 * IQN is a technique for minimizing a function which
 * can be expressed as a sum of other functions.  That is, suppose we have
 *
 * \f[
 * f(A) = \sum_{i = 0}^{n} f_i(A)
 * \f]
 * IQN is the first stochastic quasi- Newton method proven to converge
 * superlinearly in a local neighborhood of the optimal solution.
 *
 * For more information, please refer to:
 *
 * @code
 * @misc{1106.5730,
 *   author = {Mokhtari, Aryan and Eisen, Mark and Ribeiro, Alejandro},
 *   title  = {IQN: An Incremental Quasi-Newton Method with Local Superlinear
 *             Convergence Rate},
 *   year   = {2017},
 *   eprint = {arXiv:1702.00709},
 * }
 * @endcode
 *
 * This class is useful for data-dependent functions whose objective function
 * can be expressed as a sum of objective functions operating on an individual
 * point.  Then, IQN considers the gradient of the objective function operating
 * on an individual point in its update of \f$ A \f$.
 *
 * IQN can optimize differentiable separable functions.  For more details, see
 * the documentation on function types included with this distribution or on the
 * ensmallen website.
 */
class IQN
{
 public:
  /**
   * Construct the IQN optimizer with the given function and parameters.  The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Size of each batch.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   */
  IQN(const double stepSize = 0.01,
      const size_t batchSize = 10,
      const size_t maxIterations = 100000,
      const double tolerance = 1e-5);

  /**
   * Optimize the given function using IQN. The given starting point will be
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

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

 private:
  //! The step size for each example.
  double stepSize;

  //! The size of each batch.
  size_t batchSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;
};

} // namespace ens

// Include implementation.
#include "iqn_impl.hpp"

#endif
