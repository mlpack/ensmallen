/**
 * @file cd.hpp
 * @author Shikhar Bhardwaj
 *
 * Coordinate Descent (CD).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CD_CD_HPP
#define ENSMALLEN_CD_CD_HPP

#include "descent_policies/cyclic_descent.hpp"
#include "descent_policies/random_descent.hpp"
#include "descent_policies/greedy_descent.hpp"

namespace ens {

/**
 * Stochastic Coordinate descent is a technique for minimizing a function by
 * doing a line search along a single direction at the current point in the
 * iteration. The direction (or "coordinate") can be chosen cyclically, randomly
 * or in a greedy fashion(depending on the DescentPolicy).
 *
 * This optimizer is useful for problems with a smooth multivariate function
 * where computing the entire gradient for an update is infeasable. CD method
 * typically significantly outperform GD, especially on sparse problems with a
 * very large number variables/coordinates.
 *
 * For more information, see the following.
 * @code
 * @inproceedings{Shalev-Shwartz2009,
 *   author    = {Shalev-Shwartz, Shai and Tewari, Ambuj},
 *   title     = {Stochastic Methods for L1 Regularized Loss Minimization},
 *   booktitle = {Proceedings of the 26th Annual International Conference on
 *                Machine Learning},
 *   series    = {ICML '09},
 *   year      = {2009},
 *   isbn = {978-1-60558-516-1}
 * }
 * @endcode
 *
 * CD can optimize partially differentiable functions.  For more details, see
 * the documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam DescentPolicy Descent policy to decide the order in which the
 *     coordinate for descent is selected.
 */
template <typename DescentPolicyType = RandomDescent>
class CD
{
 public:
  /**
   * Construct the CD optimizer with the given function and parameters. The
   * default value here are not necessarily good for every problem, so it is
   * suggested that the values used are tailored for the task at hand. The
   * maximum number of iterations refers to the maximum number of "descents"
   * the algorithm does (in one iteration, the algorithm updates the
   * decision variable numFeatures times).
   *
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means to
   *    limit).
   * @param tolerance Maximum absolute tolerance to terminate the algorithm.
   * @param updateInterval The interval at which the objective is to be
   *    reported and checked for convergence.
   * @param descentPolicy The policy to use for picking up the coordinate to
   *    descend on.
   */
  CD(const double stepSize = 0.01,
     const size_t maxIterations = 100000,
     const double tolerance = 1e-5,
     const size_t updateInterval = 1e3,
     const DescentPolicyType descentPolicy = DescentPolicyType());

  /**
   * Optimize the given function using stochastic coordinate descent. The
   * given starting point will be modified to store the finishing point of
   * the optimization, and the final objective value is returned.
   *
   * @tparam ResolvableFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value at the final point.
   */
  template<typename ResolvableFunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(ResolvableFunctionType& function,
           MatType& iterate,
           CallbackTypes&&... callbacks);

  //! Forward arma::SpMat<typename MatType::elem_type> as GradType.
  template<typename SeparableFunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(SeparableFunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks)
  {
    return Optimize<SeparableFunctionType, MatType,
        arma::SpMat<typename MatType::elem_type>, CallbackTypes...>(
        function, iterate, std::forward<CallbackTypes>(callbacks)...);
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

  //! Get the update interval for reporting objective.
  size_t UpdateInterval() const { return updateInterval; }
  //! Modify the update interval for reporting objective.
  size_t& UpdateInterval() { return updateInterval; }

  //! Get the descent policy.
  DescentPolicyType DescentPolicy() const { return descentPolicy; }
  //! Modify the descent policy.
  DescentPolicyType& DescentPolicy() { return descentPolicy; }

 private:
  //! The step size for each example.
  double stepSize;

  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The tolerance for termination.
  double tolerance;

  //! The update interval for reporting objective and testing for convergence.
  size_t updateInterval;

  //! The descent policy used to pick the coordinates for the update.
  DescentPolicyType descentPolicy;
};

} // namespace ens

// Include implementation.
#include "cd_impl.hpp"

namespace ens {

/**
 * Backwards-compatibility alias; this can be removed after ensmallen 3.10.0.
 * The history here is that CD was originally named SCD, but that is an
 * inaccurate name because this is not a stochastic technique; thus, it was
 * renamed SCD.
 */
template<typename DescentPolicyType = RandomDescent>
using SCD = CD<DescentPolicyType>;

// Convenience typedefs.
using RandomCD = CD<RandomDescent>;
using GreedyCD = CD<GreedyDescent>;
using CyclicCD = CD<CyclicDescent>;

} // namespace ens

#endif
