/**
 * @file parallel_sgd.hpp
 * @author Shikhar Bhardwaj
 *
 * Parallel Stochastic Gradient Descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PARALLEL_SGD_HPP
#define ENSMALLEN_PARALLEL_SGD_HPP

#include "decay_policies/constant_step.hpp"
#include "decay_policies/exponential_backoff.hpp"

namespace ens {

/**
 * An implementation of parallel stochastic gradient descent using the lock-free
 * HOGWILD! approach.
 *
 * For more information, see the following.
 *
 * @code
 * @misc{1106.5730,
 *   Author = {Feng Niu and Benjamin Recht and Christopher Re and Stephen J.
 *             Wright},
 *   Title  = {HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic
 *             Gradient Descent},
 *   Year   = {2011},
 *   Eprint = {arXiv:1106.5730},
 * }
 * @endcode
 *
 * ParallelSGD can optimize sparse differentiable separable functions.  For more
 * details, see the documentation on function types included with this
 * distribution or on the ensmallen website.
 *
 * @tparam DecayPolicyType Step size update policy used by parallel SGD
 *     to update the stepsize after each iteration.
 */
template <typename DecayPolicyType = ConstantStep>
class ParallelSGD
{
 public:
  /**
   * Construct the parallel SGD optimizer to optimize the given function with
   * the given parameters. One iteration means one batch of datapoints processed
   * by each thread.
   *
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.
   *
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param threadShareSize Number of datapoints to be processed in one
   *     iteration by each thread.
   * @param tolerance Maximum absolute tolerance to terminate the algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   * @param decayPolicy The step size update policy to use.
  */
  ParallelSGD(const size_t maxIterations,
              const size_t threadShareSize,
              const double tolerance = 1e-5,
              const bool shuffle = true,
              const DecayPolicyType& decayPolicy = DecayPolicyType());

  /**
   * Optimize the given function using the parallel SGD algorithm. The given
   * starting point will be modified to store the finishing point of the
   * algorithm, and the value of the loss function at the final point is
   * returned.
   *
   * @tparam SparseFunctionType Type of function to be optimized.
   * @tparam MatType Type of the objective function.
   * @tparam GradType Type of gradient (it is strongly suggested that this be a
   *     sparse matrix of some sort!).
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to be optimized(minimized).
   * @param iterate Starting point(will be modified).
   * @param callbacks Callback functions.
   * @return Objective value at the final point.
   */
  template <typename SparseFunctionType,
            typename MatType,
            typename GradType,
            typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(SparseFunctionType& function,
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

  //! Get the maximum number of iterations (0 indicates no limits).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limits).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the number of datapoints to be processed in one iteration by each
  //! thread.
  size_t ThreadShareSize() const { return threadShareSize; }
  //! Modify the number of datapoints to be processed in one iteration by each
  //! thread.
  size_t& ThreadShareSize() { return threadShareSize; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

  //! Get whether or not the individual functions are shuffled.
  bool Shuffle() const { return shuffle; }
  //! Modify whether or not the individual functions are shuffled.
  bool& Shuffle() { return shuffle; }

  //! Get the step size decay policy.
  DecayPolicyType& DecayPolicy() const { return decayPolicy; }
  //! Modify the step size decay policy.
  DecayPolicyType& DecayPolicy() { return decayPolicy; }

 private:
  //! The maximum number of allowed iterations.
  size_t maxIterations;

  //! The number of datapoints to be processed in one iteration by each thread.
  size_t threadShareSize;

  //! The tolerance for termination.
  double tolerance;

  //! Controls whether or not the individual functions are shuffled when
  //! iterating.
  bool shuffle;

  //! The step size decay policy.
  DecayPolicyType decayPolicy;
};

} // namespace ens

// Include implementation.
#include "parallel_sgd_impl.hpp"

#endif
