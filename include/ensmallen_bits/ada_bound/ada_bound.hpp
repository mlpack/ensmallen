/**
 * @file ada_bound.hpp
 * @author Marcus Edel
 *
 * Class wrapper for the AdaBound and AMSBound update Policy.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_BOUND_HPP
#define ENSMALLEN_ADA_BOUND_HPP

#include <ensmallen_bits/sgd/sgd.hpp>
#include "ada_bound_update.hpp"
#include "ams_bound_update.hpp"

namespace ens {

/**
 * AdaBound and AMSBound, employ dynamic bounds on learning rates to achieve a
 * gradual and smooth transition from adaptive methods to SGD.
 *
 * For more information, see the following.
 *
 * @code
 * @inproceedings{Luo2019AdaBound,
 *   author    = {Luo, Liangchen and Xiong, Yuanhao and Liu, Yan and Sun, Xu},
 *   title     = {Adaptive Gradient Methods with Dynamic Bound of Learning
 *                Rate},
 *   booktitle = {Proceedings of the 7th International Conference on Learning
 *                Representations},
 *   month     = {May},
 *   year      = {2019},
 *   address   = {New Orleans, Louisiana}
 * }
 * @endcode
 *
 * AdaBound and AMSBound can optimize differentiable separable functions.
 * For more details, see the documentation on function types included with this
 * distribution or on the ensmallen website.
 */
template<typename UpdatePolicyType = AdaBoundUpdate,
         typename DecayPolicyType = NoDecay>
class AdaBoundType
{
 public:
  /**
   * Construct the AdaBoundType optimizer with the given function and
   * parameters. AdaBoundType is sensitive to its parameters and hence a good
   * hyper paramater selection is necessary as its default may not fit every
   * case.
   *
   * The maximum number of iterations refers to the maximum number of
   * points that are processed (i.e., one iteration equals one point; one
   * iteration does not equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param finalLr The final (SGD) learning rate.
   * @param gamma The convergence speed of the bound functions.
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
   *     estimates.
   * @param epsilon Value used to initialise the mean squared gradient
   *     parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   * @param resetPolicy If true, parameters are reset before every Optimize
   *     call; otherwise, their values are retained.
   * @param exactObjective Calculate the exact objective (Default: estimate the
   *        final objective obtained on the last pass over the data).
   */
  AdaBoundType(const double stepSize = 0.001,
               const size_t batchSize = 32,
               const double finalLr = 0.1,
               const double gamma = 1e-3,
               const double beta1 = 0.9,
               const double beta2 = 0.999,
               const double epsilon = 1e-8,
               const size_t maxIterations = 100000,
               const double tolerance = 1e-5,
               const bool shuffle = true,
               const bool resetPolicy = true,
               const bool exactObjective = false);

  /**
   * Optimize the given function using AdaBoundType. The given starting point
   * will be modified to store the finishing point of the algorithm, and the
   * final objective value is returned. The DecomposableFunctionType is checked
   * for API consistency at compile time.
   *
   * @tparam DecomposableFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @param callbacks Callback functions.
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType,
           typename MatType,
           typename GradType,
           typename... CallbackTypes>
  typename std::enable_if<IsArmaType<GradType>::value,
      typename MatType::elem_type>::type
  Optimize(DecomposableFunctionType& function,
           MatType& iterate,
           CallbackTypes&&... callbacks)
  {
    return optimizer.template Optimize<DecomposableFunctionType, MatType,
        GradType, CallbackTypes...>(function, iterate,
        std::forward<CallbackTypes>(callbacks)...);
  }

  //! Forward the MatType as GradType.
  template<typename DecomposableFunctionType,
           typename MatType,
           typename... CallbackTypes>
  typename MatType::elem_type Optimize(DecomposableFunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks)
  {
    return Optimize<DecomposableFunctionType, MatType, MatType,
        CallbackTypes...>(function, iterate,
        std::forward<CallbackTypes>(callbacks)...);
  }

  //! Get the final (SGD) learning rate.
  double FinalLr() const { return optimizer.UpdatePolicy().FinalLr(); }
  //! Modify the final (SGD) learning rate.
  double& FinalLr() { return optimizer.UpdatePolicy().FinalLr(); }

  //! Get the convergence speed of the bound functions.
  double Gamma() const { return optimizer.UpdatePolicy().Gamma(); }
  //! Modify the convergence speed of the bound functions.
  double& Gamma() { return optimizer.UpdatePolicy().Gamma(); }

  //! Get the step size.
  double StepSize() const { return optimizer.StepSize(); }
  //! Modify the step size.
  double& StepSize() { return optimizer.StepSize(); }

  //! Get the batch size.
  size_t BatchSize() const { return optimizer.BatchSize(); }
  //! Modify the batch size.
  size_t& BatchSize() { return optimizer.BatchSize(); }

  //! Get the smoothing parameter.
  double Beta1() const { return optimizer.UpdatePolicy().Beta1(); }
  //! Modify the smoothing parameter.
  double& Beta1() { return optimizer.UpdatePolicy().Beta1(); }

  //! Get the second moment coefficient.
  double Beta2() const { return optimizer.UpdatePolicy().Beta2(); }
  //! Modify the second moment coefficient.
  double& Beta2() { return optimizer.UpdatePolicy().Beta2(); }

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

  //! Get whether or not the update policy parameters are reset before
  //! Optimize call.
  bool ResetPolicy() const { return optimizer.ResetPolicy(); }
  //! Modify whether or not the update policy parameters
  //! are reset before Optimize call.
  bool& ResetPolicy() { return optimizer.ResetPolicy(); }

 private:
  //! The Stochastic Gradient Descent object with QHAdam policy.
  SGD<UpdatePolicyType, DecayPolicyType> optimizer;
};

using AdaBound = AdaBoundType<AdaBoundUpdate>;
using AMSBound = AdaBoundType<AMSBoundUpdate>;

} // namespace ens

// Include implementation.
#include "ada_bound_impl.hpp"

#endif
