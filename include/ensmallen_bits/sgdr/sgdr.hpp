/**
 * @file sgdr.hpp
 * @author Marcus Edel
 *
 * Definition of the Stochastic Gradient Descent with Restarts (SGDR) as
 * described in: "SGDR: Stochastic Gradient Descent with Warm Restarts" by
 * I. Loshchilov et al.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGDR_SGDR_HPP
#define ENSMALLEN_SGDR_SGDR_HPP

#include <ensmallen_bits/sgd/sgd.hpp>
#include <ensmallen_bits/sgd/update_policies/momentum_update.hpp>
#include "cyclical_decay.hpp"

namespace ens {

/**
 * This class is based on Mini-batch Stochastic Gradient Descent class and
 * simulates a new warm-started run/restart once a number of epochs are
 * performed.
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Loshchilov2016,
 *   title   = {{SGDR:} Stochastic Gradient Descent with Restarts},
 *   author  = {Ilya Loshchilov and Frank Hutter},
 *   journal = {CoRR},
 *   year    = {2016},
 *   url     = {https://arxiv.org/abs/1608.03983}
 * }
 * @endcode
 *
 * SGDR can optimize differentiable separable functions.  For more details, see
 * the documentation on function types included with this distribution or on the
 * ensmallen website.
 *
 * @tparam UpdatePolicyType Update policy used during the iterative update
 *         process. By default the momentum update policy (see
 *         ens::MomentumUpdate) is used.
 */
template<typename UpdatePolicyType = MomentumUpdate>
class SGDR
{
 public:
  //! Convenience typedef for the internal optimizer construction.
  using OptimizerType = SGD<UpdatePolicyType, CyclicalDecay>;

  /**
   * Construct the SGDR optimizer with the given function and
   * parameters.  The defaults here are not necessarily good for the given
   * problem, so it is suggested that the values used be tailored for the task
   * at hand.  The maximum number of iterations refers to the maximum number of
   * mini-batches that are processed.
   *
   * @param epochRestart Initial epoch where decay is applied.
   * @param batchSize Size of each mini-batch.
   * @param stepSize Step size for each iteration.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *        limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the mini-batch order is shuffled; otherwise, each
   *        mini-batch is visited in linear order.
   * @param updatePolicy Instantiated update policy used to adjust the given
   *        parameters.
   * @param resetPolicy If true, parameters are reset before every Optimize
   *        call; otherwise, their values are retained.
   * @param exactObjective Calculate the exact objective (Default: estimate the
   *        final objective obtained on the last pass over the data).
   */
  SGDR(const size_t epochRestart = 50,
       const double multFactor = 2.0,
       const size_t batchSize = 1000,
       const double stepSize = 0.01,
       const size_t maxIterations = 100000,
       const double tolerance = 1e-5,
       const bool shuffle = true,
       const UpdatePolicyType& updatePolicy = UpdatePolicyType(),
       const bool resetPolicy = true,
       const bool exactObjective = false);

  /**
   * Optimize the given function using SGDR.  The given starting point
   * will be modified to store the finishing point of the algorithm, and the
   * final objective value is returned.
   *
   * @tparam SeparableFunctionType Type of the function to be optimized.
   * @tparam MatType Type of matrix to optimize with.
   * @tparam GradType Type of matrix to use to represent function gradients.
   * @tparam CallbackTypes Types of callback functions.
   * @param function Function to be optimized.
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

  //! Get the batch size.
  size_t BatchSize() const { return optimizer.BatchSize(); }
  //! Modify the batch size.
  size_t& BatchSize() { return optimizer.BatchSize(); }

  //! Get the step size.
  double StepSize() const { return optimizer.StepSize(); }
  //! Modify the step size.
  double& StepSize() { return optimizer.StepSize(); }

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

  //! Get the update policy.
  const UpdatePolicyType& UpdatePolicy() const
  {
    return optimizer.UpdatePolicy();
  }
  //! Modify the update policy.
  UpdatePolicyType& UpdatePolicy()
  {
    return optimizer.UpdatePolicy();
  }

  //! Get whether or not the update policy parameters
  //! are reset before Optimize call.
  bool ResetPolicy() const { return optimizer.ResetPolicy(); }
  //! Modify whether or not the update policy parameters
  //! are reset before Optimize call.
  bool& ResetPolicy() { return optimizer.ResetPolicy(); }

 private:
  //! The size of each mini-batch.
  size_t batchSize;

  //! Locally-stored optimizer instance.
  OptimizerType optimizer;
};

} // namespace ens

// Include implementation.
#include "sgdr_impl.hpp"

#endif
