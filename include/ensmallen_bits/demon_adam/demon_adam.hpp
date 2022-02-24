/**
 * @file demon_adam.hpp
 * @author Marcus Edel
 *
 * Definition of DemonAdam.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_DEMON_ADAM_DEMON_ADAM_HPP
#define ENSMALLEN_DEMON_ADAM_DEMON_ADAM_HPP

#include "../sgd/sgd.hpp"
#include "../adam/adam_update.hpp"
#include "../adam/adamax_update.hpp"
#include "../adam/amsgrad_update.hpp"
#include "../adam/nadam_update.hpp"
#include "../adam/nadamax_update.hpp"
#include "../adam/optimisticadam_update.hpp"
#include "demon_adam_update.hpp"

namespace ens {

/**
 * DemonAdam automatically decays momentum, motivated by decaying the total
 * contribution of a gradient to all future updates.
 *
 * For more information, see the following.
 *
 * @code
 * @misc{
 *   title   = {Decaying momentum helps neural network training},
 *   author  = {John Chen and Cameron Wolfe and Zhao Li
 *              and Anastasios Kyrillidis},
 *   url     = {https://arxiv.org/abs/1910.04952}
 *   year    = {2019}
 * }
 *
 * DemonAdam can optimize differentiable separable functions. For more details,
 * see the documentation on function types include with this distribution or on
 * the ensmallen website.
 *
 * @tparam UpdateRule Adam optimizer update rule to be used.
 */
template<typename UpdateRule = AdamUpdate>
class DemonAdamType
{
 public:
  /**
   * Construct the DemonAdam optimizer with the given function and parameters.
   * The defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param momentum The initial momentum coefficient.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *     limit).
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
   *     estimates.
   * @param eps Value used to initialise the mean squared gradient parameter.
      * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *     function is visited in linear order.
   * @param resetPolicy If true, parameters are reset before every Optimize
   *     call; otherwise, their values are retained.
   * @param exactObjective Calculate the exact objective (Default: estimate the
   *     final objective obtained on the last pass over the data).
   */
  DemonAdamType(const double stepSize = 0.001,
                const size_t batchSize = 32,
                const double momentum = 0.9,
                const double beta1 = 0.9,
                const double beta2 = 0.999,
                const double eps = 1e-8,
                const size_t maxIterations = 100000,
                const double tolerance = 1e-5,
                const bool shuffle = true,
                const bool resetPolicy = true,
                const bool exactObjective = false) :
      optimizer(stepSize,
                batchSize,
                maxIterations,
                tolerance,
                shuffle,
                DemonAdamUpdate<UpdateRule>(maxIterations * batchSize,
                    momentum, UpdateRule(eps, beta1, beta2)),
                NoDecay(),
                resetPolicy,
                exactObjective)
  { /* Nothing to do here. */ }

  /**
   * Optimize the given function using DemonAdam. The given starting point will
   * be modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam SeparableFunctionType Type of the function to optimize.
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
  typename MatType::elem_type Optimize(SeparableFunctionType& function,
                                       MatType& iterate,
                                       CallbackTypes&&... callbacks)
  {
    return optimizer.template Optimize<
        SeparableFunctionType, MatType, GradType, CallbackTypes...>(
        function, iterate, std::forward<CallbackTypes>(callbacks)...);
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

  //! Get the moment coefficient.
  double Momentum() const { return optimizer.UpdatePolicy().Momentum(); }
  //! Modify the moment coefficient.
  double& Momentum() { return optimizer.UpdatePolicy().Momentum(); }

  //! Get the momentum iteration number.
  size_t MomentumIterations() const
  { return optimizer.UpdatePolicy().MomentumIterations(); }
  //! Modify the momentum iteration number.
  size_t& MomentumIterations()
  { return optimizer.UpdatePolicy().MomentumIterations(); }

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
  //! The Stochastic Gradient Descent object with DemonAdam policy.
  SGD<DemonAdamUpdate<UpdateRule>> optimizer;
};

using DemonAdam = DemonAdamType<AdamUpdate>;

using DemonAdaMax = DemonAdamType<AdaMaxUpdate>;

using DemonAMSGrad = DemonAdamType<AMSGradUpdate>;

using DemonNadam = DemonAdamType<NadamUpdate>;

using DemonNadaMax = DemonAdamType<NadaMaxUpdate>;

using DemonOptimisticAdam = DemonAdamType<OptimisticAdamUpdate>;

} // namespace ens

#endif
