/**
 * @file ada_belief.hpp
 * @author Marcus Edel
 *
 * Class wrapper for the AdaBelief update Policy. The intuition for AdaBelief is
 * to adapt the stepsize according to the "belief" in the current gradient
 * direction.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_BELIEF_HPP
#define ENSMALLEN_ADA_BELIEF_HPP

#include <ensmallen_bits/sgd/sgd.hpp>
#include "ada_belief_update.hpp"

namespace ens {

/**
 * The intuition for AdaBelief is to adapt the stepsize according to the
 * "belief" in the current gradient direction. For more information, see the
 * following.
 *
 * @code
 * @misc{zhuang2020adabelief,
 *   title         = {AdaBelief Optimizer: Adapting Stepsizes by the Belief in
 *                    Observed Gradients},
 *   author        = {Juntang Zhuang and Tommy Tang and Sekhar Tatikonda and
 *                    Nicha Dvornek and Yifan Ding and Xenophon Papademetris
 *                    and James S. Duncan},
 *   year          = {2020},
 *   eprint        = {2010.07468},
 *   archivePrefix = {arXiv},
 * }
 * @endcode
 *
 * AdaBelief can optimize differentiable separable functions. For more details,
 * see the documentation on function types included with this distribution or
 * on the ensmallen website.
 */
class AdaBelief
{
 public:
  /**
   * Construct the AdaBelief optimizer with the given function and parameters.
   * AdaBelief is sensitive to its parameters and hence a good hyperparameter
   * selection is necessary as its default may not fit every case.
   *
   * The maximum number of iterations refers to the maximum number of
   * points that are processed (i.e., one iteration equals one point; one
   * iteration does not equal one pass over the dataset).
   *
   * @param stepSize Step size for each iteration.
   * @param batchSize Number of points to process in a single step.
   * @param beta1 The exponential decay rate for the 1st moment estimates.
   * @param beta2 The exponential decay rate for the 2nd moment estimates.
   * @param epsilon A small constant for numerical stability.
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
  AdaBelief(const double stepSize = 0.001,
            const size_t batchSize = 32,
            const double beta1 = 0.9,
            const double beta2 = 0.999,
            const double epsilon = 1e-12,
            const size_t maxIterations = 100000,
            const double tolerance = 1e-5,
            const bool shuffle = true,
            const bool resetPolicy = true,
            const bool exactObjective = false);

  /**
   * Optimize the given function using AdaBelief. The given starting point will
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

  //! Get the exponential decay rate for the 1st moment estimates.
  double Beta1() const { return optimizer.UpdatePolicy().Beta1(); }
  //! Modify the exponential decay rate for the 1st moment estimates.
  double& Beta1() { return optimizer.UpdatePolicy().Beta1(); }

  //! Get the exponential decay rate for the 2nd moment estimates.
  double Beta2() const { return optimizer.UpdatePolicy().Beta2(); }
  //! Get the second moment coefficient.
  double& Beta2() { return optimizer.UpdatePolicy().Beta2(); }

  //! Get the value for numerical stability.
  double Epsilon() const { return optimizer.UpdatePolicy().Epsilon(); }
  //! Modify the value used for numerical stability.
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
  //! The Stochastic Gradient Descent object with AdaBelief policy.
  SGD<AdaBeliefUpdate> optimizer;
};

} // namespace ens

// Include implementation.
#include "ada_belief_impl.hpp"

#endif
