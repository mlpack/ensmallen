/**
 * @file adamr.hpp
 * @author Niteya Shah
 * Declaration of the AdamR optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_ADAMR_HPP
#define ENSMALLEN_ADAM_ADAMR_HPP

#include <ensmallen_bits/adam/adam_update.hpp>
#include <ensmallen_bits/sgdr/cyclical_decay.hpp>
#include <ensmallen_bits/sgd/sgd.hpp>
#include <ensmallen_bits/adam/adamw_update.hpp>

namespace ens {
/**
 * This class is based on the Adam Update where the Optimiser
 * simulates a new warm-started run/restart once a number of epochs are
 * performed.
 *
 * @code
 * @article{
 *   title   = {Decoupled Weight Decay Regularization},
 *   author  = {{Ilya}, L. and {Frank}, H.},
 *   journal = {ArXiv e-prints},
 *   url     = {https://arxiv.org/pdf/1711.05101.pdf}
 *   year    = {2019}
 * }
 * @endcode
 *
 * AdamR can optimize differentiable separable functions.  For more details, see
 * the documentation on function types included with this distribution or on the
 * ensmallen website.
 * @tparam UpdateRule Adam optimizer update rule to be used.
 * @tparam DecayPolicy The StepSize decay policy to be used by default
 * CyclicalDecay.
*/
template<typename UpdateRule = AdamUpdate,typename DecayPolicyType = CyclicalDecay>
class AdamRType
{
 public:
  /**
   * Construct the AdamR optimizer with the given function and parameters. The
   * defaults here are not necessarily good for the given problem, so it is
   * suggested that the values used be tailored to the task at hand.  The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSize Maximum and initial step size for each batch of warm
                     restart.
   * @param epochRestart Restart Rate for Warm Restarts
   * @param multFactor Multiplier for epochRestart
   * @param stepSizeMin Minimum and final step size for each batch of warm
                        restart (Use DecayPolicy() method with CyclicalDecay).
   * @param batchSize Number of points to process in a single step.
   * @param beta1 Exponential decay rate for the first moment estimates.
   * @param beta2 Exponential decay rate for the weighted infinity norm
                  estimates.
   * @param eps Value used to initialise the mean squared gradient parameter.
   * @param maxIterations Maximum number of iterations allowed (0 means no
   *                      limit).
   * @param tolerance Maximum absolute tolerance to terminate algorithm.
   * @param shuffle If true, the function order is shuffled; otherwise, each
   *        function is visited in linear order.
   * @param resetPolicy If true, parameters are reset before every Optimize
   *        call; otherwise, their values are retained.
   * @param updateRule Update Policy to be used for AdamR.
   * @param decayPolicy Decay Policy to be used for AdamR.
   */
  AdamRType(const double stepSize = 0.001,
            const size_t epochRestart = 50,
            const double multFactor = 2.0,
            const size_t batchSize = 32,
            const double beta1 = 0.9,
            const double beta2 = 0.999,
            const double eps = 1e-8,
            const size_t maxIterations = 100000,
            const double tolerance = 1e-5,
            const bool shuffle = true,
            const bool resetPolicy = true);


  AdamRType(const UpdateRule& updateRule,
            const DecayPolicyType& decayPolicy = DecayPolicyType(),
            const double stepSize = 0.001,
            const size_t batchSize = 32,
            const size_t maxIterations = 100000,
            const double tolerance = 1e-5,
            const bool shuffle = true,
            const bool resetPolicy = true);

  /**
   * Optimize the given function using AdamR. The given starting point will be
   * modified to store the finishing point of the algorithm, and the final
   * objective value is returned.
   *
   * @tparam DecomposableFunctionType Type of the function to optimize.
   * @param function Function to optimize.
   * @param iterate Starting point (will be modified).
   * @return Objective value of the final point.
   */
  template<typename DecomposableFunctionType>
  double Optimize(DecomposableFunctionType& function, arma::mat& iterate);
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

  //! Get whether or not the update policy parameters
  //! are reset before Optimize call.
  bool ResetPolicy() const { return optimizer.ResetPolicy(); }
  //! Modify whether or not the update policy parameters
  //! are reset before Optimize call.
  bool& ResetPolicy() { return optimizer.ResetPolicy(); }

  //! Get the update policy.
  const UpdateRule& UpdatePolicy() const { return optimizer.UpdatePolicy(); }
  //! Modify the update policy.
  UpdateRule& UpdatePolicy() { return optimizer.UpdatePolicy(); }

  //! Get the step size decay policy.
  const DecayPolicyType& DecayPolicy() const { return optimizer.DecayPolicy(); }
  //! Modify the step size decay policy.
  DecayPolicyType& DecayPolicy() { return optimizer.DecayPolicy(); }

 private:
  size_t batchSize;
  // The SGDR object with AdamR policy.
  SGD<UpdateRule, DecayPolicyType> optimizer;
};

using AdamWR = AdamRType<AdamWUpdate>;

using AdamR = AdamRType<AdamUpdate>;

}

#include "adamr_impl.hpp"

#endif
