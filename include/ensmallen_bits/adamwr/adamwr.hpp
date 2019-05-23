/**
 * @file adamwr.hpp
 * @author Niteya Shah
 *
 * Declaration of the AdamWR optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_ADAMWR_HPP
#define ENSMALLEN_ADAM_ADAMWR_HPP

#include <ensmallen_bits/adamw/adamw_update.hpp>
#include <ensmallen_bits/sgdr/cyclical_decay.hpp>
#include <ensmallen_bits/sgd/sgd.hpp>

namespace ens {
/*
 * This is a wrapper class for the AdamW update policy with CyclicalDecay on the
 * SGD object, combining the benefits of both the strategies to provide better
 * convergence.
 *
 * @code
 * @article{
 *   title   = {Decoupled Weight Decay Regularization},
 *   author  = {Loschilov, I. and Hutter, F.},
 *   journal = {ArXiv e-prints},
 *   url     = {https://arxiv.org/pdf/1711.05101.pdf}
 *   year    = {2019}
 * }
 * @endcode
 *
 * AdamWR can optimize differentiable separable functions.  For more details, see
 * the documentation on function types included with this distribution or on the
 * ensmallen website.
*/
class AdamWR
{
 public:
  /**
   * Construct the AdamWR optimizer with the given function and parameters.The
   * maximum number of iterations refers to the maximum number of points that
   * are processed (i.e., one iteration equals one point; one iteration does not
   * equal one pass over the dataset).
   *
   * @param stepSizeMax Maximum and initial step size for each batch of warm
     restart.
   * @param stepSizeMin Minimum and final step size for each batch of warm
     restart.
   * @param epochRestart Restart Rate for Warm Restarts
   * @param multFactor Multiplier for epochRestart
   * @param weightDecay The rate at which the update regularises the iterate.
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
   */
  AdamWR(const double stepSizeMax = 0.002,
         const double stepSizeMin = 0.001,
         const size_t epochRestart = 50,
         const double multFactor = 2.0,
         const double weightDecay = 0.0005,
         const size_t batchSize = 32,
         const double beta1 = 0.9,
         const double beta2 = 0.999,
         const double eps = 1e-8,
         const size_t maxIterations = 100000,
         const double tolerance = 1e-5,
         const bool shuffle = true,
         const bool resetPolicy = true);

  /**
   * Optimize the given function using AdamWR. The given starting point will be
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

  //! Get the minimum step size.
  double StepSizeMin() const { return optimizer.DecayPolicy().StepSizeMin(); }
  //! Modify the minimum step size.
  double& StepSizeMin() { return optimizer.DecayPolicy().StepSizeMin(); }

  //! Get the maximum step size.
  double StepSizeMax() const { return optimizer.DecayPolicy().StepSizeMax(); }
  //! Modify the maximum step size.
  double& StepSizeMax() { return optimizer.DecayPolicy().StepSizeMax(); }

  //! Get the multiplier factor
  double MultFactor() const { return optimizer.DecayPolicy().MultFactor(); }
  //! Modify the multiplier factor
  double& MultFactor() { return optimizer.DecayPolicy().MultFactor(); }

  //! Get the epoch restart
  size_t EpochRestart() const { return optimizer.DecayPolicy().EpochRestart(); }
  //! Modify the epoch restart
  size_t& EpochRestart() { return optimizer.DecayPolicy().EpochRestart(); }

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

  //! Get the weight decay parameter.
  double WeightDecay() const { return optimizer.UpdatePolicy().WeightDecay(); }
  //! Modify the weight decay parameter.
  double& WeightDecay() { return optimizer.UpdatePolicy().WeightDecay(); }

  //! Get whether or not the update policy parameters
  //! are reset before Optimize call.
  bool ResetPolicy() const { return optimizer.ResetPolicy(); }
  //! Modify whether or not the update policy parameters
  //! are reset before Optimize call.
  bool& ResetPolicy() { return optimizer.ResetPolicy(); }

 private:
  size_t batchSize;
  // The SGD object with AdamW update policy and the cyclical decay policy.
  SGD<AdamWUpdate, CyclicalDecay> optimizer;
};

}

#include "adamwr_impl.hpp"

#endif
