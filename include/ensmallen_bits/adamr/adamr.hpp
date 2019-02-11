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

namespace ens{
template<typename UpdateRule = AdamUpdate>
class AdamRType
{
 public:
  //Overloaded Constructor to Allow for min stepSize
  AdamRType(const size_t epochRestart = 50,
            const double multFactor = 2.0,
            const double stepSize = 0.001,
            const size_t batchSize = 32,
            const double beta1 = 0.9,
            const double beta2 = 0.999,
            const double eps = 1e-8,
            const size_t maxIterations = 100000,
            const double tolerance = 1e-5,
            const bool shuffle = true,
            const bool resetPolicy = true);

  AdamRType(const size_t epochRestart = 50,
            const double multFactor = 2.0,
            const double stepSize = 0.001,
            const double stepSizeMin = 0,
            const size_t batchSize = 32,
            const double beta1 = 0.9,
            const double beta2 = 0.999,
            const double eps = 1e-8,
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

 private:
  size_t batchSize;
  // The SGDR object with AdamR policy.
  SGD<UpdateRule, CyclicalDecay> optimizer;
};

using AdamWR = AdamRType<AdamWUpdate>;

using AdamR = AdamRType<>;

}

#include "adamr_impl.hpp"

#endif
