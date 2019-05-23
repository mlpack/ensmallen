/**
 * @file adamr_impl.hpp
 * @author Niteya Shah
 * Implementation of the AdamR optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_ADAMWR_IMPL_HPP
#define ENSMALLEN_ADAM_ADAMWR_IMPL_HPP

// In case it hasn't been included yet.
#include "adamwr.hpp"

namespace ens {

inline AdamWR::AdamWR(
    const double stepSizeMax,
    const double stepSizeMin,
    const size_t epochRestart,
    const double multFactor,
    const double weightDecay,
    const size_t batchSize,
    const double beta1,
    const double beta2,
    const double epsilon,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const bool resetPolicy) :
    batchSize(batchSize),
    optimizer(stepSizeMax,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              AdamWUpdate(epsilon, beta1, beta2, weightDecay),
              CyclicalDecay(
                  epochRestart,
                  multFactor,
                  stepSizeMax,
                  stepSizeMin),
              resetPolicy)
{ /* Nothing to do. */ }

template<typename DecomposableFunctionType>
double AdamWR::Optimize(
    DecomposableFunctionType& function,
    arma::mat& iterate)
{
  // If a user changed the step size he hasn't update the step size of the
  // cyclical decay instantiation, so we have to do it here.
  if (optimizer.StepSize() != optimizer.DecayPolicy().StepSizeMax())
  {
    optimizer.DecayPolicy().StepSizeMax() = optimizer.StepSize();
  }

  optimizer.DecayPolicy().EpochBatches() = function.NumFunctions() /
      double(optimizer.BatchSize());

  // If a user changed the batch size we have to update the restart fraction
  // of the cyclical decay instantiation.
  if (optimizer.BatchSize() != batchSize)
  {
    batchSize = optimizer.BatchSize();
  }

  return optimizer.Optimize(function, iterate);
}

} // namespace ens

#endif
