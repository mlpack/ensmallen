/**
 * @file sgdr_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of SGDR method.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGDR_SGDR_IMPL_HPP
#define ENSMALLEN_SGDR_SGDR_IMPL_HPP

// In case it hasn't been included yet.
#include "sgdr.hpp"

namespace ens {

template<typename UpdatePolicyType>
SGDR<UpdatePolicyType>::SGDR(
    const size_t epochRestart,
    const double multFactor,
    const size_t batchSize,
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const UpdatePolicyType& updatePolicy,
    const bool resetPolicy,
    const bool exactObjective) :
    batchSize(batchSize),
    optimizer(OptimizerType(stepSize,
                            batchSize,
                            maxIterations,
                            tolerance,
                            shuffle,
                            updatePolicy,
                            CyclicalDecay(
                                epochRestart,
                                multFactor,
                                stepSize),
                            resetPolicy,
                            exactObjective))
{
  /* Nothing to do here */
}

template<typename UpdatePolicyType>
template<typename SeparableFunctionType, typename MatType, typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
SGDR<UpdatePolicyType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterate,
    CallbackTypes&&... callbacks)
{
  // If a user changed the step size he hasn't update the step size of the
  // cyclical decay instantiation, so we have to do it here.
  if (optimizer.StepSize() != optimizer.DecayPolicy().StepSize())
  {
    optimizer.DecayPolicy().StepSize() = optimizer.StepSize();
  }

  optimizer.DecayPolicy().EpochBatches() = function.NumFunctions() /
      double(optimizer.BatchSize());

  // If a user changed the batch size we have to update the restart fraction
  // of the cyclical decay instantiation.
  if (optimizer.BatchSize() != batchSize)
  {
    batchSize = optimizer.BatchSize();
  }

  return optimizer.Optimize(function, iterate, callbacks...);
}

} // namespace ens

#endif
