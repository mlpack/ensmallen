/**
 * @file qhadam_impl.hpp
 * @author Niteya Shah
 *
 * Implementation of QHAdam class wrapper.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_BOUND_IMPL_HPP
#define ENSMALLEN_ADA_BOUND_IMPL_HPP

// In case it hasn't been included yet.
#include "ada_bound.hpp"

namespace ens {

template<typename UpdatePolicyType, typename DecayPolicyType>
AdaBoundType<UpdatePolicyType, DecayPolicyType>::AdaBoundType(
    const double stepSize,
    const size_t batchSize,
    const double finalLr,
    const double gamma,
    const double beta1,
    const double beta2,
    const double epsilon,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const bool resetPolicy,
    const bool exactObjective) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              UpdatePolicyType(finalLr, gamma, epsilon, beta1, beta2),
              NoDecay(),
              resetPolicy,
              exactObjective)
{ /* Nothing to do. */ }

} // namespace ens

 #endif
