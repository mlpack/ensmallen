/**
 * @file ada_belief_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of AdaBelief class wrapper.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_BELIEF_ADA_BELIEF_IMPL_HPP
#define ENSMALLEN_ADA_BELIEF_ADA_BELIEF_IMPL_HPP

// In case it hasn't been included yet.
#include "ada_belief.hpp"

namespace ens {

inline AdaBelief::AdaBelief(
    const double stepSize,
    const size_t batchSize,
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
              AdaBeliefUpdate(epsilon, beta1, beta2),
              NoDecay(),
              resetPolicy,
              exactObjective)
{ /* Nothing to do. */ }

} // namespace ens

 #endif
