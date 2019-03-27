/**
 * @file adamw_impl.hpp
 * @author Niteya Shah
 *
 * Implmentation of AdamW optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADAM_ADAMW_IMPL_HPP
#define ENSMALLEN_ADAM_ADAMW_IMPL_HPP

// In case it hasn't been included yet.
#include "adamw.hpp"

namespace ens {

inline AdamW::AdamW(
    const double stepSize,
    const size_t batchSize,
    const double weightDecay,
    const double beta1,
    const double beta2,
    const double epsilon,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const bool resetPolicy) :
    optimizer(stepSize,
              batchSize,
              maxIterations,
              tolerance,
              shuffle,
              AdamWUpdate(epsilon, beta1, beta2, weightDecay),
              NoDecay(),
              resetPolicy)
{ /* Nothing to do. */ }

} // namespace ens

#endif
