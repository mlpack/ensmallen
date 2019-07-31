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
#ifndef ENSMALLEN_ADAM_QHADAM_IMPL_HPP
#define ENSMALLEN_ADAM_QHADAM_IMPL_HPP

// In case it hasn't been included yet.
#include "qhadam.hpp"

namespace ens {

inline QHAdam::QHAdam(
    const double stepSize,
    const size_t batchSize,
    const double v1,
    const double v2,
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
              QHAdamUpdate(epsilon, beta1, beta2, v1, v2),
              NoDecay(),
              resetPolicy,
              exactObjective)
{ /* Nothing to do. */ }

} // namespace ens

 #endif
