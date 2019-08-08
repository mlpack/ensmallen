/**
 * @file wn_grad_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the WNGrad optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_WN_GRAD_WN_GRAD_IMPL_HPP
#define ENSMALLEN_WN_GRAD_WN_GRAD_IMPL_HPP

// In case it hasn't been included yet.
#include "wn_grad.hpp"

namespace ens {

inline WNGrad::WNGrad(
    const double stepSize,
    const size_t batchSize,
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
              WNGradUpdate(),
              NoDecay(),
              resetPolicy,
              exactObjective)
{ /* Nothing to do. */ }

} // namespace ens

#endif
