/**
 * @file ada_sqrt_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of AdaSqrt optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_ADA_SQRT_ADA_SQRT_IMPL_HPP
#define ENSMALLEN_ADA_SQRT_ADA_SQRT_IMPL_HPP

namespace ens {

inline AdaSqrt::AdaSqrt(const double stepSize,
                        const size_t batchSize,
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
              AdaSqrtUpdate(epsilon),
              NoDecay(),
              resetPolicy,
              exactObjective)
{ /* Nothing to do. */ }

} // namespace ens

#endif
