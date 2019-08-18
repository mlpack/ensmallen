/**
 * @file smorms3_impl.hpp
 * @author Vivek Pal
 *
 * Implementation of the SMORMS3 constructor.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SMORMS3_SMORMS3_IMPL_HPP
#define ENSMALLEN_SMORMS3_SMORMS3_IMPL_HPP

// In case it hasn't been included yet.
#include "smorms3.hpp"

namespace ens {

inline SMORMS3::SMORMS3(const double stepSize,
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
              SMORMS3Update(epsilon),
              NoDecay(),
              resetPolicy,
              exactObjective)
{ /* Nothing to do. */ }

} // namespace ens

#endif
