/**
 * @file ftml_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Follow the Moving Leader (FTML) optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FTML_FTML_IMPL_HPP
#define ENSMALLEN_FTML_FTML_IMPL_HPP

// In case it hasn't been included yet.
#include "ftml.hpp"

namespace ens {

inline FTML::FTML(const double stepSize,
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
              FTMLUpdate(epsilon, beta1, beta2),
              NoDecay(),
              resetPolicy,
              exactObjective)
{ /* Nothing to do. */ }

} // namespace ens

#endif
