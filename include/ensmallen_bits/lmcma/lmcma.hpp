//
/**
 * @file lmcma.hpp
 * @author Oleksandr Nikolskyy
 *
 * Implementation of the LM CMA algorithm - useful in a derivative-free large-scale
 * black-box optimization scenario. Eg, where CMA-ES fails to scale.
 *
 * For details see "LM-CMA: an Alternative to L-BFGS for Large Scale Black-box Optimization" by Loshchilov
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */



#ifndef ENSMALLEN_LMCMA_LMCMA_HPP
#define ENSMALLEN_LMCMA_LMCMA_HPP

#include <ensmallen_bits/lmcma/sampling/mirror_sampling.hpp>

namespace ens {

template<typename SamplingType = MirrorSampling>
class LMCMA 
{
  public:

    LMCMA(const size_t lambda,
          const size_t maxIterations,
          const double tolerance);

    template<typename ArbitraryFunctionType,
             typename MatType,
             typename... CallbackTypes>
    typename MatType::elem_type Optimize(ArbitraryFunctionType& f,
                                         MatType& iterateIn,
                                         CallbackTypes&&... callbacks);
  private:
    template <typename ElemType, typename MatType>
    ElemType PopulationSuccess(const std::vector<MatType>&objectives);
    size_t lambda;
    size_t maxIterations;
    double tolerance;
    SamplingType sampler;
  };


template<typename SamplingType = MirrorSampling>
using RadermacherLMCMA = LMCMA<SamplingType>;

} // namespace ens

#include "lmcma_impl.hpp"

#endif
