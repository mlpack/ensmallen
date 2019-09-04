/**
 * @file lrsdp.cpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SDP_LRSDP_IMPL_HPP
#define ENSMALLEN_SDP_LRSDP_IMPL_HPP

#include "lrsdp.hpp"

namespace ens {

template<typename SDPType>
LRSDP<SDPType>::LRSDP(const size_t numSparseConstraints,
                      const size_t numDenseConstraints,
                      const arma::Mat<typename SDPType::ElemType>& initialPoint,
                      const size_t maxIterations) :
    function(numSparseConstraints, numDenseConstraints, initialPoint),
    maxIterations(maxIterations)
{ }

template<typename SDPType>
template<typename MatType, typename... CallbackTypes>
typename MatType::elem_type LRSDP<SDPType>::Optimize(
    MatType& coordinates, CallbackTypes&&... callbacks)
{
  function.RRTAny().Clean();
  function.RRTAny().template Set<MatType>(
      new MatType(coordinates * coordinates.t()));

  augLag.Sigma() = 10;
  augLag.MaxIterations() = maxIterations;
  augLag.Optimize(function, coordinates, callbacks...);

  return function.Evaluate(coordinates);
}

} // namespace ens

#endif
