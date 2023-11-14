/**
 * @file l1_penalty_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of the proximal operator for the L1 penalty (also known as
 * the shrinkage operator).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FBS_L1_PENALTY_IMPL_HPP
#define ENSMALLEN_FBS_L1_PENALTY_IMPL_HPP

// In case it hasn't been included yet.
#include "l1_penalty.hpp"

namespace ens {

inline L1Penalty::L1Penalty(const double lambda) : lambda(lambda)
{
  // Nothing to do.
}

template<typename MatType>
typename MatType::elem_type L1Penalty::Evaluate(const MatType& coordinates)
    const
{
  // Compute the L1 penalty.
  return norm(vectorise(coordinates), 1) * lambda;
}

template<typename MatType>
void L1Penalty::ProximalStep(MatType& coordinates,
                             const double stepSize) const
{
  // Generic implementation; used for dense matrices and other objects that
  // implement the Armadillo API.
  coordinates = sign(coordinates) % clamp(abs(coordinates) - lambda * stepSize,
      0, std::numeric_limits<typename MatType::elem_type>::max());
}

template<typename eT>
void L1Penalty::ProximalStep(arma::SpMat<eT>& coordinates,
                             const double stepSize) const
{
  // Specific implementation for sparse coordinates, optimized to skip
  // processing of all zero-valued coordinates.
  typename arma::SpMat<eT>::iterator it = coordinates.begin();
  while (it != coordinates.end())
  {
    const eT val = (*it);
    if (val > 0.0)
      (*it) = std::max(0.0, val - lambda * stepSize);
    else
      (*it) = std::min(0.0, val + lambda * stepSize);

    ++it;
  }
}

} // namespace ens

#endif
