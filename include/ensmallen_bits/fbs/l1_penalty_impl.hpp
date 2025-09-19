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
  return norm(vectorise(coordinates), 1) * typename MatType::elem_type(lambda);
}

template<typename MatType>
void L1Penalty::ProximalStep(MatType& coordinates,
                             const double stepSize) const
{
  // Apply the backwards step coordinate-wise.  If `MatType` is sparse, this
  // only applies to nonzero elements, which is just fine.
  typedef typename MatType::elem_type eT;

  // This is equivalent to the following .transform() implementation (which is
  // easier to read but will not work with Bandicoot):
  //
  //arma::Mat<typename MatType::elem_type> c2 = conv_to<arma::Mat<typename MatType::elem_type>>::from(coordinates);
  //c2.transform([this, stepSize](eT val) { return (val > eT(0)) ?
  //    (std::max(eT(0), val - eT(lambda * stepSize))) :
  //    (std::min(eT(0), val + eT(lambda * stepSize))); });
  // coordinates.transform([this, stepSize](eT val) { return (val > eT(0)) ?
  //     (std::max(eT(0), val - eT(lambda * stepSize))) :
  //     (std::min(eT(0), val + eT(lambda * stepSize))); });
  //
  coordinates = sign(coordinates) % clamp(
      abs(coordinates) - eT(lambda * stepSize), eT(0),
      std::numeric_limits<eT>::max());

  //coordinates.print("coordinates");
  //c2.print("c2");
}

} // namespace ens

#endif
