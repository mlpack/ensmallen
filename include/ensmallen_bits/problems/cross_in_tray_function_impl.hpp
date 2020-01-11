/**
 * @file cross_in_tray_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Cross-in-Tray function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_CROSS_IN_TRAY_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_CROSS_IN_TRAY_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "cross_in_tray_function.hpp"

namespace ens {
namespace test {

inline CrossInTrayFunction::CrossInTrayFunction() { /* Nothing to do here */ }

inline void CrossInTrayFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type CrossInTrayFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = -0.0001 * std::pow(std::abs(std::sin(x1) *
      std::sin(x2) * std::exp(std::abs(100 - (std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2)) / arma::datum::pi))) + 1), 0.1);
  return objective;
}

template<typename MatType>
typename MatType::elem_type CrossInTrayFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
