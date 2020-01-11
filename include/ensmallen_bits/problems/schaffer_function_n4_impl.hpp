/**
 * @file schaffer_function_n4_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of Schaffer function N.4.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N4_IMPL_HPP
#define ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N4_IMPL_HPP

// In case it hasn't been included yet.
#include "schaffer_function_n4.hpp"

namespace ens {
namespace test {

inline SchafferFunctionN4::SchafferFunctionN4() { /* Nothing to do here */ }

inline void SchafferFunctionN4::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type SchafferFunctionN4::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = 0.5 + (std::pow(std::cos(std::sin(std::abs(
      std::pow(x1, 2) - std::pow(x2, 2)))), 2) - 0.5) / std::pow(1 + 0.001 *
      (std::pow(x1, 2) + std::pow(x2, 2)), 2);

  return objective;
}

template<typename MatType>
typename MatType::elem_type SchafferFunctionN4::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
