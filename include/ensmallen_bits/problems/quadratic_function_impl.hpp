/**
 * @file quadratic_function_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of QuadraticFunction, f(x) = | x |.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_QUADRATIC_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_QUADRATIC_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "quadratic_function.hpp"

namespace ens {
namespace test {

inline QuadraticFunction::QuadraticFunction() { /* Nothing to do here */ }

inline void QuadraticFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type QuadraticFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  return coordinates[0] * coordinates[0];
}

template<typename MatType>
typename MatType::elem_type QuadraticFunction::Evaluate(const MatType& coordinates)
    const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void QuadraticFunction::Gradient(const MatType& coordinates,
                                  const size_t /* begin */,
                                  GradType& gradient,
                                  const size_t /* batchSize */) const
{
  gradient.set_size(1, 1);
  gradient(0, 0) = 2 * coordinates[0];
}

template<typename MatType, typename GradType>
inline void QuadraticFunction::Gradient(const MatType& coordinates,
                                  GradType& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
