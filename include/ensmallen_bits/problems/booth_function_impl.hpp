/**
 * @file booth_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Booth function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_BOOTH_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_BOOTH_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "booth_function.hpp"

namespace ens {
namespace test {

inline BoothFunction::BoothFunction() { /* Nothing to do here */ }

inline void BoothFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type BoothFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = std::pow(x1 + 2 * x2 - 7, 2) +
      std::pow(2 * x1 + x2 - 5, 2);

  return objective;
}

template<typename MatType>
typename MatType::elem_type BoothFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void BoothFunction::Gradient(const MatType& coordinates,
                                    const size_t /* begin */,
                                    GradType& gradient,
                                    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = 10 * x1 + 8 * x2 - 34;
  gradient(1) = 8 * x1 + 10 * x2 - 38;
}

template<typename MatType, typename GradType>
inline void BoothFunction::Gradient(const MatType& coordinates,
                                    GradType& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
