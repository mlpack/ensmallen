/**
 * @file mc_cormick_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the McCormick function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_MC_CORMICK_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_MC_CORMICK_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "mc_cormick_function.hpp"

namespace ens {
namespace test {

inline McCormickFunction::McCormickFunction() { /* Nothing to do here */ }

inline void McCormickFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type McCormickFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const typename MatType::elem_type x1 = coordinates(0);
  const typename MatType::elem_type x2 = coordinates(1);

  const typename MatType::elem_type objective = std::sin(x1 + x2) +
      std::pow(x1 - x2, 2) - 1.5 * x1 + 2.5 * x2 + 1;

  return objective;
}

template<typename MatType>
typename MatType::elem_type McCormickFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void McCormickFunction::Gradient(const MatType& coordinates,
                                        const size_t /* begin */,
                                        GradType& gradient,
                                        const size_t /* batchSize */) const
{
  // For convenience; we assume these temporaries will be optimized out.
  const typename MatType::elem_type x1 = coordinates(0);
  const typename MatType::elem_type x2 = coordinates(1);

  gradient.set_size(2, 1);
  gradient(0) = std::cos(x1 + x2) + 2 * x1 - 2 * x2 - 1.5;
  gradient(1) = std::cos(x1 + x2) - 2 * x1 + 2 * x2 + 2.5;
}

template<typename MatType, typename GradType>
inline void McCormickFunction::Gradient(const MatType& coordinates,
                                        GradType& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
