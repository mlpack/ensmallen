/**
 * @file beale_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Beale function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_BEALE_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_BEALE_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "beale_function.hpp"

namespace ens {
namespace test {

inline BealeFunction::BealeFunction() { /* Nothing to do here */ }

inline void BealeFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type BealeFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = std::pow(1.5 - x1 + x1 * x2, 2) +
      std::pow(2.25 - x1 + x1 * x2 * x2, 2) +
      std::pow(2.625 - x1 + x1 * pow(x2, 3), 2);

  return objective;
}

template<typename MatType>
typename MatType::elem_type BealeFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void BealeFunction::Gradient(const MatType& coordinates,
                                    const size_t /* begin */,
                                    GradType& gradient,
                                    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  // Aliases for different terms in the expression of the gradient.
  const ElemType x2Sq = x2 * x2;
  const ElemType x2Cub = pow(x2, 3);

  gradient.set_size(2, 1);
  gradient(0) = ((2 * x2 - 2) * (x1 * x2 - x1 + 1.5)) +
      ((2 * x2Sq - 2) * (x1 * x2Sq - x1 + 2.25)) +
      ((2 * x2Cub - 2) * (x1 * x2Cub - x1 + 2.625));
  gradient(1) = (6 * x1 * x2Sq * (x1 * x2Cub - x1 + 2.625)) +
      (4 * x1 * x2 * (x1 * x2Sq - x1 + 2.25)) +
      (2 * x1 * (x1 * x2 - x1 + 1.5));
}

template<typename MatType, typename GradType>
inline void BealeFunction::Gradient(const MatType& coordinates,
                                    GradType& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
