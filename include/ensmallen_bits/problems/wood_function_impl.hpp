/**
 * @file wood_function_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Wood function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_WOOD_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_WOOD_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "wood_function.hpp"

namespace ens {
namespace test {

inline WoodFunction::WoodFunction() { /* Nothing to do here */ }

inline void WoodFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type WoodFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);
  const ElemType x3 = coordinates(2);
  const ElemType x4 = coordinates(3);

  const ElemType objective =
      /* f1(x) */ 100 * std::pow(x2 - std::pow(x1, 2), 2) +
      /* f2(x) */ std::pow(1 - x1, 2) +
      /* f3(x) */ 90 * std::pow(x4 - std::pow(x3, 2), 2) +
      /* f4(x) */ std::pow(1 - x3, 2) +
      /* f5(x) */ 10 * std::pow(x2 + x4 - 2, 2) +
      /* f6(x) */ (1.0 / 10.0) * std::pow(x2 - x4, 2);

  return objective;
}

template<typename MatType>
typename MatType::elem_type WoodFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void WoodFunction::Gradient(const MatType& coordinates,
                                   const size_t /* begin */,
                                   GradType& gradient,
                                   const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);
  const ElemType x3 = coordinates(2);
  const ElemType x4 = coordinates(3);

  gradient.set_size(4, 1);
  gradient(0) = 400 * (std::pow(x1, 3) - x2 * x1) - 2 * (1 - x1);
  gradient(1) = 200 * (x2 - std::pow(x1, 2)) + 20 * (x2 + x4 - 2) +
      (1.0 / 5.0) * (x2 - x4);
  gradient(2) = 360 * (std::pow(x3, 3) - x4 * x3) - 2 * (1 - x3);
  gradient(3) = 180 * (x4 - std::pow(x3, 2)) + 20 * (x2 + x4 - 2) -
      (1.0 / 5.0) * (x2 - x4);
}

template<typename MatType, typename GradType>
inline void WoodFunction::Gradient(const MatType& coordinates,
                                   GradType& gradient) const
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
