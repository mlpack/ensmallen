/**
 * @file goldstein_price_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Goldstein-Price function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_GOLDSTEIN_PRICE_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_GOLDSTEIN_PRICE_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "goldstein_price_function.hpp"

namespace ens {
namespace test {

inline GoldsteinPriceFunction::GoldsteinPriceFunction()
{ /* Nothing to do here */ }

inline void GoldsteinPriceFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type GoldsteinPriceFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType x1Sq = std::pow(x1, 2);
  const ElemType x2Sq = std::pow(x2, 2);
  const ElemType x1x2 = x1 * x2;
  const ElemType objective = (1 + std::pow(x1 + x2 + 1, 2) * (19 - 14 * x1 + 3 *
      x1Sq - 14 * x2 + 6 * x1x2 + 3 * x2Sq)) * (30 + std::pow(2 * x1 - 3 * x2,
      2) * (18 - 32 * x1 + 12 * x1Sq + 48 * x2 - 36 * x1x2 + 27 * x2Sq));

  return objective;
}

template<typename MatType>
typename MatType::elem_type GoldsteinPriceFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void GoldsteinPriceFunction::Gradient(const MatType& coordinates,
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
  gradient(0) = (std::pow(2 * x1 - 3 * x2, 2) * (24 * x1 - 36 * x2 - 32) + (8 *
      x1 - 12 * x2) * (12 * x1 * x1 - 36 * x1 * x2 - 32 * x1 + 27 * x2 * x2 +
      48 * x2 + 18)) * (std::pow(x1 + x2 + 1, 2) * (3 * x1 * x1 + 6 * x1 * x2 -
      14 * x1 + 3 * x2 * x2 - 14 * x2 + 19) + 1) + (std::pow(2 * x1 - 3 * x2,
      2) * (12 * x1 * x1 - 36 * x1 * x2 - 32 * x1 + 27 * x2 * x2 + 48 * x2 +
      18) + 30) * (std::pow(x1 + x2 + 1, 2) * (6 * x1 + 6 * x2 - 14) + (2 * x1 +
      2 * x2 + 2) * (3 * x1 * x1 + 6 * x1 * x2 - 14 * x1 + 3 * x2 * x2 - 14 *
      x2 + 19));
  gradient(1) = ((- 12 * x1 + 18 * x2) * (12 * x1 * x1 - 36 * x1 * x2 - 32 *
      x1 + 27 * x2 * x2 + 48 * x2 + 18) + std::pow(2 * x1 - 3 * x2, 2) * (-36 *
      x1 + 54 * x2 + 48)) * (std::pow(x1 + x2 + 1, 2) * (3 * x1 * x1 + 6 * x1 *
      x2 - 14 * x1 + 3 * x2 * x2 - 14 * x2 + 19) + 1) + (std::pow(2 * x1 - 3 *
      x2, 2) * (12 * x1 * x1 - 36 * x1 * x2 - 32 * x1 + 27 * x2 * x2 + 48 * x2 +
      18) + 30) * (std::pow(x1 + x2 + 1, 2) * (6 * x1 + 6 * x2 - 14) + (2 * x1 +
      2 * x2 + 2) * (3 * x1 * x1 + 6 * x1 * x2 - 14 * x1 + 3 * x2 * x2 - 14 *
      x2 + 19));
}

template<typename MatType, typename GradType>
inline void GoldsteinPriceFunction::Gradient(const MatType& coordinates,
                                             GradType& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
