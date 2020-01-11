/**
 * @file schaffer_function_n2_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of Schaffer function N.2.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N2_IMPL_HPP
#define ENSMALLEN_PROBLEMS_SCHAFFER_FUNCTION_N2_IMPL_HPP

// In case it hasn't been included yet.
#include "schaffer_function_n2.hpp"

namespace ens {
namespace test {

inline SchafferFunctionN2::SchafferFunctionN2() { /* Nothing to do here */ }

inline void SchafferFunctionN2::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type SchafferFunctionN2::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = 0.5 + (std::pow(std::sin(std::pow(x1, 2) -
      std::pow(x2, 2)), 2) - 0.5) / std::pow(1 + 0.001 *
      (std::pow(x1, 2) + std::pow(x2, 2)), 2);

  return objective;
}

template<typename MatType>
typename MatType::elem_type SchafferFunctionN2::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void SchafferFunctionN2::Gradient(const MatType& coordinates,
                                         const size_t /* begin */,
                                         GradType& gradient,
                                         const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  // Aliases for different terms in the expression of the gradient
  const ElemType x1Sq = x1 * x1;
  const ElemType x2Sq = x2 * x2;
  const ElemType sum1 = x1Sq - x2Sq;
  const ElemType sinSum1 = sin(sum1);
  const ElemType sum2 = 0.001 * (x1Sq + x2Sq) + 1;
  const ElemType trigExpression = 4 * sinSum1 * cos(sum1);
  const ElemType numerator1 = - 0.004 * (pow(sinSum1, 2) - 0.5);
  const ElemType expr1 = numerator1 / pow(sum2, 3);
  const ElemType expr2 = trigExpression / pow(sum2, 2);

  gradient.set_size(2, 1);
  gradient(0) = x1 * (expr1 + expr2);
  gradient(1) = x2 * (expr1 - expr2);
}

template<typename MatType, typename GradType>
inline void SchafferFunctionN2::Gradient(const MatType& coordinates,
                                         GradType& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
