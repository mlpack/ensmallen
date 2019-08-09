/**
 * @file eggholder_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Eggholder function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_EGGHOLDER_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_EGGHOLDER_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "eggholder_function.hpp"

namespace ens {
namespace test {

inline EggholderFunction::EggholderFunction() { /* Nothing to do here */ }

inline void EggholderFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type EggholderFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = -1.0 * (x2 + 47) * std::sin(std::sqrt(
      std::abs(x2 + x1 / 2 + 47))) - x1 * std::sin(std::sqrt(
      std::abs(x1 - (x2 + 47))));

  return objective;
}

template<typename MatType>
typename MatType::elem_type EggholderFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void EggholderFunction::Gradient(const MatType& coordinates,
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
  gradient(0) = -1.0 * std::sin(std::sqrt(std::abs(x1 - x2 - 47))) -
      (x1 * (x1 - x2 - 47) * std::cos(std::sqrt(std::abs(x1 - x2 - 47)))) /
      std::pow(2 * std::abs(x1 - x2 - 47), 1.5) -
      ((x1 + 47) * (x1 / 2 + x2 + 47) *
      std::cos(std::sqrt(std::abs(x1 / 2 + x2  + 47)))) /
      (4 * std::pow(std::abs(x1 / 2 + x2  + 47), 1.5));

  gradient(1) = -1.0 * std::sin(std::sqrt(std::abs(x1 / 2 + x2 + 47))) -
      (x1 * (x1 - x2 - 47) * std::cos(std::sqrt(std::abs(x1 - x2 - 47)))) /
      std::pow(2 * std::abs(x1 - x2 - 47), 1.5) -
      ((x1 + 47) * (x1 / 2 + x2 + 47) *
      std::cos(std::sqrt(std::abs(x1 / 2 + x2  + 47)))) /
      (4 * std::pow(std::abs(x1 / 2 + x2  + 47), 1.5));
}

template<typename MatType, typename GradType>
inline void EggholderFunction::Gradient(const MatType& coordinates,
                                        GradType& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
