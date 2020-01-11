/**
 * @file easom_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Easom function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_EASOM_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_EASOM_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "easom_function.hpp"

namespace ens {
namespace test {

inline EasomFunction::EasomFunction() { /* Nothing to do here */ }

inline void EasomFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type EasomFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = -std::cos(x1) * std::cos(x2) *
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 - arma::datum::pi, 2));

  return objective;
}

template<typename MatType>
typename MatType::elem_type EasomFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void EasomFunction::Gradient(const MatType& coordinates,
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
  gradient(0) = 2 * (x1 - arma::datum::pi) *
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 - arma::datum::pi, 2)) *
      std::cos(x1) * std::cos(x2) +
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 -  arma::datum::pi, 2)) *
      std::sin(x1) * std::cos(x2);

  gradient(1) = 2 * (x2 - arma::datum::pi) *
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 - arma::datum::pi, 2)) *
      std::cos(x1) * std::cos(x2) +
      std::exp(-1.0 * std::pow(x1 - arma::datum::pi, 2) -
                      std::pow(x2 - arma::datum::pi, 2)) *
      std::cos(x1) * std::sin(x2);
}

template<typename MatType, typename GradType>
inline void EasomFunction::Gradient(const MatType& coordinates,
                                    GradType& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
