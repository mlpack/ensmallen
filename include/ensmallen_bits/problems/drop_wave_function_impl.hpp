/**
 * @file drop_wave_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Drop-Wave function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_DROP_WAVE_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_DROP_WAVE_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "drop_wave_function.hpp"

namespace ens {
namespace test {

inline DropWaveFunction::DropWaveFunction() { /* Nothing to do here */ }

inline void DropWaveFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type DropWaveFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = -1.0 * (1.0 + std::cos(12.0 *
      std::sqrt(std::pow(x1, 2) + std::pow(x2, 2)))) /
      (0.5 * (std::pow(x1, 2) + std::pow(x2, 2)) + 2.0);

  return objective;
}

template<typename MatType>
typename MatType::elem_type DropWaveFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, 1);
}

template<typename MatType, typename GradType>
inline void DropWaveFunction::Gradient(const MatType& coordinates,
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
  gradient(0) = (12.0 * x1 * std::sin(12.0 * std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2)))) / (std::sqrt(std::pow(x1, 2) + std::pow(x2, 2)) *
      (0.5 * (std::pow(x1, 2) + std::pow(x2, 2)) + 2)) -
      (x1 * (-1.0 * std::cos(12.0 * std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2))) -1.0)) / std::pow(0.5 *
      (std::pow(x1, 2) + std::pow(x2, 2)) + 2, 2);

  gradient(1) = (12.0 * x2 * std::sin(12.0 * std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2)))) / (std::sqrt(std::pow(x1, 2) + std::pow(x2, 2)) *
      (0.5 * (std::pow(x1, 2) + std::pow(x2, 2)) + 2)) -
      (x2 * (-1.0 * std::cos(12.0 * std::sqrt(std::pow(x1, 2) +
      std::pow(x2, 2))) -1.0)) / std::pow(0.5 *
      (std::pow(x1, 2) + std::pow(x2, 2)) + 2, 2);
}

template<typename MatType, typename GradType>
inline void DropWaveFunction::Gradient(const MatType& coordinates,
                                       GradType& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
