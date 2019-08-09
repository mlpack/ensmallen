/**
 * @file rosenbrock_function_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Rosenbrock function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_ROSENBROCK_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_ROSENBROCK_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "rosenbrock_function.hpp"

namespace ens {
namespace test {

inline RosenbrockFunction::RosenbrockFunction() { /* Nothing to do here */ }

inline void RosenbrockFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type RosenbrockFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective =
      /* f1(x) */ 100 * std::pow(x2 - std::pow(x1, 2), 2) +
      /* f2(x) */ std::pow(1 - x1, 2);

  return objective;
}

template<typename MatType>
typename MatType::elem_type RosenbrockFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
void RosenbrockFunction::Gradient(const MatType& coordinates,
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
  gradient(0) = -2 * (1 - x1) + 400 * (std::pow(x1, 3) - x2 * x1);
  gradient(1) = 200 * (x2 - std::pow(x1, 2));
}

template<typename MatType, typename GradType>
void RosenbrockFunction::Gradient(const MatType& coordinates,
                                  GradType& gradient) const
{
  Gradient(coordinates, 0, gradient, 1);
}

/**
 * Evaluate the function and gradient at the given coordinates.
 */
template<typename MatType, typename GradType>
typename MatType::elem_type RosenbrockFunction::EvaluateWithGradient(
    const MatType& coordinates,
    GradType& gradient) const
{
  Gradient(coordinates, gradient);
  return Evaluate(coordinates);
}

} // namespace test
} // namespace ens

#endif
