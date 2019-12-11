/**
 * @file ackley_function_impl.hpp
 * @author Suryoday Basak
 *
 * Implementation of the Ackley function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_ACKLEY_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_ACKLEY_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "ackley_function.hpp"

namespace ens {
namespace test {

inline AckleyFunction::AckleyFunction(const double c, const double epsilon) :
    c(c), epsilon(epsilon)
{ /* Nothing to do here */}

inline void AckleyFunction::Shuffle() { /* Nothing to do here */ }

template<typename MatType>
typename MatType::elem_type AckleyFunction::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  // For convenience; we assume these temporaries will be optimized out.
  const ElemType x1 = coordinates(0);
  const ElemType x2 = coordinates(1);

  const ElemType objective = -20 * std::exp(
      -0.2 * std::sqrt(0.5 * (x1 * x1 + x2 * x2))) -
      std::exp(0.5 * (std::cos(c * x1) + std::cos(c * x2))) + std::exp(1) + 20;

  return objective;
}

template<typename MatType>
typename MatType::elem_type AckleyFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
inline void AckleyFunction::Gradient(const MatType& coordinates,
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
  const ElemType t0 = std::sqrt(0.5 * (x1 * x1 + x2 * x2));
  const ElemType t1 = 2.0 * std::exp(- 0.2 * t0) / (t0 + epsilon);
  const ElemType t2 = 0.5 * c *
      std::exp(0.5 * (std::cos(c * x1) + std::cos(c * x2)));

  gradient.set_size(2, 1);
  gradient(0) = (x1 * t1) + (t2 * std::sin(c * x1));
  gradient(1) = (x2 * t1) + (t2 * std::sin(c * x2));
}

template<typename MatType, typename GradType>
inline void AckleyFunction::Gradient(const MatType& coordinates,
                                     GradType& gradient)
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
