/**
 * @file rastrigin_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Rastrigin function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_RASTRIGIN_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_RASTRIGIN_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "rastrigin_function.hpp"

namespace ens {
namespace test {

template<typename MatType, typename IndexVecType>
RastriginFunctionType<MatType, IndexVecType>::RastriginFunctionType(
    const size_t n) :
    n(n),
    visitationOrder(linspace<IndexVecType>(0, n - 1, n))

{
  initialPoint.set_size(n, 1);
  initialPoint.fill(-3);
}

template<typename MatType, typename IndexVecType>
void RastriginFunctionType<MatType, IndexVecType>::Shuffle()
{
  visitationOrder = shuffle(linspace<IndexVecType>(0, n - 1, n));
}

template<typename MatType, typename IndexVecType>
typename MatType::elem_type
RastriginFunctionType<MatType, IndexVecType>::Evaluate(
    const MatType& coordinates,
    const size_t begin,
    const size_t batchSize) const
{
  // Convenience typedef.
  typedef typename MatType::elem_type ElemType;

  ElemType objective = 0.0;
  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    objective += std::pow(coordinates(p), 2) - 10.0 *
        std::cos(2.0 * arma::datum::pi * coordinates(p));
  }
  objective += 10.0 * n;

  return objective;
}

template<typename MatType, typename IndexVecType>
typename MatType::elem_type
RastriginFunctionType<MatType, IndexVecType>::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename IndexVecType>
template<typename InputMatType, typename InputGradType>
void RastriginFunctionType<MatType, IndexVecType>::Gradient(
    const InputMatType& coordinates,
    const size_t begin,
    InputGradType& gradient,
    const size_t batchSize) const
{
  gradient.zeros(n, 1);

  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    gradient(p) += (10.0 * n) * (2 * (coordinates(p) + 10.0 * arma::datum::pi *
        std::sin(2.0 * arma::datum::pi * coordinates(p))));
  }
}

template<typename MatType, typename IndexVecType>
template<typename InputMatType, typename InputGradType>
inline void RastriginFunctionType<MatType, IndexVecType>::Gradient(
    const InputMatType& coordinates, InputGradType& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
