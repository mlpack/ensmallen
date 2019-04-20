/**
 * @file styblinski_tang_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Styblinski-Tang function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_STYBLINSKI_TANG_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_STYBLINSKI_TANG_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "styblinski_tang_function.hpp"

namespace ens {
namespace test {

inline StyblinskiTangFunction::StyblinskiTangFunction(const size_t n) :
    n(n),
    visitationOrder(arma::linspace<arma::Row<size_t> >(0, n - 1, n))

{
  initialPoint.set_size(n, 1);
  initialPoint.fill(-5);
}

inline void StyblinskiTangFunction::Shuffle()
{
  visitationOrder = arma::shuffle(
      arma::linspace<arma::Row<size_t> >(0, n - 1, n));
}

template<typename MatType>
typename MatType::elem_type StyblinskiTangFunction::Evaluate(
    const MatType& coordinates,
    const size_t begin,
    const size_t batchSize) const
{
  typename MatType::elem_type objective = 0.0;
  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    objective += std::pow(coordinates(p), 4) - 16 *
        std::pow(coordinates(p), 2) + 5 * coordinates(p);
  }
  objective /= 2;

  return objective;
}

template<typename MatType>
typename MatType::elem_type StyblinskiTangFunction::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename GradType>
void StyblinskiTangFunction::Gradient(const MatType& coordinates,
                                      const size_t begin,
                                      GradType& gradient,
                                      const size_t batchSize) const
{
  gradient.zeros(n, 1);

  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    gradient(p) += 0.5 * (4 * std::pow(coordinates(p), 3) -
        32.0 * coordinates(p) + 5.0);
  }
}

template<typename MatType, typename GradType>
void StyblinskiTangFunction::Gradient(const MatType& coordinates,
                                      GradType& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
