/**
 * @file generalized_rosenbrock_function_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Generalized-Rosenbrock function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_GENERALIZED_ROSENBROC_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_GENERALIZED_ROSENBROC_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "generalized_rosenbrock_function.hpp"

namespace ens {
namespace test {

inline GeneralizedRosenbrockFunction::GeneralizedRosenbrockFunction(
    const size_t n) :
    n(n),
    visitationOrder(arma::linspace<arma::Row<size_t> >(0, n - 1, n))

{
  initialPoint.set_size(n, 1);
  for (size_t i = 0; i < n; i++) // Set to [-1.2 1 -1.2 1 ...].
  {
    if (i % 2 == 1)
    {
      initialPoint(i) = -1.2;
    }
    else
    {
      initialPoint(i) = 1;
    }
  }
}

inline void GeneralizedRosenbrockFunction::Shuffle()
{
  visitationOrder = arma::shuffle(arma::linspace<arma::Row<size_t>>(0, n - 2,
      n - 1));
}

template<typename MatType>
typename MatType::elem_type GeneralizedRosenbrockFunction::Evaluate(
    const MatType& coordinates,
    const size_t begin,
    const size_t batchSize) const
{
  typename MatType::elem_type objective = 0.0;
  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    objective += 100 * std::pow((std::pow(coordinates[p], 2)
        - coordinates[p + 1]), 2) + std::pow(1 - coordinates[p], 2);
  }

  return objective;
}

template<typename MatType>
typename MatType::elem_type GeneralizedRosenbrockFunction::Evaluate(
    const MatType& coordinates) const
{
  typename MatType::elem_type fval = 0;
  for (size_t i = 0; i < (n - 1); i++)
  {
    fval += 100 * std::pow(std::pow(coordinates[i], 2) -
        coordinates[i + 1], 2) + std::pow(1 - coordinates[i], 2);
  }

  return fval;
}

template<typename MatType, typename GradType>
inline void GeneralizedRosenbrockFunction::Gradient(
    const MatType& coordinates,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize) const
{
  gradient.zeros(n);
  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    gradient[p] = 400 * (std::pow(coordinates[p], 3) - coordinates[p] *
        coordinates[p + 1]) + 2 * (coordinates[p] - 1);
    gradient[p + 1] = 200 * (coordinates[p + 1] - std::pow(coordinates[p], 2));
  }
}

template<typename MatType, typename GradType>
inline void GeneralizedRosenbrockFunction::Gradient(
    const MatType& coordinates,
    GradType& gradient) const
{
  gradient.zeros(n);
  for (size_t i = 0; i < (n - 1); i++)
  {
    gradient[i] = 400 * (std::pow(coordinates[i], 3) - coordinates[i] *
        coordinates[i + 1]) + 2 * (coordinates[i] - 1);

    if (i > 0)
      gradient[i] += 200 * (coordinates[i] - std::pow(coordinates[i - 1], 2));
  }

  gradient[n - 1] = 200 * (coordinates[n - 1] -
      std::pow(coordinates[n - 2], 2));
}

} // namespace test
} // namespace ens

#endif
