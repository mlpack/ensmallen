/**
 * @file sphere_function_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Sphere function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SPHERE_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_SPHERE_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "sphere_function.hpp"

namespace ens {
namespace test {

template<typename PointMatType, typename LabelsType>
SphereFunctionType<PointMatType, LabelsType>::SphereFunctionType(const size_t n) :
    n(n),
    visitationOrder(linspace<LabelsType>(0, n - 1, n))
{
  initialPoint.set_size(n, 1);

  for (size_t i = 0; i < n; ++i) // Set to [-5 5 -5 5 -5 5...].
  {
    if (i % 2 == 1)
      initialPoint(i) = 5;
    else
      initialPoint(i) = -5;
  }
}

template<typename PointMatType, typename LabelsType>
void SphereFunctionType<PointMatType, LabelsType>::Shuffle()
{
  visitationOrder = shuffle(linspace<LabelsType>(0, n - 1, n));
}

template<typename PointMatType, typename LabelsType>
template<typename MatType>
typename MatType::elem_type SphereFunctionType<
    PointMatType, LabelsType>::Evaluate(
    const MatType& coordinates,
    const size_t begin,
    const size_t batchSize) const
{
  typedef typename MatType::elem_type ElemType;

  ElemType objective = 0;
  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    objective += std::pow(coordinates(p), ElemType(2));
  }

  //std::cout << "SphereFunction objective " << objective << " coordinates " << coordinates.t();
  return objective;
}

template<typename PointMatType, typename LabelsType>
template<typename MatType>
typename MatType::elem_type SphereFunctionType<
    PointMatType, LabelsType>::Evaluate(
    const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename PointMatType, typename LabelsType>
template<typename MatType, typename GradType>
void SphereFunctionType<PointMatType, LabelsType>::Gradient(
    const MatType& coordinates,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize) const
{
  gradient.zeros(n, 1);

  for (size_t j = begin; j < begin + batchSize; ++j)
  {
    const size_t p = visitationOrder[j];
    gradient(p) += 2 * coordinates[p];
  }

  //std::cout << "SphereFunction coordinates " << coordinates.t() << "               gradient " << gradient.t();
}

template<typename PointMatType, typename LabelsType>
template<typename MatType, typename GradType>
void SphereFunctionType<PointMatType, LabelsType>::Gradient(
    const MatType& coordinates, GradType& gradient)
{
  Gradient(coordinates, 0, gradient, NumFunctions());
}

} // namespace test
} // namespace ens

#endif
