/**
 * @file rosenbrock_wood_function_impl.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Implementation of the Rosenbrock-Wood function.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "rosenbrock_wood_function.hpp"

namespace ens {
namespace test {

template<typename MatType, typename LabelsType>
RosenbrockWoodFunctionType<MatType, LabelsType>::RosenbrockWoodFunctionType() :
    rf(4), wf()
{
  initialPoint.set_size(4, 2);
  initialPoint.col(0) = rf.GetInitialPoint();
  initialPoint.col(1) = wf.GetInitialPoint<MatType>();
}

template<typename MatType, typename LabelsType>
void RosenbrockWoodFunctionType<MatType, LabelsType>::Shuffle()
{ /* Nothing to do here */ }

template<typename MatType, typename LabelsType>
typename MatType::elem_type RosenbrockWoodFunctionType<
    MatType, LabelsType>::Evaluate(
    const MatType& coordinates,
    const size_t /* begin */,
    const size_t /* batchSize */) const
{
  return rf.Evaluate(MatType(coordinates.col(0))) +
      wf.Evaluate(MatType(coordinates.col(1)));
}

template<typename MatType, typename LabelsType>
typename MatType::elem_type RosenbrockWoodFunctionType<
    MatType, LabelsType>::Evaluate(const MatType& coordinates) const
{
  return Evaluate(coordinates, 0, NumFunctions());
}

template<typename MatType, typename LabelsType>
template<typename GradType>
void RosenbrockWoodFunctionType<MatType, LabelsType>::Gradient(
    const MatType& coordinates,
    const size_t /* begin */,
    GradType& gradient,
    const size_t /* batchSize */) const
{
  // Convenience typedef.
  typedef typename ForwardType<MatType>::bcol BaseColType;

  gradient.set_size(4, 2);

  BaseColType grf(4);
  BaseColType gwf(4);

  rf.Gradient(MatType(coordinates.col(0)), grf);
  wf.Gradient(MatType(coordinates.col(1)), gwf);

  gradient.col(0) = grf;
  gradient.col(1) = gwf;
}

template<typename MatType, typename LabelsType>
template<typename GradType>
inline void RosenbrockWoodFunctionType<MatType, LabelsType>::Gradient(
    const MatType& coordinates, GradType& gradient) const
{
  Gradient(coordinates, 0, gradient, 1);
}

} // namespace test
} // namespace ens

#endif
