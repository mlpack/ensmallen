/**
 * @file aug_lagrangian_function_impl.hpp
 * @author Ryan Curtin
 *
 * Simple, naive implementation of AugLagrangianFunction.  Better
 * specializations can probably be given in many cases, but this is the most
 * general case.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_FUNCTION_IMPL_HPP
#define ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_FUNCTION_IMPL_HPP

// In case it hasn't been included.
#include "aug_lagrangian_function.hpp"

namespace ens {

// Initialize the AugLagrangianFunction.
template<typename LagrangianFunction, typename VecType>
AugLagrangianFunction<LagrangianFunction, VecType>::AugLagrangianFunction(
    LagrangianFunction& function,
    VecType& lambda,
    double& sigma) :
    function(function),
    lambda(lambda),
    sigma(sigma)
{
  // Nothing else to do.
}

// Evaluate the AugLagrangianFunction at the given coordinates.
template<typename LagrangianFunction, typename VecType>
template<typename MatType>
typename MatType::elem_type
AugLagrangianFunction<LagrangianFunction, VecType>::Evaluate(
    const MatType& coordinates) const
{
  // The augmented Lagrangian is evaluated as
  //    f(x) + {-lambda_i * c_i(x) + (sigma / 2) c_i(x)^2} for all constraints

  typedef typename MatType::elem_type ElemType;

  // First get the function's objective value.
  ElemType objective = function.Evaluate(coordinates);

  // Now loop for each constraint.
  for (size_t i = 0; i < function.NumConstraints(); ++i)
  {
    ElemType constraint = function.EvaluateConstraint(i, coordinates);

    objective += (-ElemType(lambda[i]) * constraint) +
        ElemType(sigma) * std::pow(constraint, ElemType(2)) / 2;
  }

  return objective;
}

// Evaluate the gradient of the AugLagrangianFunction at the given coordinates.
template<typename LagrangianFunction, typename VecType>
template<typename MatType, typename GradType>
void AugLagrangianFunction<LagrangianFunction, VecType>::Gradient(
    const MatType& coordinates,
    GradType& gradient) const
{
  typedef typename MatType::elem_type ElemType;

  // The augmented Lagrangian's gradient is evaluted as
  // f'(x) + {(-lambda_i + sigma * c_i(x)) * c'_i(x)} for all constraints
  gradient.zeros();
  function.Gradient(coordinates, gradient);

  GradType constraintGradient; // Temporary for constraint gradients.
  for (size_t i = 0; i < function.NumConstraints(); i++)
  {
    function.GradientConstraint(i, coordinates, constraintGradient);

    // Now calculate scaling factor and add to existing gradient.
    GradType tmpGradient;
    tmpGradient = (ElemType(-lambda[i]) + ElemType(sigma) *
        function.EvaluateConstraint(i, coordinates)) * constraintGradient;
    gradient += tmpGradient;
  }
}

// Get the initial point.
template<typename LagrangianFunction, typename VecType>
template<typename MatType>
const MatType&
AugLagrangianFunction<LagrangianFunction, VecType>::GetInitialPoint()
    const
{
  return function.template GetInitialPoint<MatType>();
}

} // namespace ens

#endif

