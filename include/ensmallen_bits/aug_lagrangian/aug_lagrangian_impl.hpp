/**
 * @file aug_lagrangian_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of AugLagrangian class (Augmented Lagrangian optimization
 * method).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP
#define ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP

#include <ensmallen_bits/lbfgs/lbfgs.hpp>
#include <ensmallen_bits/function.hpp>
#include "aug_lagrangian_function.hpp"

namespace ens {

inline AugLagrangian::AugLagrangian(const size_t maxIterations,
                                    const double penaltyThresholdFactor,
                                    const double sigmaUpdateFactor,
                                    const L_BFGS& lbfgs) :
    maxIterations(maxIterations),
    penaltyThresholdFactor(penaltyThresholdFactor),
    sigmaUpdateFactor(sigmaUpdateFactor),
    lbfgs(lbfgs),
    terminate(false),
    sigma(0.0)
{
}

template<typename LagrangianFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value, bool>::type
AugLagrangian::Optimize(LagrangianFunctionType& function,
                        MatType& coordinates,
                        const arma::vec& initLambda,
                        const double initSigma,
                        CallbackTypes&&... callbacks)
{
  lambda = initLambda;
  sigma = initSigma;

  AugLagrangianFunction<LagrangianFunctionType> augfunc(function,
      lambda, sigma);

  return Optimize(augfunc, coordinates, callbacks...);
}

template<typename LagrangianFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value, bool>::type
AugLagrangian::Optimize(LagrangianFunctionType& function,
                        MatType& coordinates,
                        CallbackTypes&&... callbacks)
{
  // If the user did not specify the right size for sigma and lambda, we will
  // use defaults.
  if (!lambda.is_empty())
  {
    AugLagrangianFunction<LagrangianFunctionType> augfunc(function, lambda,
        sigma);
    return Optimize(augfunc, coordinates, callbacks...);
  }
  else
  {
    AugLagrangianFunction<LagrangianFunctionType> augfunc(function);
    return Optimize(augfunc, coordinates, callbacks...);
  }
}

template<typename LagrangianFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value, bool>::type
AugLagrangian::Optimize(
    AugLagrangianFunction<LagrangianFunctionType>& augfunc,
    MatType& coordinatesIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  BaseMatType& coordinates = (BaseMatType&) coordinatesIn;

  // Check that the types satisfy our needs.
  traits::CheckConstrainedFunctionTypeAPI<LagrangianFunctionType, BaseMatType,
      BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  LagrangianFunctionType& function = augfunc.Function();

  // Ensure that we update lambda immediately.
  ElemType penaltyThreshold = std::numeric_limits<ElemType>::max();

  // Track the last objective to compare for convergence.
  ElemType lastObjective = function.Evaluate(coordinates);

  // Convergence tolerance---depends on the epsilon of the type we are using for
  // optimization.
  ElemType tolerance = 1e3 * std::numeric_limits<ElemType>::epsilon();

  // Then, calculate the current penalty.
  ElemType penalty = 0;
  for (size_t i = 0; i < function.NumConstraints(); i++)
  {
    const ElemType p = std::pow(function.EvaluateConstraint(i, coordinates), 2);
    terminate |= Callback::EvaluateConstraint(*this, function, coordinates, i,
        p, callbacks...);

    penalty += p;
  }

  Info << "Penalty is " << penalty << " (threshold " << penaltyThreshold
      << ")." << std::endl;

  // The odd comparison allows user to pass maxIterations = 0 (i.e. no limit on
  // number of iterations).
  size_t it;
  Callback::BeginOptimization(*this, function, coordinates, callbacks...);
  for (it = 0; it != (maxIterations - 1) && !terminate; it++)
  {
    Info << "AugLagrangian on iteration " << it
        << ", starting with objective "  << lastObjective << "." << std::endl;

    if (!lbfgs.Optimize(augfunc, coordinates, callbacks...))
      Info << "L-BFGS reported an error during optimization." << std::endl;
    Info << "Done with L-BFGS." << std::endl;

    const ElemType objective = function.Evaluate(coordinates);

    terminate |= Callback::Evaluate(*this, function, coordinates, objective,
        callbacks...);

    // Check if we are done with the entire optimization (the threshold we are
    // comparing with is arbitrary).
    if (std::abs(lastObjective - objective) < tolerance &&
        augfunc.Sigma() > 500000)
    {
      lambda = std::move(augfunc.Lambda());
      sigma = augfunc.Sigma();

      Callback::EndOptimization(*this, function, coordinates, callbacks...);
      return true;
    }

    lastObjective = objective;

    // Assuming that the optimization has converged to a new set of coordinates,
    // we now update either lambda or sigma.  We update sigma if the penalty
    // term is too high, and we update lambda otherwise.

    // First, calculate the current penalty.
    ElemType penalty = 0;
    for (size_t i = 0; i < function.NumConstraints(); i++)
    {
      const ElemType p = std::pow(function.EvaluateConstraint(i, coordinates),
          2);
      terminate |= Callback::EvaluateConstraint(*this, function, coordinates, i,
          p, callbacks...);

      penalty += p;
    }

    Info << "Penalty is " << penalty << " (threshold " << penaltyThreshold
        << ")." << std::endl;

    if (terminate)
      break;

    if (penalty < penaltyThreshold) // We update lambda.
    {
      // We use the update: lambda_{k + 1} = lambda_k - sigma * c(coordinates),
      // but we have to write a loop to do this for each constraint.
      for (size_t i = 0; i < function.NumConstraints(); i++)
      {
        const ElemType p = function.EvaluateConstraint(i, coordinates);
        terminate |= Callback::EvaluateConstraint(*this, function, coordinates,
            i, p, callbacks...);

        augfunc.Lambda()[i] -= augfunc.Sigma() * p;
      }

      // We also update the penalty threshold to be a factor of the current
      // penalty.
      penaltyThreshold = penaltyThresholdFactor * penalty;
      Info << "Lagrange multiplier estimates updated." << std::endl;
    }
    else
    {
      // We multiply sigma by a constant value.
      augfunc.Sigma() *= sigmaUpdateFactor;
      Info << "Updated sigma to " << augfunc.Sigma() << "." << std::endl;
      if (augfunc.Sigma() >= std::numeric_limits<ElemType>::max() / 2.0)
      {
        Warn << "AugLagrangian::Optimize(): sigma too large for element type; "
            << "terminating." << std::endl;
        Callback::EndOptimization(*this, function, coordinates, callbacks...);
        return false;
      }
    }

    terminate |= Callback::StepTaken(*this, function, coordinates,
        callbacks...);
  }

  Callback::EndOptimization(*this, function, coordinates, callbacks...);
  return false;
}

} // namespace ens

#endif // ENSMALLEN_AUG_LAGRANGIAN_AUG_LAGRANGIAN_IMPL_HPP

