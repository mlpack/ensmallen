/**
 * @file cd_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Implementation of coordinate descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CD_CD_IMPL_HPP
#define ENSMALLEN_CD_CD_IMPL_HPP

// In case it hasn't been included yet.
#include "cd.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template <typename DescentPolicyType>
CD<DescentPolicyType>::CD(
    const double stepSize,
    const size_t maxIterations,
    const double tolerance,
    const size_t updateInterval,
    const DescentPolicyType descentPolicy) :
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    updateInterval(updateInterval),
    descentPolicy(descentPolicy)
{ /* Nothing to do */ }

//! Optimize the function (minimize).
template <typename DescentPolicyType>
template <typename ResolvableFunctionType,
          typename MatType,
          typename GradType,
          typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
CD<DescentPolicyType>::Optimize(
    ResolvableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  // Make sure we have the methods that we need.
  traits::CheckResolvableFunctionTypeAPI<ResolvableFunctionType, BaseMatType,
      BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();

  ElemType overallObjective = 0;
  ElemType lastObjective = std::numeric_limits<ElemType>::max();

  BaseMatType& iterate = (BaseMatType&) iterateIn;
  BaseGradType gradient;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Start iterating.
  Callback::BeginOptimization(*this, function, iterate, callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    // Get the coordinate to descend on.
    size_t featureIdx = descentPolicy.template DescentFeature<
        ResolvableFunctionType, BaseMatType, BaseGradType>(i, iterate,
        function);

    // Get the partial gradient with respect to this feature.
    function.PartialGradient(iterate, featureIdx, gradient);

    terminate |= Callback::Gradient(*this, function, iterate, overallObjective,
        gradient, callbacks...);
    if (terminate)
      break;

    // Update the decision variable with the partial gradient.
    iterate.col(featureIdx) -= stepSize * gradient.col(featureIdx);
    terminate |= Callback::StepTaken(*this, function, iterate, callbacks...);

    // Check for convergence.
    if (i % updateInterval == 0)
    {
      overallObjective = function.Evaluate(iterate);
      terminate |= Callback::Evaluate(*this, function, iterate,
          overallObjective, callbacks...);

      // Output current objective function.
      Info << "CD: iteration " << i << ", objective " << overallObjective
          << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Warn << "CD: converged to " << overallObjective << "; terminating"
            << " with failure.  Try a smaller step size?" << std::endl;

        Callback::EndOptimization(*this, function, iterate, callbacks...);
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Info << "CD: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;

        Callback::EndOptimization(*this, function, iterate, callbacks...);
        return overallObjective;
      }

      lastObjective = overallObjective;
    }
  }

  Info << "CD: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate and return final objective.  No need to pay attention to the
  // result of the callback.
  const ElemType objective = function.Evaluate(iterate);
  (void) Callback::Evaluate(*this, function, iterate, objective, callbacks...);

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return objective;
}

} // namespace ens

#endif
