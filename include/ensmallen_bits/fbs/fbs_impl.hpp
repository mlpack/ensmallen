/**
 * @file fbs_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of Forward-Backward Splitting (FBS).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FBS_FBS_IMPL_HPP
#define ENSMALLEN_FBS_FBS_IMPL_HPP

// In case it hasn't been included yet.
#include "fbs.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

//! Constructor of the FBS class.
template<typename BackwardStepType>
FBS<BackwardStepType>::FBS(const double stepSize,
                           const size_t maxIterations,
                           const double tolerance) :
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }

template<typename BackwardStepType>
FBS<BackwardStepType>::FBS(BackwardStepType backwardStep,
                           const double stepSize,
                           const size_t maxIterations,
                           const double tolerance) :
    backwardStep(std::move(backwardStep)),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename BackwardStepType>
template<typename FunctionType, typename MatType, typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
    typename MatType::elem_type>::type
FBS<BackwardStepType>::Optimize(FunctionType& function,
                                MatType& iterateIn,
                                CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  typedef Function<FunctionType, BaseMatType, BaseGradType> FullFunctionType;
  FullFunctionType& f = static_cast<FullFunctionType&>(function);

  // Make sure we have all necessary functions.
  traits::CheckFunctionTypeAPI<FullFunctionType, BaseMatType, BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // To keep track of the function value.
  ElemType currentObjective = std::numeric_limits<ElemType>::max();
  ElemType currentFObjective = currentObjective;
  ElemType currentGObjective = currentObjective;
  ElemType lastObjective = currentObjective;

  BaseGradType gradient(iterate.n_rows, iterate.n_cols);

  // Controls early termination of the optimization process.
  bool terminate = false;

  terminate |= Callback::BeginOptimization(*this, f, iterate, callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    // During this optimization, we want to optimize h(x) = f(x) + g(x).
    // f(x) is `f`, but g(x) is specified by `BackwardStepType`.

    // First compute f(x) and f'(x).
    currentFObjective = f.EvaluateWithGradient(iterate, gradient);
    // Now compute g(x) to get the full objective.
    currentGObjective = backwardStep.Evaluate(iterate);

    lastObjective = currentObjective;
    currentObjective = currentFObjective + currentGObjective;

    terminate |= Callback::EvaluateWithGradient(*this, f, iterate,
        currentObjective, gradient, callbacks...);

    // Output current objective function.
    Info << "FBS::Optimize(): iteration " << i << ", combined objective "
        << currentObjective << " (f(x) = " << currentFObjective << ", g(x) = "
        << currentGObjective << ")." << std::endl;

    // Check for convergence.
    if ((i > 1) && (std::abs(currentObjective - lastObjective) < tolerance))
    {
      Info << "FBS::Optimize(): minimized within objective tolerance "
          << tolerance << "; terminating optimization." << std::endl;

      Callback::EndOptimization(*this, f, iterate, callbacks...);
      return currentObjective;
    }

    if ((i > 1) && !std::isfinite(currentObjective))
    {
      Warn << "FBS::Optimize(): objective diverged to " << currentObjective
          << "; terminating optimization." << std::endl;

      Callback::EndOptimization(*this, f, iterate, callbacks...);
      return currentObjective;
    }

    // Perform forward update.
    iterate -= stepSize * gradient;
    // Now perform backward step (proximal update).
    backwardStep.ProximalStep(iterate, stepSize);

    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);
  }

  if (!terminate)
  {
    Info << "FBS::Optimize(): maximum iterations (" << maxIterations
        << ") reached; terminating optimization." << std::endl;
  }

  Callback::EndOptimization(*this, f, iterate, callbacks...);
  return currentObjective;
} // Optimize()

} // namespace ens

#endif
