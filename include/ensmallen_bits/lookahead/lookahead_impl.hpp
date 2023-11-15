/**
 * @file lookahead_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of Lookahead Optimizer: k steps forward, 1 step back.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_LOOKAHEAD_LOOKAHEAD_IMPL_HPP
#define ENSMALLEN_LOOKAHEAD_LOOKAHEAD_IMPL_HPP

// In case it hasn't been included yet.
#include "lookahead.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename BaseOptimizerType, typename DecayPolicyType>
inline Lookahead<BaseOptimizerType, DecayPolicyType>::Lookahead(
    const double stepSize,
    const size_t k,
    const size_t maxIterations,
    const double tolerance,
    const DecayPolicyType& decayPolicy,
    const bool resetPolicy,
    const bool exactObjective) :
    baseOptimizer(BaseOptimizerType()),
    stepSize(stepSize),
    k(k),
    maxIterations(maxIterations),
    tolerance(tolerance),
    decayPolicy(decayPolicy),
    resetPolicy(resetPolicy),
    exactObjective(exactObjective),
    isInitialized(false)
{ /* Nothing to do. */ }

template<typename BaseOptimizerType, typename DecayPolicyType>
inline Lookahead<BaseOptimizerType, DecayPolicyType>::Lookahead(
    const BaseOptimizerType& baseOptimizer,
    const double stepSize,
    const size_t k,
    const size_t maxIterations,
    const double tolerance,
    const DecayPolicyType& decayPolicy,
    const bool resetPolicy,
    const bool exactObjective) :
    baseOptimizer(baseOptimizer),
    stepSize(stepSize),
    k(k),
    maxIterations(maxIterations),
    tolerance(tolerance),
    decayPolicy(decayPolicy),
    resetPolicy(resetPolicy),
    exactObjective(exactObjective),
    isInitialized(false)
{ /* Nothing to do. */ }

template<typename BaseOptimizerType, typename DecayPolicyType>
inline Lookahead<BaseOptimizerType, DecayPolicyType>::~Lookahead()
{
  instDecayPolicy.Clean();
}

//! Optimize the function (minimize).
template<typename BaseOptimizerType, typename DecayPolicyType>
template<typename SeparableFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
Lookahead<BaseOptimizerType, DecayPolicyType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  typedef Function<SeparableFunctionType, BaseMatType, BaseGradType>
      FullFunctionType;
  FullFunctionType& f(static_cast<FullFunctionType&>(function));

  // The decay policy internally use a templated class so that
  // we can know MatType and GradType only when Optimize() is called.
  typedef typename DecayPolicyType::template Policy<BaseMatType, BaseGradType>
      InstDecayPolicyType;

  // Make sure we have all the methods that we need.
  traits::CheckSeparableFunctionTypeAPI<FullFunctionType, BaseMatType,
      BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Check if the optimizer implements HasMaxIterations() and override the
  // parameter with k.
  SetMaxIterations(baseOptimizer, k);

  // Check if the optimizer implements ResetPolicy() and override the reset
  // policy.
  if (traits::HasResetPolicySignature<BaseOptimizerType>::value &&
      baseOptimizer.ResetPolicy())
  {
    Warn << "Parameters are reset before every Optimize call; set "
        << "ResetPolicy() to false.";
    baseOptimizer.ResetPolicy() = resetPolicy;
  }

  // To keep track of where we are and how things are going.
  ElemType overallObjective = 0;
  ElemType lastOverallObjective = DBL_MAX;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Initialize the decay policy if needed.
  if (!isInitialized || !instDecayPolicy.Has<InstDecayPolicyType>())
  {
    instDecayPolicy.Clean();
    instDecayPolicy.Set<InstDecayPolicyType>(
        new InstDecayPolicyType(decayPolicy));
    isInitialized = true;
  }

  // Now iterate!
  Callback::BeginOptimization(*this, f, iterate, callbacks...);
  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t i = 0; i < actualMaxIterations && !terminate; i++)
  {
    BaseMatType iterateModel = iterate;

    overallObjective = baseOptimizer.Optimize(f, iterateModel,
        callbacks...);

    // Now update the learning rate if requested by the user, note we pass the
    // latest inner model coordinates instead of the gradient.
    instDecayPolicy.As<InstDecayPolicyType>().Update(iterate, stepSize,
        iterateModel);

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "Lookahead: converged to " << overallObjective
          << "; terminating with failure.  Try a smaller step size?"
          << std::endl;

      iterate = iterateModel;
      Callback::EndOptimization(*this, f, iterate, callbacks...);
      return overallObjective;
    }

    if (std::abs(lastOverallObjective - overallObjective) < tolerance)
    {
      Info << "Lookahead: minimized within tolerance " << tolerance << "; "
          << "terminating optimization." << std::endl;

      iterate = iterateModel;
      Callback::EndOptimization(*this, f, iterate, callbacks...);
      return overallObjective;
    }

    iterate += stepSize * (iterateModel - iterate);
    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);

    // Save the current objective.
    lastOverallObjective = overallObjective;
  }

  Info << "Lookahead: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate final objective if exactObjective is set to true.
  if (exactObjective)
  {
    // Find the number of functions to use.
    const size_t numFunctions = f.NumFunctions();

    size_t batchSize = 1;
    // Check if the optimizer implements the BatchSize() method and use the
    // parameter for the objective calculation.
    if (traits::HasBatchSizeSignature<BaseOptimizerType>::value)
      batchSize = baseOptimizer.BatchSize();

    overallObjective = 0;
    for (size_t i = 0; i < numFunctions; i += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
      const ElemType objective = f.Evaluate(iterate, i, effectiveBatchSize);
      overallObjective += objective;

      // The optimization is over, so we don't need to care about the result of
      // the callback.
      (void) Callback::Evaluate(*this, f, iterate, objective, callbacks...);
    }
  }

  Callback::EndOptimization(*this, f, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif
