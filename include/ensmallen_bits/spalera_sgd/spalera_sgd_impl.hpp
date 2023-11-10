/**
 * @file spalera_sgd_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of SPALeRA SGD.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SPALERA_SGD_SPALERA_SGD_IMPL_HPP
#define ENSMALLEN_SPALERA_SGD_SPALERA_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "spalera_sgd.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename DecayPolicyType>
SPALeRASGD<DecayPolicyType>::SPALeRASGD(const double stepSize,
                                        const size_t batchSize,
                                        const size_t maxIterations,
                                        const double tolerance,
                                        const double lambda,
                                        const double alpha,
                                        const double epsilon,
                                        const double adaptRate,
                                        const bool shuffle,
                                        const DecayPolicyType& decayPolicy,
                                        const bool resetPolicy,
                                        const bool exactObjective) :
    stepSize(stepSize),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    lambda(lambda),
    shuffle(shuffle),
    exactObjective(exactObjective),
    updatePolicy(SPALeRAStepsize(alpha, epsilon, adaptRate)),
    decayPolicy(decayPolicy),
    resetPolicy(resetPolicy),
    isInitialized(false)
{ /* Nothing to do. */ }

template<typename DecayPolicyType>
SPALeRASGD<DecayPolicyType>::~SPALeRASGD()
{
  instUpdatePolicy.Clean();
  instDecayPolicy.Clean();
}

//! Optimize the function (minimize).
template<typename DecayPolicyType>
template<typename SeparableFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
SPALeRASGD<DecayPolicyType>::Optimize(
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

  traits::CheckSeparableFunctionTypeAPI<FullFunctionType, BaseMatType,
      BaseGradType>();
  RequireDenseFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // The update policy and decay policy internally use a templated class so that
  // we can know MatType and GradType only when Optimize() is called.
  typedef typename SPALeRAStepsize::Policy<BaseMatType, BaseGradType>
      InstUpdatePolicyType;
  typedef typename DecayPolicyType::template Policy<BaseMatType, BaseGradType>
      InstDecayPolicyType;

  // Find the number of functions to use.
  const size_t numFunctions = f.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  size_t epoch = 1;
  ElemType overallObjective = 0;
  ElemType lastObjective = DBL_MAX;

  // Controls early termination of the optimization process.
  bool terminate = false;

  Callback::BeginOptimization(*this, f, iterate, callbacks...);

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; i += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
    const ElemType objective = f.Evaluate(iterate, i, effectiveBatchSize);
    overallObjective += objective;

    terminate |= Callback::Evaluate(*this, f, iterate, objective, callbacks...);
  }

  ElemType currentObjective = overallObjective / numFunctions;

  // Initialize the decay policy if needed.
  if (!isInitialized || !instDecayPolicy.Has<InstDecayPolicyType>())
  {
    instDecayPolicy.Clean();
    instDecayPolicy.Set<InstDecayPolicyType>(
        new InstDecayPolicyType(decayPolicy));
  }

  // Initialize the update policy.
  if (resetPolicy || !isInitialized ||
      !instUpdatePolicy.Has<InstUpdatePolicyType>())
  {
    instUpdatePolicy.Clean();
    instUpdatePolicy.Set<InstUpdatePolicyType>(
        new InstUpdatePolicyType(updatePolicy, iterate.n_rows, iterate.n_cols,
                                 currentObjective * lambda));
    isInitialized = true;
  }

  // Now iterate!
  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  terminate |= Callback::BeginEpoch(*this, f, iterate, epoch, overallObjective,
      callbacks...);
  for (size_t i = 0; i < actualMaxIterations && !terminate;
      /* incrementing done manually */)
  {
    // Calculate gradient and objective.
    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of iterations left
    //       before actualMaxIterations is hit;
    // - the batch size can't be larger than the number of functions left.
    const size_t effectiveBatchSize = std::min(
        std::min(batchSize, actualMaxIterations - i),
        numFunctions - currentFunction);

    currentObjective = f.EvaluateWithGradient(iterate, currentFunction,
        gradient, effectiveBatchSize);

    terminate |= Callback::EvaluateWithGradient(*this, f, iterate,
        currentObjective, gradient, callbacks...);
    if (terminate)
      break;

    // Use the update policy to take a step.
    if (!instUpdatePolicy.As<InstUpdatePolicyType>().Update(stepSize,
        currentObjective, effectiveBatchSize, numFunctions, iterate, gradient))
    {
      Warn << "SPALeRA SGD: converged to " << overallObjective << "; "
          << "terminating with failure.  Try a smaller step size?"
          << std::endl;

      Callback::EndOptimization(*this, f, iterate, callbacks...);
      return overallObjective;
    }
    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);

    // Now update the learning rate if requested by the user.
    instDecayPolicy.As<InstDecayPolicyType>().Update(iterate, stepSize,
        gradient);

    i += effectiveBatchSize;
    currentFunction += effectiveBatchSize;
    overallObjective += currentObjective;

    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      terminate |= Callback::EndEpoch(*this, f, iterate, epoch++,
          overallObjective / (ElemType) numFunctions, callbacks...);

      // Output current objective function.
      Info << "SPALeRA SGD: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Warn << "SPALeRA SGD: converged to " << overallObjective
            << "; terminating with failure.  Try a smaller step size?"
            << std::endl;

        Callback::EndOptimization(*this, f, iterate, callbacks...);
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Info << "SPALeRA SGD: minimized within tolerance " << tolerance
            << "; terminating optimization." << std::endl;

        Callback::EndOptimization(*this, f, iterate, callbacks...);
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      terminate |= Callback::BeginEpoch(*this, f, iterate, epoch,
          overallObjective, callbacks...);

      if (shuffle) // Determine order of visitation.
        f.Shuffle();
    }
  }

  Info << "SPALeRA SGD: maximum iterations (" << maxIterations
      << ") reached; terminating optimization." << std::endl;

  // Calculate final objective if exactObjective is set to true.
  if (exactObjective)
  {
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
