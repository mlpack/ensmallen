/**
 * @file sgd_impl.hpp
 * @author Ryan Curtin
 * @author Arun Reddy
 * @author Abhinav Moudgil
 *
 * Implementation of stochastic gradient descent.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SGD_SGD_IMPL_HPP
#define ENSMALLEN_SGD_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "sgd.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename UpdatePolicyType, typename DecayPolicyType>
SGD<UpdatePolicyType, DecayPolicyType>::SGD(
    const double stepSize,
    const size_t batchSize,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const UpdatePolicyType& updatePolicy,
    const DecayPolicyType& decayPolicy,
    const bool resetPolicy,
    const bool exactObjective) :
    stepSize(stepSize),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    exactObjective(exactObjective),
    updatePolicy(updatePolicy),
    decayPolicy(decayPolicy),
    resetPolicy(resetPolicy),
    isInitialized(false)
{ /* Nothing to do. */ }

template<typename UpdatePolicyType, typename DecayPolicyType>
SGD<UpdatePolicyType, DecayPolicyType>::~SGD()
{
  // Clean decay and update policies, if they were initialized.
  instDecayPolicy.Clean();
  instUpdatePolicy.Clean();
}

//! Optimize the function (minimize).
template<typename UpdatePolicyType, typename DecayPolicyType>
template<typename SeparableFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
SGD<UpdatePolicyType, DecayPolicyType>::Optimize(
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

  // The update policy and decay policy internally use a templated class so that
  // we can know MatType and GradType only when Optimize() is called.
  typedef typename UpdatePolicyType::template Policy<BaseMatType, BaseGradType>
      InstUpdatePolicyType;
  typedef typename DecayPolicyType::template Policy<BaseMatType, BaseGradType>
      InstDecayPolicyType;

  // Make sure we have all the methods that we need.
  traits::CheckSeparableFunctionTypeAPI<FullFunctionType, BaseMatType,
      BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Find the number of functions to use.
  const size_t numFunctions = f.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  size_t epoch = 1;
  ElemType overallObjective = 0;
  ElemType lastObjective = DBL_MAX;

  // Controls early termination of the optimization process.
  bool terminate = false;

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
        new InstUpdatePolicyType(updatePolicy, iterate.n_rows, iterate.n_cols));
    isInitialized = true;
  }

  // Now iterate!
  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  Callback::BeginOptimization(*this, f, iterate, callbacks...);
  terminate |= Callback::BeginEpoch(*this, f, iterate, epoch,
      overallObjective, callbacks...);
  for (size_t i = 0; i < actualMaxIterations && !terminate;
      /* incrementing done manually */)
  {
    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of iterations left
    //       before actualMaxIterations is hit;
    // - the batch size can't be larger than the number of functions left.
    const size_t effectiveBatchSize = std::min(
        std::min(batchSize, actualMaxIterations - i),
        numFunctions - currentFunction);

    // Technically we are computing the objective before we take the step, but
    // for many FunctionTypes it may be much quicker to do it like this.
    const ElemType objective = f.EvaluateWithGradient(iterate, currentFunction,
        gradient, effectiveBatchSize);
    overallObjective += objective;

    terminate |= Callback::EvaluateWithGradient(*this, f, iterate, objective,
        gradient, callbacks...);

    // Use the update policy to take a step.
    instUpdatePolicy.As<InstUpdatePolicyType>().Update(iterate, stepSize,
        gradient);

    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);

    // Now update the learning rate if requested by the user.
    instDecayPolicy.As<InstDecayPolicyType>().Update(iterate, stepSize,
        gradient);

    i += effectiveBatchSize;
    currentFunction += effectiveBatchSize;

    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      terminate |= Callback::EndEpoch(*this, f, iterate, epoch++,
          overallObjective / (ElemType) numFunctions, callbacks...);

      // Output current objective function.
      Info << "SGD: iteration " << i << ", objective " << overallObjective
         << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Warn << "SGD: converged to " << overallObjective << "; terminating"
            << " with failure.  Try a smaller step size?" << std::endl;

        Callback::EndOptimization(*this, f, iterate, callbacks...);
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Info << "SGD: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;

        Callback::EndOptimization(*this, f, iterate, callbacks...);
        return overallObjective;
      }

      terminate |= Callback::BeginEpoch(*this, f, iterate, epoch,
          overallObjective, callbacks...);

      // Reset the counter variables.
      lastObjective = overallObjective;
      overallObjective = 0;
      currentFunction = 0;

      if (shuffle) // Determine order of visitation.
        f.Shuffle();
    }
  }

  if (!terminate)
  {
    Info << "SGD: maximum iterations (" << maxIterations << ") reached; "
        << "terminating optimization." << std::endl;
  }

  // Calculate final objective if exactObjective is set to true.
  if (exactObjective)
  {
    overallObjective = 0;
    for (size_t i = 0; i < numFunctions; i += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
      const ElemType objective = f.Evaluate(iterate, i, effectiveBatchSize);
      overallObjective += objective;

      // The optimization is over, so it doesn't matter what the callback
      // returns.
      (void) Callback::Evaluate(*this, f, iterate, objective, callbacks...);
    }
  }

  Callback::EndOptimization(*this, f, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif
