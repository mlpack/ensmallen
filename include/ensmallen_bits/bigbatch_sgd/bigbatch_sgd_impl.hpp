/**
 * @file bigbatch_sgd_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of big-batch SGD.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_BIGBATCH_SGD_BIGBATCH_SGD_IMPL_HPP
#define ENSMALLEN_BIGBATCH_SGD_BIGBATCH_SGD_IMPL_HPP

// In case it hasn't been included yet.
#include "bigbatch_sgd.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename UpdatePolicyType>
BigBatchSGD<UpdatePolicyType>::BigBatchSGD(
    const size_t batchSize,
    const double stepSize,
    const double batchDelta,
    const size_t maxIterations,
    const double tolerance,
    const bool shuffle,
    const bool exactObjective) :
    batchSize(batchSize),
    stepSize(stepSize),
    batchDelta(batchDelta),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    exactObjective(exactObjective),
    updatePolicy(UpdatePolicyType())
{ /* Nothing to do. */ }

template<typename UpdatePolicyType>
BigBatchSGD<UpdatePolicyType>::~BigBatchSGD()
{
  instUpdatePolicy.Clean();
}

//! Optimize the function (minimize).
template<typename UpdatePolicyType>
template<typename SeparableFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
BigBatchSGD<UpdatePolicyType>::Optimize(
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

  // Make sure we have all the methods that we need.
  traits::CheckSeparableFunctionTypeAPI<FullFunctionType, BaseMatType,
      BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  typedef typename UpdatePolicyType::template Policy<BaseMatType>
      InstUpdatePolicyType;

  if (!instUpdatePolicy.Has<InstUpdatePolicyType>())
  {
    instUpdatePolicy.Clean();
    instUpdatePolicy.Set<InstUpdatePolicyType>(
        new InstUpdatePolicyType(updatePolicy));
  }

  // Find the number of functions to use.
  const size_t numFunctions = f.NumFunctions();

  // To keep track of where we are and how things are going.
  size_t currentFunction = 0;
  size_t epoch = 1;
  ElemType overallObjective = 0;
  ElemType lastObjective = DBL_MAX;
  bool reset = false;
  BaseGradType delta0, delta1;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Now iterate!
  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  BaseGradType functionGradient(iterate.n_rows, iterate.n_cols);
  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  Callback::BeginOptimization(*this, f, iterate, callbacks...);
  for (size_t i = 0; i < actualMaxIterations && !terminate;
      /* incrementing done manually */)
  {
    // Find the effective batch size; we have to take the minimum of three
    // things:
    // - the batch size can't be larger than the user-specified batch size;
    // - the batch size can't be larger than the number of iterations left
    //       before actualMaxIterations is hit;
    // - the batch size can't be larger than the number of functions left.
    size_t effectiveBatchSize = std::min(
        std::min(batchSize, actualMaxIterations - i),
        numFunctions - currentFunction);

    size_t k = 1;
    double vB = 0;

    // Compute the stochastic gradient estimation.
    f.Gradient(iterate, currentFunction, gradient, 1);

    terminate |= Callback::Gradient(*this, f, iterate, gradient, callbacks...);

    delta1 = gradient;
    for (size_t j = 1; j < effectiveBatchSize; ++j, ++k)
    {
      f.Gradient(iterate, currentFunction + j, functionGradient, 1);

      terminate |= Callback::Gradient(*this, f, iterate, functionGradient,
          callbacks...);

      delta0 = delta1 + (functionGradient - delta1) / k;

      // Compute sample variance.
      vB += arma::norm(functionGradient - delta1, 2.0) *
          arma::norm(functionGradient - delta0, 2.0);

      delta1 = delta0;
      gradient += functionGradient;
    }
    double gB = std::pow(arma::norm(gradient / effectiveBatchSize, 2), 2.0);

    // Reset the batch size update process counter.
    reset = false;

    // Increase batchSize only if there are more samples left.
    if (effectiveBatchSize == batchSize)
    {
      // Update batch size.
      while (gB <= ((1 / ((double) batchSize - 1) * vB) / batchSize))
      {
        // Increase batch size at least by one.
        size_t batchOffset = batchDelta * batchSize;
        if (batchOffset <= 0)
          batchOffset = 1;

        if ((currentFunction + batchSize + batchOffset) >= numFunctions)
          break;

        // Update the stochastic gradient estimation.
        const size_t batchStart = (currentFunction + batchSize + batchOffset
            - 1) < numFunctions ? currentFunction + batchSize - 1 : 0;
        for (size_t j = 0; j < batchOffset; ++j, ++k)
        {
          f.Gradient(iterate, batchStart + j, functionGradient, 1);
          terminate |= Callback::Gradient(*this, f, iterate,
              functionGradient, callbacks...);

          delta0 = delta1 + (functionGradient - delta1) / (k + 1);

          // Compute sample variance.
          vB += arma::norm(functionGradient - delta1, 2.0) *
              arma::norm(functionGradient - delta0, 2.0);

          delta1 = delta0;
          gradient += functionGradient;
        }
        gB = std::pow(arma::norm(gradient / (batchSize + batchOffset), 2), 2.0);

        // Update the batchSize.
        batchSize += batchOffset;
        effectiveBatchSize += batchOffset;

        // Batch size updated.
        reset = true;
      }
    }

    if (terminate)
      break;

    instUpdatePolicy.As<InstUpdatePolicyType>().Update(f, stepSize, iterate,
        gradient, gB, vB, currentFunction, batchSize, effectiveBatchSize,
        reset);

    // Update the iterate.
    iterate -= stepSize * gradient;
    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);

    const ElemType objective = f.Evaluate(iterate, currentFunction,
        effectiveBatchSize);
    overallObjective += objective;

    terminate |= Callback::Evaluate(*this, f, iterate, objective,
        callbacks...);

    i += effectiveBatchSize;
    currentFunction += effectiveBatchSize;

    // Is this iteration the start of a sequence?
    if ((currentFunction % numFunctions) == 0)
    {
      terminate |= Callback::EndEpoch(*this, f, iterate, epoch++,
          overallObjective / (ElemType) numFunctions, callbacks...);

      // Output current objective function.
      Info << "Big-batch SGD: iteration " << i << ", objective "
          << overallObjective << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Warn << "Big-batch SGD: converged to " << overallObjective
            << "; terminating with failure.  Try a smaller step size?"
            << std::endl;

        Callback::EndOptimization(*this, f, iterate, callbacks...);
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Info << "Big-batch SGD: minimized within tolerance " << tolerance
            << "; terminating optimization." << std::endl;

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
    Info << "Big-batch SGD: maximum iterations (" << maxIterations << ") "
        << "reached; terminating optimization." << std::endl;
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

      // The optimization is finished, so we don't need to care what the
      // callback returns.
      (void) Callback::Evaluate(*this, f, iterate, objective, callbacks...);
    }
  }

  Callback::EndOptimization(*this, f, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif
