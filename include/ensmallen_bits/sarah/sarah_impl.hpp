/**
 * @file sarah_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of StochAstic Recusive gRadient algoritHm (SARAH).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SARAH_SARAH_IMPL_HPP
#define ENSMALLEN_SARAH_SARAH_IMPL_HPP

// In case it hasn't been included yet.
#include "sarah.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename UpdatePolicyType>
SARAHType<UpdatePolicyType>::SARAHType(
    const double stepSize,
    const size_t batchSize,
    const size_t maxIterations,
    const size_t innerIterations,
    const double tolerance,
    const bool shuffle,
    const UpdatePolicyType& updatePolicy,
    const bool exactObjective) :
    stepSize(stepSize),
    batchSize(batchSize),
    maxIterations(maxIterations),
    innerIterations(innerIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    exactObjective(exactObjective),
    updatePolicy(updatePolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename UpdatePolicyType>
template<typename SeparableFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
SARAHType<UpdatePolicyType>::Optimize(
    SeparableFunctionType& functionIn,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  typedef Function<SeparableFunctionType, BaseMatType, BaseGradType>
      FullFunctionType;
  FullFunctionType& function(static_cast<FullFunctionType&>(functionIn));

  traits::CheckSeparableFunctionTypeAPI<SeparableFunctionType,
      BaseMatType, BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  ElemType overallObjective = 0;
  ElemType lastObjective = DBL_MAX;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Set epoch length to n / b if the user asked for.
  if (innerIterations == 0)
    innerIterations = numFunctions;

  // Now iterate!
  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  BaseGradType v(iterate.n_rows, iterate.n_cols);
  BaseGradType gradient0(iterate.n_rows, iterate.n_cols);
  BaseMatType iterate0;

  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  Callback::BeginOptimization(*this, function, iterate, callbacks...);
  for (size_t i = 0; i < actualMaxIterations && !terminate; ++i)
  {
    // Calculate the objective function.
    overallObjective = 0;
    for (size_t f = 0; f < numFunctions; f += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
      const ElemType objective = function.Evaluate(iterate, f,
          effectiveBatchSize);
      overallObjective += objective;

      terminate |= Callback::Evaluate(*this, function, iterate, objective,
          callbacks...);
    }

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "SARAH: converged to " << overallObjective
          << "; terminating  with failure.  Try a smaller step size?"
          << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Info << "SARAH: minimized within tolerance " << tolerance
          << "; terminating optimization." << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    lastObjective = overallObjective;

    // Compute the full gradient.
    size_t effectiveBatchSize = std::min(batchSize, numFunctions);
    function.Gradient(iterate, 0, v, effectiveBatchSize);

    terminate |= Callback::Gradient(*this, function, iterate, v, callbacks...);

    for (size_t f = effectiveBatchSize; f < numFunctions;
        /* incrementing done manually */)
    {
      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - f);

      function.Gradient(iterate, f, gradient, effectiveBatchSize);
      v += gradient;

      f += effectiveBatchSize;
    }
    v /= (double) numFunctions;

    if (terminate)
      break;

    // Update iterate with full gradient (v).
    iterate -= stepSize * v;

    const ElemType vNorm = arma::norm(v);

    for (size_t f = 0, currentFunction = 0; f < innerIterations;
        /* incrementing done manually */)
    {
      // Is this iteration the start of a sequence?
      if ((currentFunction % numFunctions) == 0)
      {
        currentFunction = 0;

        // Determine order of visitation.
        if (shuffle)
          function.Shuffle();
      }

      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - currentFunction);

      // Calculate variance reduced gradient.
      function.Gradient(iterate, currentFunction, gradient,
          effectiveBatchSize);

      terminate |= Callback::Gradient(*this, function, iterate, gradient,
          callbacks...);

      // Avoid an unnecessary copy on the first iteration.
      if (f > 0)
      {
        function.Gradient(iterate0, currentFunction, gradient0,
            effectiveBatchSize);

        terminate |= Callback::Gradient(*this, function, iterate0, gradient0,
            callbacks...);

        // Store current parameter for the calculation of the variance reduced
        // gradient.
        iterate0 = iterate;

        // Use the update policy to take a step.
        if (terminate || updatePolicy.Update(iterate, v, gradient, gradient0,
            effectiveBatchSize, stepSize, vNorm))
        {
          break;
        }
      }
      else
      {
        // Store current parameter for the calculation of the variance reduced
        // gradient.
        iterate0 = iterate;

        // Use the update policy to take a step.
        if (terminate || updatePolicy.Update(iterate, v, gradient, gradient,
            effectiveBatchSize, stepSize, vNorm))
        {
          break;
        }
      }

      terminate |= Callback::StepTaken(*this, function, iterate, callbacks...);
      currentFunction += effectiveBatchSize;
      f += effectiveBatchSize;
    }
  }

  Info << "SARAH: maximum iterations (" << maxIterations << ") reached; "
      << "terminating optimization." << std::endl;

  // Calculate final objective.
  if (exactObjective)
  {
    overallObjective = 0;
    for (size_t i = 0; i < numFunctions; i += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
      const ElemType objective = function.Evaluate(iterate, i, effectiveBatchSize);
      overallObjective += objective;

      // The optimization is finished, so we don't need to care about the result
      // of the callback.
      (void) Callback::Evaluate(*this, function, iterate, objective,
          callbacks...);
    }
  }

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif
