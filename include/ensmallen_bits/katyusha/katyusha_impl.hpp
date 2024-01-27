/**
 * @file katyusha_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of Katyusha a direct, primal-only stochastic gradient method.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_KATYUSHA_KATYUSHA_IMPL_HPP
#define ENSMALLEN_KATYUSHA_KATYUSHA_IMPL_HPP

// In case it hasn't been included yet.
#include "katyusha.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

template<bool Proximal>
KatyushaType<Proximal>::KatyushaType(
    const double convexity,
    const double lipschitz,
    const size_t batchSize,
    const size_t maxIterations,
    const size_t innerIterations,
    const double tolerance,
    const bool shuffle,
    const bool exactObjective) :
    convexity(convexity),
    lipschitz(lipschitz),
    batchSize(batchSize),
    maxIterations(maxIterations),
    innerIterations(innerIterations),
    tolerance(tolerance),
    shuffle(shuffle),
    exactObjective(exactObjective)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<bool Proximal>
template<typename SeparableFunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
KatyushaType<Proximal>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  traits::CheckSeparableFunctionTypeAPI<SeparableFunctionType,
      BaseMatType, BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // Set epoch length to n / b if the user asked for.
  if (innerIterations == 0)
    innerIterations = numFunctions;

  // Find the number of batches.
  size_t numBatches = innerIterations / batchSize;
  if (numFunctions % batchSize != 0)
    ++numBatches; // Capture last few.

  const double tau1 = std::min(0.5,
      std::sqrt(batchSize * convexity / (3.0 * lipschitz)));
  const double tau2 = 0.5;
  const double alpha = 1.0 / (3.0 * tau1 * lipschitz);
  const double r = 1.0 + std::min(alpha * convexity, 1.0 /
      (4.0 / innerIterations));

  // sum_{j=0}^{m-1} 1 + std::min(alpha * convexity, 1 / (4 * m)^j).
  double normalizer = 1;
  for (size_t i = 0; i < numBatches; i++)
  {
    normalizer = r * (normalizer + 1.0);
  }
  normalizer = 1.0 / normalizer;

  // To keep track of where we are and how things are going.
  ElemType overallObjective = 0;
  ElemType lastObjective = DBL_MAX;

  // Now iterate!
  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  BaseGradType fullGradient(iterate.n_rows, iterate.n_cols);
  BaseGradType gradient0(iterate.n_rows, iterate.n_cols);

  BaseMatType iterate0 = iterate;
  BaseMatType y = iterate;
  BaseMatType z = iterate;
  BaseMatType w(iterate.n_rows, iterate.n_cols);
  w.zeros();

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
      const ElemType objective = function.Evaluate(iterate0, f,
          effectiveBatchSize);
      overallObjective += objective;

      terminate |= Callback::Evaluate(*this, function, iterate0, objective,
          callbacks...);
    }

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "Katyusha: converged to " << overallObjective
          << "; terminating  with failure.  Try a smaller step size?"
          << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Info << "Katyusha: minimized within tolerance " << tolerance
          << "; terminating optimization." << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    lastObjective = overallObjective;

    // Compute the full gradient.
    size_t effectiveBatchSize = std::min(batchSize, numFunctions);
    function.Gradient(iterate, 0, fullGradient, effectiveBatchSize);
    terminate |= Callback::Gradient(*this, function, iterate, fullGradient,
          callbacks...);
    for (size_t f = effectiveBatchSize; f < numFunctions;
        /* incrementing done manually */)
    {
      // Find the effective batch size (the last batch may be smaller).
      effectiveBatchSize = std::min(batchSize, numFunctions - f);

      function.Gradient(iterate0, f, gradient, effectiveBatchSize);
      fullGradient += gradient;

      terminate |= Callback::Gradient(*this, function, iterate0, gradient,
          callbacks...);

      f += effectiveBatchSize;
    }
    fullGradient /= (double) numFunctions;

    // To keep track of where we are and how things are going.
    double cw = 1;
    w.zeros();

    for (size_t f = 0, currentFunction = 0; (f < innerIterations) && !terminate;
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
      iterate = tau1 * z + tau2 * iterate0 + (1 - tau1 - tau2) * y;

      terminate |= Callback::StepTaken(*this, function, iterate,
            callbacks...);

      // Calculate variance reduced gradient.
      function.Gradient(iterate, currentFunction, gradient,
          effectiveBatchSize);
      terminate |= Callback::Gradient(*this, function, iterate, gradient,
          callbacks...);

      function.Gradient(iterate0, currentFunction, gradient0,
          effectiveBatchSize);
      terminate |= Callback::Gradient(*this, function, iterate0, gradient0,
          callbacks...);

      // By the minimality definition of z_{k + 1}, we have that:
      // z_{k+1} − z_k + \alpha * \sigma_{k+1} + \alpha g = 0.
      BaseMatType zNew = z - alpha * (fullGradient + (gradient - gradient0) /
          (double) batchSize);

      // Proximal update, choose between Option I and Option II. Shift relative
      // to the Lipschitz constant or take a constant step using the given step
      // size.
      if (Proximal)
      {
        // yk = x0 − 1 / (3L) * \delta1, k = 1
        // yk = x0 − 1 / (3L) * \delta2 - ((1 - tau) / (3L)) + tau * alpha)
        // * \delta1, k = 2
        // yk = x0 − 1 / (3L) * \delta3 - ((1 - tau) / (3L)) + tau * alpha)
        // * \delta2 - ((1-tau)^2 / (3L) + (1 - (1 - tau)^2) * alpha) * \delta1,
        // k = 3.
        y = iterate + 1.0 / (3.0 * lipschitz) * w;
      }
      else
      {
        y = iterate + tau1 * (zNew - z);
      }

      z = std::move(zNew);

      // sum_{j=0}^{m-1} 1 + std::min(alpha * convexity, 1 / (4 * m)^j * ys).
      w += cw * iterate;
      cw *= r;

      currentFunction += effectiveBatchSize;
      f += effectiveBatchSize;
    }
    iterate0 = normalizer * w;
  }

  Info << "Katyusha: maximum iterations (" << maxIterations << ") reached"
      << "; terminating optimization." << std::endl;

  // Calculate final objective if exactObjective is set to true.
  if (exactObjective)
  {
    overallObjective = 0;
    for (size_t i = 0; i < numFunctions; i += batchSize)
    {
      const size_t effectiveBatchSize = std::min(batchSize, numFunctions - i);
      const ElemType objective = function.Evaluate(iterate, i,
          effectiveBatchSize);
      overallObjective += objective;

      // The optimization is finished, so we don't need to care about the
      // callback result.
      (void) Callback::Evaluate(*this, function, iterate, objective,
          callbacks...);
    }
  }

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif
