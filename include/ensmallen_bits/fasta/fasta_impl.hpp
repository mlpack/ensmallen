/**
 * @file fasta_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of FASTA (Fast Adaptive Shrinkage/Thresholding Algorithm).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FASTA_FASTA_IMPL_HPP
#define ENSMALLEN_FASTA_FASTA_IMPL_HPP

// In case it hasn't been included yet.
#include "fasta.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

//! Constructor of the FBS class.
template<typename BackwardStepType>
FASTA<BackwardStepType>::FASTA(const size_t maxIterations,
                               const double tolerance,
                               const size_t maxLineSearchSteps,
                               const double stepSizeAdjustment,
                               const size_t lineSearchLookback,
                               const bool estimateStepSize,
                               const size_t estimateTrials,
                               const double maxStepSize) :
    maxIterations(maxIterations),
    tolerance(tolerance),
    maxLineSearchSteps(maxLineSearchSteps),
    stepSizeAdjustment(stepSizeAdjustment),
    lineSearchLookback(lineSearchLookback),
    estimateStepSize(estimateStepSize),
    estimateTrials(estimateTrials),
    maxStepSize(maxStepSize)
{
  // Check estimateSteps parameter.
  if (estimateStepSize && estimateTrials == 0)
  {
    throw std::invalid_argument("FASTA::FASTA(): estimateTrials must be greater"
        " than 0!");
  }

  if (lineSearchLookback == 0)
  {
    throw std::invalid_argument("FASTA::FASTA(): lineSearchLookback cannot be "
        "0!");
  }
}

template<typename BackwardStepType>
FASTA<BackwardStepType>::FASTA(BackwardStepType backwardStep,
                               const size_t maxIterations,
                               const double tolerance,
                               const size_t maxLineSearchSteps,
                               const double stepSizeAdjustment,
                               const size_t lineSearchLookback,
                               const bool estimateStepSize,
                               const size_t estimateTrials,
                               const double maxStepSize) :
    backwardStep(std::move(backwardStep)),
    maxIterations(maxIterations),
    tolerance(tolerance),
    maxLineSearchSteps(maxLineSearchSteps),
    stepSizeAdjustment(stepSizeAdjustment),
    lineSearchLookback(lineSearchLookback),
    estimateStepSize(estimateStepSize),
    estimateTrials(estimateTrials),
    maxStepSize(maxStepSize)
{
  // Check estimateSteps parameter.
  if (estimateStepSize && estimateTrials == 0)
  {
    throw std::invalid_argument("FASTA::FASTA(): estimateTrials must be greater"
        " than 0!");
  }

  if (lineSearchLookback == 0)
  {
    throw std::invalid_argument("FASTA::FASTA(): lineSearchLookback cannot be "
        "0!");
  }
}

//! Optimize the function (minimize).
template<typename BackwardStepType>
template<typename FunctionType, typename MatType, typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsMatrixType<GradType>::value,
    typename MatType::elem_type>::type
FASTA<BackwardStepType>::Optimize(FunctionType& function,
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

  // Sanity check: make sure lineSearchLookback is valid.
  if (lineSearchLookback == 0)
  {
    throw std::invalid_argument("FASTA::FASTA(): lineSearchLookback cannot be "
        "0!");
  }

  // Here we make a copy because we will use std::move() internally, and if
  // iterateIn is an alias, this is unsafe.  We will copy the final result back
  // to iterateIn at the end.
  BaseMatType x(iterateIn);

  // To keep track of the function value.
  ElemType currentFObj = f.Evaluate(x);
  ElemType currentGObj = backwardStep.Evaluate(x);
  ElemType currentObj = currentFObj + currentGObj;

  // This will be the denominator of the normalized residual termination
  // condition.
  ElemType firstResidual = ElemType(0);

  // This will be used in the non-monotone line search, to track the last
  // several function values.
  arma::Col<ElemType> lastFObjs(lineSearchLookback);
  lastFObjs.fill(std::numeric_limits<ElemType>::min());
  size_t currentObjPos = 0;

  BaseGradType g(x.n_rows, x.n_cols);
  BaseMatType lastXHat; // Used for residual checks.
  BaseMatType lastX; // Used for residual and alpha reset checks.
  BaseMatType xHat; // Used for residual checks.
  BaseMatType lpaX = x; // Used for alpha reset check.
  ElemType alpha = ElemType(1); // Initialize alpha^1 = 1.
  ElemType lastAlpha = alpha;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // First, estimate the Lipschitz constant to set the initial/maximum step
  // size, if the user asked us to.
  if (estimateStepSize)
    EstimateLipschitzStepSize(f, x);

  // Keep track of the last step size we used.
  ElemType currentStepSize = (ElemType) maxStepSize;
  ElemType lastStepSize = (ElemType) maxStepSize;

  Callback::BeginOptimization(*this, f, x, callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    // During this optimization, we want to optimize h(x) = f(x) + g(x).
    // f(x) is `f`, but g(x) is specified by `BackwardStepType`.

    // The first step is to compute a step size via a non-monotone line search.
    // To do this, we need to compute the gradient f'(y) as required by the line
    // search condition in Eq. (38).  Note that our code does a little sleight
    // of hand, and so `x` stores what the paper calls `y^k` here.  (See the
    // code for the adaptive step below.)
    currentFObj = f.EvaluateWithGradient(x, g);
    terminate |= Callback::EvaluateWithGradient(*this, f, x, currentFObj, g,
        callbacks...);

    // Use backtracking non-monotone line search to find the best step size.
    // This is the version from the FASTA paper, but with a minor modification:
    // we start our search at the last step size, and allow the search to
    // increase the step size up to the maximum step size if it can.  This is a
    // more effective heuristic than simply starting at the largest allowable
    // step size and shrinking from there, especially in regions where the
    // gradient norm is small.  It is also more effective than simply starting
    // at the last step size and shrinking from there, as it prevents getting
    // "stuck" with a very small step size.
    bool lsDone = false;
    size_t lsTrial = 0;
    bool increasing = false; // Will be set during the first iteration.
    ElemType lastFObj = ElemType(0);
    BaseMatType lsLastX; // Only used in increasing mode.
    BaseMatType lsLastXHat; // Only used in increasing mode.
    BaseMatType xDiff;

    lastX = std::move(x);
    lastStepSize = currentStepSize;
    currentStepSize = std::min(currentStepSize, (ElemType) maxStepSize);

    // Ensure that the last `lineSearchLookback` objective values are recorded
    // properly.
    lastFObjs[currentObjPos] = currentFObj;
    currentObjPos = (currentObjPos + 1) % lineSearchLookback;
    const ElemType strictMaxFObj = currentFObj;
    const ElemType maxFObj = lastFObjs.max();

    while (!lsDone && !terminate)
    {
      if (lsTrial == maxLineSearchSteps)
      {
        if (increasing)
        {
          Warn << "FASTA::Optimize(): line search reached maximum number of "
              << "steps (" << maxLineSearchSteps << "); using step size "
              << currentStepSize << "." << std::endl;
          break; // The step size is still valid.
        }
        else
        {
          Warn << "FASTA::Optimize(): could not find valid step size in range "
              << "(0, " << maxStepSize << "]!  Terminating optimization."
              << std::endl;
          terminate = true;
          break;
        }
      }

      // If the step size has converged to zero, we are done.
      if (currentStepSize == ElemType(0))
      {
        Warn << "FASTA::Optimize(): computed zero step size; terminating "
            << "optimization." << std::endl;
        terminate = true;
        break;
      }

      // Perform forward update into x.
      xHat = lastX - currentStepSize * g;
      // (We must store xHat separately for the residual, so this copy is
      // necessary.)
      x = xHat;
      backwardStep.ProximalStep(x, currentStepSize);

      // Compute objective of new point.
      const ElemType fObj = f.Evaluate(x);
      terminate |= Callback::Evaluate(*this, f, x, fObj, callbacks...);

      // Compute the quadratic approximation of the objective (the condition in
      // Eq. (38)).
      xDiff = (x - lastX);

      // Note: since we allow the step size to increase, we have to modify the
      // non-monotone line search a little bit to keep things from diverging.
      // Specifically, if we are increasing the step size, then we force a
      // monotone line search (by looking only at the previous function value).
      // It is only when we are decreasing the step size that we allow
      // relaxation.
      const ElemType relaxedCond = maxFObj + dot(xDiff, g) +
          (1 / (2 * currentStepSize)) * dot(xDiff, xDiff);
      const ElemType strictCond = strictMaxFObj + dot(xDiff, g) +
          (1 / (2 * currentStepSize)) * dot(xDiff, xDiff);

      // If we're on the first iteration, we don't know if we should be
      // searching for a step size by increasing or decreasing the step size.
      // (Remember that our valid ranges of step sizes are [0, maxStepSize], and
      // we are starting at lastStepSize.)
      //
      // Thus, if the condition is satisfied, let's try increasing the step size
      // until it's no longer satisfied.  Otherwise, we will have to decrease
      // the step size.
      if (lsTrial == 0)
      {
        increasing = ((fObj <= strictCond) && (std::isfinite(fObj)));
      }

      if (increasing)
      {
        // If we are in "increasing" mode, then termination occurs on the first
        // iteration when the strict condition is *not* satisfied (and we use
        // the last step size).
        if ((fObj > strictCond) || (!std::isfinite(fObj)))
        {
          lsDone = true;
          x = std::move(lsLastX);
          xHat = std::move(lsLastXHat);
          currentFObj = lastFObj;
          currentStepSize = lastStepSize; // Take one step backwards.
        }
        else if (currentStepSize == (ElemType) maxStepSize)
        {
          // The condition is still satisfied, but the step size will be too big
          // if we take another step.  Go back to the maximum step size.
          lsDone = true;
          currentFObj = fObj;
        }
        else
        {
          // The condition is still satisfied; increase the step size.
          lastStepSize = currentStepSize;
          currentStepSize *= ElemType(stepSizeAdjustment);
          lsLastX = std::move(x);
          lsLastXHat = std::move(xHat);
          lastFObj = fObj;
          ++lsTrial;
        }
      }
      else
      {
        // If we are in "decreasing" mode, then termination occurs on the first
        // iteration when the relaxed condition is satisfied.
        if ((fObj <= relaxedCond) && (std::isfinite(fObj)))
        {
          lsDone = true;
          currentFObj = fObj;
        }
        else
        {
          // The condition is not yet satisfied; decrease the step size.
          currentStepSize /= ElemType(stepSizeAdjustment);
          ++lsTrial;
        }
      }
    }

    if (!lsDone)
    {
      // The line search failed, so terminate.
      Warn << "FASTA::Optimize(): non-monotone line search failed after "
          << maxLineSearchSteps << " steps; terminating optimization."
          << std::endl;
      x = std::move(lastX);
      terminate = true;
    }

    // If we terminated during the line search, we are done.
    if (terminate)
      break;

    // Now that we have taken a step, compute the full objective by computing
    // g(x).
    currentGObj = backwardStep.Evaluate(x);
    currentObj = currentFObj + currentGObj;

    // Output current objective function.
    Info << "FASTA::Optimize(): iteration " << i << ", combined objective "
        << currentObj << " (f(x) = " << currentFObj << ", g(x) = "
        << currentGObj << "), step size " << currentStepSize << "."
        << std::endl;

    // Sanity check for divergence.
    if ((i > 1) && !std::isfinite(currentObj))
    {
      Warn << "FASTA::Optimize(): objective diverged to "
          << currentObj << "; terminating optimization." << std::endl;
      terminate = true;
      break;
    }

    // Now, check for convergence.  The FASTA convergence check uses both the
    // normalized residual and the relative residual, stopping when either
    // becomes sufficiently small.  The check depends on x before and after the
    // proximal step.

    // Compute residual.  This is Eq. (40) in the paper.
    const ElemType residual = norm(g + (xHat - x) / currentStepSize, 2);

    // If this is the first iteration, store the residual as the first residual.
    if (i == 1)
      firstResidual = residual;

    // First, check the normalized residual for convergence.  This is Eq. (43)
    // in the paper.
    const ElemType eps = 20 * std::numeric_limits<ElemType>::epsilon();
    const ElemType normalizedResidual = residual / (firstResidual + eps);

    if ((i < 10) && (normalizedResidual < ElemType(1e-5)))
    {
      // Heuristic: sometimes the optimization starts in such an awful place
      // that we are able to make huge amounts of progress in the first few
      // iterations.  In this case, reset the firstResidual to the slightly
      // better point we get to by the tenth iterate.
      firstResidual = residual;
    }
    else if ((i > 10) && (normalizedResidual < tolerance))
    {
      Info << "FASTA::Optimize(): normalized residual minimized within "
          << "tolerance " << tolerance << "; terminating optimization."
          << std::endl;
      break;
    }

    // Next, check the relative residual for convergence.  This is Eq. (42) in
    // the paper.
    const ElemType gNorm = norm(g, 2);
    const ElemType proxStepNorm = norm((xHat - x) / currentStepSize, 2);

    const ElemType relativeResidual = residual /
        (std::max(gNorm, proxStepNorm) + 20 * eps);

    if (relativeResidual < tolerance)
    {
      Info << "FASTA::Optimize(): relative residual minimized within "
          << "tolerance " << tolerance << "; terminating optimization."
          << std::endl;
      break;
    }

    // Compute updated prediction parameter alpha.
    lastAlpha = alpha;
    alpha = (1 + std::sqrt(1 + 4 * std::pow(alpha, ElemType(2)))) / 2;

    // Take a predictive step.
    BaseMatType y = x + ((lastAlpha - 1) / alpha) * (x - lpaX);

    // Sometimes alpha can get to be too large; this restart scheme is taken
    // originally from O'Donoghue and Candes, "Adaptive restart for accelerated
    // gradient schemes", 2012.
    //
    // The notation is confusing here when compared with Eq. (37) in the paper.
    // This is because the paper is poorly notated, although it's not clear much
    // has been done here to improve things.  To translate:
    //
    //    Paper   Code    Explanation
    //
    //    y^k     lastX   This is the result of the predictive step on the
    //                    previous iteration.  In our code, we apply the
    //                    predictive step to x, which next iteration becomes
    //                    lastX.
    //
    //    x^k     x       This is the iterate before the predictive step, this
    //                    iteration.
    //
    //    x^k-1   lpaX    "Last Pre-Accelerated X"---we have to take a specific
    //                    step to store this.
    //
    const ElemType restartCheck = dot(lastX - x, x - lpaX);
    if (restartCheck > 0)
    {
      Info << "FASTA::Optimize(): alpha too large (" << alpha << "); reset to "
          << "1." << std::endl;
      alpha = ElemType(1);
      lastAlpha = ElemType(1);
    }

    lpaX = std::move(x);
    x = std::move(y);

    terminate |= Callback::StepTaken(*this, f, x, callbacks...);
  }

  if (!terminate)
  {
    Info << "FASTA::Optimize(): maximum iterations (" << maxIterations
        << ") reached; terminating optimization." << std::endl;
  }

  Callback::EndOptimization(*this, f, x, callbacks...);

  ((BaseMatType&) iterateIn) = x;
  return currentObj;
} // Optimize()

template<typename BackwardStepType>
template<typename MatType>
void FASTA<BackwardStepType>::RandomFill(
    MatType& x,
    const size_t rows,
    const size_t cols,
    const typename MatType::elem_type maxVal)
{
  x.randu(rows, cols);
  x *= maxVal;
}

template<typename BackwardStepType>
template<typename eT>
void FASTA<BackwardStepType>::RandomFill(
    arma::SpMat<eT>& x,
    const size_t rows,
    const size_t cols,
    const eT maxVal)
{
  eT density = eT(0.1);
  // Try and keep the matrix from having too many elements.
  if (rows * cols > 100000)
    density = eT(0.01);
  else if (rows * cols > 1000000)
    density = eT(0.001);
  else if (rows * cols > 10000000)
    density = eT(0.0001);

  x.sprandu(rows, cols, density);

  // Make sure we got at least some nonzero elements...
  while (x.n_nonzero == 0)
  {
    if (x.n_elem < 10)
      x.sprandu(rows, cols, 1.0);
    else
      x.sprandu(rows, cols, 0.5);
  }

  x *= maxVal;
}

template<typename BackwardStepType>
template<typename FunctionType, typename MatType>
void FASTA<BackwardStepType>::EstimateLipschitzStepSize(
    FunctionType& f,
    const MatType& x)
{
  typedef typename MatType::elem_type ElemType;

  // Sanity check for estimateSteps parameter.
  if (estimateTrials == 0)
  {
    throw std::invalid_argument("FASTA::Optimize(): estimateTrials must be "
        "greater than 0!");
  }

  const ElemType xMax = std::max(ElemType(1), 2 * x.max());
  ElemType sum = ElemType(0);
  MatType x1, x2, gx1, gx2;

  for (size_t t = 0; t < estimateTrials; ++t)
  {
    RandomFill(x1, x.n_rows, x.n_cols, xMax);
    RandomFill(x2, x.n_rows, x.n_cols, xMax);

    f.Gradient(x1, gx1);
    f.Gradient(x2, gx2);

    // Compute a Lipschitz constant estimate.
    const ElemType lEst = norm(gx1 - gx2, 2) / norm(x1 - x2, 2);
    sum += lEst;
  }

  sum /= estimateTrials;
  if (sum == 0)
    maxStepSize = std::numeric_limits<ElemType>::max();
  else
    maxStepSize = (10 / sum);

  Info << "FASTA::Optimize(): estimated a maximum step size of "
      << maxStepSize << "." << std::endl;
}

} // namespace ens

#endif
