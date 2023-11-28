/**
 * @file fista_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_FISTA_FISTA_IMPL_HPP
#define ENSMALLEN_FISTA_FISTA_IMPL_HPP

// In case it hasn't been included yet.
#include "fista.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

//! Constructor of the FBS class.
template<typename BackwardStepType>
FISTA<BackwardStepType>::FISTA(const size_t maxIterations,
                               const double tolerance,
                               const size_t maxLineSearchSteps,
                               const double stepSizeAdjustment,
                               const bool estimateStepSize,
                               const size_t estimateTrials,
                               const double maxStepSize) :
    maxIterations(maxIterations),
    tolerance(tolerance),
    maxLineSearchSteps(maxLineSearchSteps),
    stepSizeAdjustment(stepSizeAdjustment),
    estimateStepSize(estimateStepSize),
    estimateTrials(estimateTrials),
    maxStepSize(maxStepSize)
{
  // Check estimateSteps parameter.
  if (estimateStepSize && estimateTrials == 0)
  {
    throw std::invalid_argument("FISTA::FISTA(): estimateTrials must be greater"
        " than 0!");
  }
}

template<typename BackwardStepType>
FISTA<BackwardStepType>::FISTA(BackwardStepType backwardStep,
                               const size_t maxIterations,
                               const double tolerance,
                               const size_t maxLineSearchSteps,
                               const double stepSizeAdjustment,
                               const bool estimateStepSize,
                               const size_t estimateTrials,
                               const double maxStepSize) :
    backwardStep(std::move(backwardStep)),
    maxIterations(maxIterations),
    tolerance(tolerance),
    maxLineSearchSteps(maxLineSearchSteps),
    stepSizeAdjustment(stepSizeAdjustment),
    estimateStepSize(estimateStepSize),
    estimateTrials(estimateTrials),
    maxStepSize(maxStepSize)
{
  // Check estimateSteps parameter.
  if (estimateStepSize && estimateTrials == 0)
  {
    throw std::invalid_argument("FISTA::FISTA(): estimateTrials must be greater"
        " than 0!");
  }
}

//! Optimize the function (minimize).
template<typename BackwardStepType>
template<typename FunctionType, typename MatType, typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
    typename MatType::elem_type>::type
FISTA<BackwardStepType>::Optimize(FunctionType& function,
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

  // To keep track of the function value.  This is ignored during the iteration
  // if computeIterationLoss is false and ENS_PRINT_INFO is not defined.  (It
  // will be computed at the end, though.)
  ElemType lastObjective = std::numeric_limits<ElemType>::max();;
  ElemType currentFObjective = f.Evaluate(iterate);
  ElemType currentGObjective = backwardStep.Evaluate(iterate);
  ElemType currentObjective = currentFObjective + currentGObjective;

  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  BaseMatType y = iterate; // Initialize y^1 = x^0.
  BaseMatType lastIterate = iterate;
  ElemType alpha = 1; // Initialize alpha^1 = 1.
  ElemType lastAlpha = alpha;

  // Controls early termination of the optimization process.
  bool terminate = false;

  // First, estimate the Lipschitz constant to set the initial/maximum step
  // size, if the user asked us to.
  if (estimateStepSize)
  {
    // Sanity check for estimateSteps parameter.
    if (estimateTrials == 0)
    {
      throw std::invalid_argument("FISTA::FISTA(): estimateTrials must be "
          "greater than 0!");
    }

    const ElemType iterateMax = std::max(1.0, 2.0 * iterate.max());
    ElemType sum = 0.0;
    BaseMatType x1, x2, gx1, gx2;

    for (size_t t = 0; t < estimateTrials; ++t)
    {
      RandomFill(x1, iterate.n_rows, iterate.n_cols, iterateMax);
      RandomFill(x2, iterate.n_rows, iterate.n_cols, iterateMax);

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
      maxStepSize = (10.0 / sum);

    Info << "FISTA::Optimize(): estimated a maximum step size of "
        << maxStepSize << "." << std::endl;
  }

  // Keep track of the last step size we used.
  ElemType currentStepSize = (ElemType) maxStepSize;
  ElemType lastStepSize = (ElemType) maxStepSize;

  terminate |= Callback::BeginOptimization(*this, f, iterate, callbacks...);
  for (size_t i = 1; i != maxIterations && !terminate; ++i)
  {
    // During this optimization, we want to optimize h(x) = f(x) + g(x).
    // f(x) is `f`, but g(x) is specified by `BackwardStepType`.

    // Output current objective function.
    Info << "FISTA::Optimize(): iteration " << i << ", combined objective "
        << currentObjective << " (f(x) = " << currentFObjective
        << ", g(x) = " << currentGObjective << "), step size "
        << currentStepSize << "." << std::endl;

    if ((i > 1) && !std::isfinite(currentObjective))
    {
      Warn << "FISTA::Optimize(): objective diverged to "
          << currentObjective << "; terminating optimization." << std::endl;
      terminate = true;
      break;
    }

    // Compute the gradient on y.  Note that FISTA uses f'(y) to take a step,
    // not f'(x).  We will also need f(y) for the line search, so compute it
    // now.
    const ElemType yObj = f.EvaluateWithGradient(y, gradient);
    terminate |= Callback::EvaluateWithGradient(*this, f, y, yObj, gradient,
        callbacks...);

    // Check for convergence.  This is a simple check on the objective.
    if ((i > 1) && (std::abs(currentObjective - lastObjective) < tolerance))
    {
      Info << "FISTA::Optimize(): minimized within objective tolerance "
          << tolerance << "; terminating optimization." << std::endl;
      terminate = true;
    }

    // Terminate before the line search, if we need to.
    if (terminate)
      break;

    // Use backtracking line search to find the best step size.  This is not the
    // version from the FASTA paper (non-monotone line search) but instead the
    // version proposed by Beck and Teboulle, with a minor modification: we
    // start our search at the last step size, and allow the search to increase
    // the step size up to the maximum step size if it can.  This is a more
    // effective heuristic than simply starting at the largest allowable step
    // size and shrinking from there, especially in regions where the gradient
    // norm is small.  It is also more effective than simply starting at the
    // last step size and shrinking from there, as it prevents getting "stuck"
    // with a very small step size.
    bool lineSearchSatisfied = false;
    size_t lineSearchTrial = 0;
    bool increasing = false; // Will be set during the first iteration.
    ElemType lastFObj = 0.0;
    ElemType lastGObj = 0.0;
    BaseMatType lastLineSearchIterate; // Only used in increasing mode.

    lastIterate = iterate;
    lastStepSize = currentStepSize;
    currentStepSize = std::min(currentStepSize, (ElemType) maxStepSize);

    while (!lineSearchSatisfied && !terminate)
    {
      if (lineSearchTrial == maxLineSearchSteps)
      {
        if (increasing)
        {
          Warn << "FISTA::Optimize(): line search reached maximum number of "
              << "steps (" << maxLineSearchSteps << "); using step size "
              << currentStepSize << "." << std::endl;
          break; // The step size is still valid.
        }
        else
        {
          Warn << "FISTA::Optimize(): could not find valid step size in range "
              << "(0, " << maxStepSize << "]!  Terminating optimization."
              << std::endl;
          iterate = lastIterate; // Revert to previous coordinates.
          terminate = true;
          break;
        }
      }

      // If the step size has converged to zero, we are done.
      if (currentStepSize == 0.0)
      {
        Warn << "FISTA::Optimize(): computed zero step size; terminating "
            << "optimization." << std::endl;
        iterate = lastIterate; // Revert to previous coordinates.
        terminate = true;
        break;
      }

      // Perform forward update into x.
      iterate = y - currentStepSize * gradient;
      backwardStep.ProximalStep(iterate, currentStepSize);

      // Compute line search improvement condition.
      // Q_s(x1, x2) = f(x2) + < x1 - x2, f'(x2) > + (1 / 2s) || x1 - x2 ||^2 +
      //     g(x1)
      // Here, x1 is the proposal, and x1 is the last y (prediction).
      const ElemType fObj = f.Evaluate(iterate);
      const ElemType gObj = backwardStep.Evaluate(iterate);
      terminate |= Callback::Evaluate(*this, f, iterate, fObj, callbacks...);

      const ElemType q = yObj + dot(iterate - y, gradient) +
          (1.0 / (2.0 * currentStepSize)) * dot(iterate - y, iterate - y) +
          gObj;

      // If we're on the first iteration, we don't know if we should be
      // searching for a step size by increasing or decreasing the step size.
      // (Remember that our valid ranges of step sizes are [0, maxStepSize], and
      // we are starting at lastStepSize.)
      //
      // Thus, if the condition is satisfied, let's try increasing the step size
      // until it's no longer satisfied.  Otherwise, we will have to decrease
      // the step size.
      if (lineSearchTrial == 0)
      {
        increasing = (fObj <= q);
      }

      if (increasing)
      {
        // If we are in "increasing" mode, then termination occurs on the first
        // iteration when the condition is *not* satisfied (and we use the last
        // step size).
        if (fObj > q)
        {
          lineSearchSatisfied = true;
          iterate = lastLineSearchIterate;
          currentFObjective = lastFObj;
          currentGObjective = lastGObj;
          lastObjective = currentObjective;
          currentObjective = currentFObjective + currentGObjective;
          currentStepSize = lastStepSize; // Take one step backwards.
        }
        else if (currentStepSize == (ElemType) maxStepSize)
        {
          // The condition is still satisfied, but the step size will be too big
          // if we take another step.  Go back to the maximum step size.
          lineSearchSatisfied = true;
          currentFObjective = fObj;
          currentGObjective = gObj;
          lastObjective = currentObjective;
          currentObjective = currentFObjective + currentGObjective;
        }
        else
        {
          // The condition is still satisfied; increase the step size.
          lastStepSize = currentStepSize;
          currentStepSize *= stepSizeAdjustment;
          lastLineSearchIterate = iterate;
          lastFObj = fObj;
          lastGObj = gObj;
          ++lineSearchTrial;
        }
      }
      else
      {
        // If we are in "decreasing" mode, then termination occurs on the first
        // iteration when the condition is satisfied.
        if (fObj <= q)
        {
          lineSearchSatisfied = true;
          currentFObjective = fObj;
          currentGObjective = gObj;
          lastObjective = currentObjective;
          currentObjective = currentFObjective + currentGObjective;
        }
        else
        {
          // The condition is not yet satisfied; decrease the step size.
          currentStepSize /= stepSizeAdjustment;
          ++lineSearchTrial;
        }
      }
    }

    if (!lineSearchSatisfied)
    {
      // The line search failed, so terminate.
      Warn << "FISTA::Optimize(): line search failed after "
          << maxLineSearchSteps << " steps; terminating optimization."
          << std::endl;
      iterate = lastIterate;
      terminate = true;
    }

    // If we terminated during the line search, we are done.
    if (terminate)
      break;

    // Compute updated prediction parameter alpha.
    lastAlpha = alpha;
    alpha = (1.0 + std::sqrt(1 + 4 * std::pow(alpha, 2.0))) / 2.0;

    // Sometimes alpha can get to be too large; this restart scheme is taken
    // originally from O'Donoghue and Candes, "Adaptive restart for accelerated
    // gradient schemes", 2012.
    const ElemType restartCheck = dot(y - iterate, iterate - lastIterate);
    if (restartCheck > 0)
    {
      Info << "FISTA::Optimize(): alpha too large (" << alpha << "); reset to "
          << "1." << std::endl;
      alpha = 1;
      lastAlpha = 1;
    }

    // Update prediction.
    y = iterate + ((lastAlpha - 1.0) / alpha) * (iterate - lastIterate);

    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);
  }

  if (!terminate)
  {
    Info << "FISTA::Optimize(): maximum iterations (" << maxIterations
        << ") reached; terminating optimization." << std::endl;
  }

  Callback::EndOptimization(*this, f, iterate, callbacks...);

  return currentObjective;
} // Optimize()

template<typename BackwardStepType>
template<typename MatType>
void FISTA<BackwardStepType>::RandomFill(
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
void FISTA<BackwardStepType>::RandomFill(
    arma::SpMat<eT>& x,
    const size_t rows,
    const size_t cols,
    const eT maxVal)
{
  eT density = 0.1;
  // Try and keep the matrix from having too many elements.
  if (rows * cols > 100000)
    density = 0.01;
  else if (rows * cols > 1000000)
    density = 0.001;
  else if (rows * cols > 10000000)
    density = 0.0001;

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

} // namespace ens

#endif
