/**
 * @file lbfgs_impl.hpp
 * @author Dongryeol Lee (dongryel@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * The implementation of the L_BFGS optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_LBFGS_LBFGS_IMPL_HPP
#define ENSMALLEN_LBFGS_LBFGS_IMPL_HPP

// In case it hasn't been included yet.
#include "lbfgs.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

/**
 * Initialize the L_BFGS object.
 *
 * @param numBasis Number of memory points to be stored (default 5).
 * @param maxIterations Maximum number of iterations for the optimization
 *     (0 means no limit and may run indefinitely).
 * @param armijoConstant Controls the accuracy of the line search routine for
 *     determining the Armijo condition.
 * @param wolfe Parameter for detecting the Wolfe condition.
 * @param minGradientNorm Minimum gradient norm required to continue the
 *     optimization.
 * @param factr Minimum relative function value decrease to continue
 *     the optimization.
 * @param maxLineSearchTrials The maximum number of trials for the line search
 *     (before giving up).
 * @param minStep The minimum step of the line search.
 * @param maxStep The maximum step of the line search.
 */
inline L_BFGS::L_BFGS(const size_t numBasis,
                      const size_t maxIterations,
                      const double armijoConstant,
                      const double wolfe,
                      const double minGradientNorm,
                      const double factr,
                      const size_t maxLineSearchTrials,
                      const double minStep,
                      const double maxStep) :
    numBasis(numBasis),
    maxIterations(maxIterations),
    armijoConstant(armijoConstant),
    wolfe(wolfe),
    minGradientNorm(minGradientNorm),
    factr(factr),
    maxLineSearchTrials(maxLineSearchTrials),
    minStep(minStep),
    maxStep(maxStep),
    terminate(false)
{
  // Nothing to do.
}

/**
 * Calculate the scaling factor, gamma, which is used to scale the Hessian
 * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal
 * (1989).
 *
 * @return The calculated scaling factor.
 * @param gradient The gradient at the initial point.
 * @param s Differences between the iterate and old iterate matrix.
 * @param y Differences between the gradient and the old gradient matrix.
 */
template<typename MatType, typename CubeType>
double L_BFGS::ChooseScalingFactor(const size_t iterationNum,
                                   const MatType& gradient,
                                   const CubeType& s,
                                   const CubeType& y)
{
  typedef typename CubeType::elem_type CubeElemType;

  constexpr const CubeElemType tol =
      100 * std::numeric_limits<CubeElemType>::epsilon();

  double scalingFactor;
  if (iterationNum > 0)
  {
    int previousPos = (iterationNum - 1) % numBasis;
    // Get s and y matrices once instead of multiple times.
    const arma::Mat<CubeElemType>& sMat = s.slice(previousPos);
    const arma::Mat<CubeElemType>& yMat = y.slice(previousPos);

    const CubeElemType tmp   = arma::dot(yMat, yMat);
    const CubeElemType denom = (tmp >= tol) ? tmp : CubeElemType(1);

    scalingFactor = arma::dot(sMat, yMat) / denom;
  }
  else
  {
    const CubeElemType tmp = arma::norm(gradient, "fro");

    scalingFactor = (tmp >= tol) ? (1.0 / tmp) : 1.0;
  }

  return scalingFactor;
}

/**
 * Find the L_BFGS search direction.
 *
 * @param gradient The gradient at the current point.
 * @param iterationNum The iteration number.
 * @param scalingFactor Scaling factor to use (see ChooseScalingFactor_()).
 * @param s Differences between the iterate and old iterate matrix.
 * @param y Differences between the gradient and the old gradient matrix.
 * @param searchDirection Vector to store search direction in.
 */
template<typename MatType, typename CubeType>
void L_BFGS::SearchDirection(const MatType& gradient,
                             const size_t iterationNum,
                             const double scalingFactor,
                             const CubeType& s,
                             const CubeType& y,
                             MatType& searchDirection)
{
  // Start from this point.
  searchDirection = gradient;

  // See "A Recursive Formula to Compute H * g" in "Updating quasi-Newton
  // matrices with limited storage" (Nocedal, 1980).
  typedef typename CubeType::elem_type CubeElemType;

  // Temporary variables.
  arma::Col<CubeElemType> rho(numBasis);
  arma::Col<CubeElemType> alpha(numBasis);

  size_t limit = (numBasis > iterationNum) ? 0 : (iterationNum - numBasis);
  for (size_t i = iterationNum; i != limit; i--)
  {
    int translatedPosition = (i + (numBasis - 1)) % numBasis;

    const arma::Mat<CubeElemType>& sMat = s.slice(translatedPosition);
    const arma::Mat<CubeElemType>& yMat = y.slice(translatedPosition);

    const CubeElemType tmp = arma::dot(yMat, sMat);

    rho[iterationNum - i] = (tmp != CubeElemType(0)) ? (1.0 / tmp) :
        CubeElemType(1);

    alpha[iterationNum - i] = rho[iterationNum - i] *
        arma::dot(sMat, searchDirection);

    searchDirection -= alpha[iterationNum - i] * yMat;
  }

  searchDirection *= scalingFactor;

  for (size_t i = limit; i < iterationNum; i++)
  {
    int translatedPosition = i % numBasis;
    double beta = rho[iterationNum - i - 1] *
        arma::dot(y.slice(translatedPosition), searchDirection);
    searchDirection += (alpha[iterationNum - i - 1] - beta) *
        s.slice(translatedPosition);
  }

  // Negate the search direction so that it is a descent direction.
  searchDirection *= -1;
}

/**
 * Update the y and s matrices, which store the differences between
 * the iterate and old iterate and the differences between the gradient and the
 * old gradient, respectively.
 *
 * @param iterationNum Iteration number.
 * @param iterate Current point.
 * @param oldIterate Point at last iteration.
 * @param gradient Gradient at current point (iterate).
 * @param oldGradient Gradient at last iteration point (oldIterate).
 * @param s Differences between the iterate and old iterate matrix.
 * @param y Differences between the gradient and the old gradient matrix.
 */
template<typename MatType, typename GradType, typename CubeType>
void L_BFGS::UpdateBasisSet(const size_t iterationNum,
                            const MatType& iterate,
                            const MatType& oldIterate,
                            const GradType& gradient,
                            const GradType& oldGradient,
                            CubeType& s,
                            CubeType& y)
{
  // Overwrite a certain position instead of pushing everything in the vector
  // back one position.
  int overwritePos = iterationNum % numBasis;
  s.slice(overwritePos) = iterate - oldIterate;
  y.slice(overwritePos) = gradient - oldGradient;
}

/**
 * Perform a back-tracking line search along the search direction to calculate a
 * step size satisfying the Wolfe conditions.
 *
 * @param function Function to optimize.
 * @param functionValue Value of the function at the initial point.
 * @param iterate The initial point to begin the line search from.
 * @param gradient The gradient at the initial point.
 * @param searchDirection A vector specifying the search direction.
 * @param finalStepSize The resulting step size used.
 * @param callbacks Callback functions.
 *
 * @return false if no step size is suitable, true otherwise.
 */
template<typename FunctionType,
         typename ElemType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
bool L_BFGS::LineSearch(FunctionType& function,
                        ElemType& functionValue,
                        MatType& iterate,
                        GradType& gradient,
                        MatType& newIterateTmp,
                        const GradType& searchDirection,
                        double& finalStepSize,
                        CallbackTypes&... callbacks)
{
  // Default first step size of 1.0.
  double stepSize = 1.0;
  finalStepSize = 0.0; // Set only when we take the step.

  // The initial linear term approximation in the direction of the
  // search direction.
  ElemType initialSearchDirectionDotGradient =
      arma::dot(gradient, searchDirection);

  // If it is not a descent direction, just report failure.
  if ( (initialSearchDirectionDotGradient > 0.0)
    || (std::isfinite(initialSearchDirectionDotGradient) == false) )
  {
    Warn << "L-BFGS line search direction is not a descent direction "
        << "(terminating)!" << std::endl;
    return false;
  }

  // Save the initial function value.
  ElemType initialFunctionValue = functionValue;

  // Unit linear approximation to the decrease in function value.
  ElemType linearApproxFunctionValueDecrease = armijoConstant *
      initialSearchDirectionDotGradient;

  // The number of iteration in the search.
  size_t numIterations = 0;

  // Armijo step size scaling factor for increase and decrease.
  const double inc = 2.1;
  const double dec = 0.5;
  double width = 0;
  double bestStepSize = 1.0;
  ElemType bestObjective = std::numeric_limits<ElemType>::max();

  while (true)
  {
    // Perform a step and evaluate the gradient and the function values at that
    // point.
    newIterateTmp = iterate;
    newIterateTmp += stepSize * searchDirection;
    functionValue = function.EvaluateWithGradient(newIterateTmp, gradient);

    if (std::isnan(functionValue))
    {
      Warn << "L-BFGS objective value is NaN (terminating)!" << std::endl;
      return false;
    }

    terminate |= Callback::EvaluateWithGradient(*this, function, newIterateTmp,
        functionValue, gradient, callbacks...);

    if (functionValue < bestObjective)
    {
      bestStepSize = stepSize;
      bestObjective = functionValue;
    }
    numIterations++;

    if (functionValue > initialFunctionValue + stepSize *
        linearApproxFunctionValueDecrease)
    {
      width = dec;
    }
    else
    {
      // Check Wolfe's condition.
      ElemType searchDirectionDotGradient = arma::dot(gradient,
          searchDirection);

      if (searchDirectionDotGradient < wolfe *
          initialSearchDirectionDotGradient)
      {
        width = inc;
      }
      else
      {
        if (searchDirectionDotGradient > -wolfe *
            initialSearchDirectionDotGradient)
        {
          width = dec;
        }
        else
        {
          break;
        }
      }
    }

    // Terminate when the step size gets too small or too big or it
    // exceeds the max number of iterations.
    const bool cond1 = (stepSize < minStep);
    const bool cond2 = (stepSize > maxStep);
    const bool cond3 = (numIterations >= maxLineSearchTrials);
    if (cond1 || cond2 || cond3)
      break;

    // Scale the step size.
    stepSize *= width;
  }

  // Move to the new iterate.
  iterate += bestStepSize * searchDirection;
  finalStepSize = bestStepSize;
  return true;
}

/**
 * Use L_BFGS to optimize the given function, starting at the given iterate
 * point and performing no more than the specified number of maximum iterations.
 * The given starting point will be modified to store the finishing point of the
 * algorithm.
 *
 * @param numIterations Maximum number of iterations to perform
 * @param iterate Starting point (will be modified)
 * @param callbacks Callback functions.
 */
template<typename FunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsArmaType<GradType>::value,
typename MatType::elem_type>::type
L_BFGS::Optimize(FunctionType& function,
                 MatType& iterateIn,
                 CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;

  // Use the Function<> wrapper to ensure the function has all of the functions
  // that we need.
  typedef Function<FunctionType, BaseMatType, BaseGradType> FullFunctionType;
  FullFunctionType& f = static_cast<FullFunctionType&>(function);

  // Check that we have all the functions we will need.
  traits::CheckFunctionTypeAPI<FullFunctionType, BaseMatType, BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Ensure that the cubes holding past iterations' information are the right
  // size.  Also set the current best point value to the maximum.
  const size_t rows = iterate.n_rows;
  const size_t cols = iterate.n_cols;

  BaseMatType newIterateTmp(rows, cols);
  arma::Cube<ElemType> s(rows, cols, numBasis);
  arma::Cube<ElemType> y(rows, cols, numBasis);

  // The old iterate to be saved.
  BaseMatType oldIterate(iterate.n_rows, iterate.n_cols);
  oldIterate.zeros();

  // Whether to optimize until convergence.
  bool optimizeUntilConvergence = (maxIterations == 0);

  // The gradient: the current and the old.
  BaseGradType gradient(iterate.n_rows, iterate.n_cols);
  gradient.zeros();
  BaseGradType oldGradient(iterate.n_rows, iterate.n_cols);
  oldGradient.zeros();

  // The search direction.
  BaseGradType searchDirection(iterate.n_rows, iterate.n_cols);
  searchDirection.zeros();

  // The initial function value and gradient.
  ElemType functionValue = f.EvaluateWithGradient(iterate, gradient);

  terminate |= Callback::EvaluateWithGradient(*this, f, iterate,
        functionValue, gradient, callbacks...);

  ElemType prevFunctionValue;

  // The main optimization loop.
  Callback::BeginOptimization(*this, f, iterate, callbacks...);
  for (size_t itNum = 0; (optimizeUntilConvergence || (itNum != maxIterations))
      && !terminate; ++itNum)
  {
    prevFunctionValue = functionValue;

    // Break when the norm of the gradient becomes too small.
    //
    // But don't do this on the first iteration to ensure we always take at
    // least one descent step.
    // TODO: to speed this up, investigate use of arma::norm2est() in Armadillo
    // 12.4
    if (arma::norm(gradient, 2) < minGradientNorm)
    {
      Info << "L-BFGS gradient norm too small (terminating successfully)."
          << std::endl;
      break;
    }

    // Break if the objective is not a number.
    if (std::isnan(functionValue))
    {
      Warn << "L-BFGS terminated with objective " << functionValue << "; "
          << "are the objective and gradient functions implemented correctly?"
          << std::endl;
      break;
    }

    // Choose the scaling factor.
    double scalingFactor = ChooseScalingFactor(itNum, gradient, s, y);
    if (scalingFactor == 0.0)
    {
      Info << "L-BFGS scaling factor computed as 0 (terminating successfully)."
          << std::endl;
      break;
    }

    if (std::isfinite(scalingFactor) == false)
      {
      Warn << "L-BFGS scaling factor is not finite.  Stopping optimization."
           << std::endl;
      break;
      }

    // Build an approximation to the Hessian and choose the search
    // direction for the current iteration.
    SearchDirection(gradient, itNum, scalingFactor, s, y, searchDirection);

    // Save the old iterate and the gradient before stepping.
    oldIterate = iterate;
    oldGradient = gradient;

    double stepSize; // Set by LineSearch().
    if (!LineSearch(f, functionValue, iterate, gradient, newIterateTmp,
        searchDirection, stepSize, callbacks...))
    {
      Warn << "Line search failed.  Stopping optimization." << std::endl;
      break; // The line search failed; nothing else to try.
    }

    // It is possible that the difference between the two coordinates is zero.
    // In this case we terminate successfully.
    if (stepSize == 0.0)
    {
      Info << "L-BFGS step size of 0 (terminating successfully)."
          << std::endl;
      break;
    }

    // If we can't make progress on the gradient, then we'll also accept
    // a stable function value.
    const double denom = std::max(
        std::max(std::abs(prevFunctionValue), std::abs(functionValue)),
        (ElemType) 1.0);
    if ((prevFunctionValue - functionValue) / denom <= factr)
    {
      Info << "L-BFGS function value stable (terminating successfully)."
          << std::endl;
      break;
    }

    // Overwrite an old basis set.
    UpdateBasisSet(itNum, iterate, oldIterate, gradient, oldGradient, s, y);

    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);
  } // End of the optimization loop.

  Callback::EndOptimization(*this, f, iterate, callbacks...);
  return functionValue;
}

} // namespace ens

#endif // ENSMALLEN_LBFGS_LBFGS_IMPL_HPP

