/**
 * @file lbfgsb_impl.hpp
 * @author Khizir Siddiqui
 *
 * The implementation of the L_BFGS_B optimizer.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_LBFGSB_LBFGSB_IMPL_HPP
#define ENSMALLEN_LBFGSB_LBFGSB_IMPL_HPP

// In case it hasn't been included yet.
#include "lbfgsb.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

inline L_BFGS_B::L_BFGS_B(const size_t numBasis,
                          const arma::mat& lowerBound,
                          const arma::mat& upperBound,
                          const size_t maxIterations,
                          const double armijoConstant,
                          const double wolfe,
                          const double minGradientNorm,
                          const double factr,
                          const size_t maxLineSearchTrials,
                          const double minStep,
                          const double maxStep) :
    numBasis(numBasis),
    lowerBound(lowerBound),
    upperBound(upperBound),
    maxIterations(maxIterations),
    armijoConstant(armijoConstant),
    wolfe(wolfe),
    minGradientNorm(minGradientNorm),
    factr(factr),
    maxLineSearchTrials(maxLineSearchTrials),
    minStep(minStep),
    maxStep(maxStep),
    terminate(false),
    usingScalarBounds(false)
{
  // Nothing to do.
}

inline L_BFGS_B::L_BFGS_B(const size_t numBasis,
                          const double lowerBound,
                          const double upperBound,
                          const size_t maxIterations,
                          const double armijoConstant,
                          const double wolfe,
                          const double minGradientNorm,
                          const double factr,
                          const size_t maxLineSearchTrials,
                          const double minStep,
                          const double maxStep) :
    numBasis(numBasis),
    lowerBound(arma::vec(1).fill(lowerBound)),
    upperBound(arma::vec(1).fill(upperBound)),
    maxIterations(maxIterations),
    armijoConstant(armijoConstant),
    wolfe(wolfe),
    minGradientNorm(minGradientNorm),
    factr(factr),
    maxLineSearchTrials(maxLineSearchTrials),
    minStep(minStep),
    maxStep(maxStep),
    terminate(false),
    usingScalarBounds(true)
{
  // Nothing to do.
}

template<typename MatType>
void L_BFGS_B::ProjectPoint(MatType& iterate)
{
  if (usingScalarBounds)
  {
    iterate.clamp(lowerBound(0), upperBound(0));
  }
  else if (!lowerBound.is_empty() && !upperBound.is_empty())
  {
    for (size_t i = 0; i < iterate.n_elem; ++i)
    {
      if (iterate[i] < lowerBound[i])
        iterate[i] = lowerBound[i];
      else if (iterate[i] > upperBound[i])
        iterate[i] = upperBound[i];
    }
  }
}

template<typename FunctionType,
         typename MatType,
         typename GradType,
         typename... CallbackTypes>
typename std::enable_if<IsMatrixType<GradType>::value,
    typename MatType::elem_type>::type
L_BFGS_B::Optimize(FunctionType& function,
                   MatType& iterateIn,
                   CallbackTypes&&... callbacks)
{
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;
  typedef typename MatTypeTraits<GradType>::BaseMatType BaseGradType;
  typedef Function<FunctionType, BaseMatType, BaseGradType> FullFunctionType;

  FullFunctionType& f = static_cast<FullFunctionType&>(function);
  traits::CheckFunctionTypeAPI<FullFunctionType, BaseMatType, BaseGradType>();
  RequireFloatingPointType<BaseMatType>();
  RequireFloatingPointType<BaseGradType>();
  RequireSameInternalTypes<BaseMatType, BaseGradType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;
  const size_t rows = iterate.n_rows;
  const size_t cols = iterate.n_cols;

  // Verify bound matrix sizes if not using scalar bounds.
  if (!usingScalarBounds)
  {
    if (!lowerBound.is_empty() && (lowerBound.n_rows != rows || lowerBound.n_cols != cols))
      throw std::invalid_argument("L_BFGS_B: lowerBound matrix size does not match iterate matrix size.");
    if (!upperBound.is_empty() && (upperBound.n_rows != rows || upperBound.n_cols != cols))
      throw std::invalid_argument("L_BFGS_B: upperBound matrix size does not match iterate matrix size.");
  }

  ProjectPoint(iterate);

  BaseMatType newIterateTmp(rows, cols);
  typedef typename ForwardType<MatType>::bcube BaseCubeType;
  BaseCubeType s(rows, cols, numBasis);
  BaseCubeType y(rows, cols, numBasis);

  BaseMatType oldIterate(rows, cols);
  oldIterate.zeros();
  bool optimizeUntilConvergence = (maxIterations == 0);

  BaseGradType gradient(rows, cols);
  gradient.zeros();
  BaseGradType oldGradient(rows, cols);
  oldGradient.zeros();
  BaseGradType searchDirection(rows, cols);
  searchDirection.zeros();

  typedef typename ForwardType<MatType>::bmat DenseMatType;

  ElemType functionValue = f.EvaluateWithGradient(iterate, gradient);
  terminate |= Callback::EvaluateWithGradient(*this, f, iterate,
        functionValue, gradient, callbacks...);

  ElemType prevFunctionValue;

  // L-BFGS-B specific variables.
  arma::mat W, M;
  size_t memorySlotsUsed = 0;

  Callback::BeginOptimization(*this, f, iterate, callbacks...);
  for (size_t itNum = 0; (optimizeUntilConvergence || (itNum != maxIterations))
      && !terminate; ++itNum)
  {
    prevFunctionValue = functionValue;

    // Check convergence: projected gradient norm
    // L-BFGS-B uses the infinity norm of the projected gradient.
    ElemType projectedGradNorm = 0.0;
    for (size_t i = 0; i < iterate.n_elem; ++i)
    {
      ElemType lb = usingScalarBounds ? lowerBound(0) :
                    (!lowerBound.is_empty() ? lowerBound[i] : -std::numeric_limits<ElemType>::infinity());
      ElemType ub = usingScalarBounds ? upperBound(0) :
                    (!upperBound.is_empty() ? upperBound[i] : std::numeric_limits<ElemType>::infinity());

      ElemType g = gradient[i];
      ElemType pgrad;
      if (iterate[i] <= lb + 1e-12 && g > 0)
        pgrad = 0;
      else if (iterate[i] >= ub - 1e-12 && g < 0)
        pgrad = 0;
      else
        pgrad = g;

      projectedGradNorm = std::max(projectedGradNorm, std::abs(pgrad));
    }

    if (projectedGradNorm < minGradientNorm)
    {
      Info << "L-BFGS-B: projected gradient norm too small (terminating successfully)."
          << std::endl;
      break;
    }

    if (std::isnan(functionValue))
    {
      Warn << "L-BFGS-B: objective value is NaN (terminating)!" << std::endl;
      break;
    }

    // Determine theta (L-BFGS scaling factor).
    // \theta = \frac{y_k^T y_k}{s_k^T y_k}
    ElemType theta = 1.0;
    if (itNum > 0)
    {
      int previousPos = (itNum - 1) % numBasis;
      const DenseMatType& sMat = s.slice(previousPos);
      const DenseMatType& yMat = y.slice(previousPos);
      ElemType sy = arma::dot(sMat, yMat);
      ElemType yy = arma::dot(yMat, yMat);
      if (sy > 1e-10 * yy)
        theta = yy / sy;
    }

    // 1. Cauchy Point Computation
    // The generalized Cauchy point x^c is defined as the first local minimizer
    // of the quadratic model along the projected gradient direction P(x - t * g, l, u).
    // Here we use an active-set projection approach over the free variables.
    searchDirection = gradient;
    arma::uvec isFree(iterate.n_elem, arma::fill::ones);

    for (size_t i = 0; i < iterate.n_elem; ++i)
    {
      ElemType lb = usingScalarBounds ? lowerBound(0) :
                    (!lowerBound.is_empty() ? lowerBound[i] : -std::numeric_limits<ElemType>::infinity());
      ElemType ub = usingScalarBounds ? upperBound(0) :
                    (!upperBound.is_empty() ? upperBound[i] : std::numeric_limits<ElemType>::infinity());

      if ((iterate[i] <= lb + 1e-12 && gradient[i] > 0) ||
          (iterate[i] >= ub - 1e-12 && gradient[i] < 0))
      {
        searchDirection[i] = 0;
        isFree[i] = 0;
      }
    }

    // 2. Reconstruct SearchDirection using the Subspace Minimization step.
    // L-BFGS two-loop recursion over the subspace of free variables to compute H_k \nabla f(x_k).
    if (memorySlotsUsed > 0)
    {
      arma::vec rho(numBasis, arma::fill::zeros);
      arma::vec alpha(numBasis, arma::fill::zeros);
      size_t limit = (numBasis > itNum) ? 0 : (itNum - numBasis);

      // First loop: \alpha_i = \rho_i s_i^T q, q = q - \alpha_i y_i
      for (size_t i = itNum; i != limit; i--)
      {
        int pos = (i + (numBasis - 1)) % numBasis;
        // Zero out bound variables from s and y computations to keep them in subspace
        DenseMatType sFree = s.slice(pos) % isFree;
        DenseMatType yFree = y.slice(pos) % isFree;

        ElemType sy = arma::dot(yFree, sFree);
        rho[itNum - i] = (sy != 0) ? (1.0 / sy) : 1.0;
        alpha[itNum - i] = rho[itNum - i] * arma::dot(sFree, searchDirection);
        searchDirection -= alpha[itNum - i] * yFree;
      }

      searchDirection /= theta;

      // Second loop: \beta_i = \rho_i y_i^T r, r = r + (\alpha_i - \beta_i) s_i
      for (size_t i = limit; i < itNum; i++)
      {
        int pos = i % numBasis;
        DenseMatType sFree = s.slice(pos) % isFree;
        DenseMatType yFree = y.slice(pos) % isFree;

        ElemType beta = rho[itNum - i - 1] * arma::dot(yFree, searchDirection);
        searchDirection += (alpha[itNum - i - 1] - beta) * sFree;
      }
    }

    // Fallback: if search direction isn't a descent direction, use steepest descent.
    if (arma::dot(searchDirection, gradient) <= 0)
      searchDirection = gradient % isFree;

    searchDirection *= -1;

    oldIterate = iterate;
    oldGradient = gradient;

    // 4. Projected Backtracking Line Search
    // Find step size \alpha that satisfies the Armijo condition for the projected point:
    // f(P(x_k + \alpha d_k, l, u)) \le f(x_k) + c_1 \nabla f(x_k)^T (P(x_k + \alpha d_k, l, u) - x_k)
    ElemType stepSize = 1.0;
    if (stepSize > ElemType(maxStep)) stepSize = ElemType(maxStep);
    if (stepSize < ElemType(minStep)) stepSize = ElemType(minStep);

    ElemType initialSearchDirectionDotGradient = arma::dot(gradient, searchDirection);
    ElemType initialFunctionValue = functionValue;

    size_t numIterations = 0;
    const ElemType dec = 0.5;

    while (true)
    {
      newIterateTmp = iterate + stepSize * searchDirection;
      ProjectPoint(newIterateTmp);

      functionValue = f.EvaluateWithGradient(newIterateTmp, gradient);

      if (std::isnan(functionValue))
      {
        Warn << "L-BFGS-B: objective value is NaN!" << std::endl;
        break;
      }

      terminate |= Callback::EvaluateWithGradient(*this, f, newIterateTmp,
          functionValue, gradient, callbacks...);

      // Armijo condition check
      // directional derivative = \nabla f(x_k)^T (x_{k+1} - x_k)
      // For projected search, this is <grad, x_new - x>
      ElemType actualMoveDotGrad = arma::dot(oldGradient, newIterateTmp - iterate);

      if (functionValue <= initialFunctionValue + armijoConstant * actualMoveDotGrad)
        break;

      if (stepSize < minStep || numIterations >= maxLineSearchTrials)
        break;

      stepSize *= dec;
      numIterations++;
    }

    if (arma::norm(newIterateTmp - iterate, "inf") == 0)
    {
      Info << "L-BFGS-B: step size is effectively zero (terminating successfully)." << std::endl;
      break;
    }

    iterate = newIterateTmp;

    const ElemType denom = std::max(ElemType(1),
        std::max(std::abs(prevFunctionValue), std::abs(functionValue)));
    if ((prevFunctionValue - functionValue) / denom <= factr)
    {
      Info << "L-BFGS-B: function value stable (terminating successfully)." << std::endl;
      break;
    }

    // 5. Update L-BFGS basis matrices
    // s_k = x_{k+1} - x_k
    // y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
    int overwritePos = itNum % numBasis;
    s.slice(overwritePos) = iterate - oldIterate;
    y.slice(overwritePos) = gradient - oldGradient;
    memorySlotsUsed = std::min(numBasis, memorySlotsUsed + 1);

    terminate |= Callback::StepTaken(*this, f, iterate, callbacks...);
  }

  Callback::EndOptimization(*this, f, iterate, callbacks...);
  return functionValue;
}

} // namespace ens

#endif // ENSMALLEN_LBFGSB_LBFGSB_IMPL_HPP
