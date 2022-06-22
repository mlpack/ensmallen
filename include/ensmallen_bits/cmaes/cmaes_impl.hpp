/**
 * @file cmaes_impl.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 * @author John Hoang
 * Implementation of the Covariance Matrix Adaptation Evolution Strategy as
 * proposed by N. Hansen et al. in "Completely Derandomized Self-Adaptation in
 * Evolution Strategies".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_CMAES_IMPL_HPP
#define ENSMALLEN_CMAES_CMAES_IMPL_HPP

// In case it hasn't been included yet.
#include "cmaes.hpp"
#include "cmaparameters.hpp"
#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename SelectionPolicyType>
CMAES<SelectionPolicyType>::CMAES(const size_t lambda,
                                  const double lowerBound,
                                  const double upperBound,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const size_t negativeWeight,
                                  const SelectionPolicyType& selectionPolicy) :
    lambda(lambda),
    lowerBound(lowerBound),
    upperBound(upperBound),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    negativeWeight(negativeWeight),
    selectionPolicy(selectionPolicy)
{ /* Nothing to do. */ }

//! Optimize the function (minimize).
template<typename SelectionPolicyType>
template<typename SeparableFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type CMAES<SelectionPolicyType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitrarySeparableFunctionTypeAPI<
      SeparableFunctionType, BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;
  // Intantiated the algorithm params
  CMAparameters<MatType> params(iterate.n_elem, lambda, negativeWeight);

  BaseMatType sigma(2, 1); // sigma is vector-shaped.
  sigma(0) = 0.3 * (upperBound - lowerBound);

  std::vector<BaseMatType> mPosition(2, BaseMatType(iterate.n_rows, iterate.n_cols));
  mPosition[0] = lowerBound + arma::randu<BaseMatType>(
      iterate.n_rows, iterate.n_cols) * (upperBound - lowerBound);

  BaseMatType step(iterate.n_rows, iterate.n_cols);
  step.zeros();

  // Calculate the first objective function.
  size_t numFunctions = function.NumFunctions();
  ElemType currentObjective = 0;
  for (size_t f = 0; f < numFunctions; f += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
    const ElemType objective = function.Evaluate(mPosition[0], f,
        effectiveBatchSize);
    currentObjective += objective;

    Callback::Evaluate(*this, function, mPosition[0], objective,
        callbacks...);
  }

  ElemType overallObjective = currentObjective;
  ElemType lastObjective = std::numeric_limits<ElemType>::max();

  // Population parameters.
  std::vector<BaseMatType> pStep(params.lambda, BaseMatType(iterate.n_rows, iterate.n_cols));
  /**< pPosition is vector of x_i(g) with x_i(g) is either a column or row vector but we generalize the type of data here) */
  std::vector<BaseMatType> pPosition(params.lambda, BaseMatType(iterate.n_rows, iterate.n_cols)); 
  BaseMatType pObjective(params.lambda, 1); // pObjective is vector-shaped.
  std::vector<BaseMatType> ps(2, BaseMatType(iterate.n_rows, iterate.n_cols));
  ps[0].zeros();
  ps[1].zeros();
  std::vector<BaseMatType> pc = ps;
  std::vector<BaseMatType> C(2, BaseMatType(params.dim, params.dim));
  C[0].eye();

  // Covariance matrix parameters.
  arma::Col<ElemType> eigval; // TODO: might need a more general type.
  BaseMatType eigvec;
  BaseMatType eigvalZero(params.dim, 1); // eigvalZero is vector-shaped.
  eigvalZero.zeros();

  // The current visitation order (sorted by population objectives).
  arma::uvec idx = arma::linspace<arma::uvec>(0, params.lambda - 1, params.lambda);

  // Controls early termination of the optimization process.
  bool terminate = false;

  // Now iterate!
  terminate |= Callback::BeginOptimization(*this, function, iterate,
      callbacks...);
  for (size_t i = 1; i < maxIterations && !terminate; ++i)
  {
    // To keep track of where we are.
    const size_t idx0 = (i - 1) % 2;
    const size_t idx1 = i % 2;

    // Perform Cholesky decomposition. If the matrix is not positive definite,
    // add a small value and try again.
    BaseMatType covLower;
    while (!arma::chol(covLower, C[idx0], "lower"))
      C[idx0].diag() += std::numeric_limits<ElemType>::epsilon();
    std::vector<BaseMatType> z(params.lambda, BaseMatType(iterate.n_rows, iterate.n_cols));
    for (size_t j = 0; j < params.lambda; ++j)
    {
      if (iterate.n_rows > iterate.n_cols)
      {
        z[j] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
        pStep[idx(j)] = covLower * z[j];
      }
      else
      {
        z[j] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
        pStep[idx(j)] = z[j] * covLower;
      }

      pPosition[idx(j)] = mPosition[idx0] + sigma(idx0) * pStep[idx(j)];

      // Calculate the objective function.
      pObjective(idx(j)) = selectionPolicy.Select(function, batchSize,
          pPosition[idx(j)], callbacks...);
    }

    // Sort population.
    idx = arma::sort_index(pObjective);
    step = params.weights(0) * pStep[idx(0)];
    for (size_t j = 1; j < params.mu; ++j)
      step += params.weights(j) * pStep[idx(j)];

    mPosition[idx1] = mPosition[idx0] + sigma(idx0) * step;

    // Calculate the objective function.
    currentObjective = selectionPolicy.Select(function, batchSize,
        mPosition[idx1], callbacks...);

    // Update best parameters.
    if (currentObjective < overallObjective)
    {
      overallObjective = currentObjective;
      iterate = mPosition[idx1];

      terminate |= Callback::StepTaken(*this, function, iterate, callbacks...);
    }

    // Update Step Size.
    if (iterate.n_rows > iterate.n_cols)
    {
      ps[idx1] = (1 - params.csigma) * ps[idx0] + std::sqrt(
          params.csigma * (2 - params.csigma) * params.muw) * covLower.t() * step;
    }
    else
    {
      ps[idx1] = (1 - params.csigma) * ps[idx0] + std::sqrt(
          params.csigma * (2 - params.csigma) * params.muw) * step * covLower.t();
    }

    const ElemType psNorm = arma::norm(ps[idx1]);
    const size_t hs = (psNorm / sqrt(1 - std::pow(1 - params.csigma, 2 * i)) < params.hsigma) ? 1 : 0;
    const double deltahs = (1 - hs) * params.cc * (2 - params.cc);

    // Update covariance matrix.
    sigma(idx1) = sigma(idx0) * std::exp(params.csigma / params.dsigma * ( psNorm / params.chi - 1));
    pc[idx1] = (1 - params.cc) * pc[idx0] + hs * std::sqrt(params.cc * (2 - params.cc) * params.muw) * step; 
    C[idx1] = (1 + params.c1 * deltahs - params.c1 - params.cmu * arma::accu(params.weights)) * C[idx0];
  
    if (iterate.n_rows > iterate.n_cols)
    {
      C[idx1] = C[idx1] + params.c1 * (pc[idx1] * pc[idx1].t());
      for (size_t j = 0; j < params.offsprings; ++j)
      {
        if(params.weights(j) < 0) params.weights(j) *= params.dim/std::pow(arma::norm(z[j]), 2);
        C[idx1] = C[idx1] + params.cmu * params.weights(j) *
            pStep[idx(j)] * pStep[idx(j)].t();
      }
    }
    else
    {
      C[idx1] = C[idx1] + params.c1 * (pc[idx1].t() * pc[idx1]);
      for (size_t j = 0; j < params.offsprings; ++j)
      {
        if(params.weights(j) < 0) params.weights(j) *= params.dim/std::pow(arma::norm(z[j]), 2);
        C[idx1] = C[idx1] + params.cmu * params.weights(j) *
            pStep[idx(j)].t() * pStep[idx(j)];
      }
    }

    arma::eig_sym(eigval, eigvec, C[idx1]);
    const arma::uvec negativeEigval = arma::find(eigval < 0, 1);
    if (!negativeEigval.is_empty())
    {
      if (negativeEigval(0) == 0)
      {
        C[idx1].zeros();
      }
      else
      {
        C[idx1] = eigvec.cols(0, negativeEigval(0) - 1) *
            arma::diagmat(eigval.subvec(0, negativeEigval(0) - 1)) *
            eigvec.cols(0, negativeEigval(0) - 1).t();
      }
    }

    // Output current objective function.
    Info << "CMA-ES: iteration " << i << ", objective " << overallObjective
        << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "CMA-ES: converged to " << overallObjective << "; "
          << "terminating with failure.  Try a smaller step size?" << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Info << "CMA-ES: minimized within tolerance " << tolerance << "; "
          << "terminating optimization." << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    lastObjective = overallObjective;
  }

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return overallObjective;
}

} // namespace ens

#endif
