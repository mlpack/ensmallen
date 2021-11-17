/**
 * @file cmaes_impl.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 *
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

#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename SelectionPolicyType>
CMAES<SelectionPolicyType>::CMAES(const size_t lambda,
                                  const double lowerBound,
                                  const double upperBound,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const SelectionPolicyType& selectionPolicy) :
    lambda(lambda),
    lowerBound(lowerBound),
    upperBound(upperBound),
    batchSize(batchSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
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

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // Population size.
  if (lambda == 0)
    lambda = (4 + std::round(3 * std::log(iterate.n_elem))) * 10;

  // Parent weights.
  const size_t mu = std::round(lambda / 2);
  BaseMatType w = std::log(mu + 0.5) - arma::log(
      arma::linspace<BaseMatType>(0, mu - 1, mu) + 1.0);
  w /= arma::accu(w);

  // Number of effective solutions.
  const double muEffective = 1 / arma::accu(arma::pow(w, 2));

  // Step size control parameters.
  BaseMatType sigma(2, 1); // sigma is vector-shaped.
  sigma(0) = 0.3 * (upperBound - lowerBound);
  const double cs = (muEffective + 2) / (iterate.n_elem + muEffective + 5);
  const double ds = 1 + cs + 2 * std::max(std::sqrt((muEffective - 1) /
      (iterate.n_elem + 1)) - 1, 0.0);
  const double enn = std::sqrt(iterate.n_elem) * (1.0 - 1.0 /
      (4.0 * iterate.n_elem) + 1.0 / (21 * std::pow(iterate.n_elem, 2)));

  // Covariance update parameters.
  // Cumulation for distribution.
  const double cc = (4 + muEffective / iterate.n_elem) /
      (4 + iterate.n_elem + 2 * muEffective / iterate.n_elem);
  const double h = (1.4 + 2.0 / (iterate.n_elem + 1.0)) * enn;

  const double c1 = 2 / (std::pow(iterate.n_elem + 1.3, 2) + muEffective);
  const double alphaMu = 2;
  const double cmu = std::min(1 - c1, alphaMu * (muEffective - 2 + 1 /
      muEffective) / (std::pow(iterate.n_elem + 2, 2) +
      alphaMu * muEffective / 2));

  std::vector<BaseMatType> mPosition(2, BaseMatType(iterate.n_rows,
      iterate.n_cols));
  mPosition[0] = lowerBound + arma::randu<BaseMatType>(
      iterate.n_rows, iterate.n_cols) * (upperBound - lowerBound);

  BaseMatType step(iterate.n_rows, iterate.n_cols);
  step.zeros();

  // Calculate the first objective function.
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
  std::vector<BaseMatType> pStep(lambda, BaseMatType(iterate.n_rows,
      iterate.n_cols));
  std::vector<BaseMatType> pPosition(lambda, BaseMatType(iterate.n_rows,
      iterate.n_cols));
  BaseMatType pObjective(lambda, 1); // pObjective is vector-shaped.
  std::vector<BaseMatType> ps(2, BaseMatType(iterate.n_rows, iterate.n_cols));
  ps[0].zeros();
  ps[1].zeros();
  std::vector<BaseMatType> pc = ps;
  std::vector<BaseMatType> C(2, BaseMatType(iterate.n_elem, iterate.n_elem));
  C[0].eye();

  // Covariance matrix parameters.
  arma::Col<ElemType> eigval; // TODO: might need a more general type.
  BaseMatType eigvec;
  BaseMatType eigvalZero(iterate.n_elem, 1); // eigvalZero is vector-shaped.
  eigvalZero.zeros();

  // The current visitation order (sorted by population objectives).
  arma::uvec idx = arma::linspace<arma::uvec>(0, lambda - 1, lambda);

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

    for (size_t j = 0; j < lambda; ++j)
    {
      if (iterate.n_rows > iterate.n_cols)
      {
        pStep[idx(j)] = covLower *
            arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols);
      }
      else
      {
        pStep[idx(j)] = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols)
            * covLower;
      }

      pPosition[idx(j)] = mPosition[idx0] + sigma(idx0) * pStep[idx(j)];

      // Calculate the objective function.
      pObjective(idx(j)) = selectionPolicy.Select(function, batchSize,
          pPosition[idx(j)], callbacks...);
    }

    // Sort population.
    idx = arma::sort_index(pObjective);

    step = w(0) * pStep[idx(0)];
    for (size_t j = 1; j < mu; ++j)
      step += w(j) * pStep[idx(j)];

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
      ps[idx1] = (1 - cs) * ps[idx0] + std::sqrt(
          cs * (2 - cs) * muEffective) * covLower.t() * step;
    }
    else
    {
      ps[idx1] = (1 - cs) * ps[idx0] + std::sqrt(
          cs * (2 - cs) * muEffective) * step * covLower.t();
    }

    const ElemType psNorm = arma::norm(ps[idx1]);
    sigma(idx1) = sigma(idx0) * std::exp(cs / ds * ( psNorm / enn - 1));

    // Update covariance matrix.
    if ((psNorm / sqrt(1 - std::pow(1 - cs, 2 * i))) < h)
    {
      pc[idx1] = (1 - cc) * pc[idx0] + std::sqrt(cc * (2 - cc) *
        muEffective) * step;

      if (iterate.n_rows > iterate.n_cols)
      {
        C[idx1] = (1 - c1 - cmu) * C[idx0] + c1 *
          (pc[idx1] * pc[idx1].t());
      }
      else
      {
        C[idx1] = (1 - c1 - cmu) * C[idx0] + c1 *
          (pc[idx1].t() * pc[idx1]);
      }
    }
    else
    {
      pc[idx1] = (1 - cc) * pc[idx0];

      if (iterate.n_rows > iterate.n_cols)
      {
        C[idx1] = (1 - c1 - cmu) * C[idx0] + c1 * (pc[idx1] *
            pc[idx1].t() + (cc * (2 - cc)) * C[idx0]);
      }
      else
      {
        C[idx1] = (1 - c1 - cmu) * C[idx0] + c1 *
            (pc[idx1].t() * pc[idx1] + (cc * (2 - cc)) * C[idx0]);
      }
    }

    if (iterate.n_rows > iterate.n_cols)
    {
      for (size_t j = 0; j < mu; ++j)
      {
        C[idx1] = C[idx1] + cmu * w(j) *
            pStep[idx(j)] * pStep[idx(j)].t();
      }
    }
    else
    {
      for (size_t j = 0; j < mu; ++j)
      {
        C[idx1] = C[idx1] + cmu * w(j) *
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
