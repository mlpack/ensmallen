/**
 * @file cmaes_impl.hpp
 * @author Marcus Edel
 * @author Kartik Nighania
 * @author John Hoang
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

template <typename SelectionPolicyType,
          typename WeightPolicyType,
          typename UpdatePolicyType>
CMAES<SelectionPolicyType, WeightPolicyType, UpdatePolicyType>::
CMAES(const size_t lambda,
      const double lowerBound,
      const double upperBound,
      const size_t batchSize,
      const size_t maxIterations,
      const double tolerance,
      const SelectionPolicyType& selectionPolicy,
      const WeightPolicyType& weightPolicy,
      const UpdatePolicyType& updatePolicy):
      lambda(lambda),
      lowerBound(lowerBound),
      upperBound(upperBound),
      batchSize(batchSize),
      maxIterations(maxIterations),
      tolerance(tolerance),
      selectionPolicy(selectionPolicy),
      weightPolicy(weightPolicy),
      updatePolicy(updatePolicy)
{ /* Nothing to do. */ }
  

//! Optimize the function (minimize).
template <typename SelectionPolicyType,
          typename WeightPolicyType,
          typename UpdatePolicyType>
template <typename SeparableFunctionType,
          typename MatType,
          typename... CallbackTypes>
typename MatType::elem_type 
    CMAES<SelectionPolicyType, WeightPolicyType, UpdatePolicyType>::
Optimize(SeparableFunctionType &function,
          MatType &iterateIn,
          CallbackTypes &&...callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  // Make sure that we have the methods that we need.  Long name...
  traits::CheckArbitrarySeparableFunctionTypeAPI<
      SeparableFunctionType, BaseMatType>();
  RequireDenseFloatingPointType<BaseMatType>();

  BaseMatType& iterate = (BaseMatType&) iterateIn;

  // Intantiated the algorithm parameters.
  double sigma = 0.3 * (upperBound - lowerBound);
  Initialize(iterate);

  BaseMatType mCandidate(iterate.n_rows, iterate.n_cols);
  
  mCandidate = lowerBound + arma::randu<BaseMatType>(
      iterate.n_rows, iterate.n_cols) * (upperBound - lowerBound);

  // Calculate the first objective function.
  size_t numFunctions = function.NumFunctions();
  ElemType currentObjective = 0;
  for (size_t f = 0; f < numFunctions; f += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
    const ElemType objective = function.Evaluate(mCandidate, f,
        effectiveBatchSize);
    currentObjective += objective;

    Callback::Evaluate(*this, function, mCandidate, objective,
        callbacks...);
  }

  ElemType overallObjective = currentObjective;
  ElemType lastObjective = std::numeric_limits<ElemType>::max();

  // Population parameters.
  std::vector<BaseMatType> z(lambda, 
      BaseMatType(iterate.n_rows, iterate.n_cols));
  std::vector<BaseMatType> y(lambda, 
      BaseMatType(iterate.n_rows, iterate.n_cols));
  // candidates is vector of x_i(g) with x_i(g) 
  // is either a column or row vector but we generalize the type of data here).
  std::vector<BaseMatType> candidates(lambda, 
      BaseMatType(iterate.n_rows, iterate.n_cols)); 
  BaseMatType pObjective(lambda, 1); // pObjective is vector-shaped.

  BaseMatType ps(iterate.n_rows, iterate.n_cols);
  ps.zeros();
  BaseMatType pc = ps;
  // Sep and Vd update parameters.
  BaseMatType sepCov(iterate.n_rows, iterate.n_cols, arma::fill::ones);
  BaseMatType sepCovinv = sepCov;
  // v ~ N(0, I/d).
  BaseMatType v = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols) / 
      std::sqrt(iterate.n_elem);

  // Vanilda update parameters.
  BaseMatType C = arma::eye<BaseMatType>(iterate.n_elem, iterate.n_elem);
  BaseMatType B = arma::eye<BaseMatType>(iterate.n_elem, iterate.n_elem);
  BaseMatType D = arma::eye<BaseMatType>(iterate.n_elem, iterate.n_elem);
  // The current visitation order (sorted by population objectives).
  arma::uvec idx = arma::linspace<arma::uvec>(0, lambda - 1, lambda);

  // Controls early termination of the optimization process.
  bool terminate = false;
  // Now iterate!
  terminate |= Callback::BeginOptimization(*this, function, iterate,
      callbacks...);
  for (size_t i = 1; i < maxIterations && !terminate; ++i)
  { 
    niter++;
    // Sampling population from current parameters.
    candidates = updatePolicy.SamplePop(sigma, lambda, iterate, z, y, 
        candidates, mCandidate, B, D, sepCovinv, sepCov, v, idx);

    // Evaluate the sampled canidate.
    for (size_t j = 0; j < lambda; ++j)
    {
      countval++;
      // Calculate the objective function.
      pObjective(idx(j)) = selectionPolicy.Select(function, batchSize,
          candidates[idx(j)], callbacks...);
    }

    // Sort population.
    idx = arma::sort_index(pObjective);

    mCandidate.zeros();
    // Update mCandidate.
    for (size_t j = 0; j < mu; ++j)
    {
      mCandidate += weights(j) * candidates[idx(j)];
    }
    // Calculate the objective function.
    currentObjective = selectionPolicy.Select(function, batchSize,
        mCandidate, callbacks...);

    // Update best candidate
    if (currentObjective < overallObjective)
    {
      overallObjective = currentObjective;
      iterate = mCandidate;

      terminate |= Callback::StepTaken(*this, function, iterate, callbacks...);
    }

    // Update new parameters for sampling distribution.
    Update(iterate, ps, pc, sigma, z, y, B, D, C, sepCovinv, sepCov, v, idx);

    // Output current objective function. So this is the termination criteria.
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

//! Initialize parameters.
template <typename SelectionPolicyType,
          typename WeightPolicyType,
          typename UpdatePolicyType>
template<typename MatType>
inline void CMAES<SelectionPolicyType, WeightPolicyType, UpdatePolicyType>::
Initialize(MatType& iterate)
{
  niter = 0;
  countval = 0;
  eigenval = 0;
  chi = std::sqrt(iterate.n_elem) * (1.0 - 1.0 / (4.0 * iterate.n_elem) + 
      1.0 / (21 * std::pow(iterate.n_elem, 2)));
  if(lambda == 0)
    lambda = (4 + std::round(3 * std::log(iterate.n_elem))) * 10;

  mu = std::round(lambda / 2);

  // Generate raw weights first to compute mueffective.
  weightPolicy.GenerateRaw(lambda);
  mueff = weightPolicy.MuEff();

  // Strategy parameter setting: Adaption.
  cc = (4.0 + mueff / iterate.n_elem) / 
      (4.0 + iterate.n_elem + 2 * mueff / iterate.n_elem);
  csigma = (mueff + 2.0) / (iterate.n_elem + mueff + 5.0);
  c1 = 2 / (std::pow(iterate.n_elem + 1.3, 2) + mueff);
  alphacov = 2.0;
  cmu = 2.0 * (mueff - 2.0 + 1.0 / mueff) / 
      (std::pow(iterate.n_elem + 2.0, 2) + alphacov * mueff / 2);
  cmu = std::min(1.0 - c1, cmu);

  // Finalize the weights vector by computed paramters.
  weights = weightPolicy.Generate(iterate.n_elem, c1, cmu);

  // Controlling.
  dsigma = 1 + csigma + 2 * 
      std::max(std::sqrt((mueff-1)/(iterate.n_elem+1)) - 1, 0.0);
  hsigma = (1.4 + 2.0 / (iterate.n_elem + 1.0)) * chi;
}

//! Update the algorithm's parameters.
template <typename SelectionPolicyType,
          typename WeightPolicyType,
          typename UpdatePolicyType>
template<typename MatType, typename BaseMatType>
inline void CMAES<SelectionPolicyType, WeightPolicyType, UpdatePolicyType>::
Update(MatType& iterate,
       BaseMatType& ps, 
       BaseMatType& pc, 
       double& sigma, 
       std::vector<BaseMatType>& z,
       std::vector<BaseMatType>& y,
       BaseMatType& B,
       BaseMatType& D,
       BaseMatType& C,
       BaseMatType& sepCovinv,
       BaseMatType& sepCov,  
       BaseMatType& v,
       arma::uvec& idx)
{     
  // Reusable variables.
  BaseMatType stepY(iterate.n_rows, iterate.n_cols);
  stepY.zeros();
  BaseMatType stepZ(iterate.n_rows, iterate.n_cols);
  stepZ.zeros();
  for (size_t j = 0; j < mu; ++j)
  {
    stepZ += weights(j) * z[idx(j)];
    stepY += weights(j) * y[idx(j)];
  }
  ps = updatePolicy.UpdatePs(iterate, ps, B, sepCovinv, 
      v, stepZ, stepY, mueff);

  double psNorm = arma::norm(ps);
  size_t hs = (psNorm < hsigma*sqrt(1.0 - 
      std::pow(1.0 - csigma, 2.0 * (niter + 1)))) ? 1 : 0;
  // Rescale parameters c1, cmu, csigma for vd-update.
  updatePolicy.RescaleParam(iterate, c1, cmu, csigma, mueff);
  // Update sigma.
  sigma = sigma * std::exp(csigma / dsigma * (psNorm / chi - 1));

  // Update pc.
  pc = updatePolicy.UpdatePc(cc, pc, hs, mueff, stepY);

  //Update covariance matrix
  updatePolicy.UpdateC(iterate, cc, c1, cmu, mueff, lambda, 
      hs, C, B, D, pc, idx, z, y, weights, sepCov, sepCovinv, v,
      eigenval, countval);
}

} // namespace ens

#endif
