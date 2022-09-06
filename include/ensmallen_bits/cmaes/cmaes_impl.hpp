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
typename MatType::elem_type CMAES<SelectionPolicyType, WeightPolicyType, UpdatePolicyType>::
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
  // Intantiated the algorithm params
  initialize(iterate);

  double sigma;
  sigma = 0.3 * (upperBound - lowerBound);

  BaseMatType mCandidate(iterate.n_rows, iterate.n_cols);
  
  mCandidate = lowerBound + arma::randu<BaseMatType>(
      iterate.n_rows, iterate.n_cols) * (upperBound - lowerBound);

  // Calculate the first objective function.
  size_t numFunctions = function.NumFunctions();
  ElemType currentObjective = 0;
  for (size_t f = 0; f < numFunctions; f += batchSize)
  {
    const size_t effectiveBatchSize = std::min(batchSize, numFunctions - f);
    const ElemType objective = function.Evaluate(mCandidate[0], f,
        effectiveBatchSize);
    currentObjective += objective;

    Callback::Evaluate(*this, function, mCandidate[0], objective,
        callbacks...);
  }

  ElemType overallObjective = currentObjective;
  ElemType lastObjective = std::numeric_limits<ElemType>::max();

  // Population parameters.
  std::vector<BaseMatType> z(lambda, BaseMatType(iterate.n_rows, iterate.n_cols));
  std::vector<BaseMatType> y(lambda, BaseMatType(iterate.n_rows, iterate.n_cols));
  // candidates is vector of x_i(g) with x_i(g) 
  // is either a column or row vector but we generalize the type of data here)
  std::vector<BaseMatType> candidates(lambda, BaseMatType(iterate.n_rows, iterate.n_cols)); 
  BaseMatType pObjective(lambda, 1); // pObjective is vector-shaped.

  BaseMatType ps(iterate.n_rows, iterate.n_cols);
  ps.zeros();
  BaseMatType pc = ps;

  // Sep and Vd update parameters
  BaseMatType sepCov(iterate.n_rows, iterate.n_cols, fill:value(1.0));
  BaseMatType sepCovsqinv = sepCov;
  // v ~ N(0, I/d)
  BaseMatType v = arma::randn<BaseMatType>(iterate.n_rows, iterate.n_cols) / std::sqrt(iterate.n_elem);

  // Vanilda update parameters 
  BaseMatType C(iterate.n_elem, iterate.n_elem);
  C.eye();
  BaseMatType B(iterate.n_elem, iterate.n_elem); B.eye();
  BaseMatType D(iterate.n_elem, iterate.n_elem); D.eye();
  BaseMatType BD = B*D;

  // The current visitation order (sorted by population objectives).
  arma::uvec idx = arma::linspace<arma::uvec>(0, lambda - 1, lambda);

  // Controls early termination of the optimization process.
  bool terminate = false;
  size_t eigenval = 0;
  size_t countval = 0;
  // Now iterate!
  terminate |= Callback::BeginOptimization(*this, function, iterate,
      callbacks...);
  for (size_t i = 1; i < maxIterations && !terminate; ++i)
  { 
    candidates = updatePolicy.samplePop(sigma, lambda, iterate, z, y, mCandidate, 
        B, D, sepCovsqinv, sepCov, C);

    // Evaluate the sampled canidates
    for (size_t j = 0; j < lambda; ++j)
    {
      countval++;
      // Calculate the objective function.
      pObjective(idx(j)) = selectionPolicy.Select(function, batchSize,
          candidates[idx(j)], callbacks...);
    }
    // Sort population.
    idx = arma::sort_index(pObjective);

    mCandidate.zeros(); // Reset the mean
    // update mCandidate
    for (size_t j = 0; j < mu; ++j)
    {
      mCandidate += weights(j) * candidates[idx(j)];
    }
    // Calculate the objective function.
    currentObjective = selectionPolicy.Select(function, batchSize,
        mCandidate, callbacks...);

    // Update best parameters.
    if (currentObjective < overallObjective)
    {
      overallObjective = currentObjective;
      iterate = mCandidate;

      terminate |= Callback::StepTaken(*this, function, iterate, callbacks...);
    }

    // Update part which it used to placed
    update(iterate, ps, pc, sigma, z, y, mCandidate, B, D, sepCovsqinv, sepCov, idx, C);
    
    //! Eigen DecomCandidate covariance matrix C
    // Covariance matrix parameters.
    arma::Col<ElemType> eigval; // TODO: might need a more general type.
    BaseMatType eigvec;
    BaseMatType eigvalZero(iterate.n_elem, 1); // eigvalZero is vector-shaped.
    eigvalZero.zeros();

    if (countval - eigenval > lambda / ((c1 + cmu) * iterate.n_elem * 10))
    {
      eigenval = countval;
      C = arma::trimatu(C) + arma::trimatu(C).t();
      arma::eig_sym(eigval, eigvec, C);
      B = eigvec;
      D = arma::diagmat(arma::sqrt(eigval));
    }

    // Output current objective function. So this is the termination criteria 
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
  //! Initialize parameters
template <typename SelectionPolicyType,
          typename WeightPolicyType,
          typename UpdatePolicyType>
template<typename MatType>
inline void CMAES<SelectionPolicyType, WeightPolicyType, UpdatePolicyType>::
initialize(MatType& iterate)
{
  chi = std::sqrt(iterate.n_elem)*(1.0 - 1.0 / (4.0 * iterate.n_elem) + 
      1.0 / (21 * std::pow(iterate.n_elem, 2)));
  if(lambda == 0)
    lambda = (4 + std::round(3 * std::log(iterate.n_elem))) * 10;

  mu = std::round(lambda / 2);

  weightPolicy.GenerateRaw(lambda);
  mu_eff = weightPolicy.MuEff();

  // Strategy parameter setting: Adaption
  cc = (4.0 + mu_eff / iterate.n_elem) / (4.0 + iterate.n_elem + 2 * mu_eff / iterate.n_elem);
  csigma = (mu_eff + 2.0) / (iterate.n_elem + mu_eff + 5.0);
  c1 = 2 / (std::pow(iterate.n_elem + 1.3, 2) + mu_eff);
  alphacov = 2.0;
  cmu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / 
      (std::pow(iterate.n_elem + 2.0, 2) + alphacov * mu_eff / 2);
  cmu = std::min(1.0 - c1, cmu);

  weights = weightPolicy.Generate(iterate.n_elem, c1, cmu);

  // Controlling
  dsigma = 1 + csigma + 2 * 
      std::max(std::sqrt((mu_eff-1)/(iterate.n_elem+1)) - 1, 0.0);
  hsigma = (1.4 + 2.0 / (iterate.n_elem + 1.0)) * chi;

}

//! Update the algorithm's parameters
template <typename SelectionPolicyType,
          typename WeightPolicyType,
          typename UpdatePolicyType>
template<typename MatType, typename BaseMatType>
inline void CMAES<SelectionPolicyType, WeightPolicyType, UpdatePolicyType>::
update(MatType& iterate,
       BaseMatType& ps, 
       BaseMatType& pc, 
       BaseMatType& sigma, 
       std::vector<BaseMatType>& z,
       std::vector<BaseMatType>& y,
       BaseMatType& B,
       BaseMatType& D,
       BaseMatType& sepCovsqinv,
       BaseMatType& sepCov,  
       arma::uvec& idx
       BaseMatType& C)
{     
  // Reusable variables
  BaseMatType step(iterate.n_rows, iterate.n_cols); 
  step.zeros();
  BaseMatType stepz(iterate.n_rows, iterate.n_cols);
  stepz.zeros();

  for (size_t j = 0; j < mu; ++j)
  {
    step += weights(j) * y[idx(j)];
    stepz += weights(j) * z[idx(j)];
  }

  ps = updatePolicy.updatePs(iterate, ps, B, stepz, mu_eff);

  double psNorm = arma::norm(ps);
  size_t hs = (psNorm / sqrt(1 - std::pow(1 - csigma, 2 * i)) < hsigma) ? 1 : 0;

  // Update sigma.
  sigma = sigma * std::exp(csigma / dsigma * (psNorm / chi - 1));

  // Update pc 
  pc = updatePolicy.updatePc(cc, pc, hs, mu_eff, step);

  //Update covariance matrix
  C = updatePolicy.updateC(iterate, cc, c1, cmu, mu_eff, lambda, 
      hs, C, pc, idx, z, y, weights);
}

} // namespace ens

#endif
