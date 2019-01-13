/**
 * @file spsa.hpp
 * @author N Rajiv Vaidyanathan
 * @author Marcus Edel
 *
 * SPSA (Simultaneous perturbation stochastic approximation)
 * update for faster convergence.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_SPSA_SPSA_IMPL_HPP
#define ENSMALLEN_SPSA_SPSA_IMPL_HPP

// In case it hasn't been included yet.
#include "spsa.hpp"

#include <ensmallen_bits/function.hpp>

namespace ens {

inline SPSA::SPSA(const double alpha,
                  const size_t batchSize,
                  const double gamma,
                  const double stepSize,
                  const double evaluationStepSize,
                  const size_t maxIterations,
                  const double tolerance,
                  const bool shuffle) :
    alpha(alpha),
    batchSize(batchSize),
    gamma(gamma),
    stepSize(stepSize),
    evaluationStepSize(evaluationStepSize),
    Ak(0.001 * maxIterations),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle)
{ /* Nothing to do. */ }

template<typename DecomposableFunctionType>
inline double SPSA::Optimize(
    DecomposableFunctionType& function, arma::mat& iterate)
{
  // Make sure that we have the methods that we need.
  traits::CheckNonDifferentiableDecomposableFunctionTypeAPI<
      DecomposableFunctionType>();

  arma::mat gradient(iterate.n_rows, iterate.n_cols);
  arma::mat spVector(iterate.n_rows, iterate.n_cols);

  // To keep track of where we are and how things are going.
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  const size_t actualMaxIterations = (maxIterations == 0) ?
      std::numeric_limits<size_t>::max() : maxIterations;
  for (size_t k = 0; k < actualMaxIterations; /* incrementing done manually */)
  {
    // Is this iteration the start of a sequence?
    if (k > 0)
    {
      // Output current objective function.
      Info << "SPSA: iteration " << k << ", objective " << overallObjective
          << "." << std::endl;

      if (std::isnan(overallObjective) || std::isinf(overallObjective))
      {
        Warn << "SPSA: converged to " << overallObjective << "; terminating"
            << " with failure.  Try a smaller step size?" << std::endl;
        return overallObjective;
      }

      if (std::abs(lastObjective - overallObjective) < tolerance)
      {
        Info << "SPSA: minimized within tolerance " << tolerance << "; "
            << "terminating optimization." << std::endl;
        return overallObjective;
      }

      // Reset the counter variables.
      lastObjective = overallObjective;

      if (shuffle) // Determine order of visitation.
        function.Shuffle();
    }

    // Gain sequences.
    const double ak = stepSize / std::pow(k + 1 + Ak, alpha);
    const double ck = evaluationStepSize / std::pow(k + 1, gamma);

    gradient.zeros();
    for (size_t b = 0; b < batchSize; b++)
    {
      // Stochastic directions.
      spVector = arma::conv_to<arma::mat>::from(
          arma::randi(iterate.n_rows, iterate.n_cols,
          arma::distr_param(0, 1))) * 2 - 1;

      iterate += ck * spVector;
      const double fPlus = function.Evaluate(iterate, 0, iterate.n_elem);

      iterate -= 2 * ck * spVector;
      const double fMinus = function.Evaluate(iterate, 0, iterate.n_elem);
      iterate += ck * spVector;

      gradient += (fPlus - fMinus) * (1 / (2 * ck * spVector));
    }

    gradient /= (double) batchSize;
    iterate -= ak * gradient;

    overallObjective = function.Evaluate(iterate, 0, iterate.n_elem);
    k += batchSize;
  }

  // Calculate final objective.
  return function.Evaluate(iterate, 0, iterate.n_elem);
}

} // namespace ens

#endif
