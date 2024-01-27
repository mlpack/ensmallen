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
                  const double gamma,
                  const double stepSize,
                  const double evaluationStepSize,
                  const size_t maxIterations,
                  const double tolerance) :
    alpha(alpha),
    gamma(gamma),
    stepSize(stepSize),
    evaluationStepSize(evaluationStepSize),
    ak(0.001 * maxIterations),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }

template<typename ArbitraryFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type SPSA::Optimize(ArbitraryFunctionType& function,
                                           MatType& iterate,
                                           CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  // Make sure that we have the methods that we need.
  traits::CheckArbitraryFunctionTypeAPI<ArbitraryFunctionType,
      MatType>();
  RequireFloatingPointType<MatType>();

  BaseMatType gradient(iterate.n_rows, iterate.n_cols);
  arma::Mat<ElemType> spVector(iterate.n_rows, iterate.n_cols);

  // To keep track of where we are and how things are going.
  ElemType overallObjective = 0;
  ElemType lastObjective = DBL_MAX;

  // Controls early termination of the optimization process.
  bool terminate = false;

  Callback::BeginOptimization(*this, function, iterate, callbacks...);
  for (size_t k = 0; k < maxIterations && !terminate; ++k)
  {
    // Output current objective function.
    Info << "SPSA: iteration " << k << ", objective " << overallObjective
        << "." << std::endl;

    if (std::isnan(overallObjective) || std::isinf(overallObjective))
    {
      Warn << "SPSA: converged to " << overallObjective << "; terminating"
          << " with failure.  Try a smaller step size?" << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    if (std::abs(lastObjective - overallObjective) < tolerance)
    {
      Warn << "SPSA: minimized within tolerance " << tolerance << "; "
          << "terminating optimization." << std::endl;
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      return overallObjective;
    }

    // Reset the counter variables.
    lastObjective = overallObjective;

    // Gain sequences.
    const double akLocal = stepSize / std::pow(k + 1 + ak, alpha);
    const double ck = evaluationStepSize / std::pow(k + 1, gamma);

    // Choose stochastic directions.
    spVector = arma::conv_to<arma::Mat<ElemType>>::from(
        arma::randi(iterate.n_rows, iterate.n_cols,
        arma::distr_param(0, 1))) * 2 - 1;

    iterate += ck * spVector;
    const double fPlus = function.Evaluate(iterate);
    terminate |= Callback::Evaluate(*this, function, iterate, fPlus,
        callbacks...);

    iterate -= 2 * ck * spVector;
    const double fMinus = function.Evaluate(iterate);
    terminate |= Callback::Evaluate(*this, function, iterate, fMinus,
        callbacks...);

    iterate += ck * spVector;
    if (terminate)
      break;

    gradient = (fPlus - fMinus) * (1 / (2 * ck * spVector));
    iterate -= akLocal * gradient;

    terminate |= Callback::StepTaken(*this, function, iterate, callbacks...);

    overallObjective = function.Evaluate(iterate);
    terminate |= Callback::Evaluate(*this, function, iterate, overallObjective,
        callbacks...);
  }

  // Calculate final objective.  The optimization is over, so the result of the
  // callback doesn'tatter.
  const ElemType objective = function.Evaluate(iterate);
  (void) Callback::Evaluate(*this, function, iterate, objective, callbacks...);

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  return objective;
}

} // namespace ens

#endif
