/**
 * @file ipop_cmaes_impl.hpp
 * @author Marcus Edel
 * @author Suvarsha Chennareddy
 *
 * Implementation of the IPOP Covariance Matrix Adaptation Evolution Strategy
 * as proposed by A. Auger and N. Hansen in "A Restart CMA Evolution
 * Strategy With Increasing Population Size".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_IPOP_CMAES_IMPL_HPP
#define ENSMALLEN_CMAES_IPOP_CMAES_IMPL_HPP

// In case it hasn't been included yet.
#include "ipop_cmaes.hpp"
#include "count_function_evaluations.hpp"
#include <ensmallen_bits/function.hpp>

namespace ens {

template<typename CMAESType>
IPOPCMAES<CMAESType>::IPOPCMAES(const CMAESType& CMAES,
                                const double populationFactor,
                                const size_t maxRestarts) :
    cmaes(CMAES),
    populationFactor(populationFactor),
    maxRestarts(maxRestarts)
{ /* Nothing to do. */ }

template<typename CMAESType>
IPOPCMAES<CMAESType>::IPOPCMAES(const size_t lambda,
                                const typename CMAESType::transformationPolicyType&
                                      transformationPolicy,
                                const size_t batchSize,
                                const size_t maxIterations,
                                const double tolerance,
                                const typename CMAESType::selectionPolicyType&
                                      selectionPolicy,
                                double stepSize,
                                bool saveState,
                                const double populationFactor,
                                const size_t maxRestarts) :
    cmaes(lambda, transformationPolicy, batchSize, 
         maxIterations, tolerance, selectionPolicy, stepSize, saveState),
    populationFactor(populationFactor),
    maxRestarts(maxRestarts)
{ /* Nothing to do. */  }


//! Optimize the function (minimize).
template<typename CMAESType>
template<typename SeparableFunctionType,
         typename MatType,
         typename... CallbackTypes>
typename MatType::elem_type IPOPCMAES<CMAESType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{

  StoreBestCoordinates<MatType> sbc;
  CountFunctionEvaluations nfe;
  MatType iterate;
  for (int i = 0; i < maxRestarts; i++) {

    if (!cmaes.SaveState()) {
      // Use the starting point.
      iterate = iterateIn;

      // Optimize using the CMAES object.
      cmaes.Optimize(function, iterate, sbc, nfe, callbacks...);
    }
    else {
      // Optimize using the CMAES object.
      cmaes.Optimize(function, iterateIn, sbc, nfe, callbacks...);

      iterate = iterateIn;
    }

    // If the number of function evaluation exceeds the threshold, end 
    // the optimization.
    if (nfe.NumFunctionEvaluations() > iterateIn.n_elem * 10000) {
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      break;
    }

    // Increase the population size by the population factor for next restart.
    cmaes.PopulationSize() *= populationFactor;

  }

  // Store the best coordinates.
  iterateIn = sbc.BestCoordinates();
  // Return the best objective.
  return sbc.BestObjective();
}

} // namespace ens

#endif
