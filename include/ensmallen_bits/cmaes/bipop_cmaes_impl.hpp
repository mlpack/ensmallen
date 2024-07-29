/**
 * @file bipop_cmaes_impl.hpp
 * @author Benjami Parellada
 *
 * Implementation of the BIPOP Covariance Matrix Adaptation Evolution Strategy 
 * as proposed by N. Hansen in "Benchmarking a BI-population CMA-ES on the
 * BBOB-2009 function testbed".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef ENSMALLEN_CMAES_BIPOP_CMAES_IMPL_HPP
#define ENSMALLEN_CMAES_BIPOP_CMAES_IMPL_HPP

#include "bipop_cmaes.hpp"
#include <ensmallen_bits/function.hpp>
#include <random>

namespace ens {

template<typename CMAESType>
BIPOPCMAES<CMAESType>::BIPOPCMAES(const CMAESType& CMAES,
                                  const size_t maxRestarts,
                                  const size_t maxFunctionEvaluations,
                                  const size_t populationFactor) :
    cmaes(CMAES),
    maxRestarts(maxRestarts),
    maxFunctionEvaluations(maxFunctionEvaluations),
    populationFactor(populationFactor)
{ /* Nothing to do. */ }

template<typename CMAESType>
BIPOPCMAES<CMAESType>::BIPOPCMAES(const size_t lambda,
                                  const typename CMAESType::transformationPolicyType& transformationPolicy,
                                  const size_t batchSize,
                                  const size_t maxIterations,
                                  const double tolerance,
                                  const typename CMAESType::selectionPolicyType& selectionPolicy,
                                  double stepSize,
                                  const size_t maxRestarts,
                                  const size_t maxFunctionEvaluations,
                                  const size_t populationFactor) :
    cmaes(lambda, transformationPolicy, batchSize, maxIterations, tolerance, selectionPolicy, stepSize),
    maxRestarts(maxRestarts),
    maxFunctionEvaluations(maxFunctionEvaluations),
    populationFactor(populationFactor)
{ /* Nothing to do. */  }

template<typename CMAESType>
template<typename SeparableFunctionType, typename MatType, typename... CallbackTypes>
typename MatType::elem_type BIPOPCMAES<CMAESType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;
  typedef typename MatTypeTraits<MatType>::BaseMatType BaseMatType;

  StoreBestCoordinates<MatType> sbc;
  size_t totalFunctionEvaluations = 0;
  size_t largePopulationBudget = 0;
  size_t smallPopulationBudget = 0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // First single run with default population size
  MatType iterate = iterateIn;
  ElemType overallObjective = cmaes.Optimize(function, iterate, sbc, callbacks...);
  ElemType objective;
  size_t evaluations;

  size_t defaultLambda = cmaes.PopulationSize();
  size_t currentLargeLambda = defaultLambda;

  double stepSizeDefault = cmaes.StepSize();

  // Print out the default population size
  Info << "Default population size: " << defaultLambda << std::endl;

  size_t restart = 0;

  while (restart < maxRestarts)
  {
    if (largePopulationBudget <= smallPopulationBudget || restart == 0 || restart == maxRestarts - 1)
    {
      // Large population regime

      currentLargeLambda *= populationFactor;
      cmaes = CMAESType(currentLargeLambda, cmaes.TransformationPolicy(), cmaes.BatchSize(),
                         cmaes.MaxIterations(), cmaes.Tolerance(), cmaes.SelectionPolicy(), stepSizeDefault);

      std::cout << "BIPOP-CMA-ES: restart " << restart << ", large population size (lambda): " 
            << cmaes.PopulationSize() << std::endl;

      iterate = iterateIn;
      objective = cmaes.Optimize(function, iterate, sbc, callbacks...);

      evaluations = cmaes.FunctionEvaluations();
      largePopulationBudget += evaluations;
      ++restart;
    }
    else
    {
      // Small population regime
      double u = dis(gen);
      size_t smallLambda = static_cast<size_t>(defaultLambda * std::pow(0.5 * currentLargeLambda / defaultLambda, u * u));
      double stepSizeSmall = 2 * std::pow(10, -2*dis(gen));
      cmaes = CMAESType(smallLambda, cmaes.TransformationPolicy(), cmaes.BatchSize(),
                         cmaes.MaxIterations(), cmaes.Tolerance(), cmaes.SelectionPolicy(), stepSizeSmall);

      std::cout << "BIPOP-CMA-ES: restart " << restart << ", small population size (lambda): " 
            << cmaes.PopulationSize() << std::endl;

      iterate = iterateIn;
      objective = cmaes.Optimize(function, iterate, sbc, callbacks...);

      evaluations = cmaes.FunctionEvaluations();
      smallPopulationBudget += evaluations;
    }

    if (objective < overallObjective)
    {
      overallObjective = objective;
      Info << "BIPOP-CMA-ES: New best objective: " << overallObjective << std::endl;
    }

    totalFunctionEvaluations += evaluations;
    // Check if the total number of evaluations has exceeded the limit
    if (totalFunctionEvaluations >= maxFunctionEvaluations) {
      Warn << "BIPOP-CMA-ES: Maximum function overall evaluations reached. "
        << "terminating optimization." << std::endl;

      //iterate = transformationPolicy.Transform(iterate);
      Callback::EndOptimization(*this, function, iterate, callbacks...);
      iterateIn = sbc.BestCoordinates();
      return sbc.BestObjective();
    }
  }

  //iterate = transformationPolicy.Transform(iterate);
  Callback::EndOptimization(*this, function, iterate, callbacks...);
  iterateIn = sbc.BestCoordinates();
  return sbc.BestObjective();
}

} // namespace ens

#endif
