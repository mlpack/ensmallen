/**
 * @file ipop_cmaes_impl.hpp
 * @author Marcus Edel
 * @author Benjami Parellada
 *
 * Implementation of the IPOP Covariance Matrix Adaptation Evolution Strategy
 * as proposed by A. Auger and N. Hansen in "A Restart CMA Evolution
 * Strategy With Increasing Population Size" and BIPOP Covariance Matrix
 * Adaptation Evolution Strategy as proposed by N. Hansen in "Benchmarking 
 * a BI-population CMA-ES on the BBOB-2009 function testbed".
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_CMAES_POP_CMAES_IMPL_HPP
#define ENSMALLEN_CMAES_POP_CMAES_IMPL_HPP

#include "pop_cmaes.hpp"
#include <ensmallen_bits/function.hpp>
#include <random>

namespace ens {

template<typename SelectionPolicyType, typename TransformationPolicyType>
POPCMAES<SelectionPolicyType, TransformationPolicyType>::POPCMAES(
    const size_t lambda,
    const TransformationPolicyType& transformationPolicy,
    const size_t batchSize,
    const size_t maxIterations,
    const double tolerance,
    const SelectionPolicyType& selectionPolicy,
    double stepSize,
    const double populationFactor,
    const size_t maxRestarts,
    const size_t maxFunctionEvaluations,
    const bool useBIPOP) :
    CMAES<SelectionPolicyType, TransformationPolicyType>(
        lambda, transformationPolicy, batchSize, maxIterations,
        tolerance, selectionPolicy, stepSize),
    populationFactor(populationFactor),
    maxRestarts(maxRestarts),
    maxFunctionEvaluations(maxFunctionEvaluations),
    useBIPOP(useBIPOP)
{ /* Nothing to do. */ }

template<typename SelectionPolicyType, typename TransformationPolicyType>
template<typename SeparableFunctionType, typename MatType, typename... CallbackTypes>
typename MatType::elem_type POPCMAES<SelectionPolicyType, TransformationPolicyType>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
  // Convenience typedefs.
  typedef typename MatType::elem_type ElemType;

  StoreBestCoordinates<MatType> sbc;
  StoreBestCoordinates<MatType> overallSBC;
  size_t totalFunctionEvaluations = 0;
  size_t largePopulationBudget = 0;
  size_t smallPopulationBudget = 0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // First single run with default population size
  MatType iterate = iterateIn;
  ElemType overallObjective = CMAES<SelectionPolicyType, 
      TransformationPolicyType>::Optimize(function, iterate, sbc, 
                                          callbacks...);
  ElemType objective;
  size_t evaluations;

  size_t defaultLambda = this->PopulationSize();
  size_t currentLargeLambda = defaultLambda;

  double stepSizeDefault = this->StepSize();

  // Print out the default population size
  Info << "Default population size: " << defaultLambda << std::endl;

  size_t restart = 0;

  while (restart < maxRestarts)
  {
    if (!useBIPOP || largePopulationBudget <= smallPopulationBudget || restart == 0 || 
        restart == maxRestarts - 1)
    {
      // Large population regime (IPOP or BIPOP)
      currentLargeLambda *= populationFactor;
      this->PopulationSize() = currentLargeLambda;
      this->StepSize() = stepSizeDefault;

      Info << "POP-CMA-ES: restart " << restart << ", large population size" <<
          " (lambda): " << this->PopulationSize() << std::endl;
      
      iterate = iterateIn;

      // Optimize using the CMAES object.
      objective = CMAES<SelectionPolicyType, 
          TransformationPolicyType>::Optimize(function, iterate, sbc, 
          callbacks...);

      evaluations = this->FunctionEvaluations();
      largePopulationBudget += evaluations;
    }
    else if (useBIPOP)
    {
      // Small population regime (BIPOP only)
      double u = dis(gen);
      size_t smallLambda = static_cast<size_t>(defaultLambda * std::pow(0.5 * 
          currentLargeLambda / defaultLambda, u * u));
      double stepSizeSmall = 2 * std::pow(10, -2*dis(gen));
      
      this->PopulationSize() = smallLambda;
      this->StepSize() = stepSizeSmall;

      Info << "BIPOP-CMA-ES: restart " << restart << ", small population" <<
          " size (lambda): " << this->PopulationSize() << std::endl;

      iterate = iterateIn;
      
      // Optimize using the CMAES object.
      objective = CMAES<SelectionPolicyType, 
          TransformationPolicyType>::Optimize(function, iterate, sbc, 
                                              callbacks...);

      evaluations = this->FunctionEvaluations();
      smallPopulationBudget += evaluations;
    }

    if (objective < overallObjective)
    {
      overallObjective = objective;
      overallSBC = sbc;
      Info << "POP-CMA-ES: New best objective: " << overallObjective << std::endl;
    }

    totalFunctionEvaluations += evaluations;
    // Check if the total number of evaluations has exceeded the limit
    if (totalFunctionEvaluations >= maxFunctionEvaluations) {
      Warn << "POP-CMA-ES: Maximum function overall evaluations reached. "
           << "terminating optimization." << std::endl;

      Callback::EndOptimization(*this, function, iterate, callbacks...);
      iterateIn = overallSBC.BestCoordinates();
      return overallSBC.BestObjective();
    }

    ++restart;
  }

  Callback::EndOptimization(*this, function, iterate, callbacks...);
  iterateIn = overallSBC.BestCoordinates();
  return overallSBC.BestObjective();
}

} // namespace ens

#endif